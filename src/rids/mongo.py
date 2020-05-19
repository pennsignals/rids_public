"""Mongo.

"""

from __future__ import annotations
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from re import sub
from sys import stdout
from time import sleep as block
from uuid import uuid4

from bson.objectid import ObjectId
import pandas as pd
import numpy as np

from pymongo import MongoClient
from pymongo.errors import AutoReconnect

from .configurable import Configurable
from .constants import (
    BACKOFF,
    RETRIES,
)

try:
    from pandas import (
        DataFrame,
        isnull,
    )
except ImportError:
    DataFrame = None
    isnull = None


basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


def retry_on_reconnect(retries=RETRIES, backoff=BACKOFF):
    """Retry decorator for AutoReconnect.

    The pymongo client does not retry operations even when
        mongo reports auto reconnect is available.
    """
    def wrapper(func):
        """Return wrapped method."""
        @wraps(func)
        def wrapped(*args, **kwargs):
            """Return method result."""
            try:
                return func(*args, **kwargs)
            # coverage only with a replicaset and failover.
            except AutoReconnect as retry:
                logger.exception(retry)
                for i in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except AutoReconnect:
                        block(backoff * i)
                raise
        return wrapped
    return wrapper


class Mongo(Configurable):
    """Mongo."""

    @classmethod
    def from_cfg(cls, cfg) -> Output:
        """Return output from cfg."""
        collection = cfg['collection']
        collection_cls_name = '_Collection' + uuid4().hex

        class Collection(namedtuple(  # pylint: disable=too-few-public-methods
                collection_cls_name, collection.keys())):
            """Collection."""

            @classmethod
            def from_cfg(cls, cfg: dict) -> object:
                """Return collection from cfg."""
                return cls(**cfg)

        print("** in mongo.py Mongo from_cfg")
        # print(cfg)
        print('>>>cfg<<<')
        print([(key, len(str(value))) for key, value in cfg.items()])
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (
                ('uri', str),
                ('collection', Collection.from_cfg),
            )}
        return cls(**kwargs)

    def __init__(
            self,
            uri: str,
            collection: namedtuple) -> None:
        """Initialize Mongo."""
        self.uri = uri
        self._collection = collection

    @contextmanager
    def collections(self) -> None:
        """Contextmanager for collection."""
        with self.database() as database:
            kwargs = {
                key: database[value]
                for key, value in self._collection._asdict().items()
            }
            yield self._collection.__class__(**kwargs)

    @contextmanager
    def connection(self) -> None:
        """Contextmanager for connection."""
        with MongoClient(self.uri) as connection:
            logger.info('{"mongo": "open"}')
            try:
                yield connection
            finally:
                logger.info('{"mongo": "close"}')

    # @contextmanager
    # def my_open(filename):
    #     fin = open(filename)
    #     try:
    #         yield fin
    #     finally:
    #         fin.close()

    @contextmanager
    def database(self) -> None:
        """Contextmanager for database."""
        with self.connection() as connection:
            database = connection.get_database()
            logger.info(
                '{"mongo.open": {"database": "%s"}}',
                database.name)
            try:
                yield database
            finally:
                logger.info(
                    '{"mongo.closed": {"database": "%s"}}',
                    database.name)

    def ping(self) -> bool:
        """Ping mongo."""
        try:
            with self.database() as database:
                # is this lazy?
                # ideally a conmmand that gets some mongo status from the server
                return True
        except:
            return False


class Output(Mongo):
    """Mongo output."""

    @classmethod
    def df_to_bsonable(cls, df: DataFrame) -> list:
        """Convert a DataFrame into a mongo document that can be serialized."""
        for c in df:
            df[c] = df[c].astype('O')
            idb = isnull(df[c])
            df.loc[idb, (c)] = None

            if '.' in str(c):
                new_col = sub(r'\.', r'_dot_', str(c))
                df[new_col] = df[c]
                del df[c]

        lstDictionary = df.to_dict(orient='records')
        return [cls.fix_encoding(dictionary) for dictionary in lstDictionary]

    @classmethod
    def dict_to_bsonable(cls, dictionary: dict) -> dict:
        """Convert dict to a bsonable dict that can be serialized."""
        # print("in dict_to_bsonable")
        # print(list(dictionary.keys()))
        # TODO: Where might d[c] be None? Is it because mongo can't deal with null?
        for key in list(dictionary.keys()):
            # if isnull(d[c]):
            if dictionary[key]!=dictionary[key]:
                dictionary[key] = None
            if '.' in str(key):
                new_col = sub(r'\.', r'_dot_', str(key))
                dictionary[new_col] = dictionary[key]
                del dictionary[key]
        return cls.fix_encoding(dictionary)

    @classmethod
    def fix_encoding(cls, dictionary):
        """
        From: https://stackoverflow.com/a/57830500
        Correct the encoding of python dictionaries so they can be encoded to mongodb
        """
        new = {}
        for key1, val1 in dictionary.items():
            # Nested dictionaries
            if isinstance(val1, dict):
                val1 = cls.fix_encoding(val1)

            if isinstance(val1, np.bool_):
                val1 = bool(val1)

            if isinstance(val1, np.int64):
                val1 = int(val1)

            if isinstance(val1, np.float64):
                val1 = float(val1)

            new[key1] = val1
        return new

    @retry_on_reconnect()
    def insert_one_dict(
            self,
            collection,
            dictionary):
        """Insert one dict."""
        datum = self.dict_to_bsonable(dictionary)
        collection.insert_one(datum)  # datum is a dictionary
        return datum

    @retry_on_reconnect()
    def insert_many_dict(  # pylint: disable=no-self-use
            self,
            collection,
            dictionary) -> list:
        """Insert many dicts."""
        data = [self.fix_encoding(dictionary) for key, dictionary in dictionary.items()]
        collection.insert_many(data)
        return data

    @retry_on_reconnect()
    def insert_many_df(self, collection, df) -> list:
        """Insert many df."""
        data = self.df_to_bsonable(df) # data is list of dictionaries
        collection.insert_many(data)
        return data


class Input(Mongo):
    """Mongo input.""" 

    # ARGS = {
    #     ('INPUT_URI', '-input-uri'): {
    #         'dest': 'input_uri',
    #         'help': 'Mongo input uri.',
    #         'type': str,
    #     },
    # }

    @classmethod
    def get_wholeCollection(cls, collection) -> pd.DataFrame:
        return pd.DataFrame(list(collection.find({})))

    @classmethod
    def get_batchQueryLatest(cls, collection, mg_query=None, dt_now=None, bDebug=False) -> dict:
        if mg_query is None:
            mg_query = {}
        if dt_now is not None:
            mg_query['_id'] = {'$lt': ObjectId.from_datetime(dt_now)}
        if bDebug: print("mg_query = " + str(mg_query))
        if collection.count_documents(mg_query) == 0:
            return {}
        batch_latest = collection.find(mg_query).sort('_id', -1).limit(1)[0]
        return batch_latest


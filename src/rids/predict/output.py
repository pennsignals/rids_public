"""Output."""

from __future__ import annotations
from argparse import Namespace
from collections import namedtuple
from datetime import datetime, timedelta
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from sys import stdout
from uuid import uuid4

import numpy as np
import pandas as pd
from yaml import safe_load as yaml_loads

from ..mongo import (
    Output as BaseMongoOutput,
    retry_on_reconnect,
)

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class Output(BaseMongoOutput):
    """Output."""

    ARGS = {
        ('OUTPUT_URI', '--output-uri'): {
            'dest': 'output_uri',
            'help': 'Mongo output uri.',
            'type': str,
        },
    }

    @classmethod
    def from_cfg(cls, cfg: dict) -> Output:
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

        kwargs = {
            key: cast(cfg[key])
            for key, cast in (
                ('uri', str),
                ('collection', Collection.from_cfg),
            )}
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.output_uri),):
            if value is not None:
                cfg[key] = value
        return cfg

    def __call__(self, idfs: namedtuple, tdfs: namedtuple, pdfs: namedtuple, dt_now: datetime) -> None:
        """Emit output dfs."""
        self.dt_now = dt_now
        self.insert_predictions(pdfs.output, pdfs.model)

        # FYI:
        # pdfs = Batch_pdfs(input=df_prediction_input,
        #                   output=df_prediction_output,
        #                   model=self)
        # model
        # self.clf = path['clf']
        # self.features = path['features']
        # self.version = path['version']

    def insert_predictions(self, df_predictions, model):
        # TODO: CHANGE MODEL and MICROSERVICE VERSIONS
        """Write predictions to mongo."""
        batch = {'created_on': self.dt_now,
                 'model_version': model.version,
                 'microservice_version': '1.2'}
        with self.collections() as collections:
            # Create a dummy batch
            batch = self.insert_one_dict(collections.prediction_batches, batch)  # dict, not dataframe
            #  batch is now: {'_id': ObjectId('...')}
            # print("In insert_predictions")
            # print(batch)
            df_predictions['batch_id'] = batch_id = batch['_id']
            self.insert_many_df(collections.predictions, df_predictions)  # dataframe
        return batch_id

    def ping(self) -> bool:
        """Ping output.

        Ensure that the output is online.
        Aquire any startup state needed here:
            max_batch_id, last_inserted_date, etc.
        It is ok to read from outputs,
            but use a separate input if needed for local testing.
        """
        with self.collections() as collections:
            print(collections)
        return True



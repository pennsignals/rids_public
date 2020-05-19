"""PS1 Input.
Used to get
- labs
"""

from __future__ import annotations
import datetime

from argparse import Namespace
from collections import namedtuple
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from bson.objectid import ObjectId
import re
from sys import stdout
from pandas import DataFrame
import pandas as pd
from yaml import safe_load as yaml_loads

from ..mongo import (
    Input as BaseInput,
    retry_on_reconnect,
    )

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class Input(BaseInput):
    """Input."""

    ARGS = {
        ('ORDER_INPUT_URI', '--order-input-uri'): {
            'dest': 'order_input_uri',
            'help': 'Order input uri.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.order_input_uri),):
            if value is not None:
                cfg[key] = value
            else:
                logger.info("In order_input.py Input patch_args: %s is missing.", key)

        return cfg

    def __call__(self, df_i_cohort, dt_now) -> pd.DataFrame:
        """Return a Batch of input dfs.
        """
        self.df_i_cohort = df_i_cohort.drop_duplicates(subset='CSN').copy()
        with self.collections() as collections:
            df_i_orders = self.get_orders(collections.lab_orders, dt_now)

        return df_i_orders

    # def ping(self) -> bool:
    #     """Ping mongo."""
    #     try:
    #         with self.database() as database:
    #             return True
    #     except:
    #         return False

    def get_orders(self, collection, dt_now):
        df_i_cohort = self.df_i_cohort

        df_i_orders = pd.DataFrame()
        print("size: " + str(df_i_cohort.shape))
        for index, row in df_i_cohort.iterrows():
            if index % 50 == 0:
                print(str(datetime.datetime.now()) + '\t' + str(index))
            dt_end = dt_now
            obj_end = ObjectId.from_datetime(dt_end)
            dt_start = row['dt_visitStart_UTC']
            obj_start = ObjectId.from_datetime(dt_start)

            mg_query = {'UID': int(row['UID']),
                        '_id': {'$lt': obj_end, '$gt': obj_start},
                        # 'recorded_on': {'$gt': nTimeStart},
                        }
            sortOrder = (('_id', -1), ('OrderDate', -1))

            df_temp_ind = pd.DataFrame(list(collection.find(mg_query)))

            if df_temp_ind.shape[0] == 0:
                df_temp_ind = pd.DataFrame(columns=['CSN'], data=[row['CSN']])
            else:
                df_temp_ind['CSN'] = row['CSN']
            #             df_rows = pd.concat([pd.DataFrame(row).T[['CSN']], df_temp_ind], axis='columns')

            df_i_orders = pd.concat([df_i_orders, df_temp_ind], axis='rows', sort=False)

        df_i_orders = df_i_orders.reset_index(drop=True)

        return df_i_orders
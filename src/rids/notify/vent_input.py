"""Vent Input.
Used to get
- patient list,
- demographics,
- vitals
"""

from __future__ import annotations
from datetime import datetime, timezone

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
import pandas as pd
from time import mktime
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

Batch = namedtuple('Batch_ventInput', ('cohort','demographics','vitals'))

strTZ = 'US/Eastern'
dt_epoch = datetime(1970, 1, 1, 0, tzinfo=timezone.utc)

class Input(BaseInput):
    """Input."""

    ARGS = {
        ('VENT_INPUT_URI', '--vent-input-uri'): {
            'dest': 'vent_input_uri',
            'help': 'Vent Mongo input uri.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.vent_input_uri),):
            if value is not None:
                cfg[key] = value

        return cfg

    def __call__(self, df_i_cohort, dt_now) -> pd.DataFrame:
        """Return a Batch (a namedtuple) of input dfs.
        """
        self.df_i_cohort = df_i_cohort
        with self.collections() as collections:
            # lstintPatientUID = [int(x) for x in list(df_i_cohort['UID'].unique())]
            # print(len(lstintPatientUID))
            # df_i_mar = self.get_MAR(collections.signal_mar_events, lstintPatientUID, dt_now)
            df_i_mar = self.get_MAR(collections.signal_mar_events, dt_now)

        return df_i_mar

    # def ping(self) -> bool:
    #     """Ping mongo."""
    #     try:
    #         with self.database() as database:
    #             return True
    #     except:
    #         return False

    def get_MAR(self, collection, dt_now) -> pd.DataFrame:
        df_i_cohort = self.df_i_cohort.drop_duplicates(subset='CSN').copy()

        df_i_mar = pd.DataFrame()
        print("size: " + str(df_i_cohort.shape))
        for index, row in df_i_cohort.iterrows():
            if index % 50 == 0:
                print(str(datetime.now()) + '\t--------------' + str(index))
            # Want the MAR between the visit start and the current time
            nTimeStart = (row['dt_visitStart_UTC'] - dt_epoch).total_seconds()
            dt_end = dt_now
            obj_end = ObjectId.from_datetime(dt_end)

            mg_query = {'_id': {'$lt': obj_end},
                        'patient_id': int(row['UID']),
                        'recorded_on': {'$gt': nTimeStart},
                        }

            df_temp_ind = pd.DataFrame(list(collection.find(mg_query)))

            if df_temp_ind.shape[0] == 0:
                df_temp_ind = pd.DataFrame(columns=['CSN'], data=[row['CSN']])
            else:
                df_temp_ind['CSN'] = row['CSN']
            df_i_mar = pd.concat([df_i_mar, df_temp_ind], axis='rows', sort=False)

        df_i_mar = df_i_mar.reset_index(drop=True)

        return df_i_mar

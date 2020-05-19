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

Batch = namedtuple('Batch_ps1Input', 'labs')


class Input(BaseInput):
    """Input."""

    ARGS = {
        ('PS1_INPUT_URI', '--ps1-input-uri'): {
            'dest': 'ps1_input_uri',
            'help': 'PS1 Mongo input uri.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.ps1_input_uri),):
            if value is not None:
                cfg[key] = value

        return cfg

    def __call__(self, df_i_cohort, dt_now) -> tuple:
        """Return a Batch of input dfs.
        """
        lstintPatientUID = list(df_i_cohort['UID'].unique())
        with self.collections() as collections:
            df_i_labs = self.get_labs(collections.lab_results, dt_now,
                                      lstintPatientUID, td_lookback=pd.Timedelta(hours=6))

        return Batch(labs=df_i_labs)

    def ping(self) -> bool:
        """Ping mongo."""
        try:
            with self.database() as database:
                return True
        except:
            return False

    def get_labs(self, collection, dt_now, lstintPatientUID, td_lookback):
        # TODO: Currently testing using 6 hour lookback
        dt_end = dt_now
        dt_start = dt_end - td_lookback
        obj_start = ObjectId.from_datetime(dt_start)
        obj_end = ObjectId.from_datetime(dt_end)

        mg_query = {'_id': {'$gt': obj_start, '$lt': obj_end},
                    'UID': {'$in': [int(x) for x in lstintPatientUID]}}
        sortOrder = (('_id', -1), ('OrderDate', -1))

        temp = collection.find(mg_query).sort(sortOrder)
        df_i_labs = pd.DataFrame.from_dict(list(temp))

        return df_i_labs

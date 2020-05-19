"""PS1 Input.
Used to get
- labs
"""

from __future__ import annotations
from datetime import datetime, timezone
from dateutil.tz import gettz

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

Batch = namedtuple('Batch_predictionInput', ('outstandingMessages', 'cohort', 'notifications',
                                             'assignments', 'prediction_batch_latest'))


def convertUTCcolumns(df):
    for strCol in [x for x in list(df) if x.endswith('_UTC')]:
        df[strCol] = df[strCol].apply(lambda x: x.replace(tzinfo=timezone.utc))
    return df


class Input(BaseInput):
    """Input."""

    ARGS = {
        ('PREDICTION_INPUT_URI', '--prediction-input-uri'): {
            'dest': 'prediction_input_uri',
            'help': 'Prediction Mongo input uri.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        print("** in prediction_input.py Input patch_args")
        print([(key, len(str(value))) for key, value in cfg.items()])
        for key, value in (
                ('uri', args.prediction_input_uri),):
            if value is not None:
                cfg[key] = value
            else:
                logger.info("In prediction_input.py Input patch_args: %s is missing.", key)

        return cfg

    def __call__(self, dt_now) -> Batch:
        """Return a Batch of input dfs.
        """
        with self.collections() as collections:
            df_i_outstandingMessages = self.get_outstandingMessages(collections.outstanding_twilio)
            prediction_batch_latest = self.get_predictionBatchLatest(collections.prediction_batches, dt_now)

            df_i_cohort = self.get_cohort(collections.predictions, prediction_batch_latest)
            df_i_notifications = self.get_notifications(collections.notifications, df_i_cohort, dt_now)
            df_i_assignments = self.get_assignments(collections.assignments)

        return Batch(outstandingMessages=df_i_outstandingMessages,
                     cohort=df_i_cohort,
                     notifications=df_i_notifications,
                     assignments=df_i_assignments,
                     prediction_batch_latest=prediction_batch_latest
                     )

    def ping(self) -> bool:
        # TODO ping randomizations
        """Ping mongo."""
        try:
            with self.collections() as collections:
                # collections.randomization...count > 0
                return True
        except:
            return False

    def get_outstandingMessages(self, collection):
        """Just pulling whole table; It doesn't make sense to implement a time cutoff"""
        return self.get_wholeCollection(collection)

    def is_matching_latestPredictNotifyBatches(self, dt_now):
        with self.collections() as collections:
            prediction_batch_latest = self.get_predictionBatchLatest(collections.prediction_batches, dt_now)
            notification_batch_latest = self.get_notificationBatchLatest(collections.notification_batches, dt_now)

        if (len(notification_batch_latest) == 0) or (len(prediction_batch_latest) == 0):
            return False
        elif notification_batch_latest['prediction_batch_id'] == prediction_batch_latest['_id']:
            return True
        return False

    def get_predictionBatchLatest(self, collection, dt_now):
        return self.get_batchQueryLatest(collection, mg_query={'created_on': {'$lt': dt_now}})

    def get_notificationBatchLatest(self, collection, dt_now):
        return self.get_batchQueryLatest(collection, mg_query={'created_on': {'$lt': dt_now}})

    def get_cohort(self, collection, prediction_batch_latest) -> pd.DataFrame:
        # nNumTest = 50
        prediction_batchID_latest = prediction_batch_latest['_id']
        mg_query = {'batch_id': prediction_batchID_latest}
        temp = collection.find(mg_query)#.limit(nNumTest)
        df_i_cohort = pd.DataFrame.from_dict(list(temp))
        df_i_cohort = convertUTCcolumns(df_i_cohort)
        return df_i_cohort

    def get_notifications(self, collection, df_i_cohort, dt_now):
        lstCSN = list(df_i_cohort['CSN'].unique())
        mg_query = {'created_on': {'$lt': dt_now},
                    'CSN': {'$in': lstCSN}}

        df_i_notifications = pd.DataFrame(list(collection.find(mg_query)))
        if df_i_notifications.shape[0] == 0:
            df_i_notifications = pd.DataFrame(columns=['CSN'])

        return df_i_notifications

    def get_assignments(self, collection):
        """Just pulling whole table; It doesn't make sense to implement a time cutoff"""
        return self.get_wholeCollection(collection)

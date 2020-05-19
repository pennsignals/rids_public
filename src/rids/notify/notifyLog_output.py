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
    Input as BaseMongoInput,
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
        ('NOTIFY_OUTPUT_URI', '--notify-output-uri'): {
            'dest': 'notify_output_uri',
            'help': 'Notify Mongo output uri.',
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
        if cfg is None:
            cfg = {}
        for key, value in (
                ('uri', args.notify_output_uri),):
            if value is not None:
                cfg[key] = value
            else:
                logger.info("In notifyLog_output.py Output patch_args: %s is missing.", key)

        return cfg

    def __call__(self, notifs, df_notifications) -> namedtuple:
        """Insert logs into Mongo collection."""
        # notifs = Batch(notification_batch=notification_batch,
        #                df_notifications=df_notifications,
        #                lstCSN_notifyCandidate=lstCSN_notifyCandidate,
        #                lstCSN_notify=lstCSN_notify, )
        notification_batch = notifs.notification_batch

        with self.collections() as collections:
            notificationlog_batch = {'created_on': notification_batch['created_on'],
                                     'notification_batch_id': notification_batch['_id'],
                                     'prediction_batch_id': notification_batch['prediction_batch_id'],
                                     'notification_logs':self.df_to_bsonable(df_notifications)}
            self.insert_one_dict(collections.notification_logs, notificationlog_batch)
            # self.insert_many_df(collections.notification_logs, df_notifications)

    def ping(self) -> bool:
        with self.collections() as collections:
            print(collections)
        return True

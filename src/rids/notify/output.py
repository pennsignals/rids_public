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

from .notifyLog_output import Output as NotifyLogOutput

from ..configurable import Configurable
from ..mongo import (
    Output as BaseMongoOutput,
    retry_on_reconnect,
)

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class Output(Configurable):
    """Output."""

    ARGS = {
        **NotifyLogOutput.ARGS,
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # print("**in output.py Output patch_args")
        # print(args)
        # print(cfg)
        # print(cfg.keys())
        # for each other yml file, open and load/replace cfg[key]
        """Patch cfg from args."""
        for key, patch_args_function in (('notificationLog', NotifyLogOutput.patch_args),
                                         ):
            cfg[key] = patch_args_function(args, cfg.get(key))

        return cfg

    @classmethod
    def from_cfg(cls, cfg: dict) -> Output:
        """Return Model from cfg."""
        # print("**in output.py Output from_cfg")
        # print(cfg)
        # E.g. key:value :: 'slack':SlackWebhook.from_cfg(cfg['slack'])
        # E.g. key:value :: 'twilio_output':TwilioOutput.from_cfg(cfg['twilio_output'])
        kwargs = {key: from_cfg_function(cfg[key])
                  for key, from_cfg_function in (('notificationLog', NotifyLogOutput.from_cfg),
                                                 )
                  }

        return cls(**kwargs)

    def __init__(self,
                 notificationLog):
        self.notificationLog_output = notificationLog

    def __call__(self,
                 notifyBatch,
                 df_notifications,
                 lstNotificationResults,
                 lstErrors):
        """Emit output dfs."""
        self.notificationLog_output(notifyBatch, df_notifications)
        pass

    def ping(self) -> bool:
        return self.notificationLog_output.ping()

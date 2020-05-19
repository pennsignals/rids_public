"""RedCap."""

from __future__ import annotations
from argparse import Namespace
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
# from re import sub
from sys import stdout
# from time import sleep as block
# from uuid import uuid4

from requests import post
from json import dumps
from datetime import datetime, timezone
# import pandas as pd

from ..configurable import Configurable
# from constants import (
#     BACKOFF,
#     RETRIES,
# )

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


# def convertUTCcolumns(df):
#     for strCol in [x for x in list(df) if x.endswith('_UTC')]:
#         df[strCol] = df[strCol].apply(lambda x: x.replace(tzinfo=timezone.utc))
#     return df


# def responseToDF(response):
#     return pd.DataFrame(response.json()['Results'])

class Output(Configurable):
    """RedCap."""

    ARGS = {
        ('REDCAP_API_URI', '--redcap_api_uri'): {
            'dest': 'redcap_api_uri',
            'help': 'RedCap API URI.',
            'type': str,
        },
        ('REDCAP_TOKEN', '--redcap_api_token'): {
            'dest': 'redcap_api_token',
            'help': 'RedCap client token.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # print("** in redcap_output.py Input patch_args")
        # print(args)
        # print(cfg)
        if cfg is None:
            cfg = {}
        for key, value in (('api_uri', args.redcap_api_uri),
                           ('api_token', args.redcap_api_token),
                           ):
            if value is not None:
                cfg[key] = value
            elif cfg.get(key) is None:
                logger.info("In redcap_output.py Output patch_args: %s is missing.", key)

        return cfg

    @classmethod
    def from_cfg(cls, cfg):
        """Return output from cfg."""
        # print("** in redcap_output.py Input from_cfg")
        # print(cfg)
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (('api_uri', str),
                              ('api_token', str),
                              )}
        return cls(**kwargs)

    def __init__(
            self,
            api_uri: str,
            api_token: str,
    ) -> None:
        """Initialize RedCap."""
        self.api_uri = api_uri
        self.token = api_token

    def __call__(self, df_notify, dt_now):
        """.
        """
        # self.df_notify = df_notify.drop_duplicates(subset='CSN').copy()

        return

    def createRecord(self, dictCSNbody):
        """These are all fields in the Message Acknowledgement form"""
        record = {
            'record_id': 1,  # must have this, but if forceAutoNumber is True, it'll override whatever this is
            'message_received': 0,
            'facility': {'HUP': 0, 'PMC': 1}[dictCSNbody['facility']], # Quick and dirty mapping
            'location': dictCSNbody['location'],
            'last_name': dictCSNbody['last_name'],
            'csn': dictCSNbody['csn'],
            'uid': dictCSNbody['uid'],
            'created_on': dictCSNbody['created_on'],
            'logging': dictCSNbody['logging'],
        }
        return {key: str(record[key]) for key in record.keys()}

    def postRecord(self, dictCSNbody):
        record = self.createRecord(dictCSNbody)
        data = {
            'token': self.token,
            'content': 'record',
            'format': 'json',
            'type': 'flat',
            'data': dumps([record]),
            'forceAutoNumber': True,
            'returnContent': 'auto_ids',
            'returnFormat': 'json',
        }
        response = post(url=self.api_uri, data=data)
        # print(response.text)
        return response


"""Mongo.

"""

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

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from .configurable import Configurable
from .constants import (
    BACKOFF,
    RETRIES,
)

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class TwilioClient(Configurable):
    """TwilioClient."""

    ARGS = {
        ('TWILIO_CLIENT_SID', '--twilio_sid'): {
            'dest': 'twilio_sid',
            'help': 'Twilio account SID.',
            'type': str,
        },
        ('TWILIO_CLIENT_AUTH', '--twilio_token'): {
            'dest': 'twilio_token',
            'help': 'Twilio auth token.',
            'type': str,
        },
        ('TWILIO_FROM_NUMBER', '--twilio_from_number'): {
            'dest': 'twilio_from_number',
            'help': 'Twilio from number.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # print("** in twilio.py TwilioClient patch_args")
        # print(args)
        # print(cfg)
        if cfg is None:
            cfg = {}
        for key, value in (('sid', args.twilio_sid),
                           ('token', args.twilio_token),
                           ('from_number', args.twilio_from_number)
                           ):
            if value is not None:
                cfg[key] = value
        return cfg

    @classmethod
    def from_cfg(cls, cfg) -> TwilioClient:
        """Return output from cfg."""
        # print("** in twilio.py TwilioClient from_cfg")
        # print(cfg)
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (('sid', str),
                              ('token', str),
                              ('from_number', str)
                              )}
        return cls(**kwargs)

    def __init__(
            self,
            sid: str,
            token: str,
            from_number: str) -> None:
        """Initialize Twilio client."""
        self.sid = sid
        self.token = token
        self.from_number = from_number
        self._twilio_client = Client(sid, token)

    def status(self, sid):
        return self._twilio_client.messages(sid).fetch().status

    def send(self, to, from_, body):
        return self._twilio_client.messages.create(to=to, from_=from_, body=body)

    def getListMessages(self, **kwargs) -> list:
        """from_ = strFrom, date_sent = dt_after, limit = nLimit"""
        return self._twilio_client.messages.list(kwargs)

    # @property
    # def from_number(self):
    #     return self._from_number
    #
    # @from_number.setter
    # def from_number(self, value):
    #     self._from_value = value

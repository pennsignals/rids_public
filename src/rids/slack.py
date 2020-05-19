"""Slack.
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

import json
import requests

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




class SlackWebhook(Configurable):
    ARGS = {
        ('SLACK_ALERT_HUP_URL', '--slack-alert-hup-url'): {
            'dest': 'slack_alert_hup_url',
            'help': 'Slack alert HUP url.',
            'type': str,
        },
        ('SLACK_ALERT_PMC_URL', '--slack-alert-pmc-url'): {
            'dest': 'slack_alert_pmc_url',
            'help': 'Slack alert PMC url.',
            'type': str,
        },
        ('SLACK_STARTUP_URL', '--slack-startup-url'): {
            'dest': 'slack_startup_url',
            'help': 'Slack startup url.',
            'type': str,
        },
    }

    @classmethod
    def from_cfg(cls, cfg) -> SlackWebhook:
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (
                ('startup_url', str),
                ('alert_hup_url', str),
                ('alert_pmc_url', str),
            )}
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # for each other yml file, open and load/replace cfg[key]
        """Patch cfg from args."""
        if cfg is None:
            cfg = {}
        print("**in slack.py SlackWebhook patch_args")
        for key, value in (('startup_url', args.slack_startup_url),
                           ('alert_hup_url', args.slack_alert_hup_url),
                           ('alert_pmc_url', args.slack_alert_pmc_url),
                           ):
            if value is not None:
                cfg[key] = value
            else:
                logger.info("In slack.py SlackWebhook patch_args: %s is missing.", key)

        return cfg

    def __init__(
            self,
            startup_url: str,
            alert_hup_url: str,
            alert_pmc_url: str,
            ) -> None:
        """Initialize Slack Webhook."""
        # print("DO I EVER GET HERE???")
        self.startup_url = startup_url
        self.alert_hup_url = alert_hup_url
        self.alert_pmc_url = alert_pmc_url
        # self.slack_send(startup_url)

    def getHospitalURL(self, strHospital):
        if strHospital == 'HUP':
            return self.alert_hup_url
        elif strHospital == 'PMC':
            return self.alert_pmc_url
        return self.startup_url

    def slack_send(self, slack_url, strMessage="test message"):
        data = json.dumps({"text": strMessage})
        # slack_url = "https://hooks.slack.com/services/*********/*********/********************"
        try:
            response = requests.post(slack_url, data, timeout=2)
            return {'response': response.text}
        except:
            return {'error': 404}

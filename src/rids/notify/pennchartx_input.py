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

from requests import post
from datetime import datetime, timezone
import pandas as pd

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


def convertUTCcolumns(df):
    for strCol in [x for x in list(df) if x.endswith('_UTC')]:
        df[strCol] = df[strCol].apply(lambda x: x.replace(tzinfo=timezone.utc))
    return df


def responseToDF(response):
    return pd.DataFrame(response.json()['Results'])

class Input(Configurable):
# class PennChartX(Configurable):
    """PennChartX."""

    ARGS = {
        ('PENNCHARTX_CLIENT_ID', '--pennchartx_client_id'): {
            'dest': 'pennchartx_client_id',
            'help': 'PennChartX client ID.',
            'type': str,
        },
        ('PENNCHARTX_CLIENT_SECRET', '--pennchartx_client_secret'): {
            'dest': 'pennchartx_client_secret',
            'help': 'PennChartX client secret.',
            'type': str,
        },
        ('PENNCHARTX_USER_NAME', '--pennchartx_user_name'): {
            'dest': 'pennchartx_user_name',
            'help': 'PennChartX user name.',
            'type': str,
        },
        ('PENNCHARTX_PASSWORD', '--pennchartx_password'): {
            'dest': 'pennchartx_password',
            'help': 'PennChartX password.',
            'type': str,
        },
        ('PENNCHARTX_TOKEN_URI', '--pennchartx_token_uri'): {
            'dest': 'pennchartx_token_uri',
            'help': 'PennChartX token URI.',
            'type': str,
        },
        ('PENNCHARTX_API_URI', '--pennchartx_api_uri'): {
            'dest': 'pennchartx_api_uri',
            'help': 'PennChartX API URI.',
            'type': str,
        },
        ('PENNCHARTX_POSTMAN_TOKEN', '--pennchartx_postman_token'): {
            'dest': 'pennchartx_postman_token',
            'help': 'PennChartX postman token.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # print("** in pennchartx_input.py PennChartX patch_args")
        # print(args)
        # print(cfg)
        if cfg is None:
            cfg = {}
        print("** in pennchartx_input.py Input patch_args")
        print('>>>args<<<')
        print([(key, len(str(value))) for key, value in vars(args).items()])
        print('>>>cfg<<<')
        print([(key, len(str(value))) for key, value in cfg.items()])

        for key, value in (('client_id', args.pennchartx_client_id),
                           ('client_secret', args.pennchartx_client_secret),
                           ('user_name', args.pennchartx_user_name),
                           ('password', args.pennchartx_password),
                           ('token_uri', args.pennchartx_token_uri),
                           ('api_uri', args.pennchartx_api_uri),
                           ('postman_token', args.pennchartx_postman_token),
                           ):
            if value is not None:
                cfg[key] = value
            elif cfg.get(key) is None:
                logger.info("In pennchartx_input.py Input patch_args: %s is missing.", key)

        return cfg

    @classmethod
    def from_cfg(cls, cfg):
        """Return output from cfg."""
        print("** in pennchartx_input.py Input from_cfg")
        print('>>>cfg<<<')
        print([(key, len(str(value))) for key, value in cfg.items()])
        # print(cfg)
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (('client_id', str),
                              ('client_secret', str),
                              ('user_name', str),
                              ('password', str),
                              ('token_uri', str),
                              ('api_uri', str),
                              ('postman_token', str),
                              )}
        return cls(**kwargs)

    def __init__(
            self,
            client_id: str,
            client_secret: str,
            user_name: str,
            password: str,
            token_uri: str,
            api_uri: str,
            postman_token: str,
    ) -> None:
        """Initialize PennChartX."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_name = user_name
        self.password = password
        self.token_uri = token_uri
        self.api_uri = api_uri
        self.postman_token = postman_token

        self.token = self.createNewToken()

    def __call__(self, df_i_cohort, dt_now) -> pd.DataFrame:
        """Return the blood culture dfs.
        """
        self.df_i_cohort = df_i_cohort.drop_duplicates(subset='CSN').copy()
        df_i_orders = self.get_orders(dt_now)

        return df_i_orders

    def createNewToken(self):
        # 'https://uphsnet.uphs.upenn.edu/MedviewidentityServerV2/identity/connect/token'
        url = self.token_uri

        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Length': '166',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Host': 'uphsnet.uphs.upenn.edu',
            'Postman-Token': self.postman_token,
            'User-Agent': 'PostmanRuntime/7.17.1',
            'cache-control': 'no-cache',
        }

        data = {
            'grant_type': 'password',
            'username': self.user_name,
            'password': self.password,
            'scope': 'pennchartx-api-scope',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        token = post(url=url, headers=headers, data=data)
        # access_token = response.json()['access_token']
        # print(token.text)

        return token

    def getTokenIsValid(self) -> bool:
        response = self.getTokenStatus()
        if response.status_code == 200:
            return True
        else:
            return False

    def getTokenStatus(self):
        url = self.api_uri + 'order/GetLabOrdersByPatient'
        headers = {
            'accept': 'application/json',
            'authorization': self.token.json()['token_type'] + ' ' + self.token.json()['access_token'],
            'Content-Type': 'application/json',
        }
        data = str({'UID': ''})
        response = post(url=url, headers=headers, data=data)

        return response

    def get_orders(self, dt_now, lstnLabOrderTypeId=[447]):
        """ Filtering by time and lab type here just to keep size manageable"""
        # Lab order 447 represents blood cultures
        df_i_cohort = self.df_i_cohort
        token = self.token
        url = self.api_uri + 'order/GetLabOrdersByPatient'
        #     nLabOrderTypeId = 447

        headers = {
            'accept': 'application/json',
            'authorization': token.json()['token_type'] + ' ' + token.json()['access_token'],
            'Content-Type': 'application/json',
        }

        df_i_cohort = df_i_cohort.drop_duplicates(subset='CSN').copy()

        df_i_orders = pd.DataFrame()
        print("size: " + str(df_i_cohort.shape))
        for index, row in df_i_cohort.iterrows():
            if index % 50 == 0:
                print(str(datetime.now()) + '\t--------------' + str(index))
            dt_end = dt_now
            dt_start = row['dt_visitStart_UTC']

            for nLabOrderTypeId in lstnLabOrderTypeId:
                dictData = {'UID': row['UID'], 'LabOrderTypeId': nLabOrderTypeId}
                data = str(dictData)  # '{ "UID": "", "LabOrderTypeId": 0, "StatusId": 0, "PriorityId": 0}'
                df_orders_temp = responseToDF(post(url=url, headers=headers, data=data))
                if df_orders_temp.shape[0] == 0:
                    continue
                df_orders_temp['OrderDate_UTC'] = pd.to_datetime(df_orders_temp['OrderDate'],
                                                                 errors='coerce', utc=True)
                # Couldn't find a better way to keep the local, timezone-aware time:
                df_orders_temp['OrderDate_DT'] = pd.to_datetime(df_orders_temp['OrderDate'].apply(lambda x: x[:-6]),
                                                                errors='coerce')

                df_orders_temp = convertUTCcolumns(df_orders_temp)
                # print("df_orders_temp")
                # print(df_orders_temp)
                # print("CSN", row['CSN'], sep='\t')
                # df_orders_temp = df_orders_temp.rename(columns={'EncounterNumber':'CSN'})
                df_orders_temp['CSN'] = df_orders_temp['EncounterNumber'].apply(lambda x: str(x).zfill(12))
                df_orders_temp = df_orders_temp[df_orders_temp['CSN'] == row['CSN']]
                '''
                # If it turns out that EncounterNumber =/= CSN, then we have to merge them back from df_i_cohort
                df_orders_temp = df_orders_temp[(df_orders_temp['OrderDate_UTC'] >= dt_start) & \
                                                (df_orders_temp['OrderDate_UTC'] < dt_end)]
                df_orders_temp = df_orders_temp.merge(df_i_cohort[['UID', 'CSN']].drop_duplicates(),
                                                      on='UID', how='left')
                '''

                df_i_orders = pd.concat([df_i_orders, df_orders_temp], axis='rows', sort=False)
        return df_i_orders

    # def getOrdersByPatientList(self, lstUID):
    #     if not self.getTokenIsValid():
    #         self.createNewToken()
    #     for strUID in lstUID:
    #         response = self.getOrdersByPatient(self, strUID)
    #
    # def getOrdersByPatient(self, strUID):
    #     # strUID = '8464561802'
    #     nLabOrderTypeId = 447
    #     url = self.api_uri + 'order/GetLabOrdersByPatient'
    #
    #     headers = {
    #         'accept': 'application/json',
    #         'authorization': self.token.json()['token_type'] + ' ' + self.token.json()['access_token'],
    #         'Content-Type': 'application/json',
    #     }
    #
    #     dictData = {'UID': strUID, 'LabOrderTypeId': nLabOrderTypeId}
    #     data = str(dictData)  # '{ "UID": "", "LabOrderTypeId": 0, "StatusId": 0, "PriorityId": 0}'
    #
    #     response = post(url=url, headers=headers, data=data)
    #
    #     return response

    # @property
    # def from_number(self):
    #     return self._from_number
    #
    # @from_number.setter
    # def from_number(self, value):
    #     self._from_value = value

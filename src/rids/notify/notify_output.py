"""Output."""

from __future__ import annotations
from argparse import Namespace
from collections import namedtuple
from datetime import datetime, timedelta
from dateutil.tz import gettz
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
import pickle

Batch = namedtuple('Batch_notificationOutput', ('notification_batch', 'df_notifications',
                                                'lstCSN_notifyCandidate', 'lstCSN_notify',))



def get_TwilioRecipientRecord(collection, mg_query):
    if collection.count_documents(mg_query) == 0:
        mg_query = {'name': "Default Fallback"}
        return collection.find(mg_query).sort('_id', -1).limit(1)[0]
    else:
        return collection.find(mg_query).sort('_id', -1).limit(1)[0]  # ['_id']


def get_twilioToPersonRecord(collection, strPennID):
    return get_TwilioRecipientRecord(collection, mg_query={'pennID': strPennID})


def get_twilioToAARecord(collection, strHospital):
    return get_TwilioRecipientRecord(collection, mg_query={'hospitalLocation': strHospital})


basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name


class Output(BaseMongoInput, BaseMongoOutput):
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

        print("** in notify_output.py Output from_cfg")
        print('>>>cfg<<<')
        print([(key, len(str(value))) for key, value in cfg.items()])
        # print(cfg)
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (
                ('uri', str),
                ('collection', Collection.from_cfg),
            )}
        # print(kwargs)
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        print("** in notify_output.py Output patch_args")
        print('>>>args<<<')
        print([(key, len(str(value))) for key, value in vars(args).items()])
        print('>>>cfg<<<')
        print([(key, len(str(value))) for key, value in cfg.items()])
        # print(cfg)
        if cfg is None:
            cfg = {}
        for key, value in (
                ('uri', args.notify_output_uri),):
            if value is not None:
                cfg[key] = value
            else:
                logger.info("In notify_output.py Output patch_args: %s is missing.", key)
        # print(cfg)
        return cfg

    def __call__(self, df_notifications, idfs: namedtuple, dt_now: datetime) -> namedtuple:

        """
        Batch_idfs = namedtuple('Batch_idfs', ('outstandingMessages', 'cohort', 'notifications', 'assignments',
                                       'MAR', 'orders', 'prediction_batch'))

        Batch_tdfs = namedtuple('Batch_tdfs', ('cohort', 'notifications', 'MAR', 'orders', 'prediction_batch'))
        """

        """Create output dfs."""
        self.dt_now = dt_now
        self.df_notifications = df_notifications
        notification_batch = self.create_notificationBatch(idfs.prediction_batch)
        self.notification_batch = notification_batch
        df_notifications['notification_batch_id'] = notification_batch['_id']

        # IF RANDOMIZING
        # lstCSN_notifyCandidate, lstCSN_notify = self.divideArms()
        # df_notifications['ASSIGNED'] = np.where(df_notifications['CSN'].isin(lstCSN_notifyCandidate) == False,
        #                                         -1,
        #                                         np.where(df_notifications['CSN'].isin(lstCSN_notify),
        #                                                  1,
        #                                                  0)
        #                                         )

        # IF NOT RANDOMIZING, NOTIFY ON EVERYBODY
        lstCSN_notifyCandidate = list(df_notifications[df_notifications['SUM'] == 0]['CSN'])
        lstCSN_notify = lstCSN_notifyCandidate

        notifs = Batch(notification_batch=notification_batch,
                       df_notifications=df_notifications,
                       lstCSN_notifyCandidate=lstCSN_notifyCandidate,
                       lstCSN_notify=lstCSN_notify, )

        with self.collections() as collections:
            df_sendTo = self.create_dfToSend(notifs, collections.recipients_twilio)
            if df_sendTo.shape[0] != 0:
                self.insert_many_df(collections.notifications, df_sendTo)

        return notifs, df_sendTo

    # Update the outstanding Twilio messages
    # def get_batchQueryLatestNotification(self, mg_query, dt_now=None) -> dict:
    #     with self.collections() as collections:
    #         lstNotification_current = list(self.get_batchQueryLatest(collections.notifications, mg_query, dt_now=dt_now))
    #     if len(lstNotification_current) == 0:
    #         return {}
    #     else:
    #         return lstNotification_current[0]

    def update_one_outstandingTwilio(self, mg_query, mg_set_update):
        with self.collections() as collections:
            collections.outstanding_twilio.update_one(mg_query, mg_set_update)
        return

    # Clearing out the existing documents in the outstanding_twilio collection,
    # and re-inserting the ones that were not delivered/received
    def clearReinsert_outstandingTwilio(self, df_i_outstandingMessages_updated):
        with self.collections() as collections:
            collections.outstanding_twilio.delete_many({})
            if df_i_outstandingMessages_updated.shape[0] != 0:
                self.insert_many_df(self.collections.outstanding_twilio, df_i_outstandingMessages_updated)

    def create_notificationBatch(self, prediction_batch_latest):
        # TODO: Created batch, but if there is an error, there needs to be a way to know it didn't finish?
        #
        notification_batch = {'created_on': self.dt_now,
                              'prediction_batch_id': prediction_batch_latest['_id']}
        with self.collections() as collections:
            notification_batch = self.insert_one_dict(collections.notification_batches, notification_batch)
            # notification_batch_id = notification_batch['_id']
        return notification_batch

    # Get recipient information/who we're sending to and how
    def create_dfToSend(self, notifyBatch: namedtuple, collection) -> pd.DataFrame:
        """Weird dictionary popping of created_on is because of this bug:
            https://github.com/pandas-dev/pandas/issues/22796
        """
        lstCSN_notify = notifyBatch.lstCSN_notify
        df_notifications = notifyBatch.df_notifications
        notification_batch = notifyBatch.notification_batch
        df_sendTo = pd.DataFrame(columns=['CSN', 'method', 'to', 'body'])

        # SEND PER CSN
        for CSN in lstCSN_notify:
            # Getting CSN information
            dictCSN = df_notifications[df_notifications['CSN'] == CSN].squeeze().to_dict()
            # dictCSN['body'] = str(dictCSN['body'])
            # print("dictCSN " + CSN)
            # print(dictCSN)

            # # Twilio to Attending
            # df_sendTo = pd.concat([df_sendTo,
            #                        pd.DataFrame(
            #                            [self.create_twilioCSNToPersonDict(collection, dictCSN)])],
            #                       axis='rows', sort=False)

            # # Twilio to AA
            # df_sendTo = pd.concat([df_sendTo,
            #                        pd.DataFrame([self.create_twilioCSNToAADict(collection, dictCSN)])],
            #                       axis='rows', sort=False)

            # RedCap
            tempDict = self.create_redcapCSN(dictCSN)
            tempDict.pop('created_on', None)
            df_sendTo = pd.concat([df_sendTo,
                                   pd.DataFrame([tempDict])],
                                  axis='rows', sort=False)

        # SEND PER LOCATION
        lstHospitals = list(df_notifications[df_notifications['CSN'].isin(lstCSN_notify)]['LOCATION_HOSPITAL'].unique())
        for strHospital in lstHospitals:
            # Send 1 message to AA over Slack
            tempDict = self.create_slackLocationDict(strHospital)
            tempDict.pop('created_on', None)
            df_sendTo = pd.concat([df_sendTo,
                                   pd.DataFrame([tempDict])],
                                  axis='rows', sort=False)

        df_sendTo = df_sendTo.reset_index(drop=True)
        df_sendTo['created_on'] = notification_batch['created_on'].astimezone(gettz('US/Eastern'))

        return df_sendTo

    def create_defaultDict(self, dictCSN):
        lstKeepField = ['CSN', 'UID', 'LOCATION_HOSPITAL', 'LOCATION_DEPT', 'LOCATION_BED',
                        'LastName', 'FirstName', 'Y_PRED', 'AttendingName', 'AttendingPennID',
                        'prediction_id', 'prediction_batch_id', 'notification_batch_id']
        dictCSN_new = {key: dictCSN[key] for key in lstKeepField}
        dictCSN_new['body'] = dictCSN['FirstName'][0] + '.' + dictCSN['LastName'][0] + \
                              '. in ' + dictCSN['LOCATION_BED']  # initials + bed location
        # dictCSN_new['message_id'] = dictCSN['notification_batch_id']
        notification_batch = self.notification_batch
        dictCSN_new['created_on'] = notification_batch['created_on'].astimezone(gettz('US/Eastern'))
        return dictCSN_new

    def create_redcapCSN(self, dictCSN):
        dictSendRow = self.create_defaultDict(dictCSN)
        dictSendRow['friendly description'] = 'Create patient in RedCap'
        dictSendRow['method'] = 'redcap'
        dictSendRow['body'] = {'facility': dictCSN['LOCATION_HOSPITAL'],
                               'location': dictCSN['LOCATION_BED'],
                               'last_name': dictCSN['LastName'],
                               'csn': dictCSN['CSN'],
                               'uid': dictCSN['UID'],
                               'created_on': str(dictSendRow['created_on']),
                               'logging': str({'notification_batch_created_on_utc': self.notification_batch['created_on'],
                                               'notification_batch_id': dictCSN['notification_batch_id'],
                                               'prediction_batch_id': dictCSN['prediction_batch_id'],
                                               'prediction_id': dictCSN['prediction_id'],
                                               'y_pred': dictCSN['Y_PRED']})
                               }
        return dictSendRow

    def create_slackLocationDict(self, strHospital):
        dictSendRow = {}
        dictSendRow['LOCATION_HOSPITAL'] = strHospital
        dictSendRow['friendly description'] = 'Slack message to AA - ' + strHospital
        dictSendRow['method'] = 'slack'
        dictSendRow['body'] = "Check for new patients in RedCap"
        return dictSendRow

    def create_twilioCSNToPersonDict(self, collection, dictCSN):
        dictSendRow = self.create_defaultDict(dictCSN)
        dictSendRow['friendly description'] = 'Text to attending'
        dictSendRow['method'] = 'twilio'
        dictSendRow['to'] = get_twilioToPersonRecord(collection, dictCSN['AttendingPennID'])['cellNumber']
        dictSendRow['body'] = "Please order a BC for " + dictSendRow['body']
        return dictSendRow

    def create_twilioCSNToAADict(self, collection, dictCSN):
        dictSendRow = self.create_defaultDict(dictCSN)
        dictSendRow['friendly description'] = 'Text to AA'
        dictSendRow['method'] = 'twilio'
        dictSendRow['to'] = get_twilioToAARecord(collection, dictCSN['LOCATION_HOSPITAL'])['cellNumber']
        if dictCSN['AttendingName'] is None:
            dictSendRow['body'] = "Please ask the attending to order a BC for " + dictSendRow['body']
        else:
            dictSendRow['body'] = "Please ask Dr. " + \
                                  dictCSN['AttendingName'][:dictCSN['AttendingName'].find(',')] + \
                                  " to order a BC for " + dictSendRow['body']
        return dictSendRow

    def create_slackCSNToAADict(self, dictCSN):
        dictSendRow = self.create_defaultDict(dictCSN)
        dictSendRow['friendly description'] = 'Slack message to AA'
        dictSendRow['method'] = 'slack'
        dictSendRow['body'] = "Check your text messages"
        return dictSendRow

    def add_outstandingTwilio(self, dict_notificationInfo):
        with self.collections() as collections:
            self.insert_one_dict(collections.outstanding_twilio, dict_notificationInfo)

    #
    # def ping(self) -> bool:
    #     """Ping output.

    #     """
    #     with self.collections() as collections:
    #         print(collections)
    #     return True

    # IF RANDOMIZING
    def divideArms(self):
        df_notifications = self.df_notifications
        lstCSN_notifyCandidate = list(df_notifications[df_notifications['SUM'] == 0]['CSN'])
        print(len(lstCSN_notifyCandidate))

        lstCSN_notify = []
        for CSN in lstCSN_notifyCandidate:
            # print(CSN)

            with self.collections() as collections:
                batch_latest = self.get_batchQueryLatest(collections.assignments)
                # print(batch_latest)
            try:
                nNextRow_number = batch_latest['row_number'] + 1
            except KeyError:  # i.e. the collection is empty
                nNextRow_number = 0

            with self.collections() as collections:
                batch_arm = self.get_batchQueryLatest(collection=collections.randomizations,
                                                      mg_query={'row_number': nNextRow_number})
            # print(batch_arm)

            nStudy_arm = batch_arm['arm']
            if nStudy_arm == 1:
                lstCSN_notify.append(CSN)
            self.add_Assignment(nNextRow_number, CSN, nStudy_arm, self.notification_batch['_id'])

        return lstCSN_notifyCandidate, lstCSN_notify

    # IF RANDOMIZING
    def add_Assignment(self, row_number, CSN, nStudy_arm, notification_batch_id):
        batch = {'row_number': row_number,
                 'study_identifier': CSN,
                 'identifier_type': 'CSN',
                 'arm': nStudy_arm,
                 'dt_assigned': self.dt_now,
                 'notification_batch_id': notification_batch_id}
        with self.collections() as collections:
            self.insert_one_dict(collections.assignments, batch)

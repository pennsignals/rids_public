"""Model."""

from __future__ import annotations
from argparse import Namespace
from collections import namedtuple
from datetime import datetime
import pickle
import re
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from sys import stdout
import numpy as np
import pandas as pd

from .notify_output import Output as NotifyOutput
from .redcap_output import Output as RedcapOutput

from ..configurable import Configurable
from ..micro import garbage_collection
from ..mongo import (
    Output as BaseMongoOutput,
    retry_on_reconnect,
)

from ..slack import SlackWebhook
# from twilio.base.exceptions import TwilioRestException
# from ..twilio import TwilioClient

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name

# Batch = namedtuple('Batch_notificationOutput', ('notification_batch', 'df_notifications',
#                                                 'lstCSN_notifyCandidate', 'lstCSN_notify',))


def get_lstFromDf(df, strCollectionField='study_identifier') -> list:
    if df.shape[0] == 0:
        return []
    else:
        return list(df[strCollectionField].unique())


class Model(Configurable):  # pylint: disable=too-many-instance-attributes
    """Model."""

    ARGS = {
        # **TwilioClient.ARGS,
        **SlackWebhook.ARGS,
        **NotifyOutput.ARGS,
        **RedcapOutput.ARGS,
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # print("**in model.py Model patch_args")
        # print(args)
        # print(cfg)
        # print(cfg.keys())
        # for each other yml file, open and load/replace cfg[key]
        """Patch cfg from args."""
        # e.g. cfg['slack'] = SlackWebhook.patch_args(args, cfg['slack'])
        # e.g. cfg['twilio_input'] = TwilioInput.patch_args(args, cfg['twilio_input'])

        # cfg['slack'] = SlackWebhook.patch_args(args, cfg.get('slack'))
        # cfg['twilio_input'] = TwilioInput.patch_args(args, cfg.get('twilio_input'))
        # cfg['twilio_output'] = TwilioOutput.patch_args(args, cfg.get('twilio_output'))
        # cfg['notifications'] = NotifyOutput.patch_args(args, cfg.get('notifications'))

        for key, patch_args_function in (('slack', SlackWebhook.patch_args),
                                         # ('twilio', TwilioClient.patch_args),
                                         ('notifications', NotifyOutput.patch_args),
                                         ('redcap', RedcapOutput.patch_args),
                                         ):
            # print(key)
            cfg[key] = patch_args_function(args, cfg.get(key))
            # print(cfg[key])

        # print(cfg)

        return cfg

    @classmethod
    def from_cfg(cls, cfg: dict) -> Model:
        """Return Model from cfg."""
        # print("**in model.py Model from_cfg")
        # print(cfg)
        # E.g. key:value :: 'slack':SlackWebhook.from_cfg(cfg['slack'])
        # E.g. key:value :: 'twilio_output':TwilioOutput.from_cfg(cfg['twilio_output'])
        kwargs = {key: from_cfg_function(cfg[key])
                  for key, from_cfg_function in (('slack', SlackWebhook.from_cfg),
                                                 # ('twilio', TwilioClient.from_cfg),
                                                 ('notifications', NotifyOutput.from_cfg),
                                                 ('redcap', RedcapOutput.from_cfg),
                                                 )
                  }

        return cls(**kwargs)

    def __init__(
            self,
            slack,
            # twilio,
            notifications,
            redcap,
    ):
        """Return Inputs."""
        self.slack_client = slack
        # self.twilio_client = twilio
        self.notify_output = notifications
        self.redcap_client = redcap


    @garbage_collection
    def __call__(
            self,
            input,  # pylint: disable=redefined-builtin
            output):
        print("OK GO!")
        print(str(datetime.now()))
        idfs, tdfs, dt_now = input()
        if idfs is None: # This implies the latest prediction batch has already been processed
            return
        self.idfs = idfs
        self.tdfs = tdfs
        self.dt_now = dt_now

        # Batch = namedtuple('Batch_predictionInput', ('outstandingMessages', 'cohort', 'notifications',
        #                                              'assignments', 'prediction_batch_latest'))

        keepCol = ['UID', 'CSN', 'Y_PRED',
                   'VisitStartDTime', 'dt_visitStart', 'dt_visitStart_UTC',
                   #            'HOSPITAL','Loc_Dept','Loc_Room_bed',
                   'LastName', 'FirstName', 'AGE',
                   'AttendingName', 'AttendingPennID', 'prediction_id', 'prediction_batch_id']

        self.df_notifications = self.tdfs.cohort[keepCol].copy()

        # # Update the outstanding Twilio messages
        # print(idfs.outstandingMessages)
        # if idfs.outstandingMessages.shape[0] != 0:
        #     self.update_outstandingTwilio(idfs.outstandingMessages)
        #     # self.twilio_input.update_outstandingTwilio(idfs.outstandingMessages)

        # Do filters
        df_notifications = self.doChecks()
        df_notifications['SUM'] = \
            df_notifications[[x for x in list(df_notifications) if x.endswith('_YN')]].sum(axis='columns')

        # Create notification batch
        # ~~Assign into arms~~ [Edit: No longer assigning into arms here.]
        # Get recipient information/who we're sending to and how
        # print("list df_notifications")
        # print(list(df_notifications))
        notifyBatch, df_sendTo = self.notify_output(df_notifications, idfs, dt_now)

        # df_sendTo = self.create_dfToSend(notifyBatch, collections.recipients_twilio)
        df_notifications.to_csv('./temporary_output/df_notifications.tsv', sep='\t')
        df_notifications.to_pickle('./temporary_output/df_notifications.pickle')
        df_sendTo.to_csv('./temporary_output/df_sendTo.tsv', sep='\t')
        df_sendTo.to_pickle('./temporary_output/df_sendTo.pickle')

        print(df_notifications.shape)
        print(df_sendTo.shape)

        # Send notification
        lstNotificationResults = []
        lstErrors = []
        for index, row in df_sendTo.iterrows():
            print(str(datetime.now()) + '\t' + str(index))
            # if row['method'] == 'twilio':
            #     lstNotificationResults, lstErrors = self.send_twilio(row, lstNotificationResults, lstErrors)
            if row['method'] == 'redcap':
                lstNotificationResults, lstErrors = self.send_redcap(row, lstNotificationResults, lstErrors)
            elif row['method'] == 'slack':
                self.send_slack(row)

        output(notifyBatch, df_notifications, lstNotificationResults, lstErrors)

        return

    # # Update the outstanding Twilio messages
    # def update_outstandingTwilio(self, df_i_outstandingMessages):
    #     df_i_outstandingMessages_updated = pd.DataFrame()
    #     for index, row in df_i_outstandingMessages.iterrows():
    #         print("ROW")
    #         print(row)
    #         obj_id = row['_id']
    #
    #         status_updated = self.twilio_client.status(row['response']['sid'])
    #         row_updated = row.copy()
    #         row_updated['response']['status'] = status_updated
    #
    #         mg_query = {'_id': obj_id}
    #         if row_updated['response']['status'] != row['response']['status']:
    #             self.notify_output.update_one_outstandingTwilio(mg_query, {"$set": row_updated})
    #
    #         # with self.collections() as collections:
    #         #     mg_query = {'_id': obj_id}
    #         #     dictNotification_current = list(self.get_batchQueryLatest(collections.notifications, mg_query))[0]
    #         #     dictResponse_new = dictNotification_current['response']
    #         #     dictResponse_new['status'] = row_updated['status']
    #         #     collections.notifications.update_one(mg_query, {"$set": {'response': dictResponse_new}})
    #
    #         if row_updated['response']['status'] not in ['delivered', 'receiving', 'received', 'read']:
    #             df_i_outstandingMessages_updated = pd.concat([df_i_outstandingMessages_updated,
    #                                                           pd.DataFrame([row_updated], columns=row_updated.keys())],
    #                                                          axis='rows')
    #
    #     # Clearing out the existing documents in the outstanding_twilio collection,
    #     # and re-inserting the ones that were not delivered/received
    #     self.notify_output.clearReinsert_outstandingTwilio(df_i_outstandingMessages_updated)

    # Do filters
    def doChecks(self):
        # TODO: Move dictFacility_EDLocation to a better place
        dictFacility_EDLocation = {'HUP': 'ERSRV', 'PMC': 'ER'}

        # Batch_idfs = namedtuple('Batch_idfs', ('outstandingMessages', 'cohort', 'notifications', 'assignments',
        #                                        'MAR', 'orders', 'prediction_batch'))
        #
        # Batch_tdfs = namedtuple('Batch_dfs', ('cohort', 'notifications', 'MAR', 'orders', 'prediction_batch'))

        df_notifications = self.df_notifications
        df_notifications = self.checkAge()
        df_notifications = self.checkLocation(dictFacility_EDLocation)
        df_notifications = self.checkThreshold()
        # df_notifications = self.checkAssigned(self.idfs.assignments)
        df_notifications = self.checkNotified(self.idfs.notifications)
        df_notifications = self.checkVisitDuration()
        df_notifications = self.checkVisitReceivedTreatment(self.tdfs)
        df_notifications = self.checkVisitReceivedBCOrder(self.idfs)

        return df_notifications

    def checkAge(self, dictColOut={'TOO_YOUNG': 18, 'TOO_OLD': 123}):
        df_notifications = self.df_notifications
        df_notifications['AGE_LT_YN'] = np.where(df_notifications['AGE'] < dictColOut['TOO_YOUNG'], 1, 0)
        df_notifications['AGE_GT_YN'] = np.where(df_notifications['AGE'] > dictColOut['TOO_OLD'], 1, 0)
        return df_notifications

    def checkLocation(self, dictFacility_EDLocation):
        df_notifications = self.df_notifications
        lstFacility_Location = [key + '_' + value for key, value in dictFacility_EDLocation.items()]

        df_temp = self.idfs.cohort[['CSN', 'HOSPITAL', 'Loc_Dept', 'Loc_Room_bed']].copy()

        df_temp['TEMP_DELETE'] = df_temp['HOSPITAL'] + '_' + df_temp['Loc_Dept']
        df_temp['LOCATION_EXCLUDE_YN'] = np.where(df_temp['TEMP_DELETE'].isin(lstFacility_Location), 0, 1)
        df_temp = df_temp.drop(labels=['TEMP_DELETE'], axis='columns')
        df_temp = df_temp.rename(columns={'Loc_Dept': 'LOCATION_DEPT',
                                          'Loc_Room_bed': 'LOCATION_BED',
                                          'HOSPITAL': 'LOCATION_HOSPITAL'})
        df_temp = df_temp.drop_duplicates(subset='CSN')

        df_notifications = df_notifications.merge(df_temp, on='CSN', how='left')
        # TODO: This doesn't seem kosher, but not sure what else to do
        self.df_notifications = df_notifications
        return df_notifications

    def checkThreshold(self, fThreshold=0.45, strCol='Y_PRED', strColOut='YPRED_LOW_YN'):
        df_notifications = self.df_notifications
        df_notifications[strColOut] = np.where(df_notifications[strCol] < fThreshold, 1, 0)
        return df_notifications

    # IF RANDOMIZING

    # def checkAssigned(self, df_i_assignments, strCollectionField='study_identifier', strCol='CSN',
    #                   strColOut='ALREADY_ASSIGNED_YN'):
    #     df_notifications = self.df_notifications
    #     lstIdentifiers = get_lstFromDf(df_i_assignments, strCollectionField)
    #     df_notifications[strColOut] = np.where(df_notifications[strCol].isin(lstIdentifiers), 1, 0)
    #     return df_notifications

    def checkNotified(self, df_i_notifications, strCollectionField='CSN', strCol='CSN',
                      strColOut='ALREADY_NOTIFIED_YN'):
        df_notifications = self.df_notifications
        lstIdentifiers = get_lstFromDf(df_i_notifications, strCollectionField)
        df_notifications[strColOut] = np.where(df_notifications[strCol].isin(lstIdentifiers), 1, 0)
        return df_notifications

    def checkVisitDuration(self, strCol='dt_visitStart_UTC', dictColOut={'LT': pd.Timedelta(hours=2.75),
                                                                         'GT': pd.Timedelta(hours=5.75)}):
        df_notifications = self.df_notifications
        dt_now = self.dt_now
        df_notifications['td_elapsed'] = dt_now - df_notifications[strCol]
        df_notifications['DURATION_LT_HRS_YN'] = np.where(df_notifications['td_elapsed'] < dictColOut['LT'], 1, 0)
        df_notifications['DURATION_GT_HRS_YN'] = np.where(df_notifications['td_elapsed'] > dictColOut['GT'], 1, 0)
        df_notifications['hours_elapsed'] = df_notifications['td_elapsed'] / pd.Timedelta(hours=1)
        # df_notifications = df_notifications.drop(labels='td_elapsed', axis='columns')
        df_notifications.drop(labels='td_elapsed', axis='columns', inplace=True)
        return df_notifications

    def checkVisitReceivedTreatment(self, tdfs, strColABX='RHEE_ABX_YN', strColOut='ABX_RECEIVED_YN'):
        df_notifications = self.df_notifications
        df_t_MAR = tdfs.MAR.copy()

        df_temp = df_t_MAR[df_t_MAR[strColABX] == 1].copy()
        df_temp['AdministrationDateTime'] = pd.to_datetime(df_temp['AdministrationDateTime'], errors='coerce')
        df_temp = df_temp.sort_values(by=['CSN', 'AdministrationDateTime']).drop_duplicates(subset=['CSN'])

        df_temp = df_temp[['CSN', 'AdministrationDateTime', 'OrderNumber', strColABX]]. \
            rename(columns={'AdministrationDateTime': 'ABX_DT',
                            'OrderNumber': 'ABX_ORDERNUMBER',
                            strColABX: strColOut})
        df_notifications = df_notifications.merge(df_temp, on='CSN', how='left')

        # TODO: This doesn't seem kosher, but not sure what else to do
        self.df_notifications = df_notifications
        df_notifications[strColOut] = np.where(df_notifications[strColOut].isnull(),
                                               0,
                                               df_notifications[strColOut]
                                               )
        return df_notifications

    def checkVisitReceivedBCOrder(self, idfs, strColOut='BC_RECEIVED_YN'):
        df_notifications = self.df_notifications
        df_i_orders = idfs.orders.copy()

        df_temp = df_i_orders.copy()
        # df_temp['EffectiveOrderDtime'] = pd.to_datetime(df_temp['EffectiveOrderDtime'], errors='coerce')
        # df_temp = df_temp.sort_values(by=['CSN', 'EffectiveOrderDtime'], ascending=[True, True]).drop_duplicates(
        #     subset='CSN')
        df_temp = df_temp[df_temp['LabOrderTypeId'] == 447]

        df_temp = df_temp[['CSN', 'OrderDate_UTC', 'SrcOrderId']]. \
                        sort_values(by=['CSN','OrderDate_UTC','SrcOrderId']).\
                        drop_duplicates(subset=['CSN']).\
                        rename(columns={'OrderDate_DT': 'BC_UTC',
                                        'SrcOrderId': 'BC_ORDERNUMBER'})
        df_temp[strColOut] = 1

        df_notifications = df_notifications.merge(df_temp, on='CSN', how='left')
        # TODO: This doesn't seem kosher, but not sure what else to do
        self.df_notifications = df_notifications

        df_notifications[strColOut] = np.where(df_notifications[strColOut].isnull(),
                                               0,
                                               df_notifications[strColOut]
                                               )
        return df_notifications

    def checkEligibleTime(self):
        pass

    # # Send notification
    # def send_twilio(self, dictCSN: dict, lstNotificationResults: list, lstErrors: list):
    #     lstKeepField = ['CSN', 'prediction_id', 'prediction_batch_id', 'notification_batch_id']
    #     dict_notificationInfo = {key: dictCSN[key] for key in lstKeepField}
    #     dict_notificationInfo['message'] = {'to_number': dictCSN['to'],
    #                                         'from_number': self.twilio_client.from_number,
    #                                         'body': dictCSN['body'], }
    #     print(dict_notificationInfo)
    #     try:
    #         response = self.twilio_client.send(to=dict_notificationInfo['message']['to_number'],
    #                                            from_=dict_notificationInfo['message']['from_number'],
    #                                                      body=dict_notificationInfo['message']['body'])
    #         print(response)
    #         dict_notificationInfo['response'] = {'sid': response.sid,
    #                                              'status': response.status}
    #
    #         if dict_notificationInfo['response']['status'] not in ['delivered', 'receiving', 'received', 'read']:
    #             self.notify_output.add_outstandingTwilio(dict_notificationInfo)
    #             # self.add_outstandingTwilio(dict_notificationInfo)
    #         lstNotificationResults.append(dict_notificationInfo)
    #     except TwilioRestException as error:
    #         # TODO Does the TwilioRestException include 'no connectivity'
    #         logger.warning('%s to number %s', error, dict_notificationInfo['message']['to_number'])
    #         error = dict_notificationInfo.copy()
    #         lstErrors.append(error)
    #
    #     return lstNotificationResults, lstErrors

    def send_redcap(self, dictCSN, lstNotificationResults, lstErrors):
        lstKeepField = ['CSN', 'prediction_id', 'prediction_batch_id', 'notification_batch_id', 'body']
        dict_notificationInfo = {key: dictCSN[key] for key in lstKeepField}

        response = self.redcap_client.postRecord(dictCSN['body'])
        dict_notificationInfo['response'] = response
        dict_notificationInfo['response_text'] = response.text
        # TODO: Do better error logging
        try:
            error = dict_notificationInfo['response_text']['error']
            lstErrors.append(dict_notificationInfo)
            lstNotificationResults.append(response['added'])
        except:
            lstNotificationResults.append(dict_notificationInfo)

        return lstNotificationResults, lstErrors

    def send_slack(self, row):
        slack_alert_url = self.slack_client.getHospitalURL(row['LOCATION_HOSPITAL'])
        return self.slack_client.slack_send(slack_alert_url, strMessage=row['body'])

    # Mock send
    # def mock_send(
    #         self,
    #         to_number: str,
    #         prediction: dict,
    #         body: str,
    #         notification_batch_id: ObjectId,
    #         notifications: list,
    #         errors: list):
    #     """Return notification after send to twilio."""
    #     from_number = self.from_number
    #     notification = {
    #         'body': body,
    #         'csn': prediction['csn'],
    #         'from': from_number,
    #         'notification_batch_id': notification_batch_id,
    #         'prediction_id': prediction['_id'],
    #         'response': {
    #             'sid': '0000000000',
    #         },
    #         'to': to_number,
    #     }
    #     notifications.append(notification)

    # IF PULLING FROM CLINSTREAM:
    # def checkVisitReceivedBCOrder(self, idfs, strColOut='BC_RECEIVED_YN'):
    #     df_notifications = self.df_notifications
    #     df_i_orders = idfs.orders.copy()
    #
    #     df_temp = df_i_orders[df_i_orders['ServiceTypeMnemonic'] == 'C BC'].copy()
    #     df_temp['EffectiveOrderDtime'] = pd.to_datetime(df_temp['EffectiveOrderDtime'], errors='coerce')
    #     df_temp = df_temp.sort_values(by=['CSN', 'EffectiveOrderDtime'], ascending=[True, True]).drop_duplicates(
    #         subset='CSN')
    #
    #     df_temp = df_temp[['CSN', 'EffectiveOrderDtime', 'PlacerOrderID']]. \
    #         rename(columns={'EffectiveOrderDtime': 'BC_DT',
    #                         'PlacerOrderID': 'BC_ORDERNUMBER'})
    #     df_temp[strColOut] = 1
    #
    #     df_notifications = df_notifications.merge(df_temp, on='CSN', how='left')
    #     # TODO: This doesn't seem kosher, but not sure what else to do
    #     self.df_notifications = df_notifications
    #
    #     df_notifications[strColOut] = np.where(df_notifications[strColOut].isnull(),
    #                                            0,
    #                                            df_notifications[strColOut]
    #                                            )
    #     return df_notifications
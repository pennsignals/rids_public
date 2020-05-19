"""Vent Input.
Used to get
- patient list,
- demographics,
- vitals
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
import pandas as pd
# from time import mktime
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

Batch = namedtuple('Batch_ventInput', ('cohort', 'demographics', 'vitals'))

strTZ = 'US/Eastern'
dt_epoch = datetime(1970, 1, 1, 0, tzinfo=timezone.utc)


def calculate_age(DOB: pd.Datetime):
    today = datetime.now()
    return today.year - DOB.year - ((today.month, today.day) < (DOB.month, DOB.day))


class Input(BaseInput):
    """Input."""

    ARGS = {
        ('VENT_INPUT_URI', '--vent-input-uri'): {
            'dest': 'vent_input_uri',
            'help': 'Vent Mongo input uri.',
            'type': str,
        },
    }

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.vent_input_uri),):
            if value is not None:
                cfg[key] = value

        return cfg

    def __call__(self, dt_now=None) -> Batch:
        """Return a Batch (a namedtuple) of input dfs.
        """

        with self.collections() as collections:
            self.df_i_cohort, self.df_i_demographics = self.get_patient_list(collections.clinstream_demographic_events,
                                                                             dt_now)
            # self.df_i_cohort = df_i_cohort
            df_t_cohort_new = self.get_vitals_first(collections.signal_flowsheet_events, dt_now,
                                                    td_lookback=pd.Timedelta(hours=6))
            self.df_i_cohort = self.df_i_cohort.merge(df_t_cohort_new, on='CSN', how='left')

            self.df_i_vitals = self.get_vitals(collections.signal_flowsheet_events, dt_now,
                                               td_lookback=pd.Timedelta(hours=4))

        return Batch(
            cohort=self.df_i_cohort,
            demographics=self.df_i_demographics,
            vitals=self.df_i_vitals)

    def get_patient_list(self, collection, dt_now):
        # TODO: Only keeping patients who have been here for less than 12 hours
        td_lookback = pd.Timedelta(hours=12)
        dt_end = dt_now

        # QUERYING MONGO
        obj_end = ObjectId.from_datetime(dt_end)

        mg_query = {'_id': {'$lt': obj_end}}
        temp = list(collection.find(mg_query).sort('_id', -1).limit(1))[0]
        # objSET_ID = temp['set_id']

        keepCol = ['UID', 'CSN', 'FirstName', 'LastName', 'VisitStartDTime',
                   'DOB', 'Sex', 'maritalStatus', 'Race', ]
        keepColLocation = ['Facility', 'HospitalService', 'Loc_Dept', 'Loc_Room_bed', 'ServiceName',
                           'Nav_Location', 'Nav_Room_bed']
        keepColAdmit = ['AdmitReason', 'AdmitSourceName', 'AdmitType', 'AdmitSourceCode', ]
        keepColTeam = ['AttendingName', 'AttendingPennID', 'CoveringName', 'CoveringPennID',
                       'CoveringNurseName', 'CoveringNursePennID',
                       'CoveringCellNumber', 'CoveringPagerNumber',
                       'CoveringNurseCellNumber', 'CoveringNursePagerNumber']
        keepMiscInfo = ['EpicHAR', 'LastUpdateDtime', 'HUPMRN', 'MVPatientID', 'PAHMRN', 'PMCMRN',
                        'NameSuffix', 'MiddleName', 'AdmitHeightM', 'AdmitWeightKG']
        keepColAll = keepCol + keepColLocation + keepColAdmit + keepColTeam + keepMiscInfo

        df_temp = pd.DataFrame(temp['data'])[keepColAll].copy()
        df_temp['CSN_INT'] = pd.to_numeric(df_temp['CSN'], errors='coerce')
        df_temp = df_temp[df_temp['CSN_INT'].isnull() == False]
        df_temp['CSN_INT'] = df_temp['CSN_INT'].astype(int)

        df_temp['VisitStartDTime'] = pd.to_datetime(df_temp['VisitStartDTime'], errors='coerce')
        df_temp['VisitStartDTime_UTC'] = df_temp['VisitStartDTime']. \
            apply(lambda x: x.replace(tzinfo=gettz(strTZ)).astimezone(timezone.utc))

        df_temp['DOB'] = pd.to_datetime(df_temp['DOB'], errors='coerce')
        df_temp['AGE'] = df_temp['DOB'].apply(calculate_age)

        keepColCohort = ['CSN_INT', 'VisitStartDTime_UTC', 'AGE'] + keepColAll

        df_i_cohort = df_temp[(df_temp['Facility'].isin(['HUP', 'PAH', 'PMC'])) & \
                              #                           (df_temp['Loc_Dept'].isin(['ERSRV','EMER','ER'])) & \
                              ((dt_now - df_temp['VisitStartDTime_UTC']) < td_lookback)][keepColCohort]

        keepColDemographics = ['CSN',  # object
                               'DOB',  # object --> timestamp
                               'Sex',  # object
                               'maritalStatus',  # object
                               'Race',  # object
                               'CSN_INT',  # int
                               ]
        df_i_demographics = df_temp[df_temp['CSN'].isin(df_i_cohort['CSN'])][keepColDemographics].copy()

        return df_i_cohort, df_i_demographics

    def get_vitals_first(self, collection, dt_now, td_lookback):
        print(str(datetime.now()) + '\t' + "GETTING FIRST VITALS TO DETERMINE START TIME")
        print(self.df_i_cohort.shape)

        # TODO: Ignoring timezone shift by converting times to integers... kosher?
        df_t_cohort = self.df_i_cohort[['UID', 'CSN', 'VisitStartDTime_UTC']]. \
            sort_values(by=['UID', 'CSN', 'VisitStartDTime_UTC']). \
            drop_duplicates(subset='CSN').copy()
        df_t_cohort = df_t_cohort.reset_index(drop=True)

        df_t_cohort_new = pd.DataFrame()
        for index, row in df_t_cohort.iterrows():
            if index % 50 == 0:
                print(str(datetime.now()) + '\t--------------' + str(index))
            obj_end = ObjectId.from_datetime(dt_now)
            nTimeStart = ((row['VisitStartDTime_UTC'] - dt_epoch) - td_lookback).total_seconds()

            # print('dt_now: ' + str(dt_now))
            # print('obj_end: ' + str(obj_end))
            # print('VisitStartDTime: ' + str(row['VisitStartDTime']))
            # print('nTimeStart: ' + str(nTimeStart))
            # print('patient_id: ' + str(row['UID']))

            mg_query = {'_id': {'$lt': obj_end},
                        'patient_id': row['UID'],
                        # 'recorded_on': {'$gt': nTimeStart, '$lt': nTimeEnd},
                        'recorded_on': {'$gt': nTimeStart},
                        }
            sortOrder = (('_id', 1), ('recorded_on', 1))
            temp = collection.find(mg_query).sort(sortOrder).limit(1)
            df_temp_ind = pd.DataFrame.from_dict(list(temp))

            if df_temp_ind.shape[0] == 0:
                df_temp_ind = pd.DataFrame(columns=['CSN'], data=[row['CSN']])
            else:
                keepCol_flowsheet = ['FlowsheetDateTime', 'recorded_on', 'ObservationDateTime', 'valid_on']
                df_temp_ind = df_temp_ind[keepCol_flowsheet]
                df_temp_ind['CSN'] = row['CSN']

            df_t_cohort_new = pd.concat([df_t_cohort_new, df_temp_ind], axis='rows', ignore_index=True, sort=False)

        print(str(datetime.now()) + '\t' + "DONE GETTING FIRST VITALS TO DETERMINE START TIME")

        return df_t_cohort_new

    def get_vitals(self, collection, dt_now, td_lookback):
        # TODO: Currently just going [6] hours back
        lstintPatientUID = [int(x) for x in list(self.df_i_cohort['UID'].unique())]
        dt_end = dt_now
        dt_start = dt_end - td_lookback
        obj_start = ObjectId.from_datetime(dt_start)
        obj_end = ObjectId.from_datetime(dt_end)

        # TODO: Create aggregate group so not overwhelmed
        mg_query = {'_id': {'$gt': obj_start, '$lt': obj_end}, 'patient_id': {'$in': lstintPatientUID}}
        sortOrder = (('_id', -1), ('recorded_on', -1))
        temp = collection.find(mg_query).sort(sortOrder)
        df_i_vitals = pd.DataFrame.from_dict(list(temp))

        return df_i_vitals

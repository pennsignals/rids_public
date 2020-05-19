"""Input."""

from __future__ import annotations
from argparse import Namespace
from collections import namedtuple
from csv import reader
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from datetime import datetime, timezone
from dateutil.tz import gettz
from dateutil import parser
import pandas as pd
# import pickle
import numpy as np

from sys import stdout

from yaml import safe_load as yaml_loads

from ..configurable import Configurable
from .clarity_input import Input as ClarityInput
from .vent_input import Input as VentInput
from .ps1_input import Input as PS1Input

# Batch_idfs = namedtuple('Batch_idfs', ('vent', 'clarity', 'ps1'))
Batch_dfs = namedtuple('Batch_dfs', ('cohort', 'demographics', 'vitals', 'labs', 'dx_hx'))

strTZ = 'US/Eastern'
dt_epoch = datetime(1970, 1, 1, 0, tzinfo=timezone.utc)


class CormorbidityMap():
    @classmethod
    def from_cfg(cls, cfg: list) -> dict:
        """Return a comorbidity mapping from cfg."""
        # Comorbidity, ICD9_REGEX,  ICD10_REGEX, Points
        return {
            key: value
            for key, _, value, _ in cfg}


class ElixhauserMap(CormorbidityMap):
    """ElixhauserMap."""

    ARGS = {
        ('ELIXHAUSER', '--elixhauser'):
            {
                'dest': 'elixhauser_map',
                'help': 'Elixhauser map configuration file tsv',
                'type': str,
            }
    }


class CharlsonMap(CormorbidityMap):
    """CharlsonMap."""

    ARGS = {
        ('CHARLSON', '--charlson'):
            {
                'dest': 'charlson_map',
                'help': 'Charlson map configuration file tsv',
                'type': str,
            }
    }


class RawDataMap(dict):
    """RawDataMap"""

    # @classmethod
    # def from_cfg(cls, cfg: list) -> dict:
    #     """Return model from cfg."""
    #     # NORMALIZED_STRING	ORIGINAL_STRING	IDENTIFIER	SOURCE_CODE
    #     kwargs = {
    #         key: value
    #         for value, key, _, _ in cfg}
    #
    #     #validation here
    #
    #     return cls(kwargs)


class OrdersMap(RawDataMap):
    """OrdersMap."""

    ARGS = {
        ('ORDERS', '--orders'):
            {
                'dest': 'orders_map',
                'help': 'Orders map configuration file tsv',
                'type': str,
            }
    }

    @classmethod
    def from_cfg(cls, cfg: list) -> dict:
        """Return model from cfg."""
        # NORMALIZED_STRING	ORIGINAL_STRING	IDENTIFIER	SOURCE_CODE
        kwargs = {
            key: value
            for value, key, _, _ in cfg}

        # validation here

        return cls(kwargs)


class VitalsMap(RawDataMap):
    """VitalsMap."""

    ARGS = {
        ('VITALS', '--vitals'):
            {
                'dest': 'vitals_map',
                'help': 'Vitals map configuration file tsv',
                'type': str,
            }
    }

    @classmethod
    def from_cfg(cls, cfg: list) -> dict:
        """Return model from cfg."""
        # NORMALIZED_STRING	ORIGINAL_STRING	IDENTIFIER	SOURCE_CODE
        kwargs = {
            key: value
            for value, key, _, _ in cfg}

        # validation here

        return cls(kwargs)


class Inputs(Configurable):

    NOW = {
        ('NOW', '--now'):
            {
                'dest': 'now',
                'help': 'Now',
                'type': str,
            }
    }

    ARGS = {
        **ClarityInput.ARGS,
        **VentInput.ARGS,
        **PS1Input.ARGS,
        **ElixhauserMap.ARGS,
        **CharlsonMap.ARGS,
        **OrdersMap.ARGS,
        **VitalsMap.ARGS,
        **NOW,
    }

    @classmethod
    def patch_now(cls, args, value):
        return value

    @classmethod
    def cfg_now(cls, value):
        """Return datetime or None."""
        if value is None:
            return datetime.now(timezone.utc)
        return parser.parse(value)

    @classmethod
    def from_cfg(cls, cfg: dict) -> Inputs:
        """Return micro from cfg."""
        kwargs = {
            key: from_cfg(cfg[key])
            for key, from_cfg in (
                ('clarity', ClarityInput.from_cfg),
                ('vent', VentInput.from_cfg),
                ('ps1', PS1Input.from_cfg),
                ('elixhauser', ElixhauserMap.from_cfg),
                ('charlson', CharlsonMap.from_cfg),
                ('orders', OrdersMap.from_cfg),
                ('vitals', VitalsMap.from_cfg),
                ('now', cls.cfg_now),)}
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch cfg from args."""
        # for each other yml file, open and load/replace cfg[key]
        for key, value in (
                ('elixhauser', args.elixhauser_map),
                ('charlson', args.charlson_map),
                ('orders', args.orders_map),
                ('vitals', args.vitals_map),):
            if value:
                with open(value) as fin:
                    cfg[key] = [row for row in reader(fin, delimiter='\t')][1:]  # Indiscriminately removes header

        # for key, value in (
        #         ('order', args.order_map),
        #         ('vital', args.vital_map)):
        #     if value:
        #         with open(value) as fin:
        #             cfg[key] = yaml_loads(fin.read())

        for key, patch_args in (
                ('clarity', ClarityInput.patch_args),
                ('vent', VentInput.patch_args),
                ('ps1', PS1Input.patch_args),
                ('now', cls.patch_now),):
            cfg[key] = patch_args(args, cfg.get(key))
        return cfg

    def __init__(
            self,
            clarity,
            vent,
            ps1,
            elixhauser,
            charlson,
            orders,
            vitals,
            now):
        """Return Inputs."""
        self.clarity_input = clarity
        self.vent_input = vent
        self.ps1_input = ps1
        self.elixhauser_map = elixhauser
        self.charlson_map = charlson
        self.orders_map = orders
        self.vitals_map = vitals
        self.dt_now = now

    def __call__(self):  # extract and transform
        """Return dfs from inputs."""
        dt_now = self.dt_now
        idfs = self.extract(dt_now)
        tdfs = self.transform(idfs)
        return (idfs, tdfs, dt_now)

    def ping(self):
        """Ping inputs."""
        return self.clarity_input.ping() and self.vent_input.ping() and self.ps1_input.ping()

    def extract(self, dt_now):
        vent_idfs = self.vent_input(dt_now)
        clarity_idfs = self.clarity_input(vent_idfs.cohort, dt_now)
        ps1_idfs = self.ps1_input(vent_idfs.cohort, dt_now)

        idfs = Batch_dfs(cohort=vent_idfs.cohort,
                         demographics=vent_idfs.demographics,
                         vitals=vent_idfs.vitals,
                         labs=ps1_idfs.labs,
                         dx_hx=clarity_idfs.dx_hx,
                         )

        # for field in idfs._fields:
        #     print(">>>>>" + field)
        #     print(getattr(idfs, field).head(3))

        return idfs

    def transform(self, idfs):
        tdfs = Batch_dfs(
            cohort=self.transform_cohort(idfs.cohort),
            demographics=self.transform_demographics(idfs.demographics),
            vitals=self.transform_vitals(idfs.vitals),
            labs=self.transform_labs(idfs.labs),
            dx_hx=self.transform_dx(idfs.dx_hx),
        )

        return tdfs

    def transform_cohort(self, df_i_cohort):
        df_t_cohort = df_i_cohort.copy()
        df_t_cohort['DOB'] = pd.to_datetime(df_t_cohort['DOB'], errors='coerce')

        dictCohortHeader = {'Facility': 'HOSPITAL',
                            }
        df_t_cohort = df_t_cohort.rename(columns=dictCohortHeader)
        df_t_cohort = self.calculate_startTime(df_t_cohort)

        return df_t_cohort

    def calculate_startTime(self, df_t_cohort):
        lstStrCol = ['VisitStartDTime', 'FlowsheetDateTime', 'ObservationDateTime']
        df_t_cohort[lstStrCol] = df_t_cohort[lstStrCol].apply(pd.to_datetime, errors='coerce')

        df_t_cohort['dt_visitStart'] = df_t_cohort[['VisitStartDTime', 'ObservationDateTime']].min(axis='columns')
        df_t_cohort['dt_visitStart_UTC'] = df_t_cohort['dt_visitStart']. \
                                                apply(lambda x: x.replace(tzinfo=gettz(strTZ)).astimezone(timezone.utc))
        return df_t_cohort

    def transform_demographics(self, df_i_demographics):
        df_t_demographics = df_i_demographics.copy()

        dictDemographicCodes = {'Sex': {'F': 'FEMALE', 'M': 'MALE'},
                                'maritalStatus': {'M': 'MARRIED', 'S': 'SINGLE'},
                                'Race': {'WHITE': 'WHITE', 'AABLA': 'BLACK', 'ASIAN': 'ASIAN'}}

        dictDemographicHeader = {'Sex': 'GENDER_DESCRIPTION',
                                 'maritalStatus': 'MARITAL_STATUS_DESCRIPTION',
                                 'Race': 'RACE_CODE'}

        for strCol in dictDemographicCodes:
            df_t_demographics[strCol] = df_t_demographics[strCol].map(dictDemographicCodes[strCol]).fillna('OTHER')
        df_t_demographics = df_t_demographics.rename(columns=dictDemographicHeader)

        return df_t_demographics

    def transform_vitals(self, df_i_vitals):
        df_t_vitals = df_i_vitals.copy()

        # TRANSFORMING (MAPPING) PENNSIGNALS NAMES INTO NORMALIZED NAMES
        df_t_vitals['MAPPED_VITAL'] = df_t_vitals['name'].map(self.vitals_map)

        df_t_vitals = df_t_vitals.sort_values(by=['UID', 'recorded_on', 'valid_on'], ascending=[True, False, False])
        df_t_vitals = df_t_vitals.drop_duplicates(subset=['UID', 'MAPPED_VITAL'])

        # lstCol = [x for x in list(df_t_vitals['MAPPED_VITAL'].unique()) if x is not None]
        df_t_vitals = df_t_vitals.pivot(index='UID', columns='MAPPED_VITAL', values='value').reset_index()

        df_t_vitals = self.transformCreateMeasureColumns(df_t_vitals)
        return df_t_vitals

    def transform_labs(self, df_i_labs):
        df_t_labs = df_i_labs.copy()
        if df_t_labs.shape[0] == 0:
            for strCol in ['UID', 'OrderDate', 'ResultDate', 'ObsTypeMnemonic', 'MAPPED_LAB', 'Value']:
                df_t_labs[strCol] = None
            return df_t_labs

        # TRANSFORMING (MAPPING) PENNSIGNALS NAMES INTO NORMALIZED NAMES
        # df_t_labs['TEMP'] = df_t_labs['ServiceTypeMnemonic'] + "_" + df_t_labs['ObsTypeMnemonic']
        # df_t_labs['MAPPED_LAB'] = df_t_labs['TEMP'].map(orders_map)
        # df_t_labs = df_t_labs.drop(labels=['TEMP'], axis='columns')
        df_t_labs['MAPPED_LAB'] = df_t_labs['ObsTypeMnemonic'].map(self.orders_map)

        df_t_labs = df_t_labs.sort_values(by=['UID', 'OrderDate', 'ResultDate'], ascending=[True, False, False])
        df_t_labs = df_t_labs.drop_duplicates(subset=['UID', 'MAPPED_LAB'])

        df_t_labs = df_t_labs.pivot(index='UID', columns='MAPPED_LAB', values='Value').reset_index()

        df_t_labs = self.transformCreateMeasureColumns(df_t_labs)
        return df_t_labs

    def transformCreateMeasureColumns(self, df_input):
        df = df_input.copy()

        strPrefix = 'MEASURED_'
        strSuffix = '_YN'
        df = pd.concat([df.add_prefix("MEASURE_RESULT_"),
                        df.notna().astype(int).add_prefix(strPrefix).add_suffix(strSuffix)
                        ], axis='columns', sort=False)

        df = df. \
            rename(columns={'MEASURE_RESULT_UID': 'UID'}). \
            drop(labels='MEASURED_UID_YN', axis='columns')

        for strCol in list(df):
            if strCol.startswith('MEASURE_RESULT_'):
                df[strCol] = pd.to_numeric(df[strCol], errors='coerce')
            elif strCol.startswith('MEASURED_'):
                df[strCol] = df[strCol].astype(int, errors='ignore')

        return df

    def transform_dx(self, df_i_dx_hx):
        df_t_dx_hx = df_i_dx_hx.copy()

        if df_t_dx_hx.shape[0] == 0:
            # TODO: Might want to do something if there are no comorbidities for this batch.
            # What errors would come up?
            pass

        # df_HxDx = df_HxDx_Clarity_final.copy()
        # if df_HxDx_Clarity_final.shape[0] == 0:
        #     continue
        #

        # TRANSFORMING CLARITY FORMAT INTO PDS FORMAT BY COMBINING ICD-10 and ICD-9 COLUMNS
        df_t_dx_hx['CODE'] = np.where(df_t_dx_hx['CURRENT_ICD10_LIST'].isnull() == False,
                                      df_t_dx_hx['CURRENT_ICD10_LIST'],
                                      np.where(df_t_dx_hx['CURRENT_ICD9_LIST'].isnull() == False,
                                               df_t_dx_hx['CURRENT_ICD9_LIST'], None))
        df_t_dx_hx['CODE_STANDARD_NAME'] = np.where(df_t_dx_hx['CURRENT_ICD10_LIST'].isnull() == False, 'ICD-10',
                                                    np.where(df_t_dx_hx['CURRENT_ICD9_LIST'].isnull() == False, 'ICD-9',
                                                             None))
        df_t_dx_hx['UID'] = df_t_dx_hx['UID'].astype(int)

        # print(str(datetime.datetime.now()) + '\t' + "..Adding comorbidities")

        # CREATING UNIQUE LIST OF CODES
        lstUniqueCodes = df_t_dx_hx['CODE'].unique()
        df_codes = pd.DataFrame(lstUniqueCodes).rename(columns={0: 'CODE'}).dropna()

        # MATCHING CODES TO COMORBIDITIES
        df_codes = self.matchElixCharlson_dx(df_codes)
        # print('df_codes')
        # print(df_codes.head())
        # print('df_t_dx_hx')
        # print(df_t_dx_hx.head())

        # (df_t_dx_hx, df_codes) = mergeElixCharlson_dx(df_t_dx_hx, df_codes)
        df_t_dx_hx = df_t_dx_hx.merge(df_codes, on='CODE', how='left')
        # print('df_t_dx_hx1')
        # print(df_t_dx_hx.head())

        keepCol = ['ELIX_' + key for key in self.elixhauser_map.keys()] + \
                  ['CHARLSON_' + key for key in self.charlson_map.keys()]

        df_t_dx_hx = df_t_dx_hx[['UID'] + keepCol]
        # print('df_t_dx_hx2')
        # print(df_t_dx_hx.head())

        df_t_dx_hx = df_t_dx_hx.groupby('UID').sum()[keepCol]
        # print('df_t_dx_hx3')
        # print(df_t_dx_hx.head())

        df_t_dx_hx = df_t_dx_hx.reset_index()

        return df_t_dx_hx

    def matchElixCharlson_dx(self, df, strCodeCol='CODE'):
        """
        Takes in a dataframe and adds columns for each of
        Elixhauser, Charlson, ICD9, ICD10, and each comorbidity
        """
        elixhauser_map = self.elixhauser_map
        charlson_map = self.charlson_map

        df_codes = df.copy()
        # print(str(datetime.datetime.now()) + '\t' + "....Pattern matching")
        for strComorbidity in elixhauser_map.keys():
            df_codes['ELIX_' + strComorbidity] = np.where(
                df_codes[strCodeCol].str.match(elixhauser_map[strComorbidity]), 1, 0)

        for strComorbidity in charlson_map.keys():
            df_codes['CHARLSON_' + strComorbidity] = np.where(
                df_codes[strCodeCol].str.match(charlson_map[strComorbidity]), 1, 0)
        return df_codes

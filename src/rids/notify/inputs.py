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
from dateutil import parser
import pandas as pd
# import pickle
import numpy as np
import re

from sys import stdout

from yaml import safe_load as yaml_loads

_Batch_idfs = namedtuple('_IdfBatch', ('outstandingMessages', 'cohort', 'notifications', 'assignments',
                                       'MAR', 'orders', 'prediction_batch'))

_Batch_tdfs = namedtuple('_Batch_tdfs', ('cohort', 'notifications', 'MAR', 'orders', 'prediction_batch'))

class Batch_tdfs(_Batch_tdfs):

    def __str__(self):
        s = ''
        for field in self._fields:
            s += "\n>>>>>" + field
            try:
                s += getattr(self, field).head(3)
            except AttributeError:
                s += getattr(self, field)
            # if field == 'prediction_batch':
            #     print(getattr(tdfs, field))
            # else:
            #     print(getattr(tdfs, field).head(3))
        return s


class Batch_idfs(_Batch_idfs):

    def transform(self, normalizeABX, lstABX):
        tdfs = Batch_tdfs(
            cohort=self.transform_cohort(),
            notifications=self.notifications,
            MAR=self.transform_MARtoABX(normalizeABX, lstABX),
            orders=self.transform_orders(),
            prediction_batch=self.prediction_batch,
        )
        print("printing transform dataframes")
        print(tdfs)

    def transform_cohort(self, df_i_cohort):
        df_i_cohort = self.cohort
        df_t_cohort = df_i_cohort.copy()
        df_t_cohort = df_t_cohort.rename(columns={'_id': 'prediction_id', 'batch_id': 'prediction_batch_id'})
        return df_t_cohort

    def transform_orders(self):
        df_i_orders = self.orders
        df_t_orders = df_i_orders[df_i_orders['ServiceTypeMnemonic'] == 'C BC'].copy()
        return df_t_orders

    def transform_MARtoABX(self, normalizeABX, lstABX):
        df_i_MAR = self.MAR
        df_t_MAR = df_i_MAR.copy()
        df_t_MAR['FULL_NAME_NORM'] = df_t_MAR['AdministeredMedicationName'].apply(normalizeABX)
        df_t_MAR['ABX_FULL_YN'] = np.where((df_t_MAR['FULL_NAME_NORM'].isin(lstABX)), 1, 0)

        lstRheeABX = lstAntibioticIV + lstAntibioticPO + lstAntifungalIV + lstAntifungalPO + lstAntiviralIV + lstAntiviralPO
        df_t_MAR['RHEE_ABX_YN'] = np.where((df_t_MAR['FULL_NAME_NORM'].isin(lstRheeABX)), 1, 0)

        df_t_MAR = df_t_MAR[df_t_MAR['AdministrationStatus'].isin(lstAdministrationStatusKeep)]

        return df_t_MAR

    def __str__(self):
        s = ''
        for field in self._fields:
            s += "\n>>>>>" + field
            try:
                s += getattr(self, field).head(3)
            except AttributeError:
                s += getattr(self, field)
            # if field == 'prediction_batch':
            #     print(getattr(tdfs, field))
            # else:
            #     print(getattr(tdfs, field).head(3))
        return s


from ..configurable import Configurable
from .prediction_input import Input as PredictionInput
# from .order_input import Input as OrderInput
from .pennchartx_input import Input as OrderInput
from .vent_input import Input as VentInput
from .antibioticsLists import (
    lstAntibioticIV,
    lstAntibioticPO,
    lstAntifungalIV,
    lstAntifungalPO,
    lstAntiviralIV,
    lstAntiviralPO,
    lstRemoveDrug,
    lstRemoveStr,
    lstAdministrationStatusKeep
)


class AntibioticsSNOMED():
    ARGS = {
        ('ANTIBIOTICS_SNOMED', '--SNOMED'):
            {
                'dest': 'SNOMED',
                'help': 'SNOMED dict file tsv',
                'type': str,
            }
    }

    @classmethod
    def from_cfg(cls, cfg: list) -> pd.DataFrame:
        """Return a list from cfg."""
        # e.g. SNOMEDID	SNOMED_description
        # e.g. 387426003	Acetohydroxamic acid (substance)
        # return {key:value for key, value in cfg}
        return pd.DataFrame(cfg, columns=['SNOMED_ID', 'DESCRIPTION'])


class AntibioticsSNOMED_norm():
    ARGS = {
        ('ANTIBIOTICS_SNOMED_NORM', '--SNOMED_NORM'):
            {
                'dest': 'SNOMED_norm',
                'help': 'SNOMED mapping file tsv',
                'type': str,
            }
    }

    @classmethod
    def from_cfg(cls, cfg: list) -> dict:
        """Return a dict from cfg."""
        # e.g. SNOMEDID	SNOMED_description	normalized_form	manual_final_mapping
        # e.g. 56723006	Penicillin V potassium (substance)	PENICILLIN V POTASSIUM	PENICILLIN
        return {key: value for _, _, key, value in cfg}


class AntibioticsBrandsGenerics():
    ARGS = {
        ('ANTIBIOTICS_BRAND_GENERIC', '--BRAND_GENERIC'):
            {
                'dest': 'brandToGeneric',
                'help': 'Brand to generic mapping file tsv',
                'type': str,
            }
    }

    @classmethod
    def from_cfg(cls, cfg: list) -> pd.DataFrame:
        """Return a dict from cfg."""
        # e.g. brand_name	conceptid	drug_name
        # e.g. "artec"	"102649004"	"heptanoic acid (substance)"
        # {key.upper(): value.upper() for key, _, value in cfg}
        return pd.DataFrame(cfg, columns=['brand_name', 'conceptid', 'drug_name'])




class Inputs(Configurable):

    ARGS = {
        **PredictionInput.ARGS,
        **VentInput.ARGS,
        **OrderInput.ARGS,
        **AntibioticsSNOMED.ARGS,
        **AntibioticsSNOMED_norm.ARGS,
        **AntibioticsBrandsGenerics.ARGS,
    }

    @classmethod
    def from_cfg(cls, cfg: dict) -> Inputs:
        """Return micro from cfg."""
        # E.g. key:value :: 'predictions': PredictionInput.from_cfg(cfg['predictions'])
        # E.g PredictionInput extends Configurable:
        #       from_cfg(cls, cfg: dict) -> Configurable:
        #             return cls()
        # I.e. from_cfg calls __init__ with cfg['predictions'] parameters

        kwargs = {key: from_cfg(cfg[key]) for key, from_cfg in (('predictions', PredictionInput.from_cfg),
                                                                ('vent', VentInput.from_cfg),
                                                                ('pennchartx', OrderInput.from_cfg),
                                                                ('SNOMED', AntibioticsSNOMED.from_cfg),
                                                                ('SNOMED_norm', AntibioticsSNOMED_norm.from_cfg),
                                                                ('brandToGeneric', AntibioticsBrandsGenerics.from_cfg),
                                                                )
                  }

        value = cfg.get('now', None)  # TODO: make better
        if value:
            value = parser.parse(value)
        cfg['now'] = value
        kwargs['now'] = value

        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        # for each other yml file, open and load/replace cfg[key]
        print("**in inputs.py Inputs patch_args")
        for key, value, bHasHeader in (('SNOMED', args.SNOMED, False),
                                       ('SNOMED_norm', args.SNOMED_norm, False),
                                       ('brandToGeneric', args.brandToGeneric, True),
                                       ):
            if value:
                with open(value) as fin:
                    cfg[key] = [row for row in reader(fin, delimiter='\t')][int(bHasHeader):]

        """Patch cfg from args."""

        for key, patch_args in (
                ('predictions', PredictionInput.patch_args),
                ('vent', VentInput.patch_args),
                ('pennchartx', OrderInput.patch_args),
        ):
            cfg[key] = patch_args(args, cfg.get(key))
        return cfg

    def __init__(
            self,
            predictions,
            pennchartx,
            vent,
            SNOMED,
            SNOMED_norm,
            brandToGeneric,
            now):
        """Return Inputs."""
        self.predictions_input = predictions
        self.orders_input = pennchartx
        self.vent_input = vent
        self.dt_now = now
        self.df_SNOMEDABX = SNOMED
        self.dictABXremap = SNOMED_norm
        self.df_BrandGeneric = brandToGeneric
        self.dictBrandGeneric = {}
        self.lstABX = []

        df_SNOMEDABX, df_BrandGeneric, dictBrandGeneric, lstABX = self.normalizeSNOMED()
        self.dictBrandGeneric = dictBrandGeneric
        self.lstABX = lstABX
        # print('df_SNOMEDABX')
        # print(df_SNOMEDABX.head(2))
        # print('df_BrandGeneric')
        # print(df_BrandGeneric.head())

    def __call__(self) -> namedtuple():  # extract and transform
        """Return dfs from inputs."""
        dt_now = self.dt_now
        if dt_now is None:
            # dt_now = datetime.now()
            # dt_now = datetime.utcnow()
            # dt_now = datetime(2019, 8, 14, 21)
            dt_now = datetime.now(timezone.utc)

        # TODO: I don't know if this is the right way to do this
        # If prediction batch has already been notified on, then quit and wait for the next cycle
        bBatchesMatch = self.predictions_input.is_matching_latestPredictNotifyBatches(dt_now)
        if bBatchesMatch:
            print("Already processed the latest prediction_batch!")
            return None, None, dt_now

        print("EXTRACTING")
        idfs = self.extract(dt_now)

        # Transforms
        print("TRANSFORMING")
        tdfs = self.transform(idfs)

        return idfs, tdfs, dt_now

    def ping(self):
        """Ping inputs."""
        # TODO: More pings needed
        return self.predictions_input.ping()  # and self.orders_input.ping()

    def extract(self, dt_now):

        print("... Getting predictions")
        prediction_idfs = self.predictions_input(dt_now)
        # Batch = namedtuple('Batch_predictionInput', ('outstandingMessages', 'cohort', 'notifications',
        #                                              'assignments', 'prediction_batch_latest'))
        df_i_cohort = prediction_idfs.cohort
        print("... Getting MAR")
        df_i_mar = self.vent_input(df_i_cohort, dt_now)
        print("... Getting orders")
        df_i_orders = self.orders_input(df_i_cohort, dt_now)

        idfs = Batch_idfs(outstandingMessages=prediction_idfs.outstandingMessages,
                          cohort=prediction_idfs.cohort,
                          notifications=prediction_idfs.notifications,
                          assignments=prediction_idfs.assignments,
                          MAR=df_i_mar,
                          orders=df_i_orders,
                          prediction_batch=prediction_idfs.prediction_batch_latest
                          )

        # print("printing input dataframes")
        # for field in idfs._fields:
        #     print(">>>>>" + field)
        #     try:
        #         print(getattr(idfs, field).head(3))
        #     except AttributeError:
        #         print(getattr(idfs, field))
        #     # if field == 'prediction_batch':
        #     #     print(getattr(idfs, field))
        #     # else:
        #     #     print(getattr(idfs, field).head(3))

        return idfs

    def transform(self, idfs):
        # tdfs = idfs.transform(self.normalizeABX, self.lstABX)
        tdfs = Batch_tdfs(cohort=self.transform_cohort(idfs.cohort),
                          notifications=idfs.notifications,
                          MAR=self.transform_MARtoABX(idfs.MAR),
                          orders=self.transform_orders(idfs.orders),
                          prediction_batch=idfs.prediction_batch
                          )
        '''
        print("printing transform dataframes")
        for field in tdfs._fields:
            print(">>>>>" + field)
            try:
                print(getattr(idfs, field).head(3))
            except AttributeError:
                print(getattr(idfs, field))
            # if field == 'prediction_batch':
            #     print(getattr(tdfs, field))
            # else:
            #     print(getattr(tdfs, field).head(3))
        '''
        return tdfs

    def transform_cohort(self, df_i_cohort):
        df_t_cohort = df_i_cohort.copy()
        df_t_cohort = df_t_cohort.rename(columns={'_id': 'prediction_id', 'batch_id': 'prediction_batch_id'})
        return df_t_cohort

    def transform_orders(self, df_i_orders):
        df_t_orders = df_i_orders.copy()
        # df_t_orders = df_i_orders[df_i_orders['ServiceTypeMnemonic'] == 'C BC'].copy()
        return df_t_orders

    def transform_MARtoABX(self, df_i_MAR):
        df_t_MAR = df_i_MAR.copy()
        df_t_MAR['FULL_NAME_NORM'] = df_t_MAR['AdministeredMedicationName'].apply(self.normalizeABX)
        df_t_MAR['ABX_FULL_YN'] = np.where((df_t_MAR['FULL_NAME_NORM'].isin(self.lstABX)), 1, 0)

        lstRheeABX = lstAntibioticIV + lstAntibioticPO + lstAntifungalIV + lstAntifungalPO + lstAntiviralIV + lstAntiviralPO
        df_t_MAR['RHEE_ABX_YN'] = np.where((df_t_MAR['FULL_NAME_NORM'].isin(lstRheeABX)), 1, 0)

        df_t_MAR = df_t_MAR[df_t_MAR['AdministrationStatus'].isin(lstAdministrationStatusKeep)]

        return df_t_MAR

    def normalizeSNOMED(self):
        df_SNOMEDABX = self.df_SNOMEDABX
        df_BrandGeneric = self.df_BrandGeneric

        # df_SNOMEDABX = pd.DataFrame(list(self.dictSNOMED.keys()), columns=['SNOMED_ID'])
        # df_SNOMEDABX['DESCRIPTION'] = df_SNOMEDABX['SNOMED_ID'].map(self.dictSNOMED)
        df_SNOMEDABX['DESCRIPTION_NORM'] = df_SNOMEDABX['DESCRIPTION'].apply(self.normalizeABX)
        lstABX = [x for x in list(df_SNOMEDABX['DESCRIPTION_NORM'].values) if x != '' and x != '**REMOVE**']
        # Adding in the Rhee paper meds here
        lstABX += lstAntibioticIV + lstAntibioticPO + \
                  lstAntifungalIV + lstAntifungalPO + \
                  lstAntiviralIV + lstAntiviralPO
        lstABX = list(set(lstABX))

        # Load and normalize all the brand/generics and create dictBrandGeneric dictionary
        df_BrandGeneric = df_BrandGeneric.merge(df_SNOMEDABX, left_on='conceptid', right_on='SNOMED_ID', how='left')
        # Fill in the missing SNOMED_id (for deprecated concepts)
        df_BrandGeneric['DESCRIPTION_NORM'] = np.where(df_BrandGeneric['DESCRIPTION'].isnull(),
                                                       df_BrandGeneric['drug_name'].apply(self.normalizeABX),
                                                       df_BrandGeneric['DESCRIPTION'].apply(self.normalizeABX))
        dictBrandGeneric = dict(zip(df_BrandGeneric['brand_name'].str.upper(),
                                    df_BrandGeneric['DESCRIPTION_NORM']))

        return df_SNOMEDABX, df_BrandGeneric, dictBrandGeneric, lstABX

    def manualMapping(self, ABX):
        dictBrandGeneric = self.dictBrandGeneric
        dictABXremap = self.dictABXremap

        if ABX in dictBrandGeneric:
            ABX = dictBrandGeneric[ABX]

        if ABX in dictABXremap:
            return dictABXremap[ABX]
        else:
            return ABX

    def normalizeABX(self, ABX, bDebug=False):
        if (ABX is None) or (ABX != ABX):  # Apparently (ABX!=ABX) is a check for ABX=nan
            return None
        ABX = ABX.upper()  # uppercase everything
        if bDebug: print("1) Uppercase: " + ABX)

        dictReplace = {'HCL': 'HYDROCHLORIDE'}
        # lstRemoveDrug = ['INTRAPERITONEAL', 'INTRATHECAL',
        #                  'CREAM', 'VAGINAL', 'TOPICAL', 'INHALATION', 'IR ONLY', ' NEBU',
        #                  'OPHTHALMIC', ' OPHTH', 'DROPS', 'OINTMENT', ' OINT']
        # lstRemoveStr = ['_', 'IVPB', 'INFUSION', 'INJECTION', 'BOLUS', 'ORAL', 'LIQUID', ' CAPSULE',
        #                 'SUBSTANCE', 'CLASS OF ANTIBIOTIC', ' TABLET', ' SOLUTION', 'SYRINGE',
        #                 'SUSPENSION', 'INTRAMUSCULAR', 'CENTRAL', 'MINI-BAG PLUS', 'LIPOSOMAL',
        #                 'HYDROCHLORIDE', 'SODIUM', 'SULFATE', 'AXETIL', 'PALMITATE', 'MONOHYDRATE', 'HYCLATE',
        #                 ' DS', ' DR', ' ER', ' SS', ' INJ.', ' INJ', ' IV ', ' TAB', ' SOLN', ' SUSP',
        #                 'HUP']
        try:
            for strReplace in dictReplace:  # replace list of words in dictionary above
                ABX = ABX.replace(strReplace, dictReplace[strReplace])
            if bDebug: print("2) Dictionary replace: " + ABX)

            for strRemove in lstRemoveDrug:  # completely remove the drug
                if strRemove in ABX:
                    ABX = "**REMOVE**"
            if bDebug: print("3) Potentially remove: " + ABX)

            for strRemove in lstRemoveStr:  # remove list of words in list above
                ABX = ABX.replace(strRemove, ' ')
            if bDebug: print("4) Remove list of words: " + ABX)

            if ABX.startswith("."):
                ABX = ABX[1:]
            if ABX.endswith(".") or ABX.endswith("-"):
                ABX = ABX[:-1]
            ABX = re.sub(r'[\s][-]+', ' ', ABX)  # replace space followed by dash with a space
            ABX = re.sub(r' \+ ', '-', ABX)  # replace " + " with a dash, mainly for SNOMED drugs
            ABX = re.sub(r'/', '-', ABX)  # replace "/" with a dash
            ABX = re.sub(r'\(.*?\)', ' ', ABX)  # remove anything in parentheses
            ABX = re.sub(r'\[.*?\]', ' ', ABX)  # remove anything in brackets
            ABX = re.sub(r' IN .+', ' ', ABX)  # remove the word "IN" and anything after it
            if bDebug: print("5) Prelim cleaning: " + ABX)

            match = re.search("\d", ABX)  # look for any digit
            try:
                # remove the digit and anything after, then remove extra spaces at the beginning/end
                ABX = ABX[0:match.start()].strip()
            except:
                ABX = ABX.strip()  # if no digits, remove extra spaces at the beginning/end
            if bDebug: print("6) Remove digits: " + ABX)

            ABX = self.manualMapping(ABX)  # manual mapping (e.g. combining SNOMED drugs, brands->generics)
            if bDebug: print("7) Manual mapping: " + ABX)

            return ABX
        except:
            return None

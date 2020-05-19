"""Model."""

from __future__ import annotations
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
# from pandas import DataFrame
import pandas as pd

from ..configurable import Configurable
from ..micro import garbage_collection

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name

Batch_pdfs = namedtuple('Batch_pfs', ('input', 'output', 'model'))


# DAYS_IN_NS = 8.64e+13


def unpickle_from_file(file_name):
    """Return unpickle from file."""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


# def remove_dot(x):
#     """Remove the . from ICD code."""
#     return re.sub(r'\.', r'', str(x))


# def regex_match(icdcode, expr):
#     """Regular expressions match ICD code."""
#     return re.match(expr, icdcode, re.IGNORECASE) is not None

def replaceValues(df_temp, strCol, lstReplace, strReplaceWith="OTHER"):
    df_temp.loc[:, strCol] = df_temp[strCol].fillna(value=strReplaceWith)
    for strCategory in lstReplace:
        df_temp.loc[:, strCol] = df_temp[strCol].replace(to_replace=strCategory, value=strReplaceWith)
    return df_temp


# def timedelta_in_years(td):
#     """Compute the time in years from a time-delta."""
#     if pd.isnull(td):
#         return np.nan
#     return float(td) / (365.25 * DAYS_IN_NS)


# def safe_float(x):
#     """Cast to float."""
#     try:
#         return float(x)
#     except ValueError:
#         return np.nan

class Model(Configurable):  # pylint: disable=too-many-instance-attributes
    """Model."""

    # @classmethod
    # def from_cfg(cls, cfg: dict) -> Model:
    #     """Return model from cfg."""
    #     kwargs = {

    #         dest: from_cfg(cfg[src]) # key: from_cfg(cfg[key]) like in other files
    #         for dest, src, from_cfg in (
    #             ('model', 'path', unpickle_from_file),
    #             ('pred_floor', 'prediction_floor', float),
    #             ('n_per_prov', 'max_n_per_provider', int),
    #         )
    #     }
    #     return cls(**kwargs)

    @classmethod
    def from_cfg(cls, cfg: dict) -> Model:
        """Return model from cfg."""
        kwargs = {
            key: from_cfg(cfg[key])
            for key, from_cfg in (
                ('path', unpickle_from_file),)}
        return cls(**kwargs)

    def __init__(
            self,
            path) -> None:
        """Initialize model."""
        self.clf = path['clf']
        self.features = path['features']
        self.version = path['version']

    @garbage_collection
    def __call__(
            self,
            input,  # pylint: disable=redefined-builtin
            output):

        print("OK GO!")
        print(str(datetime.now()))
        (idfs, tdfs, dt_now) = input()

        for field in idfs._fields:
            getattr(idfs, field).to_pickle('./temporary_output/idf_' + field + '.pickle')
        for field in tdfs._fields:
            getattr(tdfs, field).to_pickle('./temporary_output/tdf_' + field + '.pickle')

        # COMBINE DISPARATE DATA FRAMES AND ONLY INCLUDE THE RELEVANT COLUMNS
        df_prediction_input, df_prediction_cohort = self.lastDataTransforms(tdfs)
        # X.to_csv('./temporary_output/df_prediction_input_1.tsv', sep='\t')
        # X.to_pickle('./temporary_output/df_prediction_input_1.pickle')
        # print(X.shape)

        df_prediction_input = self.lastModelTransforms(df_prediction_input)
        # df_prediction_input.to_csv('./temporary_output/df_prediction_input_2.tsv', sep='\t')
        # df_prediction_input.to_pickle('./temporary_output/df_prediction_input_2.pickle')
        # print(df_prediction_input.shape)

        df_prediction_output = self.predict(df_prediction_input, df_prediction_cohort)
        df_prediction_output.to_csv('./temporary_output/df_prediction_output.tsv', sep='\t')
        df_prediction_output.to_pickle('./temporary_output/df_prediction_output.pickle')
        print(df_prediction_output.shape)

        pdfs = Batch_pdfs(input=df_prediction_input,
                          output=df_prediction_output,
                          model=self)
        # FYI
        # self.clf = path['clf']
        # self.features = path['features']
        # self.version = path['version']

        output(idfs, tdfs, pdfs, dt_now)

        print("OK STOP!")
        print(str(datetime.now()))

    def lastDataTransforms(self, tdfs) -> (pd.DataFrame, pd.DataFrame):
        # tdfs = Batch_dfs(
        #     cohort=self.transform_cohort(idfs.cohort),
        #     demographics=self.transform_demographics(idfs.demographics),
        #     vitals=self.transform_vitals(idfs.vitals),
        #     labs=self.transform_labs(idfs.labs),
        #     dx_hx=self.transform_dx(idfs.dx_hx),
        # )

        # PRINTING OUT EACH OF THE TRANSFORMED DATAFRAMES
        # for field in tdfs._fields:
        #     print(">>>>>" + field)
        #     print(getattr(tdfs, field).head())

        df_prediction_input = pd.merge(tdfs.cohort.drop_duplicates(subset='CSN'),
                                       tdfs.demographics[['CSN', 'GENDER_DESCRIPTION', 'MARITAL_STATUS_DESCRIPTION',
                                                          'RACE_CODE']].drop_duplicates(subset='CSN'),
                                       on='CSN', how='left')

        for category in ['vitals', 'labs', 'dx_hx']:
            df_prediction_input = pd.merge(df_prediction_input,
                                           getattr(tdfs, category),
                                           on='UID', how='left')

        lstCohortCol = list(tdfs.cohort)
        df_prediction_input = df_prediction_input.reset_index(drop=True)
        df_prediction_cohort = df_prediction_input[lstCohortCol].copy()
        X = df_prediction_input[['HOSPITAL'] + [x for x in list(df_prediction_input) if x not in lstCohortCol]].copy()

        X = replaceValues(X, strCol='RACE_CODE',
                          lstReplace=['UNKNOWN', 'AM IND AK NATIVE', 'HI PAC ISLAND'],
                          strReplaceWith="OTHER")

        X = replaceValues(X, strCol='MARITAL_STATUS_DESCRIPTION',
                          lstReplace=['WIDOWED', 'DIVORCED', 'SEPARATED'],
                          strReplaceWith="OTHER")

        # X['FC_CALCAT'] = X['FC_CALCAT'].fillna(value='Other')
        # X = replaceValues(X, strCol='FC_CALCAT',
        #                   lstReplace=['Medicare FFS','Managed Medicare'],
        #                   strReplaceWith="Medicare")
        # X = replaceValues(X, strCol='FC_CALCAT',
        #                   lstReplace=['Managed Medicaid'],
        #                   strReplaceWith="Medicaid")

        print("Size before one-hot encoding:" + str(X.shape))
        lst_1 = list(X)
        lst_2 = list(X.describe()) + \
                [x for x in list(X) if x.startswith('LeiCat_')] + \
                [x for x in list(X) if x.startswith('PROC_MAJOR_CATEGORY_')] + \
                ['PROC_AHRQ_SEV']
        lstNeedEncoding = [x for x in lst_1 if x not in lst_2]

        for strColumn in lstNeedEncoding:
            X = pd.concat([X, pd.get_dummies(X[strColumn], prefix=strColumn)], axis=1)
            X = X.drop(labels=strColumn, axis=1)
        print("Size after one-hot encoding:" + str(X.shape))

        X = X.loc[:, ~X.columns.duplicated()]
        X = X.fillna(value=-100)

        X.to_csv('X.tsv', sep='\t')

        return X, df_prediction_cohort

    def lastModelTransforms(self, df_prediction_input: pd.DataFrame) -> pd.DataFrame:
        X = df_prediction_input.copy()
        lstFeatures = self.features

        print("Num features \t" + str(len(lstFeatures)))

        # Make sure all the features are represented, and nothing more
        print("Before feature fix \t" + str(X.shape))
        for strCol in lstFeatures:
            if strCol not in list(X):
                X[strCol] = -100
        X = X[lstFeatures]
        print("After feature fix \t" + str(X.shape))

        return X

    def predict(self, df_prediction_input: pd.DataFrame, df_prediction_cohort) -> namedtuple:
        #     keepColCohort = ['UID', 'CSN', 'LastName', 'HOSPITAL', 'VisitStartDTime', 'dt_visitStart',
        #                      'VisitStartDTime_UTC', 'dt_visitStart_UTC']
        #     keepColCohortTeam = ['AttendingName','AttendingPennID']

        #     keepCol = ['UID', 'CSN', 'FirstName', 'LastName', 'VisitStartDTime',
        #                'DOB', 'AGE', 'Sex', 'maritalStatus', 'Race',]
        #     keepColLocation = ['Facility', 'HospitalService', 'Loc_Dept', 'Loc_Room_bed', 'ServiceName',
        #                        'Nav_Location', 'Nav_Room_bed']
        #     keepColAdmit = ['AdmitReason', 'AdmitSourceName', 'AdmitType','AdmitSourceCode', ]
        #     keepColTeam = ['AttendingName', 'AttendingPennID', 'CoveringName', 'CoveringPennID',
        #                    'CoveringNurseName', 'CoveringNursePennID',
        #                    'CoveringCellNumber', 'CoveringPagerNumber',
        #                    'CoveringNurseCellNumber', 'CoveringNursePagerNumber']
        #     keepMiscInfo = ['EpicHAR', 'LastUpdateDtime', 'HUPMRN', 'MVPatientID', 'PAHMRN', 'PMCMRN',
        #                     'NameSuffix', 'MiddleName', 'AdmitHeightM', 'AdmitWeightKG']
        #     keepColAll = keepCol + keepColLocation + keepColAdmit + keepColTeam + keepMiscInfo
        keepColAll = list(df_prediction_cohort)

        X = df_prediction_input.copy()
        res_model = self.clf

        s_scores = pd.DataFrame(res_model.predict_proba(X))[1].rename(columns={0: 'Y_PRED'})

        df_scores = pd.concat([df_prediction_cohort[keepColAll],
                               s_scores.rename('Y_PRED')],
                              axis='columns', sort=True)

        return df_scores

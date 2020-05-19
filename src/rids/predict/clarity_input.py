"""Clarity Input.
Used to get
- historical diagnoses
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
import re
from sys import stdout
from pandas import DataFrame
import pandas as pd
from yaml import safe_load as yaml_loads

from ..mssql import (
    Input as BaseMssqlInput,
    retry_on_operational_error,
    Uri as MssqlUri,
)

from .sqlqueries import (
    comorbidities_sql,
    # comorbidities_pds_sql
)

basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name

Batch = namedtuple('Batch_clarityInput', 'dx_hx')

strTZ = 'US/Eastern'
dt_epoch = datetime(1970, 1, 1, 0, tzinfo=timezone.utc)

def str_time(x):
    """Str to time."""
    if pd.isnull(x):
        return None
    return x.strftime('%Y-%m-%d %H:%M:%S')


class Input(BaseMssqlInput):
    """Input."""

    ARGS = {
        ('CLARITY_INPUT_URI', '--clarity-input-uri'): {
            'dest': 'clarity_input_uri',
            'help': 'Clarity Mssql input uri.',
            'type': str,
        },
        ('CLARITY_INPUT_TABLES', '--clarity-input-tables'): {
            'dest': 'clarity_input_tables',
            'help': 'Yaml configuration file of Clarity tables.',
            'type': str,
        },
    }

    @classmethod
    def from_cfg(cls, cfg: dict) -> Input:
        """Return model from cfg."""
        kwargs = {
            key: cast(cfg[key])
            for key, cast in (
                ('uri', MssqlUri.from_cfg),
                ('tables', list))}
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch args into cfg."""
        for key, value in (
                ('uri', args.clarity_input_uri),):
            if value is not None:
                cfg[key] = value

        for key, value in (
                ('tables', args.clarity_input_tables),):
            if value is not None:
                with open(value) as fin:
                    cfg[key] = yaml_loads(fin.read())
        return cfg

    def __init__(
            self,
            uri: MssqlUri,
            tables: list) -> None:
        super().__init__(uri, tables)

    def __call__(self, df_i_cohort, dt_now) -> tuple:
        """Return a tuple of input dfs.
        """
        lstintPatientUID = list(df_i_cohort['UID'].unique())
        nTestPatients = len(lstintPatientUID)

        with self.rollback() as cursor:
            df_i_dx_hx = self.get_dx_hx(cursor, dt_now, lstintPatientUID[:nTestPatients])
            # df_i_labs_hx = get_labs_hx(cursor, now)
            # df_i_proc_hx = get_proc_hx(cursor, now)

        return Batch(dx_hx=df_i_dx_hx)

    def get_dx_hx(self, cursor, dt_now, lstintPatientUID) -> pd.DataFrame:
        # TODO: The query will be a problem when there are over 1k patients

        query = comorbidities_sql.format(LIST_STR_INT_UID="'" +
                                                          "', '".join([str(int(x)) for x in lstintPatientUID]) +
                                                          "'",
                                         DT_NOW=str(dt_now.astimezone(gettz(strTZ)))[0:len('yyyy-mm-dd')],
                                         NUM_LOOKBACKDAYS=365)
        # df_i_dx_hx = ClarityQuery(query)
        df_i_dx_hx = self.query_to_df(cursor, query)

        return df_i_dx_hx

    def get_labs_hx(self, cursor, now_dt):
        # return df_i_labs_hx
        pass

    def get_proc_hx(self, cursor, now_dt):
        # return df_i_proc_hx
        pass

    # @retry_on_operational_error()
    # def get_labs(self, cursor) -> DataFrame:
    #     """Get lab results dataframe."""
    #     q = re.sub('COHORT_TO_DATE', self.cohort_to_date, labs_sql)
    #
    #     labs_lookback_days = 6 * 30  # 28
    #     labs_from_date = (pd.to_datetime(self.cohort_to_date) -
    #                       datetime.timedelta(labs_lookback_days))
    #     labs_from_date = labs_from_date.strftime('%Y-%m-%d')
    #     q = re.sub('N_MONTHS_PRIOR', labs_from_date, q)
    #
    #     q = re.sub('LABS_LIST', self.labs_list, q)
    #     logger.info(q)
    #     sql = self.cohort_q + q
    #     labs_df = self.query_to_df(cursor, sql, params=self.pat_ids)
    #     n_out = labs_df.shape[0]
    #     logger.info('{"input.get_labs": {"n_out": %i}}', n_out)
    #     return labs_df
    #
    # @retry_on_operational_error()
    # def get_dx(self, cursor) -> DataFrame:
    #     """Get diagnoses results dataframe."""
    #     q = re.sub('COHORT_FROM_DATE', self.cohort_from_date, dx_sql)
    #     q = re.sub('COHORT_TO_DATE', self.cohort_to_date, q)
    #     logger.info(q)
    #     sql = self.cohort_q + q
    #     dx_df = self.query_to_df(cursor, sql, params=self.pat_ids)
    #
    #     n_out = dx_df.shape[0]
    #     logger.info('{"input.get_dx": {"n_out": %i}}', n_out)
    #     return dx_df
    #
    # @retry_on_operational_error()
    # def get_sicp(self, cursor) -> DataFrame:
    #     """Get SICP notes dataframe."""
    #     q = sicp_sql
    #     logger.info(q)
    #     sql = self.cohort_q + q
    #     sicp_df = self.query_to_df(cursor, sql, params=self.pat_ids)
    #
    #     n_out = sicp_df.shape[0]
    #     logger.info('{"input.get_sicp": {"n_out": %i}}', n_out)
    #     return sicp_df

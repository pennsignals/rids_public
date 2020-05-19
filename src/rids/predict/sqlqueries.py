"""Sql queries."""
# pylint: disable=invalid-name

comorbidities_sql = """select distinct identity_id.identity_id as UID,
                        hsp_acct_dx_list.line as diagnosis_sequence,
                        hsp_acct_dx_list.dx_start_dt,
                        pat_enc_hsp.HOSP_ADMSN_TIME,

                        clarity_edg.dx_name,
                        clarity_edg.current_icd9_list,
                        clarity_edg.current_icd10_list

                    from pat_enc_hsp
                    inner join identity_id
                        on identity_id.pat_id=pat_enc_hsp.pat_id
                        and identity_id.identity_type_id=105 -- UID from IDENTITY_ID_TYPE table
                    inner join hsp_acct_dx_list
                        on hsp_acct_dx_list.hsp_account_id=pat_enc_hsp.hsp_account_id
                    inner join clarity_edg
                        on clarity_edg.dx_id=hsp_acct_dx_list.dx_id

                    where identity_id.identity_id in ({LIST_STR_INT_UID})
                        and DATEADD(day,-{NUM_LOOKBACKDAYS},'{DT_NOW}') <= pat_enc_hsp.hosp_admsn_time
                        and pat_enc_hsp.hosp_admsn_time < '{DT_NOW}'
                    order by identity_id.identity_id
                        , pat_enc_hsp.hosp_admsn_time
                        , hsp_acct_dx_list.line
                    
            		"""

comorbidities_pds_sql = """select distinct patient_encounter.empi as "UID",
                            diagnosis.diagnosis_sequence,
                            diagnosis.coding_date as dx_start_dt,
                            patient_encounter.enc_date as HOSP_ADMSN_TIME,

                            r_standard_codes.code_description as dx_name,
                            null as CURRENT_ICD9_LIST,
                            r_standard_codes.code as CURRENT_ICD10_LIST

                        from mdm.patient_encounter patient_encounter
                        inner join mdm.diagnosis diagnosis on diagnosis.fk_patient_encounter_id = patient_encounter.pk_patient_encounter_id
                        inner join mdm.r_standard_codes r_standard_codes
                            on r_standard_codes.pk_standard_code_id = diagnosis.fk_standard_code_id
                        where diagnosis.diagnosis_type <> 'ADMITTING'
                            and patient_encounter.empi in ({LIST_INT_UID})
                              and (date '{DT_NOW}'-{NUM_LOOKBACKDAYS}) <= diagnosis.coding_date
                              and diagnosis.coding_date < date '{DT_NOW}'
                        order by patient_encounter.empi
                            , patient_encounter.enc_date
                            , diagnosis.diagnosis_sequence
                """


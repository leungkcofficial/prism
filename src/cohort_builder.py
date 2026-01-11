"""
PRISM Cohort Formation Module

This module implements the cohort selection criteria for the PRISM causal survival analysis:
1. Persistent eGFR <15 screening: Two outpatient eGFR <15 separated by 90-365 days
2. Index date (t₀) definition: First outpatient eGFR ≤10 after screening eligibility
3. Treatment labeling: Early dialysis (A=1) if initiated within 90 days of t₀
4. Survival outcome: Time from t₀ to death or censoring (5-year follow-up)

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CohortBuilder:
    """
    Builds PRISM cohort with t₀ definition, treatment assignment, and survival outcomes.

    Parameters
    ----------
    egfr_screen_lt15_min_days : int
        Minimum days between two eGFR <15 for persistent screening (default: 90)
    egfr_screen_lt15_max_days : int
        Maximum days between two eGFR <15 for persistent screening (default: 365)
    t0_threshold : float
        eGFR threshold for t₀ definition (default: 10.0 mL/min/1.73m²)
    early_window_days : int
        Days after t₀ to define early dialysis treatment (default: 90)
    study_end_date : str
        Study end date for censoring (default: "2023-12-31")
    max_followup_days : int
        Maximum follow-up time in days (default: 1825 = 5 years)
    require_confirmatory_egfr : bool
        Whether to require confirmatory eGFR ≤10 within 14-30 days (default: False)
    outpatient_only : bool
        Whether to use only outpatient records to avoid AKI (default: True)
    """

    def __init__(
        self,
        egfr_screen_lt15_min_days: int = 90,
        egfr_screen_lt15_max_days: int = 365,
        t0_threshold: float = 10.0,
        early_window_days: int = 90,
        study_end_date: str = "2023-12-31",
        max_followup_days: int = 1825,
        require_confirmatory_egfr: bool = False,
        outpatient_only: bool = True
    ):
        self.egfr_screen_lt15_min_days = egfr_screen_lt15_min_days
        self.egfr_screen_lt15_max_days = egfr_screen_lt15_max_days
        self.t0_threshold = t0_threshold
        self.early_window_days = early_window_days
        self.study_end_date = pd.to_datetime(study_end_date)
        self.max_followup_days = max_followup_days
        self.require_confirmatory_egfr = require_confirmatory_egfr
        self.outpatient_only = outpatient_only

    def build_cohort(
        self,
        cr_df: pd.DataFrame,
        operation_df: pd.DataFrame,
        death_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build PRISM cohort with t₀, treatment A, and survival outcomes.

        Parameters
        ----------
        cr_df : pd.DataFrame
            Creatinine data with columns: [key, date, creatinine, age, gender, egfr, is_outpatient]
            Note: eGFR should already be calculated using CKD-EPI formula
        operation_df : pd.DataFrame
            Operation/dialysis data with columns: [key, date, is_dialysis]
        death_df : pd.DataFrame
            Death data with columns: [key, death_date]

        Returns
        -------
        pd.DataFrame
            Cohort with columns: [key, t0_date, age_at_t0, gender, A, duration, event]
            - key: Patient identifier
            - t0_date: Index date (first eGFR ≤10 after screening)
            - age_at_t0: Age at t₀
            - gender: Gender (M/F)
            - A: Treatment (1=early dialysis within 90 days, 0=non-early)
            - duration: Days from t₀ to death/censoring (capped at 1825)
            - event: Event indicator (1=death, 0=censored)
        """
        logger.info("Starting PRISM cohort formation...")
        logger.info(f"Input data: {len(cr_df)} creatinine records, {len(operation_df)} operations, {len(death_df)} deaths")

        # Step 1: Persistent eGFR <15 screening
        screening_eligible_df = self._identify_persistent_egfr_lt15(cr_df)
        logger.info(f"Step 1: {len(screening_eligible_df)} patients with persistent eGFR <15")

        # Step 2: Define t₀ (first eGFR ≤10 after screening)
        t0_df = self._define_index_date(cr_df, screening_eligible_df)
        logger.info(f"Step 2: {len(t0_df)} patients with t₀ defined (eGFR ≤{self.t0_threshold})")

        # Step 3: Label treatment (early vs non-early dialysis)
        cohort_df = self._label_treatment(t0_df, operation_df)
        logger.info(f"Step 3: Treatment labeled - {cohort_df['A'].sum()} early dialysis (A=1), {(cohort_df['A']==0).sum()} non-early (A=0)")

        # Step 4: Calculate survival outcomes
        cohort_df = self._calculate_survival_outcomes(cohort_df, death_df)
        logger.info(f"Step 4: Survival outcomes - {cohort_df['event'].sum()} deaths, {(cohort_df['event']==0).sum()} censored")
        logger.info(f"Final cohort size: {len(cohort_df)} patients")

        return cohort_df

    def _identify_persistent_egfr_lt15(self, cr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify patients with persistent eGFR <15 (two measurements 90-365 days apart).

        Returns DataFrame with columns: [key, screening_date]
        """
        # Filter to outpatient records if required
        if self.outpatient_only:
            if 'is_outpatient' in cr_df.columns:
                cr_subset = cr_df[cr_df['is_outpatient'] == True].copy()
            else:
                logger.warning("is_outpatient column not found, using all records")
                cr_subset = cr_df.copy()
        else:
            cr_subset = cr_df.copy()

        # Filter to eGFR <15
        cr_subset = cr_subset[cr_subset['egfr'] < 15].copy()
        cr_subset = cr_subset.sort_values(['key', 'date'])

        # Find pairs of eGFR <15 separated by 90-365 days
        screening_eligible = []

        for patient_key, group in cr_subset.groupby('key'):
            dates = group['date'].values

            if len(dates) < 2:
                continue

            # Check all pairs
            for i in range(len(dates)):
                for j in range(i + 1, len(dates)):
                    days_apart = (pd.to_datetime(dates[j]) - pd.to_datetime(dates[i])).days

                    if self.egfr_screen_lt15_min_days <= days_apart <= self.egfr_screen_lt15_max_days:
                        # Found valid pair - use second date as screening date
                        screening_eligible.append({
                            'key': patient_key,
                            'screening_date': pd.to_datetime(dates[j])
                        })
                        break  # Only need one valid pair per patient

                if len([x for x in screening_eligible if x['key'] == patient_key]) > 0:
                    break  # Already found valid pair for this patient

        screening_df = pd.DataFrame(screening_eligible)

        # Remove duplicates (keep earliest screening date)
        if len(screening_df) > 0:
            screening_df = screening_df.sort_values('screening_date').groupby('key').first().reset_index()

        return screening_df

    def _define_index_date(
        self,
        cr_df: pd.DataFrame,
        screening_eligible_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Define t₀ as first outpatient eGFR ≤10 after screening date.

        Returns DataFrame with columns: [key, t0_date, t0_egfr, age_at_t0, gender]
        """
        # Filter to outpatient records if required
        if self.outpatient_only:
            if 'is_outpatient' in cr_df.columns:
                cr_subset = cr_df[cr_df['is_outpatient'] == True].copy()
            else:
                cr_subset = cr_df.copy()
        else:
            cr_subset = cr_df.copy()

        # Merge with screening dates
        cr_with_screening = cr_subset.merge(
            screening_eligible_df[['key', 'screening_date']],
            on='key',
            how='inner'
        )

        # Filter to eGFR ≤ threshold and after screening date
        cr_with_screening = cr_with_screening[
            (cr_with_screening['egfr'] <= self.t0_threshold) &
            (cr_with_screening['date'] >= cr_with_screening['screening_date'])
        ].copy()

        # Sort and take first eGFR ≤10 per patient
        cr_with_screening = cr_with_screening.sort_values(['key', 'date'])
        t0_df = cr_with_screening.groupby('key').first().reset_index()

        # Rename columns
        t0_df = t0_df.rename(columns={
            'date': 't0_date',
            'egfr': 't0_egfr',
            'age': 'age_at_t0'
        })

        # Select relevant columns
        columns_to_keep = ['key', 't0_date', 't0_egfr', 'age_at_t0']
        if 'gender' in t0_df.columns:
            columns_to_keep.append('gender')

        t0_df = t0_df[columns_to_keep]

        # Optional: Require confirmatory eGFR ≤10 within 14-30 days
        if self.require_confirmatory_egfr:
            t0_df = self._apply_confirmatory_egfr(t0_df, cr_subset)

        return t0_df

    def _apply_confirmatory_egfr(
        self,
        t0_df: pd.DataFrame,
        cr_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Require confirmatory eGFR ≤10 within 14-30 days of initial t₀.

        This is an optional sensitivity analysis criterion.
        """
        confirmed_patients = []

        for _, row in t0_df.iterrows():
            patient_key = row['key']
            t0_date = row['t0_date']

            # Find eGFR measurements 14-30 days after t₀
            patient_cr = cr_df[
                (cr_df['key'] == patient_key) &
                (cr_df['date'] > t0_date) &
                (cr_df['date'] <= t0_date + timedelta(days=30)) &
                (cr_df['date'] >= t0_date + timedelta(days=14))
            ]

            # Check if any confirmatory eGFR ≤10
            if len(patient_cr[patient_cr['egfr'] <= self.t0_threshold]) > 0:
                confirmed_patients.append(patient_key)

        t0_df = t0_df[t0_df['key'].isin(confirmed_patients)]
        logger.info(f"Confirmatory eGFR requirement: {len(t0_df)} patients retained")

        return t0_df

    def _label_treatment(
        self,
        t0_df: pd.DataFrame,
        operation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Label treatment A: 1 if early dialysis (within 90 days of t₀), 0 otherwise.
        """
        cohort_df = t0_df.copy()
        cohort_df['A'] = 0  # Default: non-early

        # Filter to dialysis operations
        dialysis_df = operation_df[operation_df['is_dialysis'] == True].copy()

        # For each patient, check if dialysis within 90 days of t₀
        for idx, row in cohort_df.iterrows():
            patient_key = row['key']
            t0_date = row['t0_date']

            # Find dialysis for this patient
            patient_dialysis = dialysis_df[dialysis_df['key'] == patient_key]

            if len(patient_dialysis) > 0:
                # Get earliest dialysis date
                earliest_dialysis = patient_dialysis['date'].min()

                # Check if within early window
                days_to_dialysis = (pd.to_datetime(earliest_dialysis) - pd.to_datetime(t0_date)).days

                if 0 <= days_to_dialysis <= self.early_window_days:
                    cohort_df.at[idx, 'A'] = 1

        return cohort_df

    def _calculate_survival_outcomes(
        self,
        cohort_df: pd.DataFrame,
        death_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate survival outcomes: duration (days from t₀ to event/censoring) and event indicator.
        """
        # Merge with death data
        cohort_df = cohort_df.merge(
            death_df[['key', 'death_date']],
            on='key',
            how='left'
        )

        # Calculate duration and event
        cohort_df['duration'] = np.nan
        cohort_df['event'] = 0

        for idx, row in cohort_df.iterrows():
            t0_date = pd.to_datetime(row['t0_date'])
            death_date = pd.to_datetime(row['death_date']) if pd.notna(row['death_date']) else None

            if death_date is not None:
                # Patient died
                days_to_death = (death_date - t0_date).days

                if days_to_death <= self.max_followup_days:
                    # Death within follow-up window
                    cohort_df.at[idx, 'duration'] = days_to_death
                    cohort_df.at[idx, 'event'] = 1
                else:
                    # Death after follow-up window - censor at max follow-up
                    cohort_df.at[idx, 'duration'] = self.max_followup_days
                    cohort_df.at[idx, 'event'] = 0
            else:
                # Patient censored (no death recorded)
                days_to_censor = (self.study_end_date - t0_date).days
                cohort_df.at[idx, 'duration'] = min(days_to_censor, self.max_followup_days)
                cohort_df.at[idx, 'event'] = 0

        # Remove patients with negative or zero duration
        cohort_df = cohort_df[cohort_df['duration'] > 0].copy()

        # Drop temporary columns
        cohort_df = cohort_df.drop(columns=['death_date', 't0_egfr'], errors='ignore')

        # Convert duration to integer
        cohort_df['duration'] = cohort_df['duration'].astype(int)
        cohort_df['A'] = cohort_df['A'].astype(int)
        cohort_df['event'] = cohort_df['event'].astype(int)

        return cohort_df

    def get_cohort_summary(self, cohort_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the cohort.

        Returns
        -------
        dict
            Summary statistics including sample size, treatment balance, event rates, etc.
        """
        summary = {
            'n_total': len(cohort_df),
            'n_early_dialysis': int(cohort_df['A'].sum()),
            'n_non_early': int((cohort_df['A'] == 0).sum()),
            'pct_early_dialysis': float(cohort_df['A'].mean() * 100),
            'n_events': int(cohort_df['event'].sum()),
            'n_censored': int((cohort_df['event'] == 0).sum()),
            'event_rate': float(cohort_df['event'].mean() * 100),
            'median_followup_days': float(cohort_df['duration'].median()),
            'median_age_at_t0': float(cohort_df['age_at_t0'].median()),
            'event_rate_early': float(cohort_df[cohort_df['A'] == 1]['event'].mean() * 100),
            'event_rate_non_early': float(cohort_df[cohort_df['A'] == 0]['event'].mean() * 100),
        }

        return summary

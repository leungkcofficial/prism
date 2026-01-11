"""
PRISM Feature Extraction Module

This module implements t₀-centric feature extraction with lookback windows:
1. Lab features: Closest value within 90 days before t₀
2. CCI features: ICD-10 codes within 5 years before t₀
3. UACR derivation from UPCR when needed
4. Derived features: time_since_ckd_days

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional
import logging

# Import existing data processing utilities
from src.dx_ingester import ICD10Ingester
from src.lab_result_mapper import UrineLabResultMapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features with lookback windows from index date (t₀).

    Parameters
    ----------
    lab_lookback_days : int
        Days to look back for lab values (default: 90)
    cci_lookback_years : int
        Years to look back for comorbidities (default: 5)
    derive_uacr_from_upcr : bool
        Whether to derive UACR from UPCR when UACR missing (default: True)
    lab_features : List[str]
        List of lab features to extract (default: standard CKD labs)
    """

    # Default lab features for CKD survival analysis
    DEFAULT_LAB_FEATURES = [
        'creatinine',    # Cr (μmol/L)
        'hemoglobin',    # Hb (g/dL)
        'albumin',       # Alb (g/L)
        'a1c',           # HbA1c (%)
        'phosphate',     # PO4 (mmol/L)
        'calcium',       # Ca (mmol/L)
        'bicarbonate',   # HCO3 (mmol/L)
        'uacr'           # UACR (mg/mmol)
    ]

    # CCI comorbidity flags (19 categories)
    CCI_FLAGS = [
        'myocardial_infarction',
        'congestive_heart_failure',
        'peripheral_vascular_disease',
        'cerebrovascular_disease',
        'dementia',
        'copd',
        'rheumatic_disease',
        'peptic_ulcer_disease',
        'liver_disease_mild',
        'diabetes_wo_complication',
        'diabetes_w_complication',
        'hemiplegia_paraplegia',
        'renal_disease',
        'cancer',
        'liver_disease_mod_severe',
        'metastatic_solid_tumor',
        'aids_hiv'
    ]

    def __init__(
        self,
        lab_lookback_days: int = 90,
        cci_lookback_years: int = 5,
        derive_uacr_from_upcr: bool = True,
        lab_features: Optional[List[str]] = None
    ):
        self.lab_lookback_days = lab_lookback_days
        self.cci_lookback_years = cci_lookback_years
        self.derive_uacr_from_upcr = derive_uacr_from_upcr
        self.lab_features = lab_features if lab_features is not None else self.DEFAULT_LAB_FEATURES

    def extract(
        self,
        cohort_df: pd.DataFrame,
        lab_dfs: Dict[str, pd.DataFrame],
        icd10_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features with lookback windows from t₀.

        Parameters
        ----------
        cohort_df : pd.DataFrame
            Cohort with columns: [key, t0_date, age_at_t0, gender, A, duration, event]
        lab_dfs : Dict[str, pd.DataFrame]
            Dictionary of lab DataFrames with keys matching lab_features
            Each DataFrame has columns: [key, date, value]
        icd10_df : pd.DataFrame
            ICD-10 diagnosis data with columns: [key, date, icd10_code]

        Returns
        -------
        pd.DataFrame
            Features with columns: [key, cr_at_t0, hb_at_t0, ..., cci_score_total, time_since_ckd_days]
        """
        logger.info("Starting PRISM feature extraction...")
        logger.info(f"  - Lab lookback window: {self.lab_lookback_days} days")
        logger.info(f"  - CCI lookback window: {self.cci_lookback_years} years")
        logger.info(f"  - Lab features: {', '.join(self.lab_features)}")

        # Initialize features DataFrame
        features_df = cohort_df[['key', 't0_date']].copy()

        # Step 1: Extract lab features with 90-day lookback
        for lab_name in self.lab_features:
            if lab_name in lab_dfs:
                features_df = self._extract_lab_feature(
                    features_df,
                    lab_dfs[lab_name],
                    lab_name
                )
            else:
                logger.warning(f"Lab '{lab_name}' not found in lab_dfs, skipping")

        # Step 2: Derive UACR from UPCR if needed
        if self.derive_uacr_from_upcr and 'uacr' in self.lab_features:
            if 'upcr' in lab_dfs:
                features_df = self._derive_uacr_from_upcr(
                    features_df,
                    lab_dfs['upcr']
                )
            else:
                logger.warning("UPCR data not found, cannot derive UACR")

        # Step 3: Extract CCI features with 5-year lookback
        features_df = self._extract_cci_features(features_df, icd10_df)

        # Step 4: Calculate time since CKD onset
        if 'creatinine' in lab_dfs:
            features_df = self._calculate_time_since_ckd(
                features_df,
                lab_dfs['creatinine']
            )

        # Drop t0_date (no longer needed)
        features_df = features_df.drop(columns=['t0_date'])

        # Log summary
        logger.info(f"Feature extraction complete: {len(features_df)} patients, {len(features_df.columns)-1} features")
        logger.info(f"  - Lab features: {sum([col.endswith('_at_t0') for col in features_df.columns])}")
        logger.info(f"  - CCI features: {sum([col in self.CCI_FLAGS for col in features_df.columns])}")

        return features_df

    def _extract_lab_feature(
        self,
        features_df: pd.DataFrame,
        lab_df: pd.DataFrame,
        lab_name: str
    ) -> pd.DataFrame:
        """
        Extract closest lab value within lookback window from t₀.

        Uses pandas merge_asof for efficient temporal join.
        """
        # Prepare lab data
        # Check if column is named 'value' or 'result_value'
        value_col = 'value' if 'value' in lab_df.columns else 'result_value'
        lab_subset = lab_df[['key', 'date', value_col]].copy()

        # Remove rows with null key or date values (required for merge_asof)
        lab_subset = lab_subset.dropna(subset=['key', 'date'])

        # Ensure 'key' is int64 (required for merge - must match across dataframes)
        lab_subset['key'] = lab_subset['key'].astype('int64')

        # Ensure date columns are datetime
        lab_subset['date'] = pd.to_datetime(lab_subset['date'], errors='coerce')

        # Remove any rows that became null after datetime conversion
        lab_subset = lab_subset.dropna(subset=['date'])

        # Sort by date only (not by key!) for merge_asof
        lab_subset = lab_subset.sort_values('date').reset_index(drop=True)
        lab_subset = lab_subset.rename(columns={value_col: f'{lab_name}_at_t0'})

        # Prepare features data
        features_subset = features_df[['key', 't0_date']].copy()

        # Remove rows with null key or t0_date values (required for merge_asof)
        features_subset = features_subset.dropna(subset=['key', 't0_date'])

        # Ensure 'key' is int64 (required for merge - must match across dataframes)
        features_subset['key'] = features_subset['key'].astype('int64')

        # Ensure t0_date is datetime
        features_subset['t0_date'] = pd.to_datetime(features_subset['t0_date'], errors='coerce')

        # Remove any rows that became null after datetime conversion
        features_subset = features_subset.dropna(subset=['t0_date'])

        # Sort by t0_date only (not by key!) for merge_asof
        features_subset = features_subset.sort_values('t0_date').reset_index(drop=True)

        # Calculate lookback start date
        features_subset['lookback_start'] = features_subset['t0_date'] - timedelta(days=self.lab_lookback_days)

        # Merge using merge_asof for closest value within window
        merged = pd.merge_asof(
            features_subset,
            lab_subset,
            left_on='t0_date',
            right_on='date',
            by='key',
            direction='backward',
            tolerance=pd.Timedelta(days=self.lab_lookback_days)
        )

        # Add to features_df
        features_df = features_df.merge(
            merged[['key', f'{lab_name}_at_t0']],
            on='key',
            how='left'
        )

        # Log missing rate
        missing_rate = features_df[f'{lab_name}_at_t0'].isna().mean() * 100
        logger.info(f"  - {lab_name}: {missing_rate:.1f}% missing")

        return features_df

    def _derive_uacr_from_upcr(
        self,
        features_df: pd.DataFrame,
        upcr_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Derive UACR from UPCR using published conversion formulas.

        Formula: UACR (mg/mmol) ≈ UPCR (mg/mmol) × 0.6
        (Based on correlation between urinary protein and albumin)
        """
        # Only derive for patients with missing UACR
        missing_uacr_mask = features_df['uacr_at_t0'].isna()

        if missing_uacr_mask.sum() == 0:
            logger.info("  - UACR derivation: No missing values, skipping")
            return features_df

        # Extract UPCR values for patients with missing UACR
        missing_uacr_keys = features_df.loc[missing_uacr_mask, 'key'].unique()

        upcr_subset = upcr_df[upcr_df['key'].isin(missing_uacr_keys)].copy()
        upcr_subset = upcr_subset.sort_values(['key', 'date'])

        # Use same lookback window as other labs
        features_subset = features_df.loc[missing_uacr_mask, ['key', 't0_date']].copy()
        features_subset = features_subset.sort_values(['key', 't0_date'])

        # Merge using merge_asof
        merged = pd.merge_asof(
            features_subset,
            upcr_subset[['key', 'date', 'value']],
            left_on='t0_date',
            right_on='date',
            by='key',
            direction='backward',
            tolerance=pd.Timedelta(days=self.lab_lookback_days)
        )

        # Apply conversion formula
        merged['derived_uacr'] = merged['value'] * 0.6

        # Fill missing UACR values
        for idx, row in merged.iterrows():
            if pd.notna(row['derived_uacr']):
                key = row['key']
                features_df.loc[features_df['key'] == key, 'uacr_at_t0'] = row['derived_uacr']

        n_derived = merged['derived_uacr'].notna().sum()
        logger.info(f"  - UACR derivation: Derived {n_derived} values from UPCR")

        return features_df

    def _extract_cci_features(
        self,
        features_df: pd.DataFrame,
        icd10_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract Charlson Comorbidity Index features with 5-year lookback.

        Uses existing ICD10Ingester to process ICD-10 codes.
        """
        logger.info(f"  - Extracting CCI features (5-year lookback)...")

        # Initialize CCI columns
        for flag in self.CCI_FLAGS:
            features_df[flag] = 0

        features_df['cci_score_total'] = 0

        # Process each patient
        for idx, row in features_df.iterrows():
            patient_key = row['key']
            t0_date = row['t0_date']
            lookback_start = t0_date - timedelta(days=365 * self.cci_lookback_years)

            # Filter ICD-10 codes for this patient within lookback window
            patient_icd10 = icd10_df[
                (icd10_df['key'] == patient_key) &
                (icd10_df['date'] >= lookback_start) &
                (icd10_df['date'] <= t0_date)
            ]

            if len(patient_icd10) > 0:
                # Use ICD10Ingester to compute CCI
                dx_ingester = ICD10Ingester()
                cci_result = dx_ingester.process_patient_diagnoses(patient_icd10)

                # Update features
                for flag in self.CCI_FLAGS:
                    if flag in cci_result:
                        features_df.at[idx, flag] = int(cci_result[flag])

                if 'cci_score_total' in cci_result:
                    features_df.at[idx, 'cci_score_total'] = cci_result['cci_score_total']

        # Log prevalence
        for flag in self.CCI_FLAGS:
            prevalence = features_df[flag].mean() * 100
            if prevalence > 0:
                logger.info(f"    - {flag}: {prevalence:.1f}%")

        mean_cci = features_df['cci_score_total'].mean()
        logger.info(f"  - Mean CCI score: {mean_cci:.2f}")

        return features_df

    def _calculate_time_since_ckd(
        self,
        features_df: pd.DataFrame,
        cr_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate time since CKD onset (first eGFR <60).

        Returns time_since_ckd_days: Days from first eGFR <60 to t₀.
        """
        logger.info("  - Calculating time since CKD onset (eGFR <60)...")

        features_df['time_since_ckd_days'] = np.nan

        # Assume cr_df has 'egfr' column already calculated
        if 'egfr' not in cr_df.columns:
            logger.warning("eGFR column not found in creatinine data, skipping time_since_ckd calculation")
            return features_df

        # Find first eGFR <60 for each patient
        ckd_onset_df = cr_df[cr_df['egfr'] < 60].copy()
        ckd_onset_df = ckd_onset_df.sort_values(['key', 'date'])
        ckd_onset_df = ckd_onset_df.groupby('key').first().reset_index()
        ckd_onset_df = ckd_onset_df[['key', 'date']].rename(columns={'date': 'ckd_onset_date'})

        # Merge with features
        features_df = features_df.merge(ckd_onset_df, on='key', how='left')

        # Calculate days since CKD onset
        features_df['time_since_ckd_days'] = (
            (features_df['t0_date'] - features_df['ckd_onset_date']).dt.days
        )

        # Drop temporary column
        features_df = features_df.drop(columns=['ckd_onset_date'])

        # Log summary
        median_days = features_df['time_since_ckd_days'].median()
        logger.info(f"    - Median time since CKD onset: {median_days:.0f} days ({median_days/365:.1f} years)")

        return features_df

    def get_feature_summary(self, features_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for extracted features.

        Returns
        -------
        dict
            Summary statistics including missing rates, means, medians
        """
        summary = {
            'n_patients': len(features_df),
            'n_features': len(features_df.columns) - 1,  # Exclude 'key'
            'missing_rates': {},
            'lab_features': {},
            'cci_features': {}
        }

        # Lab features
        lab_cols = [col for col in features_df.columns if col.endswith('_at_t0')]
        for col in lab_cols:
            summary['missing_rates'][col] = float(features_df[col].isna().mean() * 100)
            summary['lab_features'][col] = {
                'mean': float(features_df[col].mean()),
                'median': float(features_df[col].median()),
                'std': float(features_df[col].std())
            }

        # CCI features
        for flag in self.CCI_FLAGS:
            if flag in features_df.columns:
                prevalence = float(features_df[flag].mean() * 100)
                summary['cci_features'][flag] = prevalence

        if 'cci_score_total' in features_df.columns:
            summary['cci_score_mean'] = float(features_df['cci_score_total'].mean())
            summary['cci_score_median'] = float(features_df['cci_score_total'].median())

        return summary

"""
PRISM Feature Extraction Step

ZenML pipeline step for extracting features with t₀-centric lookback windows:
- Lab features: 90-day lookback
- CCI features: 5-year lookback
- UACR derivation from UPCR
- Time since CKD onset

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
from zenml.steps import step
from typing import Dict
import logging

# Import the FeatureExtractor from src/
from src.feature_extractor import FeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def extract_features(
    cohort_df: pd.DataFrame,
    lab_dfs: Dict[str, pd.DataFrame],
    icd10_df: pd.DataFrame,
    config: dict = None
) -> pd.DataFrame:
    """
    ZenML step: Extract features with lookback windows from t₀.

    This step wraps the FeatureExtractor class and handles configuration from the pipeline.

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort with columns: [key, t0_date, age_at_t0, gender, A, duration, event]
    lab_dfs : Dict[str, pd.DataFrame]
        Dictionary of lab DataFrames with standardized format
        Keys: lab names (creatinine, hemoglobin, albumin, etc.)
        Values: DataFrames with columns [key, date, value]
    icd10_df : pd.DataFrame
        ICD-10 diagnosis data with columns: [key, date, icd10_code]
    config : dict, optional
        Configuration dictionary with feature extraction parameters:
        - lab_lookback_days: Days to look back for labs (default: 90)
        - cci_lookback_years: Years to look back for CCI (default: 5)
        - derive_uacr_from_upcr: Derive UACR from UPCR (default: True)
        - lab_features: List of lab features to extract (default: all)

    Returns
    -------
    pd.DataFrame
        Features with columns: [key, cr_at_t0, hb_at_t0, ..., cci_score_total, time_since_ckd_days]
    """
    logger.info("=" * 80)
    logger.info("PRISM FEATURE EXTRACTION STEP")
    logger.info("=" * 80)

    # Extract feature configuration from config dict
    feature_config = config.get('features', {}) if config is not None else {}

    # Initialize FeatureExtractor with config
    extractor = FeatureExtractor(
        lab_lookback_days=feature_config.get('lab_lookback_days', 90),
        cci_lookback_years=feature_config.get('cci_lookback_years', 5),
        derive_uacr_from_upcr=feature_config.get('derive_uacr_from_upcr', True),
        lab_features=feature_config.get('lab_features', None)  # None = use defaults
    )

    logger.info(f"Configuration:")
    logger.info(f"  - Lab lookback window: {extractor.lab_lookback_days} days")
    logger.info(f"  - CCI lookback window: {extractor.cci_lookback_years} years")
    logger.info(f"  - Derive UACR from UPCR: {extractor.derive_uacr_from_upcr}")
    logger.info(f"  - Lab features: {', '.join(extractor.lab_features)}")
    logger.info("")

    # Extract features
    features_df = extractor.extract(cohort_df, lab_dfs, icd10_df)

    # Generate and log summary statistics
    summary = extractor.get_feature_summary(features_df)
    logger.info("")
    logger.info("=" * 80)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total patients: {summary['n_patients']}")
    logger.info(f"Total features: {summary['n_features']}")
    logger.info("")

    # Log missing rates for lab features
    logger.info("Lab Features - Missing Rates:")
    for feature, missing_rate in summary['missing_rates'].items():
        logger.info(f"  - {feature}: {missing_rate:.1f}% missing")

    logger.info("")

    # Log CCI prevalence
    logger.info("CCI Comorbidity Prevalence:")
    for flag, prevalence in summary['cci_features'].items():
        if prevalence > 0:
            logger.info(f"  - {flag}: {prevalence:.1f}%")

    if 'cci_score_mean' in summary:
        logger.info(f"  - Mean CCI score: {summary['cci_score_mean']:.2f}")
        logger.info(f"  - Median CCI score: {summary['cci_score_median']:.1f}")

    logger.info("=" * 80)

    # Validate output
    if 'key' not in features_df.columns:
        raise ValueError("Features DataFrame must contain 'key' column")

    if len(features_df) != len(cohort_df):
        raise ValueError(
            f"Feature extraction mismatch: {len(features_df)} features vs {len(cohort_df)} cohort patients"
        )

    # Check for completely missing features
    completely_missing = [col for col in features_df.columns if col != 'key' and features_df[col].isna().all()]
    if completely_missing:
        logger.warning(f"Warning: Completely missing features: {', '.join(completely_missing)}")

    logger.info(f"✓ Feature extraction complete: {len(features_df)} patients, {len(features_df.columns)-1} features")
    logger.info("")

    return features_df

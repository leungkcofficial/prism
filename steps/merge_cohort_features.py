"""
PRISM Cohort-Features Merge Step

ZenML pipeline step for merging cohort outcomes with extracted features
to create the master dataset for training.

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
from zenml.steps import step
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def merge_cohort_features(
    cohort_df: pd.DataFrame,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    ZenML step: Merge cohort and features into master dataset.

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort with columns: [key, t0_date, age_at_t0, gender, A, duration, event]
    features_df : pd.DataFrame
        Features with columns: [key, lab_at_t0 features, CCI features, derived features]

    Returns
    -------
    pd.DataFrame
        Master dataset with all columns merged on 'key'
        Columns: [key, t0_date, age_at_t0, gender, A, duration, event, ...all features]
    """
    logger.info("=" * 80)
    logger.info("MERGING COHORT AND FEATURES")
    logger.info("=" * 80)

    # Validate inputs
    if 'key' not in cohort_df.columns:
        raise ValueError("Cohort DataFrame must contain 'key' column")

    if 'key' not in features_df.columns:
        raise ValueError("Features DataFrame must contain 'key' column")

    logger.info(f"Cohort: {len(cohort_df)} patients, {len(cohort_df.columns)} columns")
    logger.info(f"Features: {len(features_df)} patients, {len(features_df.columns)} columns")

    # Check for key overlap
    cohort_keys = set(cohort_df['key'].unique())
    feature_keys = set(features_df['key'].unique())

    only_in_cohort = cohort_keys - feature_keys
    only_in_features = feature_keys - cohort_keys

    if only_in_cohort:
        logger.warning(f"Warning: {len(only_in_cohort)} patients in cohort but not in features")

    if only_in_features:
        logger.warning(f"Warning: {len(only_in_features)} patients in features but not in cohort")

    # Merge on key (inner join to keep only patients with both cohort and features)
    master_df = cohort_df.merge(features_df, on='key', how='inner')

    logger.info(f"Master dataset: {len(master_df)} patients, {len(master_df.columns)} columns")
    logger.info("")

    # Validate merge
    if len(master_df) == 0:
        raise ValueError("Merge resulted in empty dataset - no matching keys!")

    # Check for duplicate keys
    if master_df['key'].duplicated().any():
        n_duplicates = master_df['key'].duplicated().sum()
        raise ValueError(f"Merge resulted in {n_duplicates} duplicate keys!")

    # Validate required columns exist
    required_cohort_cols = ['key', 't0_date', 'age_at_t0', 'A', 'duration', 'event']
    missing_cols = [col for col in required_cohort_cols if col not in master_df.columns]
    if missing_cols:
        raise ValueError(f"Master dataset missing required columns: {missing_cols}")

    # Log summary
    logger.info("Master Dataset Summary:")
    logger.info(f"  - Total patients: {len(master_df)}")
    logger.info(f"  - Total features: {len(master_df.columns) - len(required_cohort_cols)}")
    logger.info(f"  - Treatment balance: A=1 ({master_df['A'].sum()}, {master_df['A'].mean()*100:.1f}%), A=0 ({(master_df['A']==0).sum()}, {(1-master_df['A'].mean())*100:.1f}%)")
    logger.info(f"  - Event rate: {master_df['event'].sum()} deaths ({master_df['event'].mean()*100:.1f}%)")
    logger.info("")

    # Check for columns with all missing values
    all_missing_cols = [col for col in master_df.columns if col not in ['key'] and master_df[col].isna().all()]
    if all_missing_cols:
        logger.warning(f"Warning: Columns with 100% missing values: {', '.join(all_missing_cols)}")
        logger.warning("These columns will likely be dropped during preprocessing")
        logger.info("")

    logger.info("✓ Cohort and features merged successfully")
    logger.info("=" * 80)
    logger.info("")

    return master_df

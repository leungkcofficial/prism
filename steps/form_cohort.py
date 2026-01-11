"""
PRISM Cohort Formation Step

ZenML pipeline step for building the PRISM cohort with:
- Persistent eGFR <15 screening
- Index date (t₀) definition
- Treatment labeling (early vs non-early dialysis)
- Survival outcomes

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
from zenml.steps import step
import logging

# Import the CohortBuilder from src/
from src.cohort_builder import CohortBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def form_cohort(
    cr_df: pd.DataFrame,
    operation_df: pd.DataFrame,
    death_df: pd.DataFrame,
    config: dict = None
) -> pd.DataFrame:
    """
    ZenML step: Build PRISM cohort with t₀, treatment A, and survival outcomes.

    This step wraps the CohortBuilder class and handles configuration from the pipeline.

    Parameters
    ----------
    cr_df : pd.DataFrame
        Creatinine data with eGFR already calculated
        Columns: [key, date, creatinine, age, gender, egfr, is_outpatient]
    operation_df : pd.DataFrame
        Operation/dialysis data
        Columns: [key, date, is_dialysis]
    death_df : pd.DataFrame
        Death data
        Columns: [key, death_date]
    config : dict, optional
        Configuration dictionary with cohort formation parameters:
        - egfr_screen_lt15_min_days: Minimum days between eGFR <15 (default: 90)
        - egfr_screen_lt15_max_days: Maximum days between eGFR <15 (default: 365)
        - t0_threshold: eGFR threshold for t₀ (default: 10.0)
        - early_window_days: Days after t₀ for early dialysis (default: 90)
        - study_end_date: Study end date (default: "2023-12-31")
        - max_followup_days: Maximum follow-up days (default: 1825)
        - require_confirmatory_egfr: Require confirmatory eGFR (default: False)
        - outpatient_only: Use only outpatient records (default: True)

    Returns
    -------
    pd.DataFrame
        Cohort with columns: [key, t0_date, age_at_t0, gender, A, duration, event]
    """
    logger.info("=" * 80)
    logger.info("PRISM COHORT FORMATION STEP")
    logger.info("=" * 80)

    # Extract cohort configuration from config dict
    cohort_config = config.get('cohort', {}) if config is not None else {}

    # Initialize CohortBuilder with config
    builder = CohortBuilder(
        egfr_screen_lt15_min_days=cohort_config.get('egfr_screen_lt15_min_days', 90),
        egfr_screen_lt15_max_days=cohort_config.get('egfr_screen_lt15_max_days', 365),
        t0_threshold=cohort_config.get('t0_threshold', 10.0),
        early_window_days=cohort_config.get('early_window_days', 90),
        study_end_date=cohort_config.get('study_end_date', "2023-12-31"),
        max_followup_days=cohort_config.get('max_followup_days', 1825),
        require_confirmatory_egfr=cohort_config.get('require_confirmatory_egfr', False),
        outpatient_only=cohort_config.get('outpatient_only', True)
    )

    logger.info(f"Configuration:")
    logger.info(f"  - eGFR screening threshold: <15 mL/min/1.73m²")
    logger.info(f"  - Screening window: {builder.egfr_screen_lt15_min_days}-{builder.egfr_screen_lt15_max_days} days")
    logger.info(f"  - t₀ threshold: ≤{builder.t0_threshold} mL/min/1.73m²")
    logger.info(f"  - Early dialysis window: {builder.early_window_days} days after t₀")
    logger.info(f"  - Maximum follow-up: {builder.max_followup_days} days (5 years)")
    logger.info(f"  - Outpatient only: {builder.outpatient_only}")
    logger.info(f"  - Require confirmatory eGFR: {builder.require_confirmatory_egfr}")
    logger.info("")

    # Build cohort
    cohort_df = builder.build_cohort(cr_df, operation_df, death_df)

    # Generate and log summary statistics
    summary = builder.get_cohort_summary(cohort_df)
    logger.info("")
    logger.info("=" * 80)
    logger.info("COHORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total patients: {summary['n_total']}")
    logger.info(f"Early dialysis (A=1): {summary['n_early_dialysis']} ({summary['pct_early_dialysis']:.1f}%)")
    logger.info(f"Non-early (A=0): {summary['n_non_early']} ({100-summary['pct_early_dialysis']:.1f}%)")
    logger.info("")
    logger.info(f"Total deaths: {summary['n_events']} ({summary['event_rate']:.1f}%)")
    logger.info(f"Total censored: {summary['n_censored']} ({100-summary['event_rate']:.1f}%)")
    logger.info(f"  - Event rate in early group: {summary['event_rate_early']:.1f}%")
    logger.info(f"  - Event rate in non-early group: {summary['event_rate_non_early']:.1f}%")
    logger.info("")
    logger.info(f"Median follow-up: {summary['median_followup_days']:.0f} days ({summary['median_followup_days']/365:.1f} years)")
    logger.info(f"Median age at t₀: {summary['median_age_at_t0']:.1f} years")
    logger.info("=" * 80)

    # Validate output
    required_columns = ['key', 't0_date', 'age_at_t0', 'A', 'duration', 'event']
    missing_columns = [col for col in required_columns if col not in cohort_df.columns]
    if missing_columns:
        raise ValueError(f"Cohort DataFrame missing required columns: {missing_columns}")

    # Check for invalid values
    if cohort_df['duration'].min() <= 0:
        raise ValueError("Invalid duration values (must be > 0)")

    if not cohort_df['A'].isin([0, 1]).all():
        raise ValueError("Treatment A must be binary (0 or 1)")

    if not cohort_df['event'].isin([0, 1]).all():
        raise ValueError("Event indicator must be binary (0 or 1)")

    logger.info(f"✓ Cohort formation complete: {len(cohort_df)} patients")
    logger.info("")

    return cohort_df

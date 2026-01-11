#!/usr/bin/env python3
"""
PRISM Core Module Testing (Without ZenML)

This script tests core modules directly without ZenML orchestration.

Usage:
    python test_core_modules.py

Author: PRISM Development Team
Date: 2026-01-11
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test basic data loading without ZenML"""
    logger.info("=" * 80)
    logger.info("TEST: DATA LOADING (Direct)")
    logger.info("=" * 80)

    try:
        # Import data ingestion modules
        from src.data_ingester import DataIngester
        from src.lab_result_mapper import StandardLabResultMapper

        # Try loading creatinine data
        logger.info("\nLoading creatinine data...")
        ingester = DataIngester(
            data_type='creatinine',
            data_path='data/Cr',
            validation_rules_path='src/default_data_validation_rules.yml'
        )

        cr_df, demographics_df = ingester.load_data()

        logger.info(f"✓ Creatinine loaded: {cr_df.shape}")
        logger.info(f"  Columns: {list(cr_df.columns)}")
        logger.info(f"  Demographics: {demographics_df.shape}")
        logger.info(f"\nFirst few rows:")
        logger.info(cr_df.head())

        return {'cr_df': cr_df, 'demographics_df': demographics_df}

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cohort_formation(cr_df):
    """Test cohort formation with sample data"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: COHORT FORMATION")
    logger.info("=" * 80)

    if cr_df is None:
        logger.error("Skipping - no data available")
        return None

    try:
        from src.cohort_builder import CohortBuilder
        from src.data_cleaning import calculate_egfr

        logger.info(f"\nInput creatinine data: {cr_df.shape}")
        logger.info(f"  Columns: {list(cr_df.columns)}")

        # Check if eGFR needs to be calculated
        if 'egfr' not in cr_df.columns:
            logger.info("\neGFR column not found - attempting to calculate...")

            # Check required columns
            required = ['creatinine', 'age', 'gender']
            missing = [col for col in required if col not in cr_df.columns]

            if missing:
                logger.error(f"Cannot calculate eGFR - missing columns: {missing}")
                logger.info(f"Available columns: {list(cr_df.columns)}")
                return None

            logger.info("Calculating eGFR...")
            cr_df = calculate_egfr(cr_df)
            logger.info(f"✓ eGFR calculated")

        # For testing, create minimal mock dataframes for operation and death
        logger.info("\nCreating mock operation_df and death_df for testing...")

        # Get unique patient keys from cr_df
        unique_keys = cr_df['key'].unique()[:100]  # Test with first 100 patients

        # Mock operation_df
        operation_df = pd.DataFrame({
            'key': unique_keys[:10],  # 10 patients have dialysis
            'date': pd.to_datetime('2020-01-01'),
            'is_dialysis': True
        })

        # Mock death_df
        death_df = pd.DataFrame({
            'key': unique_keys[:5],  # 5 patients died
            'death_date': pd.to_datetime('2021-01-01')
        })

        logger.info(f"  Mock operation_df: {operation_df.shape}")
        logger.info(f"  Mock death_df: {death_df.shape}")

        # Filter cr_df to test subset
        cr_df_test = cr_df[cr_df['key'].isin(unique_keys)].copy()
        logger.info(f"\nTest subset: {cr_df_test.shape}")

        # Initialize cohort builder
        logger.info("\nInitializing CohortBuilder...")
        builder = CohortBuilder(
            egfr_screen_lt15_min_days=90,
            egfr_screen_lt15_max_days=365,
            t0_threshold=10.0,
            early_window_days=90
        )

        logger.info("Building cohort...")
        cohort_df = builder.build_cohort(cr_df_test, operation_df, death_df)

        logger.info(f"\n✓ Cohort formation successful!")
        logger.info(f"  Cohort size: {len(cohort_df)}")
        logger.info(f"  Columns: {list(cohort_df.columns)}")
        logger.info(f"\nCohort sample:")
        logger.info(cohort_df.head())

        if len(cohort_df) > 0:
            summary = builder.get_cohort_summary(cohort_df)
            logger.info(f"\nSummary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

        return cohort_df

    except Exception as e:
        logger.error(f"✗ Cohort formation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run tests"""
    logger.info("PRISM CORE MODULE TESTING")
    logger.info("=" * 80)

    # Test 1: Data loading
    data = test_data_loading()

    if data:
        cr_df = data['cr_df']

        # Test 2: Cohort formation
        cohort_df = test_cohort_formation(cr_df)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

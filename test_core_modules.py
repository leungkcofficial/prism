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
        # Use the load_lab_data function from ingest_data.py
        import sys
        sys.path.insert(0, 'steps')
        from ingest_data import load_lab_data, load_clinical_data

        # Load creatinine data (includes demographics)
        logger.info("\nLoading creatinine data...")
        cr_df, demographics_df = load_lab_data('creatinine')

        logger.info(f"✓ Creatinine loaded: {cr_df.shape}")
        logger.info(f"  Columns: {list(cr_df.columns)}")
        logger.info(f"  Demographics: {demographics_df.shape if demographics_df is not None else 'N/A'}")
        logger.info(f"\nFirst few rows of creatinine:")
        logger.info(cr_df.head())

        # Load clinical data
        logger.info("\nLoading operation data...")
        operation_df = load_clinical_data('operation')
        logger.info(f"✓ Operation data: {operation_df.shape}")

        logger.info("\nLoading death data...")
        death_df = load_clinical_data('death')
        logger.info(f"✓ Death data: {death_df.shape}")

        return {
            'cr_df': cr_df,
            'demographics_df': demographics_df,
            'operation_df': operation_df,
            'death_df': death_df
        }

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cohort_formation(data):
    """Test cohort formation with sample data"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: COHORT FORMATION")
    logger.info("=" * 80)

    if data is None or 'cr_df' not in data:
        logger.error("Skipping - no data available")
        return None

    try:
        from src.cohort_builder import CohortBuilder
        from src.data_cleaning import calculate_egfr

        cr_df = data['cr_df']
        demographics_df = data['demographics_df']
        operation_df = data['operation_df']
        death_df = data['death_df']

        logger.info(f"\nInput data:")
        logger.info(f"  Creatinine: {cr_df.shape}")
        logger.info(f"  Demographics: {demographics_df.shape if demographics_df is not None else 'N/A'}")
        logger.info(f"  Operations: {operation_df.shape}")
        logger.info(f"  Deaths: {death_df.shape}")
        logger.info(f"\nCreatinine columns: {list(cr_df.columns)}")

        # Check if eGFR needs to be calculated
        if 'egfr' not in cr_df.columns:
            logger.info("\neGFR column not found - attempting to calculate...")

            # Prepare data for eGFR calculation
            # The calculate_egfr() function expects:
            # - cr_df with columns: key, date, creatinine (renamed from result_value)
            # - demo_df with columns: key, dob, gender

            # Note: Our cr_df already contains dob and gender columns
            # So we can extract demographics from cr_df itself

            # Prepare creatinine DataFrame for eGFR calculation
            if 'result_value' in cr_df.columns:
                # Create a copy with only the required columns (key, date, creatinine)
                # Remove dob and gender so calculate_egfr can merge them properly
                cr_df_egfr = cr_df[['key', 'date', 'result_value', 'code']].copy()
                cr_df_egfr['creatinine'] = cr_df_egfr['result_value']
                logger.info("Prepared creatinine data with columns: key, date, creatinine")
            else:
                logger.error(f"Cannot calculate eGFR - 'result_value' column not found")
                logger.info(f"Available columns: {list(cr_df.columns)}")
                return None

            # Extract demographics from creatinine DataFrame
            # (demographics are already merged into cr_df by the data mapper)
            if 'dob' in cr_df.columns and 'gender' in cr_df.columns:
                demo_df_for_egfr = cr_df[['key', 'dob', 'gender']].drop_duplicates(subset=['key']).copy()

                # Convert gender to numeric: F->0 (female), M->1 (male)
                # The calculate_egfr function expects 0 for female and 1 for male
                gender_map = {'F': 0, 'M': 1}
                demo_df_for_egfr['gender'] = demo_df_for_egfr['gender'].map(gender_map)

                logger.info(f"Extracted demographics from creatinine data: {demo_df_for_egfr.shape}")
                logger.info(f"Gender encoding: {demo_df_for_egfr['gender'].value_counts().to_dict()}")
            else:
                logger.error("Cannot calculate eGFR - dob or gender columns not in creatinine data")
                return None

            logger.info("Calculating eGFR (this may take a few minutes)...")
            # calculate_egfr merges demo_df internally and calculates age from dob
            cr_df = calculate_egfr(cr_df_egfr, demo_df_for_egfr)
            logger.info(f"✓ eGFR calculated")

            if 'egfr' in cr_df.columns:
                logger.info(f"  eGFR stats: min={cr_df['egfr'].min():.1f}, max={cr_df['egfr'].max():.1f}, mean={cr_df['egfr'].mean():.1f}")
            else:
                logger.error("eGFR column still not found after calculation!")
                return None

        # Use a test subset for speed
        logger.info("\nUsing subset for testing (first 5000 creatinine records)...")
        cr_df_test = cr_df.head(5000).copy()
        unique_keys = cr_df_test['key'].unique()
        logger.info(f"Test subset: {cr_df_test.shape}, {len(unique_keys)} unique patients")

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
        # Test 2: Cohort formation
        cohort_df = test_cohort_formation(data)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

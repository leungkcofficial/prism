#!/usr/bin/env python3
"""
PRISM Pipeline Step-by-Step Testing Script

This script tests each component of the PRISM pipeline independently
to identify and fix issues before running the full pipeline.

Usage:
    python test_pipeline_steps.py --step 1  # Test data ingestion
    python test_pipeline_steps.py --step 2  # Test cohort formation
    python test_pipeline_steps.py --all     # Run all tests

Author: PRISM Development Team
Date: 2026-01-11
"""

import sys
import os
from pathlib import Path
import click
import logging
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_1_data_ingestion():
    """Test 1: Data Ingestion"""
    logger.info("=" * 80)
    logger.info("TEST 1: DATA INGESTION")
    logger.info("=" * 80)

    try:
        from steps.ingest_data import ingest_data

        logger.info("Calling ingest_data()...")
        lab_dfs = ingest_data()

        logger.info(f"\n✓ Data ingestion successful!")
        logger.info(f"  Keys: {list(lab_dfs.keys())}")

        for key, df in lab_dfs.items():
            logger.info(f"  {key}: {df.shape} - columns: {list(df.columns)[:5]}...")

        return lab_dfs

    except Exception as e:
        logger.error(f"\n✗ Data ingestion failed: {e}")
        logger.exception("Full traceback:")
        return None


def test_2_cohort_formation(lab_dfs):
    """Test 2: Cohort Formation"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: COHORT FORMATION")
    logger.info("=" * 80)

    if lab_dfs is None:
        logger.error("Skipping - data ingestion failed")
        return None

    try:
        from src.cohort_builder import CohortBuilder

        # Get required DataFrames
        cr_df = lab_dfs.get('cr_df')
        operation_df = lab_dfs.get('operation_df')
        death_df = lab_dfs.get('death_df')

        if cr_df is None or operation_df is None or death_df is None:
            logger.error("Missing required DataFrames")
            logger.info(f"cr_df: {cr_df is not None}")
            logger.info(f"operation_df: {operation_df is not None}")
            logger.info(f"death_df: {death_df is not None}")
            return None

        logger.info(f"Input DataFrames:")
        logger.info(f"  cr_df: {cr_df.shape}")
        logger.info(f"    Columns: {list(cr_df.columns)}")
        logger.info(f"  operation_df: {operation_df.shape}")
        logger.info(f"    Columns: {list(operation_df.columns)}")
        logger.info(f"  death_df: {death_df.shape}")
        logger.info(f"    Columns: {list(death_df.columns)}")

        # Check if eGFR column exists
        if 'egfr' not in cr_df.columns:
            logger.warning("eGFR column not found in cr_df!")
            logger.info("Available columns: " + ", ".join(cr_df.columns))

            # Check if we need to calculate eGFR
            if 'creatinine' in cr_df.columns and 'age' in cr_df.columns and 'gender' in cr_df.columns:
                logger.info("Found creatinine, age, gender - can calculate eGFR")
                logger.info("Attempting to calculate eGFR using CKD-EPI formula...")

                from src.data_cleaning import calculate_egfr
                cr_df = calculate_egfr(cr_df)
                logger.info(f"✓ eGFR calculated, new shape: {cr_df.shape}")
            else:
                logger.error("Cannot calculate eGFR - missing required columns")
                return None

        # Check for is_outpatient column
        if 'is_outpatient' not in cr_df.columns:
            logger.warning("is_outpatient column not found - will use all records")

        # Initialize cohort builder
        builder = CohortBuilder()

        logger.info("\nBuilding cohort...")
        cohort_df = builder.build_cohort(cr_df, operation_df, death_df)

        logger.info(f"\n✓ Cohort formation successful!")
        logger.info(f"  Cohort size: {len(cohort_df)}")
        logger.info(f"  Columns: {list(cohort_df.columns)}")
        logger.info(f"\nCohort summary:")
        logger.info(cohort_df.head())

        # Get summary statistics
        summary = builder.get_cohort_summary(cohort_df)
        logger.info(f"\nSummary statistics:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return cohort_df

    except Exception as e:
        logger.error(f"\n✗ Cohort formation failed: {e}")
        logger.exception("Full traceback:")
        return None


def test_3_feature_extraction(cohort_df, lab_dfs):
    """Test 3: Feature Extraction"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: FEATURE EXTRACTION")
    logger.info("=" * 80)

    if cohort_df is None or lab_dfs is None:
        logger.error("Skipping - previous tests failed")
        return None

    try:
        from src.feature_extractor import FeatureExtractor

        # Get ICD-10 data
        icd10_df = lab_dfs.get('icd10_df')
        if icd10_df is None:
            logger.warning("icd10_df not found - CCI features will be empty")

        logger.info(f"Input:")
        logger.info(f"  Cohort: {cohort_df.shape}")
        logger.info(f"  ICD-10: {icd10_df.shape if icd10_df is not None else 'N/A'}")

        # Initialize extractor
        extractor = FeatureExtractor()

        logger.info("\nExtracting features...")
        features_df = extractor.extract(cohort_df, lab_dfs, icd10_df)

        logger.info(f"\n✓ Feature extraction successful!")
        logger.info(f"  Features shape: {features_df.shape}")
        logger.info(f"  Columns: {list(features_df.columns)}")
        logger.info(f"\nFeature sample:")
        logger.info(features_df.head())

        # Get summary
        summary = extractor.get_feature_summary(features_df)
        logger.info(f"\nFeature summary:")
        logger.info(f"  N patients: {summary['n_patients']}")
        logger.info(f"  N features: {summary['n_features']}")

        return features_df

    except Exception as e:
        logger.error(f"\n✗ Feature extraction failed: {e}")
        logger.exception("Full traceback:")
        return None


@click.command()
@click.option('--step', type=int, help='Test specific step (1-3)')
@click.option('--all', 'run_all', is_flag=True, help='Run all tests')
def main(step, run_all):
    """Run PRISM pipeline tests step by step."""

    logger.info("PRISM PIPELINE STEP-BY-STEP TESTING")
    logger.info("=" * 80)

    # Test 1: Data Ingestion
    if step == 1 or run_all:
        lab_dfs = test_1_data_ingestion()
        if not run_all:
            return
    else:
        lab_dfs = None

    # Test 2: Cohort Formation
    if step == 2 or run_all:
        if lab_dfs is None:
            logger.info("\nRunning Test 1 first (required for Test 2)...")
            lab_dfs = test_1_data_ingestion()

        cohort_df = test_2_cohort_formation(lab_dfs)
        if not run_all:
            return
    else:
        cohort_df = None

    # Test 3: Feature Extraction
    if step == 3 or run_all:
        if lab_dfs is None or cohort_df is None:
            logger.info("\nRunning previous tests first...")
            lab_dfs = test_1_data_ingestion()
            cohort_df = test_2_cohort_formation(lab_dfs)

        features_df = test_3_feature_extraction(cohort_df, lab_dfs)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

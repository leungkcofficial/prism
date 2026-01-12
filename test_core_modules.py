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

        # Use full dataset to ensure we capture all treatment variations
        logger.info("\nUsing full dataset for cohort formation...")
        cr_df_test = cr_df.copy()
        unique_keys = cr_df_test['key'].unique()
        logger.info(f"Full dataset: {cr_df_test.shape}, {len(unique_keys)} unique patients")

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


def test_feature_extraction(data, cohort_df):
    """Test feature extraction with lookback windows"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: FEATURE EXTRACTION")
    logger.info("=" * 80)

    if cohort_df is None or cohort_df.empty:
        logger.error("Skipping - no cohort available")
        return None

    try:
        from src.feature_extractor import FeatureExtractor
        import sys
        sys.path.insert(0, 'steps')
        from ingest_data import load_lab_data, load_clinical_data

        logger.info(f"\nCohort size: {len(cohort_df)}")
        logger.info(f"Cohort columns: {list(cohort_df.columns)}")

        # Load all lab data needed for features
        logger.info("\nLoading lab data for feature extraction...")

        lab_dfs = {}
        lab_types = ['hemoglobin', 'albumin', 'hemoglobin_a1c', 'phosphate',
                     'calcium', 'bicarbonate', 'urine_albumin_creatinine_ratio']

        # Add creatinine (already loaded with eGFR)
        lab_dfs['creatinine'] = data['cr_df']

        for lab_type in lab_types:
            logger.info(f"  Loading {lab_type}...")
            lab_df, _ = load_lab_data(lab_type)

            # Map lab type names to feature names
            if lab_type == 'hemoglobin':
                lab_dfs['hemoglobin'] = lab_df
            elif lab_type == 'albumin':
                lab_dfs['albumin'] = lab_df
            elif lab_type == 'hemoglobin_a1c':
                lab_dfs['a1c'] = lab_df
            elif lab_type == 'phosphate':
                lab_dfs['phosphate'] = lab_df
            elif lab_type == 'calcium':
                lab_dfs['calcium'] = lab_df
            elif lab_type == 'bicarbonate':
                lab_dfs['bicarbonate'] = lab_df
            elif lab_type == 'urine_albumin_creatinine_ratio':
                lab_dfs['uacr'] = lab_df

            logger.info(f"    ✓ {lab_type}: {lab_df.shape}")

        # Load ICD-10 data for CCI calculation
        logger.info("\nLoading ICD-10 diagnosis data...")
        icd10_df = load_clinical_data('icd10')
        logger.info(f"✓ ICD-10 data: {icd10_df.shape}")

        # Initialize feature extractor
        logger.info("\nInitializing FeatureExtractor...")
        extractor = FeatureExtractor(
            lab_lookback_days=90,
            cci_lookback_years=5,
            derive_uacr_from_upcr=True
        )

        # Extract features
        logger.info("\nExtracting features with lookback windows...")
        features_df = extractor.extract(cohort_df, lab_dfs, icd10_df)

        logger.info(f"\n✓ Feature extraction successful!")
        logger.info(f"  Features extracted: {len(features_df.columns)}")
        logger.info(f"  Rows: {len(features_df)}")
        logger.info(f"\nFeature columns: {list(features_df.columns)}")
        logger.info(f"\nFeature sample:")
        logger.info(features_df.head())

        # Check missing value rates
        missing_rates = (features_df.isnull().sum() / len(features_df) * 100).sort_values(ascending=False)
        logger.info(f"\nMissing value rates (%):")
        for col, rate in missing_rates.items():
            if rate > 0:
                logger.info(f"  {col}: {rate:.1f}%")

        return features_df

    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_preprocessing(cohort_df, features_df):
    """Test preprocessing with MICE imputation, log transformation, and scaling"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: PREPROCESSING")
    logger.info("=" * 80)

    if cohort_df is None or features_df is None or cohort_df.empty or features_df.empty:
        logger.error("Skipping - no cohort or features available")
        return None

    try:
        import os
        from src.ckd_preprocessor import CKDPreprocessor

        logger.info(f"\nInput data:")
        logger.info(f"  Cohort: {cohort_df.shape}")
        logger.info(f"  Features: {features_df.shape}")

        # Merge cohort and features
        logger.info("\nMerging cohort and features...")
        master_df = cohort_df.merge(features_df, on='key', how='inner')
        logger.info(f"✓ Merged master_df: {master_df.shape}")
        logger.info(f"  Columns: {list(master_df.columns)}")

        # Check missing values before preprocessing
        logger.info(f"\nMissing values before preprocessing:")
        missing_counts = master_df.isnull().sum()
        for col, count in missing_counts[missing_counts > 0].items():
            pct = count / len(master_df) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")

        # Configure environment variables to match our column names
        logger.info("\nConfiguring column mappings for PRISM data...")
        os.environ['HARD_TRUTH_COLUMNS'] = 'gender,event,t0_date'
        os.environ['LAB_COLUMNS'] = 'creatinine_at_t0,hemoglobin_at_t0,albumin_at_t0,a1c_at_t0,phosphate_at_t0,calcium_at_t0,bicarbonate_at_t0,uacr_at_t0'
        os.environ['NUMERICAL_COLUMNS'] = 'age_at_t0,duration,creatinine_at_t0,hemoglobin_at_t0,albumin_at_t0,a1c_at_t0,phosphate_at_t0,calcium_at_t0,bicarbonate_at_t0,uacr_at_t0,cci_score_total,time_since_ckd_days'
        os.environ['CATEGORICAL_COLUMNS'] = 'gender,A,event'
        # Medical history columns are auto-detected from comorbidity flags

        # Initialize preprocessor
        logger.info("\nInitializing CKDPreprocessor...")
        preprocessor = CKDPreprocessor()

        # Fit preprocessor on master_df
        logger.info("\nFitting preprocessor...")
        preprocessor.fit(master_df, random_seed=42)

        logger.info(f"\n✓ Preprocessing fit complete!")

        # Get preprocessing info
        info = preprocessor.get_preprocessing_info()
        logger.info(f"\nPreprocessing summary:")
        logger.info(f"  Total features: {info['n_features']}")
        logger.info(f"  MICE fitted: {info['imputation']['mice_fitted']}")
        logger.info(f"  Log transformed: {info['transformations']['n_log_transformed']}")
        logger.info(f"  MinMax scaled: {info['transformations']['n_minmax_scaled']}")

        # Transform the data
        logger.info("\nTransforming data...")
        master_processed = preprocessor.transform(master_df)

        logger.info(f"\n✓ Data transformation complete!")
        logger.info(f"  Processed shape: {master_processed.shape}")

        # Check missing values after preprocessing
        missing_after = master_processed.isnull().sum().sum()
        logger.info(f"\nMissing values after preprocessing: {missing_after}")

        # Show sample of processed data
        logger.info(f"\nProcessed data sample:")
        logger.info(master_processed.head())

        return master_processed

    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_s_learner(master_processed):
    """Test S-Learner training"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: S-LEARNER TRAINING")
    logger.info("=" * 80)

    if master_processed is None or master_processed.empty:
        logger.error("Skipping - no preprocessed data available")
        return None

    try:
        from src.s_learner import SLearner
        import numpy as np

        logger.info(f"\nPreparing data for S-Learner...")
        logger.info(f"  Input shape: {master_processed.shape}")

        # Extract features and outcomes
        # Exclude non-feature columns: key, t0_date, A (treatment), duration, event
        exclude_cols = ['key', 't0_date', 'A', 'duration', 'event']
        feature_cols = [col for col in master_processed.columns if col not in exclude_cols]

        # Filter to numeric features only (exclude categorical for now)
        numeric_features = []
        for col in feature_cols:
            if col in master_processed.columns:
                # Skip categorical columns (they have 'category' dtype)
                if not pd.api.types.is_categorical_dtype(master_processed[col]):
                    # Also skip columns that are 100% missing
                    if master_processed[col].notna().sum() > 0:
                        numeric_features.append(col)

        logger.info(f"  Feature columns: {len(numeric_features)}")
        logger.info(f"  Features: {numeric_features}")

        # Extract data as numpy arrays
        X = master_processed[numeric_features].values.astype(np.float32)
        A = master_processed['A'].values.astype(np.int32)
        durations = master_processed['duration'].values.astype(np.float32)
        events = master_processed['event'].values.astype(np.int32)

        # Handle NaN values (should be minimal after preprocessing)
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            logger.warning(f"  Warning: {nan_mask.sum()} samples with NaN, will be excluded")
            X = X[~nan_mask]
            A = A[~nan_mask]
            durations = durations[~nan_mask]
            events = events[~nan_mask]

        logger.info(f"\n✓ Data prepared:")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Treatment A=1: {A.sum()} ({A.mean()*100:.1f}%)")
        logger.info(f"  Events: {events.sum()} ({events.mean()*100:.1f}%)")
        logger.info(f"  Duration range: [{durations.min():.0f}, {durations.max():.0f}] days")

        # Initialize S-Learner
        logger.info("\nInitializing S-Learner...")
        input_dim = X.shape[1] + 1  # +1 for treatment A
        s_learner = SLearner(
            input_dim=input_dim,
            hidden_layers=[64, 32],  # Smaller for small dataset
            dropout=0.2,
            learning_rate=0.001,
            device='cpu',  # Use CPU for small dataset
            random_seed=42
        )

        # Train S-Learner (small epochs for testing)
        logger.info("\nTraining S-Learner...")
        log = s_learner.fit(
            X=X,
            A=A,
            durations=durations,
            events=events,
            batch_size=16,  # Small batch for small dataset
            epochs=50,  # Reduced for testing
            patience=10,
            verbose=False
        )

        logger.info(f"\n✓ S-Learner training complete!")
        # Check log structure
        if hasattr(log, 'monitors') and log.monitors:
            if 'train' in log.monitors:
                if isinstance(log.monitors['train'], dict) and 'loss' in log.monitors['train']:
                    logger.info(f"  Final training loss: {log.monitors['train']['loss'][-1]:.4f}")
                    logger.info(f"  Epochs trained: {len(log.monitors['train']['loss'])}")
                else:
                    logger.info(f"  Training log structure: {type(log.monitors['train'])}")
            else:
                logger.info(f"  Available monitors: {list(log.monitors.keys())}")

        # Compute C-index
        logger.info("\nEvaluating model...")
        cindex = s_learner.compute_cindex(X, A, durations, events)
        logger.info(f"  C-index: {cindex:.3f}")

        # Compute ATE and ATT
        logger.info("\nComputing treatment effects...")
        ate = s_learner.compute_ate(X, times=[365, 1095, 1825])
        logger.info(f"  ATE at 1 year: {ate[365]:.4f}")
        logger.info(f"  ATE at 3 years: {ate[1095]:.4f}")
        logger.info(f"  ATE at 5 years: {ate[1825]:.4f}")

        if A.sum() > 0:
            att = s_learner.compute_att(X, A, times=[365, 1095, 1825])
            logger.info(f"  ATT at 1 year: {att[365]:.4f}")
            logger.info(f"  ATT at 3 years: {att[1095]:.4f}")
            logger.info(f"  ATT at 5 years: {att[1825]:.4f}")

        return s_learner

    except Exception as e:
        logger.error(f"✗ S-Learner training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run tests"""
    logger.info("PRISM CORE MODULE TESTING")
    logger.info("=" * 80)

    # Test 1: Data loading
    data = test_data_loading()

    cohort_df = None
    features_df = None
    master_processed = None
    s_learner = None

    if data:
        # Test 2: Cohort formation
        cohort_df = test_cohort_formation(data)

        if cohort_df is not None and not cohort_df.empty:
            # Test 3: Feature extraction
            features_df = test_feature_extraction(data, cohort_df)

            if features_df is not None and not features_df.empty:
                # Test 4: Preprocessing
                master_processed = test_preprocessing(cohort_df, features_df)

                if master_processed is not None and not master_processed.empty:
                    # Test 5: S-Learner training
                    s_learner = test_s_learner(master_processed)

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

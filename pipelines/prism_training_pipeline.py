"""
PRISM Training Pipeline

Main ZenML pipeline for PRISM causal survival analysis.

This pipeline orchestrates:
1. Data ingestion
2. Cohort formation (eGFR screening, t₀ definition, treatment labeling)
3. Feature extraction (t₀-centric lookback windows)
4. Data preprocessing (imputation, scaling, splitting)
5. Model training (S/T/DR-learner)
6. Comprehensive evaluation

Author: PRISM Development Team
Date: 2026-01-11
"""

from zenml import pipeline
import logging

# Import existing steps from previous project
from steps.ingest_data import ingest_data
from steps.impute_data import impute_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data

# Import new PRISM steps
from steps.form_cohort import form_cohort
from steps.extract_features import extract_features
from steps.merge_cohort_features import merge_cohort_features
from steps.train_s_learner import train_s_learner
from steps.train_t_learner import train_t_learner
from steps.train_dr_learner import train_dr_learner
from steps.evaluate_learner import evaluate_learner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pipeline
def prism_training_pipeline(config: dict):
    """
    PRISM training pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML file
        Must specify mode: 's_learner', 't_learner', or 'dr_learner'

    Returns
    -------
    Dict
        Pipeline results including trained model and evaluation metrics
    """
    logger.info("=" * 80)
    logger.info("PRISM TRAINING PIPELINE")
    logger.info("=" * 80)

    mode = config.get('project', {}).get('mode', 's_learner')
    logger.info(f"Running in {mode.upper()} mode")

    # =========================================================================
    # PHASE 1: DATA INGESTION
    # =========================================================================
    logger.info("\n=== PHASE 1: DATA INGESTION ===")

    # Ingest raw EHR data (reuse existing step)
    # Returns dictionary with 14 DataFrames:
    # - cr_df, hb_df, alb_df, a1c_df, po4_df, ca_df, hco3_df, uacr_df, upcr_df
    # - demographics_df, icd10_df, death_df, operation_df, rrt_clinic_df
    lab_dfs = ingest_data()

    # =========================================================================
    # PHASE 2: COHORT FORMATION
    # =========================================================================
    logger.info("\n=== PHASE 2: COHORT FORMATION ===")

    # Form cohort with t₀, treatment A, and survival outcomes
    # Input: creatinine (for eGFR), operations (for dialysis), death data
    # Output: [key, t0_date, age_at_t0, gender, A, duration, event]
    cohort_df = form_cohort(
        cr_df=lab_dfs['cr_df'],
        operation_df=lab_dfs['operation_df'],
        death_df=lab_dfs['death_df'],
        config=config
    )

    # =========================================================================
    # PHASE 3: FEATURE EXTRACTION
    # =========================================================================
    logger.info("\n=== PHASE 3: FEATURE EXTRACTION ===")

    # Extract features with t₀-centric lookback windows
    # Input: cohort (for t₀ dates), lab DataFrames, ICD-10 data
    # Output: [key, cr_at_t0, hb_at_t0, ..., cci_flags, cci_score_total]
    features_df = extract_features(
        cohort_df=cohort_df,
        lab_dfs=lab_dfs,
        icd10_df=lab_dfs['icd10_df'],
        config=config
    )

    # =========================================================================
    # PHASE 4: MERGE COHORT & FEATURES
    # =========================================================================
    logger.info("\n=== PHASE 4: MERGE COHORT & FEATURES ===")

    # Merge cohort outcomes with extracted features
    # Output: [key, t0_date, age_at_t0, gender, A, duration, event, ...features]
    master_df = merge_cohort_features(
        cohort_df=cohort_df,
        features_df=features_df
    )

    # =========================================================================
    # PHASE 5: DATA SPLITTING
    # =========================================================================
    logger.info("\n=== PHASE 5: DATA SPLITTING ===")

    # Split into train/temporal_test/spatial_test
    # Temporal: t₀ ≥ 2022-01-01
    # Spatial: Random 10% of remaining
    # Train: Rest
    train_df, temporal_test_df, spatial_test_df = split_data(
        master_df=master_df,
        config=config
    )

    # =========================================================================
    # PHASE 6: DATA PREPROCESSING
    # =========================================================================
    logger.info("\n=== PHASE 6: DATA PREPROCESSING ===")

    # Imputation (MICE for labs, forward-fill for medical history)
    train_imputed, temporal_test_imputed, spatial_test_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df,
        config=config
    )

    # Preprocessing (log transform, min-max scaling, CCI binarization)
    train_processed, temporal_test_processed, spatial_test_processed = preprocess_data(
        train_df=train_imputed,
        test_dfs=[temporal_test_imputed, spatial_test_imputed],
        config=config
    )

    # Create validation set (10% of training for early stopping)
    # Note: Implement simple random split here
    import pandas as pd
    import numpy as np
    np.random.seed(config.get('random_seed', 42))
    train_indices = train_processed.index.tolist()
    np.random.shuffle(train_indices)
    val_size = int(len(train_indices) * 0.1)
    val_indices = train_indices[:val_size]
    train_final_indices = train_indices[val_size:]

    train_final = train_processed.loc[train_final_indices]
    val_df = train_processed.loc[val_indices]

    # =========================================================================
    # PHASE 7: MODEL TRAINING
    # =========================================================================
    logger.info("\n=== PHASE 7: MODEL TRAINING ===")

    # Train appropriate learner based on mode
    if mode == 's_learner':
        learner, training_log = train_s_learner(
            train_df=train_final,
            val_df=val_df,
            config=config
        )
    elif mode == 't_learner':
        learner, training_log = train_t_learner(
            train_df=train_final,
            val_df=val_df,
            config=config
        )
    elif mode == 'dr_learner':
        learner, training_log = train_dr_learner(
            train_df=train_final,
            val_df=val_df,
            config=config
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 's_learner', 't_learner', or 'dr_learner'")

    # =========================================================================
    # PHASE 8: EVALUATION
    # =========================================================================
    logger.info("\n=== PHASE 8: EVALUATION ===")

    # Evaluate on temporal test set
    temporal_results = evaluate_learner(
        learner=learner,
        test_df=temporal_test_processed,
        dataset_name='temporal_test',
        config=config
    )

    # Evaluate on spatial test set
    spatial_results = evaluate_learner(
        learner=learner,
        test_df=spatial_test_processed,
        dataset_name='spatial_test',
        config=config
    )

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PRISM PIPELINE COMPLETE")
    logger.info("=" * 80)

    return {
        'mode': mode,
        'learner': learner,
        'training_log': training_log,
        'temporal_results': temporal_results,
        'spatial_results': spatial_results,
        'cohort_size': len(cohort_df),
        'train_size': len(train_final),
        'val_size': len(val_df),
        'temporal_test_size': len(temporal_test_processed),
        'spatial_test_size': len(spatial_test_processed)
    }

"""
PRISM T-Learner Training Step

ZenML pipeline step for training T-learner model.

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Tuple
import logging
import mlflow

from src.t_learner import TLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def train_t_learner(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> Tuple[TLearner, Dict]:
    """
    ZenML step: Train T-learner model (two separate models).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with features, treatment A, duration, event
    val_df : pd.DataFrame
        Validation data
    config : dict
        Configuration dictionary with model parameters

    Returns
    -------
    Tuple[TLearner, Dict]
        Trained T-learner and training log
    """
    logger.info("=" * 80)
    logger.info("T-LEARNER TRAINING STEP")
    logger.info("=" * 80)

    # Extract model configuration
    model_config = config.get('model', {})
    t_learner_config = config.get('t_learner', {})

    # Separate features, treatment, and outcomes
    feature_cols = [col for col in train_df.columns if col not in ['key', 't0_date', 'A', 'duration', 'event', 'gender']]

    if 'gender' in train_df.columns:
        feature_cols.append('gender')

    X_train = train_df[feature_cols].values
    A_train = train_df['A'].values
    durations_train = train_df['duration'].values
    events_train = train_df['event'].values

    X_val = val_df[feature_cols].values
    A_val = val_df['A'].values
    durations_val = val_df['duration'].values
    events_val = val_df['event'].values

    logger.info(f"Training features: {len(feature_cols)} features")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Initialize T-learner
    # input_dim does NOT include treatment A (separate models)
    learner = TLearner(
        input_dim=X_train.shape[1],
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.3),
        learning_rate=model_config.get('learning_rate', 0.001),
        device=model_config.get('device', 'cuda'),
        random_seed=config.get('random_seed', 42),
        min_samples_per_group=t_learner_config.get('min_samples_per_group', 50)
    )

    # Log model architecture to MLflow
    mlflow.log_params({
        'learner_type': 't_learner',
        'input_dim': X_train.shape[1],
        'hidden_layers': str(model_config.get('hidden_layers', [128, 64, 32])),
        'dropout': model_config.get('dropout', 0.3),
        'learning_rate': model_config.get('learning_rate', 0.001),
        'batch_size': model_config.get('batch_size', 256),
        'epochs': model_config.get('epochs', 100),
        'min_samples_per_group': t_learner_config.get('min_samples_per_group', 50)
    })

    # Train model (trains both model_A0 and model_A1)
    training_log = learner.fit(
        X=X_train,
        A=A_train,
        durations=durations_train,
        events=events_train,
        val_data=(X_val, A_val, durations_val, events_val),
        batch_size=model_config.get('batch_size', 256),
        epochs=model_config.get('epochs', 100),
        patience=model_config.get('early_stopping', {}).get('patience', 10),
        verbose=True
    )

    # Log training metrics for both models
    if 'model_A0' in training_log and hasattr(training_log['model_A0'], 'monitors'):
        if 'train' in training_log['model_A0'].monitors:
            final_loss_A0 = training_log['model_A0'].monitors['train']['loss'][-1]
            mlflow.log_metric('train_final_loss_A0', final_loss_A0)
            logger.info(f"Final training loss (A=0): {final_loss_A0:.4f}")

    if 'model_A1' in training_log and hasattr(training_log['model_A1'], 'monitors'):
        if 'train' in training_log['model_A1'].monitors:
            final_loss_A1 = training_log['model_A1'].monitors['train']['loss'][-1]
            mlflow.log_metric('train_final_loss_A1', final_loss_A1)
            logger.info(f"Final training loss (A=1): {final_loss_A1:.4f}")

    # Compute C-index for both models
    train_cindex = learner.compute_cindex(X_train, A_train, durations_train, events_train)
    val_cindex = learner.compute_cindex(X_val, A_val, durations_val, events_val)

    mlflow.log_metric('train_cindex_A0', train_cindex.get('cindex_A0', 0))
    mlflow.log_metric('train_cindex_A1', train_cindex.get('cindex_A1', 0))
    mlflow.log_metric('train_cindex_overall', train_cindex.get('cindex_overall', 0))

    mlflow.log_metric('val_cindex_A0', val_cindex.get('cindex_A0', 0))
    mlflow.log_metric('val_cindex_A1', val_cindex.get('cindex_A1', 0))
    mlflow.log_metric('val_cindex_overall', val_cindex.get('cindex_overall', 0))

    logger.info(f"Training C-index (A=0): {train_cindex.get('cindex_A0', 0):.4f}")
    logger.info(f"Training C-index (A=1): {train_cindex.get('cindex_A1', 0):.4f}")
    logger.info(f"Validation C-index (A=0): {val_cindex.get('cindex_A0', 0):.4f}")
    logger.info(f"Validation C-index (A=1): {val_cindex.get('cindex_A1', 0):.4f}")

    # Save both models
    output_config = config.get('output', {})
    model_A0_path = output_config.get('model_A0_path', 'models/t_learner/model_A0.pth')
    model_A1_path = output_config.get('model_A1_path', 'models/t_learner/model_A1.pth')
    learner.save(model_A0_path, model_A1_path)

    logger.info(f"✓ T-Learner training complete")
    logger.info("=" * 80)

    return learner, training_log

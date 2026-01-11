"""
PRISM S-Learner Training Step

ZenML pipeline step for training S-learner model.

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Tuple
import logging
import mlflow

from src.s_learner import SLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def train_s_learner(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> Tuple[SLearner, Dict]:
    """
    ZenML step: Train S-learner model.

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
    Tuple[SLearner, Dict]
        Trained S-learner and training log
    """
    logger.info("=" * 80)
    logger.info("S-LEARNER TRAINING STEP")
    logger.info("=" * 80)

    # Extract model configuration
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    # Separate features, treatment, and outcomes
    # Exclude columns: key, t0_date, A, duration, event
    feature_cols = [col for col in train_df.columns if col not in ['key', 't0_date', 'A', 'duration', 'event', 'gender']]

    # Add gender if it exists and not already in feature_cols
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
    logger.info(f"Feature list: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Initialize S-learner
    # input_dim includes treatment A
    learner = SLearner(
        input_dim=X_train.shape[1] + 1,  # +1 for treatment A
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.3),
        learning_rate=model_config.get('learning_rate', 0.001),
        device=model_config.get('device', 'cuda'),
        random_seed=config.get('random_seed', 42)
    )

    # Log model architecture to MLflow
    mlflow.log_params({
        'learner_type': 's_learner',
        'input_dim': X_train.shape[1] + 1,
        'hidden_layers': str(model_config.get('hidden_layers', [128, 64, 32])),
        'dropout': model_config.get('dropout', 0.3),
        'learning_rate': model_config.get('learning_rate', 0.001),
        'batch_size': model_config.get('batch_size', 256),
        'epochs': model_config.get('epochs', 100)
    })

    # Train model
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

    # Log training metrics
    if hasattr(training_log, 'monitors') and 'train' in training_log.monitors:
        final_loss = training_log.monitors['train']['loss'][-1]
        mlflow.log_metric('train_final_loss', final_loss)
        logger.info(f"Final training loss: {final_loss:.4f}")

    # Compute training C-index
    train_cindex = learner.compute_cindex(X_train, A_train, durations_train, events_train)
    val_cindex = learner.compute_cindex(X_val, A_val, durations_val, events_val)

    mlflow.log_metric('train_cindex', train_cindex)
    mlflow.log_metric('val_cindex', val_cindex)

    logger.info(f"Training C-index: {train_cindex:.4f}")
    logger.info(f"Validation C-index: {val_cindex:.4f}")

    # Save model
    output_config = config.get('output', {})
    model_path = output_config.get('model_path', 'models/s_learner/model.pth')
    learner.save(model_path)

    logger.info(f"✓ S-Learner training complete")
    logger.info("=" * 80)

    return learner, training_log

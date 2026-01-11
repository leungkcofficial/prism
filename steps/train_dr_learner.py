"""
PRISM DR-Learner Training Step

ZenML pipeline step for training DR-learner model with propensity weighting.

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Tuple
import logging
import mlflow

from src.dr_learner import DRLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def train_dr_learner(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> Tuple[DRLearner, Dict]:
    """
    ZenML step: Train DR-learner model with propensity weighting.

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
    Tuple[DRLearner, Dict]
        Trained DR-learner and training log
    """
    logger.info("=" * 80)
    logger.info("DR-LEARNER TRAINING STEP")
    logger.info("=" * 80)

    # Extract model configuration
    model_config = config.get('model', {})
    dr_config = config.get('dr_learner', {})
    propensity_config = dr_config.get('propensity_model', {})
    iptw_config = dr_config.get('iptw', {})

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

    # Extract propensity model kwargs
    propensity_model_type = propensity_config.get('model_type', 'gbdt')
    propensity_kwargs = {}

    if propensity_model_type == 'gbdt':
        propensity_kwargs = propensity_config.get('gbdt', {})
    elif propensity_model_type == 'logistic':
        propensity_kwargs = propensity_config.get('logistic', {})

    # Initialize DR-learner
    learner = DRLearner(
        input_dim=X_train.shape[1] + 1,  # +1 for treatment A
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.3),
        learning_rate=model_config.get('learning_rate', 0.001),
        device=model_config.get('device', 'cuda'),
        propensity_model_type=propensity_model_type,
        propensity_kwargs=propensity_kwargs,
        iptw_clip_min=iptw_config.get('clip_min', 0.05),
        iptw_clip_max=iptw_config.get('clip_max', 0.95),
        iptw_stabilize=iptw_config.get('use_stabilized_weights', True),
        iptw_max_weight=iptw_config.get('max_weight', 50.0),
        random_seed=config.get('random_seed', 42)
    )

    # Log model architecture to MLflow
    mlflow.log_params({
        'learner_type': 'dr_learner',
        'input_dim': X_train.shape[1] + 1,
        'hidden_layers': str(model_config.get('hidden_layers', [128, 64, 32])),
        'dropout': model_config.get('dropout', 0.3),
        'learning_rate': model_config.get('learning_rate', 0.001),
        'batch_size': model_config.get('batch_size', 256),
        'epochs': model_config.get('epochs', 100),
        'propensity_model_type': propensity_model_type,
        'iptw_clip_min': iptw_config.get('clip_min', 0.05),
        'iptw_clip_max': iptw_config.get('clip_max', 0.95),
        'iptw_stabilize': iptw_config.get('use_stabilized_weights', True),
        'iptw_max_weight': iptw_config.get('max_weight', 50.0)
    })

    # Train model (trains propensity model + weighted survival model)
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

    # Log overlap diagnostics
    if 'propensity' in training_log and 'overlap_diagnostics' in training_log['propensity']:
        overlap = training_log['propensity']['overlap_diagnostics']
        mlflow.log_metrics({
            'overlap_pct_extreme': overlap['pct_extreme'],
            'overlap_pct_overlap': overlap['pct_overlap'],
            'overlap_n_extreme_low': overlap['n_extreme_low'],
            'overlap_n_extreme_high': overlap['n_extreme_high']
        })

    # Log training metrics
    if 'survival' in training_log and hasattr(training_log['survival'], 'monitors'):
        if 'train' in training_log['survival'].monitors:
            final_loss = training_log['survival'].monitors['train']['loss'][-1]
            mlflow.log_metric('train_final_loss', final_loss)
            logger.info(f"Final training loss: {final_loss:.4f}")

    # Compute C-index
    train_cindex = learner.compute_cindex(X_train, A_train, durations_train, events_train)
    val_cindex = learner.compute_cindex(X_val, A_val, durations_val, events_val)

    mlflow.log_metric('train_cindex', train_cindex)
    mlflow.log_metric('val_cindex', val_cindex)

    logger.info(f"Training C-index: {train_cindex:.4f}")
    logger.info(f"Validation C-index: {val_cindex:.4f}")

    # Save propensity scores and weights
    propensity_scores = learner.get_propensity_scores()
    iptw_weights = learner.get_iptw_weights()

    if propensity_scores is not None:
        output_config = config.get('output', {})
        propensity_path = output_config.get('propensity_scores_path', 'models/dr_learner/propensity_scores.csv')
        pd.DataFrame({
            'key': train_df['key'].values,
            'propensity_score': propensity_scores,
            'iptw_weight': iptw_weights if iptw_weights is not None else np.nan
        }).to_csv(propensity_path, index=False)
        logger.info(f"Propensity scores saved to {propensity_path}")

    # Save models
    output_config = config.get('output', {})
    survival_path = output_config.get('survival_model_path', 'models/dr_learner/survival_model.pth')
    propensity_model_path = output_config.get('propensity_model_path', 'models/dr_learner/propensity_model.pkl')
    learner.save(survival_path, propensity_model_path)

    # Generate propensity distribution plot
    if iptw_config.get('save_propensity_dist', True):
        plot_dir = output_config.get('plots_dir', 'results/dr_learner/plots')
        import os
        os.makedirs(plot_dir, exist_ok=True)
        fig = learner.plot_propensity_distribution(
            A_train,
            save_path=f"{plot_dir}/propensity_distribution.png"
        )
        plt.close(fig)

    logger.info(f"✓ DR-Learner training complete")
    logger.info("=" * 80)

    return learner, training_log

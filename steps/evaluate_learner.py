"""
PRISM Learner Evaluation Step

ZenML pipeline step for comprehensive evaluation of trained learners.

Author: PRISM Development Team
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Union
import logging
import mlflow
import os

from src.causal_evaluator import CausalEvaluator
from src.s_learner import SLearner
from src.t_learner import TLearner
from src.dr_learner import DRLearner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def evaluate_learner(
    learner: Union[SLearner, TLearner, DRLearner],
    test_df: pd.DataFrame,
    dataset_name: str,
    config: dict
) -> Dict:
    """
    ZenML step: Comprehensive evaluation of trained learner.

    Parameters
    ----------
    learner : SLearner, TLearner, or DRLearner
        Trained learner model
    test_df : pd.DataFrame
        Test data with features, treatment A, duration, event
    dataset_name : str
        Name of test dataset (e.g., 'temporal_test', 'spatial_test')
    config : dict
        Configuration dictionary with evaluation parameters

    Returns
    -------
    Dict
        Evaluation results
    """
    logger.info("=" * 80)
    logger.info(f"EVALUATION STEP - {dataset_name.upper()}")
    logger.info("=" * 80)

    # Extract evaluation configuration
    eval_config = config.get('evaluation', {})
    output_config = config.get('output', {})

    # Determine learner type
    learner_type = learner.get_model_summary()['learner_type']
    logger.info(f"Learner type: {learner_type}")

    # Separate features, treatment, and outcomes
    feature_cols = [col for col in test_df.columns if col not in ['key', 't0_date', 'A', 'duration', 'event', 'gender']]

    if 'gender' in test_df.columns:
        feature_cols.append('gender')

    X_test = test_df[feature_cols].values
    A_test = test_df['A'].values
    durations_test = test_df['duration'].values
    events_test = test_df['event'].values

    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    # Initialize evaluator
    evaluator = CausalEvaluator(
        times=eval_config.get('time_points', [365, 1095, 1825]),
        n_bootstrap=eval_config.get('bootstrap', {}).get('n_bootstrap', 1000),
        confidence_level=eval_config.get('bootstrap', {}).get('confidence_level', 0.95),
        random_seed=config.get('random_seed', 42)
    )

    # Run comprehensive evaluation
    results = evaluator.evaluate(
        learner=learner,
        X=X_test,
        A=A_test,
        durations=durations_test,
        events=events_test,
        learner_type=learner_type,
        dataset_name=dataset_name
    )

    # Log metrics to MLflow
    _log_metrics_to_mlflow(results, dataset_name)

    # Generate and save plots
    if eval_config.get('generate_plots', True):
        plot_dir = output_config.get('plots_dir', 'results/plots')
        os.makedirs(plot_dir, exist_ok=True)

        logger.info(f"\nGenerating evaluation plots...")
        figures = evaluator.plot_treatment_effects(save_dir=plot_dir)
        logger.info(f"Plots saved to {plot_dir}")

        # Log plots to MLflow
        for fig_name, fig in figures.items():
            mlflow.log_figure(fig, f"{dataset_name}_{fig_name}.png")

    # Save results to JSON
    results_dir = output_config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{dataset_name}_evaluation.json"
    evaluator.save_results(results_path)

    logger.info(f"\n✓ Evaluation complete - {dataset_name}")
    logger.info("=" * 80)

    return results


def _log_metrics_to_mlflow(results: Dict, dataset_name: str):
    """
    Log evaluation metrics to MLflow.

    Parameters
    ----------
    results : Dict
        Evaluation results
    dataset_name : str
        Name of dataset
    """
    prefix = f"{dataset_name}_"

    # Log basic statistics
    mlflow.log_metric(f"{prefix}n_samples", results['n_samples'])
    mlflow.log_metric(f"{prefix}pct_A1", results['treatment_balance']['pct_A1'])
    mlflow.log_metric(f"{prefix}event_rate", results['event_rate'])

    # Log predictive metrics
    if 'predictive' in results:
        pred = results['predictive']

        # C-index
        if 'cindex' in pred:
            if isinstance(pred['cindex'], dict):
                for key, value in pred['cindex'].items():
                    mlflow.log_metric(f"{prefix}{key}", value)
            else:
                mlflow.log_metric(f"{prefix}cindex", pred['cindex'])

        # Brier score
        if 'brier_score' in pred and pred['brier_score'] is not None:
            for time, score in pred['brier_score'].items():
                mlflow.log_metric(f"{prefix}brier_{time}d", score)

    # Log causal metrics
    if 'causal' in results:
        causal = results['causal']

        # ATE
        if 'ate' in causal and causal['ate'] is not None:
            for time, value in causal['ate'].items():
                mlflow.log_metric(f"{prefix}ate_{time}d", value)

        # ATT
        if 'att' in causal and causal['att'] is not None:
            for time, value in causal['att'].items():
                mlflow.log_metric(f"{prefix}att_{time}d", value)

    # Log bootstrap confidence intervals
    if 'bootstrap' in results:
        bootstrap = results['bootstrap']

        # ATE CI
        if 'ate_ci' in bootstrap:
            for time, ci in bootstrap['ate_ci'].items():
                mlflow.log_metric(f"{prefix}ate_{time}d_ci_lower", ci[0])
                mlflow.log_metric(f"{prefix}ate_{time}d_ci_upper", ci[1])

        if 'ate_se' in bootstrap:
            for time, se in bootstrap['ate_se'].items():
                mlflow.log_metric(f"{prefix}ate_{time}d_se", se)

        # ATT CI
        if 'att_ci' in bootstrap:
            for time, ci in bootstrap['att_ci'].items():
                mlflow.log_metric(f"{prefix}att_{time}d_ci_lower", ci[0])
                mlflow.log_metric(f"{prefix}att_{time}d_ci_upper", ci[1])

        if 'att_se' in bootstrap:
            for time, se in bootstrap['att_se'].items():
                mlflow.log_metric(f"{prefix}att_{time}d_se", se)

    # Log DR-specific diagnostics
    if 'dr_diagnostics' in results:
        dr = results['dr_diagnostics']

        # Overlap
        if 'overlap' in dr:
            overlap = dr['overlap']
            mlflow.log_metric(f"{prefix}overlap_pct_extreme", overlap['pct_extreme'])
            mlflow.log_metric(f"{prefix}overlap_pct_overlap", overlap['pct_overlap'])

        # Weights
        if 'weights' in dr:
            weights = dr['weights']
            mlflow.log_metric(f"{prefix}weight_mean", weights['mean'])
            mlflow.log_metric(f"{prefix}weight_median", weights['median'])
            mlflow.log_metric(f"{prefix}weight_max", weights['max'])

        # Balance
        if 'balance' in dr:
            balance = dr['balance']
            mlflow.log_metric(f"{prefix}smd_unweighted_mean", balance['smd_unweighted_mean'])
            mlflow.log_metric(f"{prefix}smd_unweighted_max", balance['smd_unweighted_max'])
            if 'smd_weighted_mean' in balance:
                mlflow.log_metric(f"{prefix}smd_weighted_mean", balance['smd_weighted_mean'])
                mlflow.log_metric(f"{prefix}smd_weighted_max", balance['smd_weighted_max'])

    logger.info(f"Metrics logged to MLflow with prefix '{prefix}'")

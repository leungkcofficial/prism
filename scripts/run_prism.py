#!/usr/bin/env python3
"""
PRISM CLI Entry Point

Command-line interface for running PRISM causal survival analysis pipeline.

Usage:
    python scripts/run_prism.py --config configs/s_learner.yaml
    python scripts/run_prism.py --config configs/t_learner.yaml
    python scripts/run_prism.py --config configs/dr_learner.yaml

Author: PRISM Development Team
Date: 2026-01-11
"""

import click
import yaml
import sys
import os
from pathlib import Path
import logging
import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.prism_training_pipeline import prism_training_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to configuration YAML file (e.g., configs/s_learner.yaml)'
)
@click.option(
    '--experiment',
    type=str,
    default=None,
    help='MLflow experiment name (overrides config)'
)
@click.option(
    '--run-name',
    type=str,
    default=None,
    help='MLflow run name'
)
@click.option(
    '--subset',
    type=int,
    default=None,
    help='Use subset of data for testing (number of patients)'
)
@click.option(
    '--epochs',
    type=int,
    default=None,
    help='Override number of training epochs'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate configuration without running pipeline'
)
def main(config, experiment, run_name, subset, epochs, dry_run):
    """
    Run PRISM causal survival analysis pipeline.

    This script orchestrates the full pipeline from data ingestion to model evaluation.
    """
    logger.info("=" * 80)
    logger.info("PRISM - Predictive Renal Intelligence Survival Modeling")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"\nLoading configuration from: {config}")
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Validate configuration
    if not _validate_config(cfg):
        logger.error("Configuration validation failed!")
        sys.exit(1)

    # Apply CLI overrides
    if subset:
        logger.info(f"Using subset of {subset} patients (for testing)")
        cfg['subset'] = subset

    if epochs:
        logger.info(f"Overriding epochs to {epochs}")
        if 'model' not in cfg:
            cfg['model'] = {}
        cfg['model']['epochs'] = epochs

    # Get experiment name
    if experiment is None:
        experiment = cfg.get('mlflow', {}).get('experiment_name', 'prism_experiment')

    logger.info(f"\nMLflow experiment: {experiment}")

    if dry_run:
        logger.info("\n✓ Dry run complete - configuration is valid")
        logger.info("=" * 80)
        return

    # Set up MLflow
    tracking_uri = cfg.get('mlflow', {}).get('tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        mlflow.log_params({
            'config_file': config,
            'mode': cfg.get('project', {}).get('mode', 'unknown')
        })

        # Log all config parameters (flatten nested dict)
        _log_config_params(cfg)

        # Run pipeline
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING PIPELINE")
            logger.info("=" * 80)

            # Execute pipeline
            results = prism_training_pipeline(config=cfg)

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETE - RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Mode: {results['mode']}")
            logger.info(f"Cohort size: {results['cohort_size']}")
            logger.info(f"Training samples: {results['train_size']}")
            logger.info(f"Validation samples: {results['val_size']}")
            logger.info(f"Temporal test samples: {results['temporal_test_size']}")
            logger.info(f"Spatial test samples: {results['spatial_test_size']}")

            # Log summary metrics
            if 'temporal_results' in results:
                temporal = results['temporal_results']
                if 'causal' in temporal and 'ate' in temporal['causal']:
                    logger.info("\nTemporal Test - ATE (1/3/5 years):")
                    for t, value in temporal['causal']['ate'].items():
                        logger.info(f"  {t} days: {value:.4f}")

            if 'spatial_results' in results:
                spatial = results['spatial_results']
                if 'causal' in spatial and 'ate' in spatial['causal']:
                    logger.info("\nSpatial Test - ATE (1/3/5 years):")
                    for t, value in spatial['causal']['ate'].items():
                        logger.info(f"  {t} days: {value:.4f}")

            logger.info("\n✓ Pipeline executed successfully!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"\n✗ Pipeline failed with error: {e}")
            logger.exception("Full traceback:")
            mlflow.log_param('pipeline_status', 'failed')
            mlflow.log_param('error_message', str(e))
            sys.exit(1)


def _validate_config(cfg: dict) -> bool:
    """
    Validate configuration file.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    logger.info("Validating configuration...")

    required_keys = ['project', 'cohort', 'features', 'model']
    for key in required_keys:
        if key not in cfg:
            logger.error(f"Missing required configuration section: {key}")
            return False

    # Validate mode
    mode = cfg.get('project', {}).get('mode')
    if mode not in ['s_learner', 't_learner', 'dr_learner']:
        logger.error(f"Invalid mode: {mode}. Must be 's_learner', 't_learner', or 'dr_learner'")
        return False

    logger.info(f"✓ Configuration valid (mode: {mode})")
    return True


def _log_config_params(cfg: dict, prefix: str = ''):
    """
    Recursively log configuration parameters to MLflow.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    prefix : str
        Prefix for parameter names
    """
    for key, value in cfg.items():
        param_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively log nested dicts
            _log_config_params(value, prefix=f"{param_name}.")
        elif isinstance(value, (list, tuple)):
            # Convert lists to strings
            mlflow.log_param(param_name, str(value))
        elif isinstance(value, (str, int, float, bool)):
            # Log primitive types directly
            mlflow.log_param(param_name, value)
        else:
            # Convert other types to string
            mlflow.log_param(param_name, str(value))


if __name__ == '__main__':
    main()

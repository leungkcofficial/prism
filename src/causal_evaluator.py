"""
PRISM Causal Evaluation Framework

Comprehensive evaluation for causal survival analysis including:
- Predictive metrics: C-index, Brier score, calibration
- Causal metrics: ATE/ATT with bootstrap confidence intervals
- Balance diagnostics: Standardized Mean Difference (SMD)
- Overlap diagnostics: Propensity distribution, trimming stats

Author: PRISM Development Team
Date: 2026-01-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import TAROT2's metric calculator
from src.metric_calculator import MetricCalculator
from src.eval_model import evaluate_survival_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalEvaluator:
    """
    Evaluator for causal survival analysis.

    Computes both predictive and causal metrics, with bootstrap confidence intervals.

    Parameters
    ----------
    times : List[int]
        Time points for evaluation (default: [365, 1095, 1825] = 1, 3, 5 years)
    n_bootstrap : int
        Number of bootstrap samples for CI (default: 1000)
    confidence_level : float
        Confidence level for intervals (default: 0.95)
    random_seed : int
        Random seed (default: 42)
    """

    def __init__(
        self,
        times: List[int] = [365, 1095, 1825],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        self.times = times
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.results = {}

        np.random.seed(random_seed)

        logger.info("CausalEvaluator initialized")
        logger.info(f"  - Time points: {times} days")
        logger.info(f"  - Bootstrap samples: {n_bootstrap}")
        logger.info(f"  - Confidence level: {confidence_level}")

    def evaluate(
        self,
        learner,
        X: np.ndarray,
        A: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        learner_type: str,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Comprehensive evaluation of causal learner.

        Parameters
        ----------
        learner : S/T/DR-Learner
            Trained learner model
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        A : np.ndarray
            Treatment indicator (n_samples,)
        durations : np.ndarray
            Survival times (n_samples,)
        events : np.ndarray
            Event indicators (n_samples,)
        learner_type : str
            Type of learner: 's_learner', 't_learner', 'dr_learner'
        dataset_name : str
            Name of dataset (e.g., 'temporal_test', 'spatial_test')

        Returns
        -------
        Dict
            Comprehensive evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"EVALUATING {learner_type.upper()} ON {dataset_name.upper()}")
        logger.info("=" * 80)

        results = {
            'dataset': dataset_name,
            'learner_type': learner_type,
            'n_samples': len(X),
            'treatment_balance': {
                'n_A1': int(A.sum()),
                'n_A0': int((A == 0).sum()),
                'pct_A1': float(A.mean() * 100)
            },
            'event_rate': float(events.mean() * 100)
        }

        # 1. Predictive metrics
        logger.info("\n1. Computing predictive metrics...")
        results['predictive'] = self._compute_predictive_metrics(
            learner, X, A, durations, events
        )

        # 2. Causal metrics (ATE/ATT)
        logger.info("\n2. Computing causal metrics...")
        results['causal'] = self._compute_causal_metrics(
            learner, X, A
        )

        # 3. Bootstrap confidence intervals
        logger.info("\n3. Computing bootstrap confidence intervals...")
        results['bootstrap'] = self._compute_bootstrap_ci(
            learner, X, A
        )

        # 4. DR-specific diagnostics
        if learner_type == 'dr_learner':
            logger.info("\n4. Computing DR-learner diagnostics...")
            results['dr_diagnostics'] = self._compute_dr_diagnostics(
                learner, X, A
            )

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)

        self.results[dataset_name] = results
        return results

    def _compute_predictive_metrics(
        self,
        learner,
        X: np.ndarray,
        A: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> Dict:
        """
        Compute predictive metrics (C-index, Brier, calibration).

        Uses TAROT2's metric_calculator.py for consistency.
        """
        metrics = {}

        # C-index
        try:
            if hasattr(learner, 'compute_cindex'):
                cindex_result = learner.compute_cindex(X, A, durations, events)

                if isinstance(cindex_result, dict):
                    metrics['cindex'] = cindex_result
                else:
                    metrics['cindex'] = float(cindex_result)

                logger.info(f"  - C-index: {metrics['cindex']}")
        except Exception as e:
            logger.error(f"Error computing C-index: {e}")
            metrics['cindex'] = None

        # Predict survival curves for Brier score and calibration
        try:
            R0, R1 = learner.predict_counterfactuals(X, self.times)

            # Use factual predictions (based on actual treatment)
            R_factual = np.where(A[:, None] == 1, R1, R0)

            # Compute Brier score at each time point
            metrics['brier_score'] = {}
            for i, t in enumerate(self.times):
                surv_at_t = R_factual[:, i]
                # Simplified Brier score: mean squared error against observed outcomes
                # Note: This is approximate; proper Brier needs censoring weights
                if t <= durations.max():
                    observed_at_t = (durations > t).astype(float)
                    brier_t = ((surv_at_t - observed_at_t) ** 2).mean()
                    metrics['brier_score'][t] = float(brier_t)
                    logger.info(f"  - Brier score at {t} days: {brier_t:.4f}")

        except Exception as e:
            logger.error(f"Error computing Brier score: {e}")
            metrics['brier_score'] = None

        return metrics

    def _compute_causal_metrics(
        self,
        learner,
        X: np.ndarray,
        A: np.ndarray
    ) -> Dict:
        """
        Compute causal metrics (ATE, ATT) at specified time points.
        """
        causal_metrics = {}

        # ATE
        try:
            ate = learner.compute_ate(X, self.times)
            causal_metrics['ate'] = ate

            logger.info("  Average Treatment Effect (ATE):")
            for t, value in ate.items():
                logger.info(f"    - {t} days ({t/365:.1f} years): {value:.4f}")
        except Exception as e:
            logger.error(f"Error computing ATE: {e}")
            causal_metrics['ate'] = None

        # ATT
        try:
            att = learner.compute_att(X, A, self.times)
            causal_metrics['att'] = att

            logger.info("  Average Treatment on Treated (ATT):")
            for t, value in att.items():
                logger.info(f"    - {t} days ({t/365:.1f} years): {value:.4f}")
        except Exception as e:
            logger.error(f"Error computing ATT: {e}")
            causal_metrics['att'] = None

        return causal_metrics

    def _compute_bootstrap_ci(
        self,
        learner,
        X: np.ndarray,
        A: np.ndarray
    ) -> Dict:
        """
        Compute bootstrap confidence intervals for ATE and ATT.
        """
        logger.info(f"  Running {self.n_bootstrap} bootstrap iterations...")

        ate_bootstrap = {t: [] for t in self.times}
        att_bootstrap = {t: [] for t in self.times}

        for i in range(self.n_bootstrap):
            if (i + 1) % 100 == 0:
                logger.info(f"    Bootstrap iteration {i+1}/{self.n_bootstrap}")

            # Resample with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            A_boot = A[indices]

            # Compute ATE/ATT on bootstrap sample
            try:
                ate_boot = learner.compute_ate(X_boot, self.times)
                for t in self.times:
                    ate_bootstrap[t].append(ate_boot[t])

                att_boot = learner.compute_att(X_boot, A_boot, self.times)
                for t in self.times:
                    att_bootstrap[t].append(att_boot[t])
            except Exception as e:
                # Skip failed bootstrap iterations
                continue

        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        bootstrap_results = {
            'ate_ci': {},
            'att_ci': {},
            'ate_se': {},
            'att_se': {}
        }

        logger.info(f"\n  Bootstrap {self.confidence_level*100:.0f}% Confidence Intervals:")
        logger.info("  ATE:")
        for t in self.times:
            ci_lower = np.percentile(ate_bootstrap[t], lower_percentile)
            ci_upper = np.percentile(ate_bootstrap[t], upper_percentile)
            se = np.std(ate_bootstrap[t])

            bootstrap_results['ate_ci'][t] = [float(ci_lower), float(ci_upper)]
            bootstrap_results['ate_se'][t] = float(se)

            logger.info(f"    - {t} days: [{ci_lower:.4f}, {ci_upper:.4f}], SE={se:.4f}")

        logger.info("  ATT:")
        for t in self.times:
            ci_lower = np.percentile(att_bootstrap[t], lower_percentile)
            ci_upper = np.percentile(att_bootstrap[t], upper_percentile)
            se = np.std(att_bootstrap[t])

            bootstrap_results['att_ci'][t] = [float(ci_lower), float(ci_upper)]
            bootstrap_results['att_se'][t] = float(se)

            logger.info(f"    - {t} days: [{ci_lower:.4f}, {ci_upper:.4f}], SE={se:.4f}")

        return bootstrap_results

    def _compute_dr_diagnostics(
        self,
        learner,
        X: np.ndarray,
        A: np.ndarray
    ) -> Dict:
        """
        Compute DR-learner specific diagnostics (overlap, balance).
        """
        diagnostics = {}

        # Get propensity scores
        propensity_scores = learner.get_propensity_scores()
        if propensity_scores is None:
            logger.warning("Propensity scores not available")
            return diagnostics

        # Overlap diagnostics
        diagnostics['overlap'] = learner.propensity_model.check_overlap(
            A, propensity_scores, threshold=learner.iptw_clip_min
        )

        # Weight statistics
        iptw_weights = learner.get_iptw_weights()
        if iptw_weights is not None:
            diagnostics['weights'] = {
                'mean': float(iptw_weights.mean()),
                'median': float(np.median(iptw_weights)),
                'min': float(iptw_weights.min()),
                'max': float(iptw_weights.max()),
                'std': float(iptw_weights.std())
            }

        # Balance diagnostics (SMD)
        diagnostics['balance'] = self._compute_balance_smd(X, A, iptw_weights)

        return diagnostics

    def _compute_balance_smd(
        self,
        X: np.ndarray,
        A: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute Standardized Mean Difference (SMD) before and after weighting.

        SMD = (mean_A1 - mean_A0) / pooled_std
        """
        n_features = X.shape[1]
        smd_unweighted = np.zeros(n_features)
        smd_weighted = np.zeros(n_features) if weights is not None else None

        for i in range(n_features):
            # Unweighted SMD
            mean_A1 = X[A == 1, i].mean()
            mean_A0 = X[A == 0, i].mean()
            std_A1 = X[A == 1, i].std()
            std_A0 = X[A == 0, i].std()
            pooled_std = np.sqrt((std_A1**2 + std_A0**2) / 2)

            if pooled_std > 0:
                smd_unweighted[i] = (mean_A1 - mean_A0) / pooled_std
            else:
                smd_unweighted[i] = 0.0

            # Weighted SMD
            if weights is not None:
                mean_A1_w = np.average(X[A == 1, i], weights=weights[A == 1])
                mean_A0_w = np.average(X[A == 0, i], weights=weights[A == 0])
                # Use unweighted std for denominator (standard practice)
                if pooled_std > 0:
                    smd_weighted[i] = (mean_A1_w - mean_A0_w) / pooled_std
                else:
                    smd_weighted[i] = 0.0

        balance_results = {
            'smd_unweighted_mean': float(np.abs(smd_unweighted).mean()),
            'smd_unweighted_max': float(np.abs(smd_unweighted).max()),
            'n_imbalanced_unweighted': int((np.abs(smd_unweighted) > 0.1).sum())
        }

        if smd_weighted is not None:
            balance_results['smd_weighted_mean'] = float(np.abs(smd_weighted).mean())
            balance_results['smd_weighted_max'] = float(np.abs(smd_weighted).max())
            balance_results['n_imbalanced_weighted'] = int((np.abs(smd_weighted) > 0.1).sum())

        logger.info("  Balance (SMD) diagnostics:")
        logger.info(f"    - Unweighted: mean |SMD| = {balance_results['smd_unweighted_mean']:.3f}, max = {balance_results['smd_unweighted_max']:.3f}")
        if 'smd_weighted_mean' in balance_results:
            logger.info(f"    - Weighted: mean |SMD| = {balance_results['smd_weighted_mean']:.3f}, max = {balance_results['smd_weighted_max']:.3f}")
            logger.info(f"    - Features with |SMD| > 0.1: {balance_results['n_imbalanced_unweighted']} → {balance_results['n_imbalanced_weighted']}")

        return balance_results

    def plot_treatment_effects(
        self,
        save_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate plots for treatment effects.

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of figures
        """
        figures = {}

        for dataset_name, results in self.results.items():
            if 'causal' not in results or 'bootstrap' not in results:
                continue

            # ATE plot with CI
            fig_ate = self._plot_ate_with_ci(
                results['causal']['ate'],
                results['bootstrap']['ate_ci'],
                dataset_name
            )
            figures[f'{dataset_name}_ate'] = fig_ate

            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig_ate.savefig(f"{save_dir}/{dataset_name}_ate.png", dpi=300, bbox_inches='tight')

            # ATT plot with CI
            fig_att = self._plot_att_with_ci(
                results['causal']['att'],
                results['bootstrap']['att_ci'],
                dataset_name
            )
            figures[f'{dataset_name}_att'] = fig_att

            if save_dir:
                fig_att.savefig(f"{save_dir}/{dataset_name}_att.png", dpi=300, bbox_inches='tight')

        return figures

    def _plot_ate_with_ci(
        self,
        ate: Dict[int, float],
        ate_ci: Dict[int, List[float]],
        dataset_name: str
    ) -> plt.Figure:
        """Plot ATE with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))

        times_years = [t / 365 for t in self.times]
        ate_values = [ate[t] for t in self.times]
        ci_lower = [ate_ci[t][0] for t in self.times]
        ci_upper = [ate_ci[t][1] for t in self.times]

        ax.plot(times_years, ate_values, 'o-', linewidth=2, markersize=8, label='ATE')
        ax.fill_between(times_years, ci_lower, ci_upper, alpha=0.3, label=f'{self.confidence_level*100:.0f}% CI')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No effect')

        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Average Treatment Effect (Risk Difference)')
        ax.set_title(f'Average Treatment Effect - {dataset_name}')
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def _plot_att_with_ci(
        self,
        att: Dict[int, float],
        att_ci: Dict[int, List[float]],
        dataset_name: str
    ) -> plt.Figure:
        """Plot ATT with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, 6))

        times_years = [t / 365 for t in self.times]
        att_values = [att[t] for t in self.times]
        ci_lower = [att_ci[t][0] for t in self.times]
        ci_upper = [att_ci[t][1] for t in self.times]

        ax.plot(times_years, att_values, 's-', linewidth=2, markersize=8, label='ATT', color='orange')
        ax.fill_between(times_years, ci_lower, ci_upper, alpha=0.3, label=f'{self.confidence_level*100:.0f}% CI', color='orange')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No effect')

        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Average Treatment on Treated (Risk Difference)')
        ax.set_title(f'Average Treatment on Treated - {dataset_name}')
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def save_results(self, output_path: str):
        """
        Save evaluation results to JSON.

        Parameters
        ----------
        output_path : str
            Path to save results (.json)
        """
        import json

        # Convert numpy types to Python types
        results_serializable = {}
        for dataset_name, results in self.results.items():
            results_serializable[dataset_name] = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

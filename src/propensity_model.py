"""
PRISM Propensity Score Model

Estimates propensity scores e(X) = P(A=1|X) for DR-learner.
Supports multiple model types: Logistic Regression, Gradient Boosting, XGBoost.

Author: PRISM Development Team
Date: 2026-01-11
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropensityModel:
    """
    Propensity score model for estimating P(A=1|X).

    Parameters
    ----------
    model_type : str
        Model type: 'logistic', 'gbdt', or 'xgboost' (default: 'gbdt')
    random_seed : int
        Random seed (default: 42)
    **model_kwargs
        Additional keyword arguments for the model
    """

    def __init__(
        self,
        model_type: str = 'gbdt',
        random_seed: int = 42,
        **model_kwargs
    ):
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_kwargs = model_kwargs

        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                penalty='l2',
                C=model_kwargs.get('C', 1.0),
                max_iter=model_kwargs.get('max_iter', 1000),
                random_state=random_seed
            )
        elif model_type == 'gbdt':
            self.model = GradientBoostingClassifier(
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', 5),
                learning_rate=model_kwargs.get('learning_rate', 0.1),
                min_samples_split=model_kwargs.get('min_samples_split', 20),
                min_samples_leaf=model_kwargs.get('min_samples_leaf', 10),
                random_state=random_seed
            )
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=model_kwargs.get('n_estimators', 100),
                    max_depth=model_kwargs.get('max_depth', 5),
                    learning_rate=model_kwargs.get('learning_rate', 0.1),
                    random_state=random_seed
                )
            except ImportError:
                logger.error("XGBoost not installed. Install with: pip install xgboost")
                logger.error("Falling back to Gradient Boosting (gbdt)")
                self.model_type = 'gbdt'
                self.model = GradientBoostingClassifier(
                    n_estimators=model_kwargs.get('n_estimators', 100),
                    max_depth=model_kwargs.get('max_depth', 5),
                    learning_rate=model_kwargs.get('learning_rate', 0.1),
                    random_state=random_seed
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'logistic', 'gbdt', or 'xgboost'")

        logger.info(f"Propensity model initialized: {self.model_type}")

    def fit(self, X: np.ndarray, A: np.ndarray) -> 'PropensityModel':
        """
        Fit propensity score model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        A : np.ndarray
            Treatment indicator (n_samples,), 0 or 1

        Returns
        -------
        PropensityModel
            Fitted model (self)
        """
        logger.info("=" * 80)
        logger.info("TRAINING PROPENSITY MODEL")
        logger.info("=" * 80)
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Treatment prevalence: {A.mean()*100:.1f}%")

        # Fit model
        self.model.fit(X, A)

        # Compute and log performance metrics
        train_proba = self.model.predict_proba(X)[:, 1]
        self._log_propensity_diagnostics(A, train_proba, dataset='Training')

        logger.info("Propensity model training complete")
        logger.info("=" * 80)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict propensity scores e(X) = P(A=1|X).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Propensity scores (n_samples,)
        """
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def compute_iptw_weights(
        self,
        A: np.ndarray,
        e_hat: np.ndarray,
        clip_min: float = 0.05,
        clip_max: float = 0.95,
        stabilize: bool = True,
        normalize: bool = True,
        max_weight: Optional[float] = 50.0
    ) -> np.ndarray:
        """
        Compute Inverse Probability of Treatment Weighting (IPTW) weights.

        Stabilized weights: w_i = P(A) / e(X_i) if A_i=1 else (1-P(A)) / (1-e(X_i))
        Unstabilized weights: w_i = 1 / e(X_i) if A_i=1 else 1 / (1-e(X_i))

        Parameters
        ----------
        A : np.ndarray
            Treatment indicator (n_samples,)
        e_hat : np.ndarray
            Propensity scores (n_samples,)
        clip_min : float
            Clip propensity scores to [clip_min, clip_max] (default: 0.05)
        clip_max : float
            Upper clip threshold (default: 0.95)
        stabilize : bool
            Use stabilized weights (default: True)
        normalize : bool
            Normalize weights to sum to n (default: True)
        max_weight : float, optional
            Cap maximum weight (default: 50.0)

        Returns
        -------
        np.ndarray
            IPTW weights (n_samples,)
        """
        logger.info("Computing IPTW weights...")
        logger.info(f"  - Clipping: [{clip_min}, {clip_max}]")
        logger.info(f"  - Stabilized: {stabilize}")
        logger.info(f"  - Normalized: {normalize}")
        logger.info(f"  - Max weight cap: {max_weight}")

        # Clip propensity scores
        e_hat_clipped = np.clip(e_hat, clip_min, clip_max)

        # Report trimming
        n_trimmed = np.sum((e_hat < clip_min) | (e_hat > clip_max))
        if n_trimmed > 0:
            logger.info(f"  - Trimmed {n_trimmed} ({n_trimmed/len(e_hat)*100:.1f}%) propensity scores")

        # Compute weights
        if stabilize:
            # Stabilized IPTW weights
            p = A.mean()  # Marginal treatment probability
            weights = np.where(
                A == 1,
                p / e_hat_clipped,
                (1 - p) / (1 - e_hat_clipped)
            )
        else:
            # Unstabilized IPTW weights
            weights = np.where(
                A == 1,
                1 / e_hat_clipped,
                1 / (1 - e_hat_clipped)
            )

        # Cap maximum weight
        if max_weight is not None:
            n_capped = np.sum(weights > max_weight)
            if n_capped > 0:
                logger.info(f"  - Capped {n_capped} ({n_capped/len(weights)*100:.1f}%) weights at {max_weight}")
            weights = np.minimum(weights, max_weight)

        # Normalize weights
        if normalize:
            weights = weights * len(weights) / weights.sum()

        # Log weight statistics
        logger.info(f"  - Weight statistics:")
        logger.info(f"    - Mean: {weights.mean():.3f}")
        logger.info(f"    - Median: {np.median(weights):.3f}")
        logger.info(f"    - Min: {weights.min():.3f}, Max: {weights.max():.3f}")
        logger.info(f"    - Std: {weights.std():.3f}")

        return weights

    def check_overlap(
        self,
        A: np.ndarray,
        e_hat: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """
        Check overlap/positivity assumption.

        Overlap is violated if propensity scores are too extreme (near 0 or 1).

        Parameters
        ----------
        A : np.ndarray
            Treatment indicator
        e_hat : np.ndarray
            Propensity scores
        threshold : float
            Threshold for extreme propensity scores (default: 0.05)

        Returns
        -------
        Dict
            Overlap diagnostics
        """
        # Separate by treatment group
        e_A0 = e_hat[A == 0]
        e_A1 = e_hat[A == 1]

        # Check for extreme values
        extreme_low = (e_hat < threshold).sum()
        extreme_high = (e_hat > 1 - threshold).sum()

        # Overlap region
        overlap_mask = (e_hat >= threshold) & (e_hat <= 1 - threshold)
        pct_overlap = overlap_mask.mean() * 100

        diagnostics = {
            'n_extreme_low': int(extreme_low),
            'n_extreme_high': int(extreme_high),
            'pct_extreme': float((extreme_low + extreme_high) / len(e_hat) * 100),
            'pct_overlap': float(pct_overlap),
            'min_propensity': float(e_hat.min()),
            'max_propensity': float(e_hat.max()),
            'min_propensity_A0': float(e_A0.min()),
            'max_propensity_A0': float(e_A0.max()),
            'min_propensity_A1': float(e_A1.min()),
            'max_propensity_A1': float(e_A1.max())
        }

        logger.info("Overlap diagnostics:")
        logger.info(f"  - Extreme low (<{threshold}): {extreme_low} ({diagnostics['pct_extreme']/2:.1f}%)")
        logger.info(f"  - Extreme high (>{1-threshold}): {extreme_high} ({diagnostics['pct_extreme']/2:.1f}%)")
        logger.info(f"  - Overlap region: {pct_overlap:.1f}%")

        if diagnostics['pct_extreme'] > 10:
            logger.warning(f"WARNING: {diagnostics['pct_extreme']:.1f}% of propensity scores are extreme")
            logger.warning("Consider stronger trimming or checking model specification")

        return diagnostics

    def plot_propensity_distribution(
        self,
        A: np.ndarray,
        e_hat: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot propensity score distribution by treatment group.

        Parameters
        ----------
        A : np.ndarray
            Treatment indicator
        e_hat : np.ndarray
            Propensity scores
        save_path : str, optional
            Path to save plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Separate by treatment group
        e_A0 = e_hat[A == 0]
        e_A1 = e_hat[A == 1]

        # Plot histograms
        ax.hist(e_A0, bins=30, alpha=0.5, label='A=0 (Control)', density=True, color='blue')
        ax.hist(e_A1, bins=30, alpha=0.5, label='A=1 (Treated)', density=True, color='red')

        # Add reference lines
        ax.axvline(0.05, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Trimming bounds')
        ax.axvline(0.95, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Propensity Score e(X) = P(A=1|X)')
        ax.set_ylabel('Density')
        ax.set_title('Propensity Score Distribution by Treatment Group')
        ax.legend()
        ax.grid(alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Propensity distribution plot saved to {save_path}")

        return fig

    def _log_propensity_diagnostics(
        self,
        A: np.ndarray,
        e_hat: np.ndarray,
        dataset: str = 'Training'
    ):
        """Log propensity score diagnostics."""
        logger.info(f"{dataset} propensity score statistics:")
        logger.info(f"  - Mean: {e_hat.mean():.3f}")
        logger.info(f"  - Median: {np.median(e_hat):.3f}")
        logger.info(f"  - Min: {e_hat.min():.3f}, Max: {e_hat.max():.3f}")
        logger.info(f"  - Std: {e_hat.std():.3f}")

        # Separate by treatment group
        e_A0 = e_hat[A == 0]
        e_A1 = e_hat[A == 1]

        logger.info(f"  - A=0 range: [{e_A0.min():.3f}, {e_A0.max():.3f}]")
        logger.info(f"  - A=1 range: [{e_A1.min():.3f}, {e_A1.max():.3f}]")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (for tree-based models).

        Returns
        -------
        np.ndarray or None
            Feature importance if available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None

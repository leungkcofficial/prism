"""
PRISM DR-Learner Implementation

DR-learner (Doubly Robust learner) combines propensity score weighting with outcome modeling.
Uses IPTW (Inverse Probability of Treatment Weighting) to balance treatment groups.

Reference: Kennedy (2020) "Towards optimal doubly robust estimation"

Author: PRISM Development Team
Date: 2026-01-11
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging

from src.deepsurv_wrapper import DeepSurvWrapper
from src.propensity_model import PropensityModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DRLearner:
    """
    DR-learner for causal survival analysis.

    Trains a propensity model to estimate e(X) = P(A=1|X), then trains
    a weighted survival model with treatment A as a feature.

    Parameters
    ----------
    input_dim : int
        Number of input features (including treatment A)
    hidden_layers : List[int]
        Hidden layer sizes for survival model (default: [128, 64, 32])
    dropout : float
        Dropout rate (default: 0.3)
    learning_rate : float
        Learning rate (default: 0.001)
    device : str
        Device ('cuda' or 'cpu', default: auto-detect)
    propensity_model_type : str
        Type of propensity model: 'logistic', 'gbdt', 'xgboost' (default: 'gbdt')
    propensity_kwargs : Dict, optional
        Additional kwargs for propensity model
    iptw_clip_min : float
        Minimum propensity score (default: 0.05)
    iptw_clip_max : float
        Maximum propensity score (default: 0.95)
    iptw_stabilize : bool
        Use stabilized IPTW weights (default: True)
    iptw_max_weight : float
        Maximum weight cap (default: 50.0)
    random_seed : int
        Random seed (default: 42)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None,
        propensity_model_type: str = 'gbdt',
        propensity_kwargs: Optional[Dict] = None,
        iptw_clip_min: float = 0.05,
        iptw_clip_max: float = 0.95,
        iptw_stabilize: bool = True,
        iptw_max_weight: float = 50.0,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.propensity_model_type = propensity_model_type
        self.iptw_clip_min = iptw_clip_min
        self.iptw_clip_max = iptw_clip_max
        self.iptw_stabilize = iptw_stabilize
        self.iptw_max_weight = iptw_max_weight
        self.random_seed = random_seed

        # Initialize propensity model
        if propensity_kwargs is None:
            propensity_kwargs = {}

        self.propensity_model = PropensityModel(
            model_type=propensity_model_type,
            random_seed=random_seed,
            **propensity_kwargs
        )

        # Initialize survival model (will include treatment A as feature)
        self.survival_model = DeepSurvWrapper(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            device=device,
            random_seed=random_seed
        )

        # Store propensity scores and weights
        self.propensity_scores_ = None
        self.iptw_weights_ = None

        logger.info("DR-Learner initialized")
        logger.info("Mode: Propensity-weighted survival model with treatment A as feature")

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple] = None,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train DR-learner model.

        Steps:
        1. Train propensity model e(X) = P(A=1|X)
        2. Compute IPTW weights
        3. Train weighted survival model

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features) WITHOUT treatment A
        A : np.ndarray
            Treatment indicator (n_samples,), 0 or 1
        durations : np.ndarray
            Survival times (n_samples,)
        events : np.ndarray
            Event indicators (n_samples,)
        val_data : Tuple, optional
            Validation data as (X_val, A_val, durations_val, events_val)
        batch_size : int
            Batch size (default: 256)
        epochs : int
            Max epochs (default: 100)
        patience : int
            Early stopping patience (default: 10)
        verbose : bool
            Print progress (default: True)

        Returns
        -------
        Dict
            Training logs for propensity and survival models
        """
        logger.info("=" * 80)
        logger.info("TRAINING DR-LEARNER")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Treatment distribution: A=1: {A.sum()} ({A.mean()*100:.1f}%), A=0: {(A==0).sum()} ({(1-A.mean())*100:.1f}%)")
        logger.info(f"Event rate: {events.sum()} ({events.mean()*100:.1f}%)")
        logger.info("")

        # Step 1: Train propensity model
        logger.info("Step 1: Training propensity model...")
        self.propensity_model.fit(X, A)

        # Step 2: Compute propensity scores and IPTW weights
        logger.info("")
        logger.info("Step 2: Computing IPTW weights...")
        self.propensity_scores_ = self.propensity_model.predict_proba(X)

        # Check overlap
        overlap_diagnostics = self.propensity_model.check_overlap(
            A, self.propensity_scores_, threshold=self.iptw_clip_min
        )

        # Compute IPTW weights
        self.iptw_weights_ = self.propensity_model.compute_iptw_weights(
            A=A,
            e_hat=self.propensity_scores_,
            clip_min=self.iptw_clip_min,
            clip_max=self.iptw_clip_max,
            stabilize=self.iptw_stabilize,
            normalize=True,
            max_weight=self.iptw_max_weight
        )

        # Step 3: Train weighted survival model
        logger.info("")
        logger.info("Step 3: Training weighted survival model...")

        # Concatenate treatment A as feature
        X_with_A = np.column_stack([X, A])

        # Process validation data if provided
        val_data_processed = None
        if val_data is not None:
            X_val, A_val, durations_val, events_val = val_data
            X_val_with_A = np.column_stack([X_val, A_val])
            val_data_processed = (X_val_with_A, (durations_val, events_val))
            logger.info(f"Validation samples: {len(X_val)}")

        # Train weighted survival model
        survival_log = self.survival_model.fit_weighted(
            X=X_with_A,
            durations=durations,
            events=events,
            weights=self.iptw_weights_,
            val_data=val_data_processed,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("DR-Learner training complete")
        logger.info("=" * 80)

        return {
            'propensity': {
                'overlap_diagnostics': overlap_diagnostics
            },
            'survival': survival_log
        }

    def predict_counterfactuals(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict counterfactual survival probabilities.

        For each patient, predict survival under both treatment conditions:
        - R0: Survival probability if A=0 (non-early dialysis)
        - R1: Survival probability if A=1 (early dialysis)

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features) WITHOUT treatment A
        times : List[int], optional
            Time points for prediction (default: [365, 1095, 1825])

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (R0, R1) where each is (n_samples, n_times)
        """
        if times is None:
            times = [365, 1095, 1825]

        # Create counterfactual datasets
        n_samples = len(X)

        # Set A=0 for all patients
        A0 = np.zeros((n_samples, 1))
        X_A0 = np.column_stack([X, A0])

        # Set A=1 for all patients
        A1 = np.ones((n_samples, 1))
        X_A1 = np.column_stack([X, A1])

        # Predict survival probabilities
        R0 = self.survival_model.predict_survival(X_A0, times)
        R1 = self.survival_model.predict_survival(X_A1, times)

        return R0, R1

    def predict_counterfactual_risks(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict counterfactual mortality risks (1 - survival).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features) WITHOUT treatment A
        times : List[int], optional
            Time points for prediction (default: [365, 1095, 1825])

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Risk0, Risk1) where each is (n_samples, n_times)
        """
        R0, R1 = self.predict_counterfactuals(X, times)
        Risk0 = 1 - R0
        Risk1 = 1 - R1
        return Risk0, Risk1

    def compute_individual_treatment_effects(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute individual treatment effects (ITE).

        ITE = Risk1(t) - Risk0(t) for each patient and time point.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features) WITHOUT treatment A
        times : List[int], optional
            Time points (default: [365, 1095, 1825])

        Returns
        -------
        np.ndarray
            Individual treatment effects (n_samples, n_times)
        """
        Risk0, Risk1 = self.predict_counterfactual_risks(X, times)
        ITE = Risk1 - Risk0
        return ITE

    def compute_ate(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute Average Treatment Effect (ATE).

        ATE(t) = E[Risk1(t) - Risk0(t)]

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : List[int], optional
            Time points (default: [365, 1095, 1825])

        Returns
        -------
        Dict[int, float]
            ATE at each time point
        """
        if times is None:
            times = [365, 1095, 1825]

        ITE = self.compute_individual_treatment_effects(X, times)

        ate_dict = {}
        for i, t in enumerate(times):
            ate_dict[t] = float(ITE[:, i].mean())

        return ate_dict

    def compute_att(
        self,
        X: np.ndarray,
        A: np.ndarray,
        times: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute Average Treatment Effect on the Treated (ATT).

        ATT(t) = E[Risk1(t) - Risk0(t) | A=1]

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        A : np.ndarray
            Treatment indicator (n_samples,)
        times : List[int], optional
            Time points (default: [365, 1095, 1825])

        Returns
        -------
        Dict[int, float]
            ATT at each time point
        """
        if times is None:
            times = [365, 1095, 1825]

        ITE = self.compute_individual_treatment_effects(X, times)

        # Filter to treated patients
        treated_mask = (A == 1)
        ITE_treated = ITE[treated_mask]

        att_dict = {}
        for i, t in enumerate(times):
            att_dict[t] = float(ITE_treated[:, i].mean())

        return att_dict

    def compute_cindex(
        self,
        X: np.ndarray,
        A: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute concordance index on test data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        A : np.ndarray
            Treatment indicator
        durations : np.ndarray
            Survival times
        events : np.ndarray
            Event indicators

        Returns
        -------
        float
            Concordance index
        """
        X_with_A = np.column_stack([X, A])
        cindex = self.survival_model.compute_concordance_index(X_with_A, durations, events)
        return cindex

    def get_propensity_scores(self) -> Optional[np.ndarray]:
        """
        Get computed propensity scores.

        Returns
        -------
        np.ndarray or None
            Propensity scores if model has been trained
        """
        return self.propensity_scores_

    def get_iptw_weights(self) -> Optional[np.ndarray]:
        """
        Get computed IPTW weights.

        Returns
        -------
        np.ndarray or None
            IPTW weights if model has been trained
        """
        return self.iptw_weights_

    def plot_propensity_distribution(
        self,
        A: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot propensity score distribution.

        Parameters
        ----------
        A : np.ndarray
            Treatment indicator
        save_path : str, optional
            Path to save plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        if self.propensity_scores_ is None:
            raise ValueError("Model must be trained before plotting propensity distribution")

        return self.propensity_model.plot_propensity_distribution(
            A, self.propensity_scores_, save_path
        )

    def save(self, survival_path: str, propensity_path: str):
        """
        Save DR-learner models.

        Parameters
        ----------
        survival_path : str
            Path to save survival model (.pth)
        propensity_path : str
            Path to save propensity model (.pkl)
        """
        import pickle

        self.survival_model.save(survival_path)

        with open(propensity_path, 'wb') as f:
            pickle.dump(self.propensity_model, f)

        logger.info(f"DR-Learner saved: survival -> {survival_path}, propensity -> {propensity_path}")

    def load(self, survival_path: str, propensity_path: str):
        """
        Load DR-learner models.

        Parameters
        ----------
        survival_path : str
            Path to load survival model from (.pth)
        propensity_path : str
            Path to load propensity model from (.pkl)
        """
        import pickle

        self.survival_model.load(survival_path)

        with open(propensity_path, 'rb') as f:
            self.propensity_model = pickle.load(f)

        logger.info(f"DR-Learner loaded: survival <- {survival_path}, propensity <- {propensity_path}")

    def get_model_summary(self) -> Dict:
        """Get model summary."""
        survival_summary = self.survival_model.get_model_summary()

        summary = {
            'learner_type': 'dr_learner',
            'treatment_as_feature': True,
            'propensity_model_type': self.propensity_model_type,
            'iptw_clip_range': [self.iptw_clip_min, self.iptw_clip_max],
            'iptw_stabilize': self.iptw_stabilize,
            'iptw_max_weight': self.iptw_max_weight,
            'survival_model': survival_summary
        }

        return summary

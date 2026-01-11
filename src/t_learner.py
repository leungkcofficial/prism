"""
PRISM T-Learner Implementation

T-learner (Two learners) trains separate survival models for treated (A=1) and control (A=0).
Counterfactual predictions are made by using the appropriate model for each group.

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment effects"

Author: PRISM Development Team
Date: 2026-01-11
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging

from src.deepsurv_wrapper import DeepSurvWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TLearner:
    """
    T-learner for causal survival analysis.

    Trains two separate DeepSurv models:
    - model_A0: For control group (A=0, non-early dialysis)
    - model_A1: For treated group (A=1, early dialysis)

    Parameters
    ----------
    input_dim : int
        Number of input features (NOT including treatment A)
    hidden_layers : List[int]
        Hidden layer sizes (default: [128, 64, 32])
    dropout : float
        Dropout rate (default: 0.3)
    learning_rate : float
        Learning rate (default: 0.001)
    device : str
        Device ('cuda' or 'cpu', default: auto-detect)
    random_seed : int
        Random seed (default: 42)
    min_samples_per_group : int
        Minimum samples required per treatment group (default: 50)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None,
        random_seed: int = 42,
        min_samples_per_group: int = 50
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.random_seed = random_seed
        self.min_samples_per_group = min_samples_per_group

        # Initialize two DeepSurv models
        self.model_A0 = DeepSurvWrapper(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            device=device,
            random_seed=random_seed
        )

        self.model_A1 = DeepSurvWrapper(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            device=device,
            random_seed=random_seed + 1  # Different seed for second model
        )

        logger.info("T-Learner initialized")
        logger.info("Mode: Two separate models for A=0 and A=1")

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
        Train T-learner models.

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
            Training logs for both models
        """
        logger.info("=" * 80)
        logger.info("TRAINING T-LEARNER")
        logger.info("=" * 80)

        # Split data by treatment group
        mask_A0 = (A == 0)
        mask_A1 = (A == 1)

        X_A0 = X[mask_A0]
        durations_A0 = durations[mask_A0]
        events_A0 = events[mask_A0]

        X_A1 = X[mask_A1]
        durations_A1 = durations[mask_A1]
        events_A1 = events[mask_A1]

        logger.info(f"Total samples: {len(X)}")
        logger.info(f"A=0 (control): {len(X_A0)} samples ({len(X_A0)/len(X)*100:.1f}%)")
        logger.info(f"A=1 (treated): {len(X_A1)} samples ({len(X_A1)/len(X)*100:.1f}%)")

        # Check minimum sample size
        if len(X_A0) < self.min_samples_per_group:
            logger.warning(f"WARNING: Control group (A=0) has only {len(X_A0)} samples (< {self.min_samples_per_group})")
            logger.warning("T-learner may not have sufficient data for reliable estimation")

        if len(X_A1) < self.min_samples_per_group:
            logger.warning(f"WARNING: Treated group (A=1) has only {len(X_A1)} samples (< {self.min_samples_per_group})")
            logger.warning("T-learner may not have sufficient data for reliable estimation")

        # Split validation data if provided
        val_data_A0 = None
        val_data_A1 = None
        if val_data is not None:
            X_val, A_val, durations_val, events_val = val_data

            mask_val_A0 = (A_val == 0)
            mask_val_A1 = (A_val == 1)

            if mask_val_A0.sum() > 0:
                val_data_A0 = (
                    X_val[mask_val_A0],
                    (durations_val[mask_val_A0], events_val[mask_val_A0])
                )

            if mask_val_A1.sum() > 0:
                val_data_A1 = (
                    X_val[mask_val_A1],
                    (durations_val[mask_val_A1], events_val[mask_val_A1])
                )

        # Train model for A=0 (control)
        logger.info("")
        logger.info("-" * 80)
        logger.info("Training model_A0 (Control: A=0)")
        logger.info("-" * 80)
        log_A0 = self.model_A0.fit(
            X=X_A0,
            durations=durations_A0,
            events=events_A0,
            val_data=val_data_A0,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        # Train model for A=1 (treated)
        logger.info("")
        logger.info("-" * 80)
        logger.info("Training model_A1 (Treated: A=1)")
        logger.info("-" * 80)
        log_A1 = self.model_A1.fit(
            X=X_A1,
            durations=durations_A1,
            events=events_A1,
            val_data=val_data_A1,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("T-Learner training complete")
        logger.info("=" * 80)

        return {
            'model_A0': log_A0,
            'model_A1': log_A1
        }

    def predict_counterfactuals(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict counterfactual survival probabilities.

        For each patient, predict survival under both treatment conditions:
        - R0: Survival probability if A=0 (using model_A0)
        - R1: Survival probability if A=1 (using model_A1)

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
            - R0: Survival probabilities under A=0
            - R1: Survival probabilities under A=1
        """
        if times is None:
            times = [365, 1095, 1825]

        # Predict using both models
        R0 = self.model_A0.predict_survival(X, times)
        R1 = self.model_A1.predict_survival(X, times)

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
    ) -> Dict[str, float]:
        """
        Compute concordance index for both models.

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
        Dict[str, float]
            C-index for each model and overall
        """
        mask_A0 = (A == 0)
        mask_A1 = (A == 1)

        # C-index for control model (on control patients)
        if mask_A0.sum() > 0:
            cindex_A0 = self.model_A0.compute_concordance_index(
                X[mask_A0], durations[mask_A0], events[mask_A0]
            )
        else:
            cindex_A0 = np.nan

        # C-index for treated model (on treated patients)
        if mask_A1.sum() > 0:
            cindex_A1 = self.model_A1.compute_concordance_index(
                X[mask_A1], durations[mask_A1], events[mask_A1]
            )
        else:
            cindex_A1 = np.nan

        return {
            'cindex_A0': cindex_A0,
            'cindex_A1': cindex_A1,
            'cindex_overall': (cindex_A0 + cindex_A1) / 2 if not np.isnan([cindex_A0, cindex_A1]).any() else np.nan
        }

    def save(self, path_A0: str, path_A1: str):
        """Save T-learner models."""
        self.model_A0.save(path_A0)
        self.model_A1.save(path_A1)
        logger.info(f"T-Learner saved: model_A0 -> {path_A0}, model_A1 -> {path_A1}")

    def load(self, path_A0: str, path_A1: str):
        """Load T-learner models."""
        self.model_A0.load(path_A0)
        self.model_A1.load(path_A1)
        logger.info(f"T-Learner loaded: model_A0 <- {path_A0}, model_A1 <- {path_A1}")

    def get_model_summary(self) -> Dict:
        """Get model summary."""
        summary_A0 = self.model_A0.get_model_summary()
        summary_A1 = self.model_A1.get_model_summary()

        summary = {
            'learner_type': 't_learner',
            'treatment_as_feature': False,
            'model_A0': summary_A0,
            'model_A1': summary_A1
        }

        return summary

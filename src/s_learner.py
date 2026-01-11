"""
PRISM S-Learner Implementation

S-learner (Single learner) trains a single survival model with treatment A as a feature.
Counterfactual predictions are made by setting A=0 and A=1 for all patients.

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


class SLearner:
    """
    S-learner for causal survival analysis.

    Trains a single DeepSurv model with treatment A included as a feature.
    Estimates individual treatment effects by predicting counterfactuals.

    Parameters
    ----------
    input_dim : int
        Number of input features (including treatment A)
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
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.random_seed = random_seed

        # Initialize DeepSurv model
        self.model = DeepSurvWrapper(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            device=device,
            random_seed=random_seed
        )

        logger.info("S-Learner initialized")
        logger.info("Mode: Single model with treatment A as feature")

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
        Train S-learner model.

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
            Training log
        """
        logger.info("=" * 80)
        logger.info("TRAINING S-LEARNER")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Treatment distribution: A=1: {A.sum()} ({A.mean()*100:.1f}%), A=0: {(A==0).sum()} ({(1-A.mean())*100:.1f}%)")
        logger.info(f"Event rate: {events.sum()} ({events.mean()*100:.1f}%)")

        # Concatenate treatment A as feature
        X_with_A = np.column_stack([X, A])

        # Process validation data if provided
        val_data_processed = None
        if val_data is not None:
            X_val, A_val, durations_val, events_val = val_data
            X_val_with_A = np.column_stack([X_val, A_val])
            val_data_processed = (X_val_with_A, (durations_val, events_val))
            logger.info(f"Validation samples: {len(X_val)}")

        # Train model
        log = self.model.fit(
            X=X_with_A,
            durations=durations,
            events=events,
            val_data=val_data_processed,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        logger.info("S-Learner training complete")
        logger.info("=" * 80)

        return log

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
            - R0: Survival probabilities under A=0
            - R1: Survival probabilities under A=1
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
        R0 = self.model.predict_survival(X_A0, times)
        R1 = self.model.predict_survival(X_A1, times)

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
            - Risk0: Mortality risks under A=0
            - Risk1: Mortality risks under A=1
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
        Positive ITE means increased risk under early dialysis.

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
        Average over all patients.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : List[int], optional
            Time points (default: [365, 1095, 1825])

        Returns
        -------
        Dict[int, float]
            ATE at each time point {time: ate_value}
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
        Average over treated patients only.

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
            ATT at each time point {time: att_value}
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
        cindex = self.model.compute_concordance_index(X_with_A, durations, events)
        return cindex

    def save(self, path: str):
        """Save S-learner model."""
        self.model.save(path)
        logger.info(f"S-Learner saved to {path}")

    def load(self, path: str):
        """Load S-learner model."""
        self.model.load(path)
        logger.info(f"S-Learner loaded from {path}")

    def get_model_summary(self) -> Dict:
        """Get model summary."""
        summary = self.model.get_model_summary()
        summary['learner_type'] = 's_learner'
        summary['treatment_as_feature'] = True
        return summary

"""
PRISM DeepSurv Wrapper

Wrapper around PyCox's CoxPH model for survival analysis.
Adapted from TAROT2's model_train_surv.py with extensions for:
- Standard training (S-learner, T-learner)
- Weighted training with IPTW (DR-learner)
- Counterfactual prediction

Author: PRISM Development Team
Date: 2026-01-11
"""

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from typing import List, Tuple, Optional, Dict
import logging

# Import neural architectures from copied TAROT2 code
from src.nn_architectures import create_network

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSurvWrapper:
    """
    Wrapper for DeepSurv (Cox Proportional Hazards with neural networks).

    This class provides a unified interface for training and prediction with
    PyCox's CoxPH model, supporting both standard and weighted training.

    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_layers : List[int]
        List of hidden layer sizes (e.g., [128, 64, 32])
    dropout : float
        Dropout rate (default: 0.3)
    batch_norm : bool
        Whether to use batch normalization (default: True)
    learning_rate : float
        Learning rate for optimizer (default: 0.001)
    weight_decay : float
        L2 regularization parameter (default: 0.0)
    device : str
        Device to use ('cuda' or 'cpu', default: 'cuda' if available)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        batch_norm: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        device: str = None,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_seed = random_seed

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Initialize network
        self.net = self._create_network()

        # Initialize optimizer
        self.optimizer = tt.optim.Adam(
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize model
        self.model = CoxPH(self.net, self.optimizer, device=self.device)

        logger.info(f"DeepSurv initialized:")
        logger.info(f"  - Input dim: {input_dim}")
        logger.info(f"  - Hidden layers: {hidden_layers}")
        logger.info(f"  - Dropout: {dropout}")
        logger.info(f"  - Batch norm: {batch_norm}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Device: {self.device}")

    def _create_network(self) -> torch.nn.Module:
        """
        Create neural network using TAROT2's architecture factory.

        Uses MLP architecture from nn_architectures.py.
        """
        # Create network configuration
        config = {
            'architecture': 'mlp',
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'output_dim': 1,  # Cox model outputs single hazard value
            'dropout': self.dropout,
            'batch_norm': self.batch_norm
        }

        # Use create_network factory from nn_architectures.py
        net = create_network(config)

        return net

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        val_data: Optional[Tuple] = None,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
        callbacks: Optional[List] = None
    ) -> Dict:
        """
        Train DeepSurv model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        durations : np.ndarray
            Survival times (n_samples,)
        events : np.ndarray
            Event indicators (n_samples,), 1=event, 0=censored
        val_data : Tuple, optional
            Validation data as (X_val, (durations_val, events_val))
        batch_size : int
            Batch size for training (default: 256)
        epochs : int
            Maximum number of epochs (default: 100)
        patience : int
            Early stopping patience (default: 10)
        verbose : bool
            Whether to print training progress (default: True)
        callbacks : List, optional
            Additional callbacks for training

        Returns
        -------
        Dict
            Training log with loss history
        """
        logger.info(f"Training DeepSurv model...")
        logger.info(f"  - Training samples: {len(X)}")
        logger.info(f"  - Events: {events.sum()} ({events.mean()*100:.1f}%)")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Max epochs: {epochs}")
        logger.info(f"  - Early stopping patience: {patience}")

        # Prepare data for PyCox
        X_tensor = torch.tensor(X, dtype=torch.float32)
        durations_tensor = torch.tensor(durations, dtype=torch.float32)
        events_tensor = torch.tensor(events, dtype=torch.float32)

        # Prepare validation data if provided
        val_data_processed = None
        if val_data is not None:
            X_val, (durations_val, events_val) = val_data
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            durations_val_tensor = torch.tensor(durations_val, dtype=torch.float32)
            events_val_tensor = torch.tensor(events_val, dtype=torch.float32)
            val_data_processed = (X_val_tensor, (durations_val_tensor, events_val_tensor))

            logger.info(f"  - Validation samples: {len(X_val)}")

        # Setup callbacks
        callback_list = [
            tt.callbacks.EarlyStopping(patience=patience, min_delta=0.001)
        ]
        if callbacks is not None:
            callback_list.extend(callbacks)

        # Train model
        log = self.model.fit(
            input=X_tensor,
            target=(durations_tensor, events_tensor),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose,
            val_data=val_data_processed
        )

        logger.info(f"Training complete. Final loss: {log.monitors['train']['loss'][-1]:.4f}")

        return log

    def fit_weighted(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        weights: np.ndarray,
        val_data: Optional[Tuple] = None,
        batch_size: int = 256,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train DeepSurv model with sample weights (for DR-learner).

        Note: PyCox CoxPH doesn't natively support sample weights in the loss function.
        This implementation uses a weighted data loader approach where samples are
        resampled according to their weights.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        durations : np.ndarray
            Survival times (n_samples,)
        events : np.ndarray
            Event indicators (n_samples,)
        weights : np.ndarray
            Sample weights (n_samples,), typically IPTW weights
        val_data : Tuple, optional
            Validation data as (X_val, (durations_val, events_val))
        batch_size : int
            Batch size for training (default: 256)
        epochs : int
            Maximum number of epochs (default: 100)
        patience : int
            Early stopping patience (default: 10)
        verbose : bool
            Whether to print training progress (default: True)

        Returns
        -------
        Dict
            Training log with loss history
        """
        logger.info(f"Training DeepSurv model with sample weights (IPTW)...")
        logger.info(f"  - Training samples: {len(X)}")
        logger.info(f"  - Weight statistics:")
        logger.info(f"    - Mean: {weights.mean():.3f}")
        logger.info(f"    - Median: {np.median(weights):.3f}")
        logger.info(f"    - Min: {weights.min():.3f}, Max: {weights.max():.3f}")

        # Normalize weights to sum to n_samples (for proper resampling)
        weights_normalized = weights * len(weights) / weights.sum()

        # Resample indices according to weights
        # Higher weight = higher probability of being sampled
        n_resamples = len(X)
        resample_probs = weights_normalized / weights_normalized.sum()

        # Create resampled dataset (with replacement)
        resample_indices = np.random.choice(
            len(X),
            size=n_resamples,
            replace=True,
            p=resample_probs
        )

        X_resampled = X[resample_indices]
        durations_resampled = durations[resample_indices]
        events_resampled = events[resample_indices]

        logger.info(f"  - Resampled dataset created: {len(X_resampled)} samples")

        # Train on resampled dataset
        log = self.fit(
            X=X_resampled,
            durations=durations_resampled,
            events=events_resampled,
            val_data=val_data,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        return log

    def predict_survival(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Predict survival probabilities at specified time points.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : List[int], optional
            Time points for prediction (default: [365, 1095, 1825] = 1, 3, 5 years)

        Returns
        -------
        np.ndarray
            Survival probabilities (n_samples, n_times)
        """
        if times is None:
            times = [365, 1095, 1825]  # 1, 3, 5 years

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Predict survival curves
        surv = self.model.predict_surv_df(X_tensor)

        # Extract survival probabilities at specified times
        # surv is a DataFrame with index=time, columns=samples
        surv_at_times = []
        for t in times:
            # Find closest time point in survival curve
            closest_time = surv.index[np.argmin(np.abs(surv.index - t))]
            surv_at_time = surv.loc[closest_time].values
            surv_at_times.append(surv_at_time)

        surv_probs = np.array(surv_at_times).T  # Transpose to (n_samples, n_times)

        return surv_probs

    def predict_risk(
        self,
        X: np.ndarray,
        times: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Predict mortality risk (1 - survival probability) at specified time points.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : List[int], optional
            Time points for prediction (default: [365, 1095, 1825])

        Returns
        -------
        np.ndarray
            Mortality risks (n_samples, n_times)
        """
        surv_probs = self.predict_survival(X, times)
        risk_probs = 1 - surv_probs
        return risk_probs

    def predict_hazard(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log hazard ratio (relative risk).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Log hazard ratios (n_samples,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        log_hazard = self.model.predict(X_tensor)
        return log_hazard

    def compute_concordance_index(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute concordance index (C-index) for model predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        durations : np.ndarray
            Survival times
        events : np.ndarray
            Event indicators

        Returns
        -------
        float
            Concordance index (0.5 = random, 1.0 = perfect)
        """
        from pycox.evaluation import EvalSurv

        X_tensor = torch.tensor(X, dtype=torch.float32)
        surv = self.model.predict_surv_df(X_tensor)

        ev = EvalSurv(
            surv,
            durations,
            events,
            censor_surv='km'
        )

        cindex = ev.concordance_td()

        return cindex

    def save(self, path: str):
        """
        Save model weights.

        Parameters
        ----------
        path : str
            Path to save model weights (.pth file)
        """
        torch.save(self.model.net.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model weights.

        Parameters
        ----------
        path : str
            Path to load model weights from (.pth file)
        """
        self.model.net.load_state_dict(torch.load(path, map_location=self.device))
        self.model.net.eval()
        logger.info(f"Model loaded from {path}")

    def get_model_summary(self) -> Dict:
        """
        Get summary of model architecture and parameters.

        Returns
        -------
        Dict
            Model summary including architecture, parameters, device
        """
        n_params = sum(p.numel() for p in self.net.parameters())
        n_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        summary = {
            'architecture': 'mlp',
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'device': self.device,
            'n_parameters': n_params,
            'n_trainable_parameters': n_trainable_params
        }

        return summary

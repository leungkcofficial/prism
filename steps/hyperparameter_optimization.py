"""
Hyperparameter Optimization Step for CKD Risk Prediction

This module contains the ZenML step for optimizing hyperparameters of deep learning
survival analysis models using Optuna.
"""

import os
import numpy as np
import pandas as pd
import torch
import mlflow
import optuna
import json
import pickle
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import neural network architectures
from src.nn_architectures import create_network

# Import PyCox models
from pycox.models import CoxPH, DeepHit
from pycox.evaluation import EvalSurv


@step
def hyperparameter_optimization(
    train_ds: Any,
    patient_ids: Optional[List[str]] = None,
    entry_dates: Optional[List[str]] = None,
    model_type: str = "deepsurv",
    input_dim: int = 0,
    max_trials: int = 50,
    n_epochs: int = 100,
    patience: int = 10,
    cv_folds: int = 5,
    seed: int = 42,
    output_dir: str = "hyperopt_results",
    time_grid: Optional[List[int]] = None,
    search_space_config: Optional[Dict[str, Any]] = None,
    target_endpoint: Optional[int] = None
) -> Any:
    """
    Perform hyperparameter optimization using Optuna with time-based cross-validation.
    
    Args:
        train_ds: Training dataset (PyCox dataset)
        patient_ids: Array of patient IDs corresponding to each row in train_ds (default: None)
        entry_dates: Array of patient entry dates (datetime objects) (default: None)
        model_type: Type of model ("deepsurv" or "deephit") (default: "deepsurv")
        input_dim: Number of input features (default: 0)
        max_trials: Maximum number of trials for optimization (default: 50)
        n_epochs: Maximum number of epochs per trial (default: 100)
        patience: Early stopping patience (default: 10)
        cv_folds: Number of cross-validation folds (default: 5)
        seed: Random seed (default: 42)
        output_dir: Directory to save optimization results (default: "hyperopt_results")
        time_grid: Time grid for DeepHit model (default: None)
        search_space_config: Dictionary containing search space configuration (default: None)
        target_endpoint: Specific endpoint to target for DeepSurv models with competing risks (default: None)
        
    Returns:
        Dictionary containing best hyperparameters and optimization results
    """
    try:
        print(f"\n=== Hyperparameter Optimization for {model_type.upper()} ===\n")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Set default time grid for DeepHit if not provided
        if model_type.lower() == "deephit" and time_grid is None:
            # Handle ZenML StepArtifact for train_ds
            from src.util import extract_from_step_artifact, access_zenml_artifact_data
            import traceback
            
            # Print detailed information about train_ds
            print(f"DEBUG: train_ds type in hyperparameter_optimization: {type(train_ds)}")
            if hasattr(train_ds, 'dir'):
                print(f"DEBUG: train_ds attributes: {dir(train_ds)}")
            
            # Try to access common attributes for debugging
            try:
                print("DEBUG: Inspecting train_ds attributes")
                for attr in ['data', '_data', 'value', '_value', 'artifact_data', 'materialize', 'read']:
                    if hasattr(train_ds, attr):
                        attr_value = getattr(train_ds, attr)
                        print(f"DEBUG: train_ds.{attr} type: {type(attr_value)}")
                        if callable(attr_value):
                            print(f"DEBUG: {attr} is callable")
            except Exception as e:
                print(f"DEBUG: Error inspecting train_ds: {e}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            # First try direct access to the raw data
            print("DEBUG: Attempting direct access to raw data with access_zenml_artifact_data")
            raw_data = access_zenml_artifact_data(train_ds)
            if raw_data is not None:
                print(f"DEBUG: Raw data type: {type(raw_data)}")
                if isinstance(raw_data, tuple):
                    print(f"DEBUG: Tuple length: {len(raw_data)}")
                    for i, item in enumerate(raw_data):
                        print(f"DEBUG: Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                        # Check for NaN values in numeric arrays
                        if hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.number):
                            print(f"DEBUG: Item {i} contains NaN: {np.isnan(item).any()}")
                            print(f"DEBUG: Item {i} contains Inf: {np.isinf(item).any()}")
            
            if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 1:
                print("Successfully accessed raw tuple data from ZenML artifact")
                train_ds_data = raw_data
            else:
                print("Direct access didn't return a tuple, trying standard extraction")
                # Try to extract the data using standard methods
                print("DEBUG: Attempting extraction with extract_from_step_artifact")
                train_ds_data = extract_from_step_artifact(
                    train_ds,
                    artifact_name="train_ds"
                )
                print(f"DEBUG: After extraction, train_ds_data type: {type(train_ds_data)}")
                
            # If train_ds_data is still a StepArtifact, try one more approach
            if 'StepArtifact' in str(type(train_ds_data)):
                print("train_ds_data is still a StepArtifact after extraction attempts")
                try:
                    # Try to access the artifact directly as a tuple
                    if hasattr(train_ds_data, '__getitem__'):
                        print("Trying to access StepArtifact as a tuple-like object")
                        # Check if it has a length
                        if hasattr(train_ds_data, '__len__'):
                            length = len(train_ds_data)
                            print(f"StepArtifact has length {length}")
                            # Try to extract the first few items
                            try:
                                print("DEBUG: Attempting to extract items from StepArtifact")
                                items = [train_ds_data[i] for i in range(min(3, length))]
                                print(f"Successfully extracted {len(items)} items from StepArtifact")
                                
                                # Print detailed information about extracted items
                                for i, item in enumerate(items):
                                    print(f"DEBUG: Item {i} type: {type(item)}")
                                    print(f"DEBUG: Item {i} attributes: {dir(item)[:10]}...")
                                    if hasattr(item, 'shape'):
                                        print(f"DEBUG: Item {i} shape: {item.shape}")
                                
                                # If it looks like a tuple of arrays, convert it
                                if all(hasattr(item, 'shape') for item in items):
                                    print("Items appear to be arrays, converting to tuple")
                                    train_ds_data = tuple(train_ds_data[i] for i in range(length))
                            except Exception as e:
                                print(f"DEBUG: Error extracting items: {e}")
                                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                except Exception as e:
                    print(f"Error accessing StepArtifact as tuple-like object: {e}")
                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            # Print information about extracted data
            print(f"Extracted train_ds_data type: {type(train_ds_data)}")
            if isinstance(train_ds_data, tuple):
                print(f"train_ds_data is a tuple of length {len(train_ds_data)}")
                for i, item in enumerate(train_ds_data):
                    print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                    # Check for NaN values in numeric arrays
                    if hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.number):
                        print(f"Item {i} contains NaN: {np.isnan(item).any()}")
                        print(f"Item {i} contains Inf: {np.isinf(item).any()}")
                        if np.isnan(item).any() or np.isinf(item).any():
                            print(f"DEBUG: Found problematic values in item {i}")
                            nan_count = np.isnan(item).sum() if hasattr(np.isnan(item), 'sum') else 'unknown'
                            inf_count = np.isinf(item).sum() if hasattr(np.isinf(item), 'sum') else 'unknown'
                            print(f"DEBUG: NaN count: {nan_count}, Inf count: {inf_count}")
            
            # Get max duration from training data
            try:
                if isinstance(train_ds_data, tuple) and len(train_ds_data) > 1:
                    max_duration = train_ds_data[1].max()
                    print(f"Successfully extracted max duration: {max_duration}")
                else:
                    max_duration = 365
                    print(f"Using default max duration: {max_duration}")
            except Exception as e:
                print(f"Error extracting max duration: {e}")
                max_duration = 365
                print(f"Using default max duration after error: {max_duration}")
            # Create time grid with 10 points
            time_grid = np.linspace(0, max_duration, 10)
            print(f"Using default time grid with 10 points up to {max_duration:.1f} days")
        
        # Initialize search space configuration if not provided
        if search_space_config is None:
            search_space_config = {
                "common": {
                    "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
                    "num_layers": {"type": "int", "min": 2, "max": 4},
                    "hidden_units_layer1": {"type": "int", "min": 16, "max": 128},
                    "hidden_units_layer2": {"type": "int", "min": 16, "max": 128},
                    "hidden_units_layer3": {"type": "int", "min": 8, "max": 64},
                    "hidden_units_layer4": {"type": "int", "min": 4, "max": 32},
                    "dropout": {"type": "float", "min": 0.0, "max": 0.5},
                    "optimizer": {"type": "categorical", "values": ["Adam", "AdamW"]},
                    "batch_size": {"type": "categorical", "values": [32, 64, 128, 256]}
                },
                "deephit": {
                    "alpha": {"type": "float", "min": 0.1, "max": 1.0},
                    "sigma": {"type": "float", "min": 0.1, "max": 1.0}
                }
            }
            print("Using default search space configuration")
        else:
            print("Using provided search space configuration")
        
        # Import the cross-validation utilities
        try:
            from steps.cv_utils import time_based_patient_cv, create_cv_datasets, create_random_cv_splits
        except ImportError:
            print("Warning: cv_utils module not found. Using simple random split for cross-validation.")
            
            # Define a simple function to create random splits
            def create_random_cv_splits(dataset, n_splits=5, seed=42):
                # Handle ZenML StepArtifact
                from src.util import extract_from_step_artifact, access_zenml_artifact_data
                
                # Print detailed information about dataset
                print(f"dataset type in create_random_cv_splits: {type(dataset)}")
                if hasattr(dataset, 'dir'):
                    print(f"dataset attributes: {dir(dataset)[:10]}...")
                
                # First try direct access to the raw data
                raw_data = access_zenml_artifact_data(dataset)
                if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 0:
                    print("Successfully accessed raw tuple data from ZenML artifact")
                    dataset_data = raw_data
                else:
                    print("Direct access didn't return a tuple, trying standard extraction")
                    # Try to extract the data using standard methods
                    dataset_data = extract_from_step_artifact(
                        dataset,
                        artifact_name="dataset"
                    )
                
                # If dataset_data is still a StepArtifact, try one more approach
                if 'StepArtifact' in str(type(dataset_data)):
                    print("dataset_data is still a StepArtifact after extraction attempts")
                    try:
                        # Try to access the artifact directly as a tuple
                        if hasattr(dataset_data, '__getitem__'):
                            print("Trying to access StepArtifact as a tuple-like object")
                            # Check if it has a length
                            if hasattr(dataset_data, '__len__'):
                                length = len(dataset_data)
                                print(f"StepArtifact has length {length}")
                                # Try to extract the first few items
                                items = [dataset_data[i] for i in range(min(3, length))]
                                print(f"Successfully extracted {len(items)} items from StepArtifact")
                                # If it looks like a tuple of arrays, convert it
                                if all(hasattr(item, 'shape') for item in items):
                                    print("Items appear to be arrays, converting to tuple")
                                    dataset_data = tuple(dataset_data[i] for i in range(length))
                    except Exception as e:
                        print(f"Error accessing StepArtifact as tuple-like object: {e}")
                
                # Print information about extracted data
                print(f"Extracted dataset_data type: {type(dataset_data)}")
                if isinstance(dataset_data, tuple):
                    print(f"dataset_data is a tuple of length {len(dataset_data)}")
                    for i, item in enumerate(dataset_data):
                        print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                
                try:
                    if isinstance(dataset_data, tuple) and len(dataset_data) > 0:
                        n_samples = len(dataset_data[0])
                        print(f"Successfully determined number of samples: {n_samples}")
                    else:
                        n_samples = 100
                        print(f"Using default number of samples: {n_samples}")
                except (TypeError, IndexError) as e:
                    print(f"Warning: Could not determine number of samples from dataset: {e}")
                    n_samples = 100
                    print(f"Using default number of samples after error: {n_samples}")
                    
                np.random.seed(seed)
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                # Create splits
                fold_size = n_samples // n_splits
                cv_splits = []
                
                for i in range(n_splits - 1):
                    val_indices = indices[i*fold_size:(i+1)*fold_size]
                    train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
                    cv_splits.append((train_indices, val_indices))
                    
                return cv_splits
            
            # Define a simple function to create datasets from indices
            def create_cv_datasets(dataset, train_indices, val_indices):
                # Handle ZenML StepArtifact
                from src.util import extract_from_step_artifact, access_zenml_artifact_data
                
                # Print detailed information about dataset
                print(f"dataset type in create_cv_datasets: {type(dataset)}")
                if hasattr(dataset, 'dir'):
                    print(f"dataset attributes: {dir(dataset)[:10]}...")
                
                # First try direct access to the raw data
                raw_data = access_zenml_artifact_data(dataset)
                if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) >= 3:
                    print("Successfully accessed raw tuple data from ZenML artifact")
                    dataset_data = raw_data
                else:
                    print("Direct access didn't return a valid tuple, trying standard extraction")
                    # Try to extract the data using standard methods
                    dataset_data = extract_from_step_artifact(
                        dataset,
                        artifact_name="dataset"
                    )
                
                # If dataset_data is still a StepArtifact, try one more approach
                if 'StepArtifact' in str(type(dataset_data)):
                    print("dataset_data is still a StepArtifact after extraction attempts")
                    try:
                        # Try to access the artifact directly as a tuple
                        if hasattr(dataset_data, '__getitem__'):
                            print("Trying to access StepArtifact as a tuple-like object")
                            # Check if it has a length
                            if hasattr(dataset_data, '__len__'):
                                length = len(dataset_data)
                                print(f"StepArtifact has length {length}")
                                # Try to extract the first few items
                                items = [dataset_data[i] for i in range(min(3, length))]
                                print(f"Successfully extracted {len(items)} items from StepArtifact")
                                # If it looks like a tuple of arrays, convert it
                                if all(hasattr(item, 'shape') for item in items):
                                    print("Items appear to be arrays, converting to tuple")
                                    dataset_data = tuple(dataset_data[i] for i in range(length))
                    except Exception as e:
                        print(f"Error accessing StepArtifact as tuple-like object: {e}")
                
                # Print information about extracted data
                print(f"Extracted dataset_data type: {type(dataset_data)}")
                if isinstance(dataset_data, tuple):
                    print(f"dataset_data is a tuple of length {len(dataset_data)}")
                    for i, item in enumerate(dataset_data):
                        print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                
                # Ensure dataset is a tuple of (X, durations, events)
                if not isinstance(dataset_data, tuple) or len(dataset_data) < 3:
                    print(f"Warning: Dataset is not in expected format. Type: {type(dataset_data)}")
                    # Create dummy datasets
                    return (np.array([]), np.array([]), np.array([])), (np.array([]), np.array([]), np.array([]))
                
                X, durations, events = dataset_data
                
                # Create training dataset
                X_train = X[train_indices]
                durations_train = durations[train_indices]
                events_train = events[train_indices]
                train_dataset = (X_train, durations_train, events_train)
                
                # Create validation dataset
                X_val = X[val_indices]
                durations_val = durations[val_indices]
                events_val = events[val_indices]
                val_dataset = (X_val, durations_val, events_val)
                
                return train_dataset, val_dataset
        
        # Define the objective function for Optuna
        def objective(trial):
            # Store target_endpoint for use in the objective function
            nonlocal target_endpoint
            try:
                # Set random seeds for reproducibility
                torch.manual_seed(seed + trial.number)
                np.random.seed(seed + trial.number)
                
                # Initialize best_val_loss with a default value
                best_val_loss = float('inf')
                
                # Get common search space
                common_space = search_space_config.get("common", {})
                
                # Sample hyperparameters based on the search space configuration
                # Learning rate
                lr_config = common_space.get("learning_rate", {"type": "float", "min": 1e-4, "max": 1e-2, "log": True})
                lr = trial.suggest_float(
                    "learning_rate",
                    lr_config.get("min", 1e-4),
                    lr_config.get("max", 1e-2),
                    log=lr_config.get("log", True)
                )
                
                # Number of layers
                layers_config = common_space.get("num_layers", {"type": "int", "min": 2, "max": 4})
                num_layers = trial.suggest_int(
                    "num_layers",
                    layers_config.get("min", 2),
                    layers_config.get("max", 4)
                )
                
                # Hidden units for each layer
                units_config_layer1 = common_space.get("hidden_units_layer1", {"type": "int", "min": 16, "max": 128})
                hidden_units_layer1 = trial.suggest_int(
                    "hidden_units_layer1",
                    units_config_layer1.get("min", 16),
                    units_config_layer1.get("max", 128)
                )
                
                units_config_layer2 = common_space.get("hidden_units_layer2", {"type": "int", "min": 16, "max": 128})
                hidden_units_layer2 = trial.suggest_int(
                    "hidden_units_layer2",
                    units_config_layer2.get("min", 16),
                    units_config_layer2.get("max", 128)
                )
                
                units_config_layer3 = common_space.get("hidden_units_layer3", {"type": "int", "min": 8, "max": 64})
                hidden_units_layer3 = trial.suggest_int(
                    "hidden_units_layer3",
                    units_config_layer3.get("min", 8),
                    units_config_layer3.get("max", 64)
                )
                
                units_config_layer4 = common_space.get("hidden_units_layer4", {"type": "int", "min": 4, "max": 32})
                hidden_units_layer4 = trial.suggest_int(
                    "hidden_units_layer4",
                    units_config_layer4.get("min", 4),
                    units_config_layer4.get("max", 32)
                )
                
                # Dropout
                dropout_config = common_space.get("dropout", {"type": "float", "min": 0.0, "max": 0.5})
                dropout = trial.suggest_float(
                    "dropout",
                    dropout_config.get("min", 0.0),
                    dropout_config.get("max", 0.5)
                )
                
                # Optimizer
                optimizer_config = common_space.get("optimizer", {"type": "categorical", "values": ["Adam", "AdamW"]})
                optimizer_name = trial.suggest_categorical(
                    "optimizer",
                    optimizer_config.get("values", ["Adam", "AdamW"])
                )
                
                # Batch size
                batch_config = common_space.get("batch_size", {"type": "categorical", "values": [32, 64, 128, 256]})
                batch_size = trial.suggest_categorical(
                    "batch_size",
                    batch_config.get("values", [32, 64, 128, 256])
                )
                
                # Model-specific hyperparameters
                if model_type.lower() == "deephit":
                    # Get DeepHit-specific search space
                    deephit_space = search_space_config.get("deephit", {})
                    
                    # Alpha
                    alpha_config = deephit_space.get("alpha", {"type": "float", "min": 0.1, "max": 1.0})
                    alpha = trial.suggest_float(
                        "alpha",
                        alpha_config.get("min", 0.1),
                        alpha_config.get("max", 1.0)
                    )
                    
                    # Sigma
                    sigma_config = deephit_space.get("sigma", {"type": "float", "min": 0.1, "max": 1.0})
                    sigma = trial.suggest_float(
                        "sigma",
                        sigma_config.get("min", 0.1),
                        sigma_config.get("max", 1.0)
                    )
                    
                    output_dim = len(time_grid)
                else:
                    alpha = None
                    sigma = None
                    output_dim = 1
                
                # Build network architecture with different units per layer
                hidden_dims = []
                if num_layers >= 1:
                    hidden_dims.append(hidden_units_layer1)
                if num_layers >= 2:
                    hidden_dims.append(hidden_units_layer2)
                if num_layers >= 3:
                    hidden_dims.append(hidden_units_layer3)
                if num_layers >= 4:
                    hidden_dims.append(hidden_units_layer4)
                # Truncate to the actual number of layers if needed
                hidden_dims = hidden_dims[:num_layers]
                
                net = create_network(
                    model_type=model_type,
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout
                )
                
                # Create optimizer
                if optimizer_name == "Adam":
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
                
                # Create model and move to device
                if model_type.lower() == "deepsurv":
                    # Print target endpoint information
                    if target_endpoint is not None:
                        print(f"Training DeepSurv model for specific target endpoint: {target_endpoint}")
                    else:
                        print("Training DeepSurv model for all event types (any non-zero event)")
                    
                    model = CoxPH(net, optimizer=optimizer)
                else:
                    model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
                
                # Move model to device
                model.net = model.net.to(device)
                
                # Create callbacks for early stopping
                class EarlyStopping:
                    def __init__(self, patience=10):
                        self.patience = patience
                        self.best_val_loss = float('inf')
                        self.counter = 0
                        self.best_epoch = 0
                        
                    def __call__(self, val_loss, epoch):
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.counter = 0
                            self.best_epoch = epoch
                            return False
                        else:
                            self.counter += 1
                            if self.counter >= self.patience:
                                return True
                            return False
                
                early_stopping = EarlyStopping(patience=patience)
                
                # Perform cross-validation
                if patient_ids is not None and entry_dates is not None:
                    # Use time-based cross-validation if patient_ids and entry_dates are provided
                    try:
                        # Print detailed information about train_ds
                        print(f"train_ds type: {type(train_ds)}")
                        if hasattr(train_ds, 'dir'):
                            print(f"train_ds attributes: {dir(train_ds)[:10]}...")
                        
                        # Extract dataset from StepArtifact with more debugging
                        from src.util import extract_from_step_artifact, access_zenml_artifact_data
                        
                        # First try direct access to the raw data
                        raw_data = access_zenml_artifact_data(train_ds)
                        if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 0:
                            print("Successfully accessed raw tuple data from ZenML artifact")
                            train_ds_data = raw_data
                        else:
                            print("Direct access didn't return a valid tuple, trying standard extraction")
                            # Try to extract the data using standard methods
                            train_ds_data = extract_from_step_artifact(
                                train_ds,
                                artifact_name="train_ds"
                            )
                        
                        # If train_ds_data is still a StepArtifact, try one more approach
                        if 'StepArtifact' in str(type(train_ds_data)):
                            print("train_ds_data is still a StepArtifact after extraction attempts")
                            try:
                                # Try to access the artifact directly as a tuple
                                if hasattr(train_ds_data, '__getitem__'):
                                    print("Trying to access StepArtifact as a tuple-like object")
                                    # Check if it has a length
                                    if hasattr(train_ds_data, '__len__'):
                                        length = len(train_ds_data)
                                        print(f"StepArtifact has length {length}")
                                        # Try to extract the first few items
                                        items = [train_ds_data[i] for i in range(min(3, length))]
                                        print(f"Successfully extracted {len(items)} items from StepArtifact")
                                        # If it looks like a tuple of arrays, convert it
                                        if all(hasattr(item, 'shape') for item in items):
                                            print("Items appear to be arrays, converting to tuple")
                                            train_ds_data = tuple(train_ds_data[i] for i in range(length))
                            except Exception as e:
                                print(f"Error accessing StepArtifact as tuple-like object: {e}")
                        
                        # Print information about extracted data
                        print(f"Extracted train_ds_data type: {type(train_ds_data)}")
                        if isinstance(train_ds_data, tuple):
                            print(f"train_ds_data is a tuple of length {len(train_ds_data)}")
                            for i, item in enumerate(train_ds_data):
                                print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                        else:
                            print(f"Warning: train_ds_data is not a tuple, type: {type(train_ds_data)}")
                        
                        # Create cross-validation splits
                        cv_splits = time_based_patient_cv(
                            dataset=train_ds_data,
                            patient_ids=patient_ids,
                            entry_dates=entry_dates,
                            cv_folds=cv_folds,
                            seed=seed + trial.number
                        )
                        print(f"Using time-based cross-validation with {len(cv_splits)} folds")
                    except Exception as e:
                        print(f"Error creating time-based CV splits: {e}")
                        print("Falling back to random cross-validation")
                        cv_splits = create_random_cv_splits(train_ds, n_splits=cv_folds, seed=seed + trial.number)
                else:
                    # Use random cross-validation if patient_ids or entry_dates are not provided
                    print("Using random cross-validation")
                    cv_splits = create_random_cv_splits(train_ds, n_splits=cv_folds, seed=seed + trial.number)
                    
                # Initialize lists to store validation metrics from each fold
                fold_val_losses = []
                fold_c_indices = []
                
                # Train and evaluate on each fold
                for fold, (train_indices, val_indices) in enumerate(cv_splits):
                    # Create training and validation datasets for this fold
                    # Extract dataset from StepArtifact if not already done
                    if 'train_ds_data' not in locals():
                        from src.util import extract_from_step_artifact, access_zenml_artifact_data
                        
                        # First try direct access to the raw data
                        raw_data = access_zenml_artifact_data(train_ds)
                        if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 0:
                            print("Successfully accessed raw tuple data from ZenML artifact")
                            train_ds_data = raw_data
                        else:
                            print("Direct access didn't return a valid tuple, trying standard extraction")
                            # Try to extract the data using standard methods
                            train_ds_data = extract_from_step_artifact(
                                train_ds,
                                artifact_name="train_ds"
                            )
                        
                        # If train_ds_data is still a StepArtifact, try one more approach
                        if 'StepArtifact' in str(type(train_ds_data)):
                            print("train_ds_data is still a StepArtifact after extraction attempts")
                            try:
                                # Try to access the artifact directly as a tuple
                                if hasattr(train_ds_data, '__getitem__'):
                                    print("Trying to access StepArtifact as a tuple-like object")
                                    # Check if it has a length
                                    if hasattr(train_ds_data, '__len__'):
                                        length = len(train_ds_data)
                                        print(f"StepArtifact has length {length}")
                                        # Try to extract the first few items
                                        items = [train_ds_data[i] for i in range(min(3, length))]
                                        print(f"Successfully extracted {len(items)} items from StepArtifact")
                                        # If it looks like a tuple of arrays, convert it
                                        if all(hasattr(item, 'shape') for item in items):
                                            print("Items appear to be arrays, converting to tuple")
                                            train_ds_data = tuple(train_ds_data[i] for i in range(length))
                            except Exception as e:
                                print(f"Error accessing StepArtifact as tuple-like object: {e}")
                        
                        # Print information about extracted data
                        print(f"Extracted train_ds_data type: {type(train_ds_data)}")
                        if isinstance(train_ds_data, tuple):
                            print(f"train_ds_data is a tuple of length {len(train_ds_data)}")
                            for i, item in enumerate(train_ds_data):
                                print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                    
                    fold_train_ds, fold_val_ds = create_cv_datasets(train_ds_data, train_indices, val_indices)
                    
                    # Reset model for each fold
                    # Build network architecture with different units per layer
                    hidden_dims = []
                    if num_layers >= 1:
                        hidden_dims.append(hidden_units_layer1)
                    if num_layers >= 2:
                        hidden_dims.append(hidden_units_layer2)
                    if num_layers >= 3:
                        hidden_dims.append(hidden_units_layer3)
                    if num_layers >= 4:
                        hidden_dims.append(hidden_units_layer4)
                    # Truncate to the actual number of layers if needed
                    hidden_dims = hidden_dims[:num_layers]
                    
                    net = create_network(
                        model_type=model_type,
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        dropout=dropout
                    )
                    
                    # Create optimizer
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
                    
                    # Create model and move to device
                    if model_type.lower() == "deepsurv":
                        # Print target endpoint information for fold model
                        if target_endpoint is not None:
                            print(f"Training fold DeepSurv model for specific target endpoint: {target_endpoint}")
                        
                        fold_model = CoxPH(net, optimizer=optimizer)
                    else:
                        fold_model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
                    
                    # Move model to device
                    fold_model.net = fold_model.net.to(device)
                    
                    # Reset early stopping
                    fold_early_stopping = EarlyStopping(patience=patience)
                    
                    # Train model on this fold
                    fold_train_losses = []
                    fold_val_losses_epoch = []
                    
                    for epoch in range(n_epochs):
                        # Train for one epoch
                        # Unpack the dataset tuple (X, durations, events)
                        x_train, durations_train, events_train = fold_train_ds
                        
                        # Convert input data to torch tensors with float32 dtype and move to device
                        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
                        durations_train_tensor = torch.tensor(durations_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if model_type.lower() == "deepsurv" and target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_train == target_endpoint).astype(float)
                            events_train_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_train_tensor = torch.tensor(events_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv (CoxPH), we need to pass the input and target separately
                        if model_type.lower() == "deepsurv":
                            fold_model.fit(x_train_tensor, (durations_train_tensor, events_train_tensor), batch_size=batch_size, epochs=1, verbose=False)
                        else:
                            # For DeepHit, we can pass the dataset directly
                            fold_model.fit(x_train_tensor, (durations_train_tensor, events_train_tensor), batch_size=batch_size, epochs=1, verbose=False)
                        
                        # Get training loss
                        train_loss = fold_model.log.to_pandas().iloc[-1]['train_loss']
                        fold_train_losses.append(train_loss)
                        
                        # Compute validation loss
                        x_val, durations_val, events_val = fold_val_ds
                        
                        # Convert validation data to torch tensors with float32 dtype and move to device
                        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
                        durations_val_tensor = torch.tensor(durations_val, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if model_type.lower() == "deepsurv" and target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_val == target_endpoint).astype(float)
                            events_val_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_val_tensor = torch.tensor(events_val, dtype=torch.float32).to(device)
                        
                        # Handle validation loss calculation differently for CoxPH and DeepHit
                        if model_type.lower() == "deepsurv":
                            # For CoxPH models, we need to use the loss function directly
                            # Get model predictions
                            with torch.no_grad():
                                phi = fold_model.net(x_val_tensor)
                                # Calculate loss using the loss function
                                val_loss = fold_model.loss(phi, durations_val_tensor, events_val_tensor)
                        else:
                            # For DeepHit, we can use compute_loss method
                            val_loss = fold_model.compute_loss(x_val_tensor, (durations_val_tensor, events_val_tensor))
                        fold_val_losses_epoch.append(val_loss)
                        
                        # Check for early stopping
                        if fold_early_stopping(val_loss, epoch):
                            break
                    
                    # Get best validation metric for this fold
                    fold_best_epoch = fold_early_stopping.best_epoch
                    fold_best_val_loss = fold_early_stopping.best_val_loss
                    fold_val_losses.append(fold_best_val_loss)
                    
                    # Calculate concordance index for this fold if using DeepSurv
                    if model_type.lower() == "deepsurv":
                        # Get survival function predictions
                        x_val, durations_val, events_val = fold_val_ds
                        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
                        
                        # For CoxPH models, we need to compute baseline hazards before predicting
                        # Use the training data to compute baseline hazards
                        x_train, durations_train, events_train = fold_train_ds
                        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
                        durations_train_tensor = torch.tensor(durations_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_train == target_endpoint).astype(float)
                            events_train_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_train_tensor = torch.tensor(events_train, dtype=torch.float32).to(device)
                        
                        # Set the training data for the model
                        fold_model.training_data = (x_train_tensor, (durations_train_tensor, events_train_tensor))
                        
                        # Compute baseline hazards
                        _ = fold_model.compute_baseline_hazards()
                        
                        # Now we can predict survival probabilities
                        surv = fold_model.predict_surv_df(x_val_tensor)
                        
                        # Create EvalSurv object
                        ev = EvalSurv(
                            surv,
                            durations_val,
                            events_val,
                            censor_surv='km'
                        )
                        
                        # Calculate concordance index
                        c_index = ev.concordance_td()
                        fold_c_indices.append(c_index)
                
                # Average validation metrics across folds
                avg_val_loss = np.mean(fold_val_losses)
                
                # Store metrics
                best_val_loss = avg_val_loss
                best_epoch = 0  # Not applicable for cross-validation
                
                # Store concordance index if available
                if model_type.lower() == "deepsurv" and fold_c_indices:
                    avg_c_index = np.mean(fold_c_indices)
                    trial.set_user_attr("c_index", float(avg_c_index))
                
                # Store cross-validation metrics
                trial.set_user_attr("cv_val_losses", fold_val_losses)
                if fold_c_indices:
                    trial.set_user_attr("cv_c_indices", fold_c_indices)
                
                # If no valid CV splits were created, create a simple random split
                if len(cv_splits) == 0:
                    print("No valid CV splits created. Using simple random split.")
                    
                    # Extract dataset from StepArtifact if not already done
                    if 'train_ds_data' not in locals():
                        from src.util import extract_from_step_artifact, access_zenml_artifact_data
                        
                        # First try direct access to the raw data
                        raw_data = access_zenml_artifact_data(train_ds)
                        if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 0:
                            print("Successfully accessed raw tuple data from ZenML artifact")
                            train_ds_data = raw_data
                        else:
                            print("Direct access didn't return a valid tuple, trying standard extraction")
                            # Try to extract the data using standard methods
                            train_ds_data = extract_from_step_artifact(
                                train_ds,
                                artifact_name="train_ds"
                            )
                            
                            # Enhanced logging for debugging
                            print(f"Extracted train_ds_data type: {type(train_ds_data)}")
                            if isinstance(train_ds_data, tuple):
                                print(f"train_ds_data is a tuple of length {len(train_ds_data)}")
                                for i, item in enumerate(train_ds_data):
                                    print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                                    # Check for NaN values
                                    if hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.number):
                                        print(f"  Contains NaN: {np.isnan(item).any()}")
                                        print(f"  Contains Inf: {np.isinf(item).any()}")
                            else:
                                print(f"WARNING: train_ds_data is not a tuple! This will cause problems.")
                    
                    # Create a simple train/val split (80/20)
                    try:
                        if isinstance(train_ds_data, tuple) and len(train_ds_data) > 0:
                            n_samples = len(train_ds_data[0])
                            print(f"Determined n_samples={n_samples} from train_ds_data[0]")
                        else:
                            print(f"WARNING: train_ds_data is not a valid tuple, using default n_samples=100")
                            n_samples = 100
                    except (TypeError, IndexError) as e:
                        print(f"Warning: Could not determine number of samples from dataset: {e}")
                        print(f"train_ds_data type: {type(train_ds_data)}")
                        if isinstance(train_ds_data, tuple):
                            print(f"train_ds_data length: {len(train_ds_data)}")
                            for i, item in enumerate(train_ds_data):
                                print(f"Item {i} type: {type(item)}")
                        n_samples = 100
                        
                    indices = np.arange(n_samples)
                    np.random.shuffle(indices)
                    split = int(n_samples * 0.8)
                    train_indices = indices[:split]
                    val_indices = indices[split:]
                    
                    # Create the simple train and validation datasets
                    simple_train_ds, simple_val_ds = create_cv_datasets(train_ds_data, train_indices, val_indices)
                    
                    # Initialize lists to store training and validation metrics
                    train_losses = []
                    val_losses = []
                    
                    # Reset early stopping for this simple split
                    early_stopping = EarlyStopping(patience=patience)
                    
                    # Reset model for the simple split
                    # Build network architecture with different units per layer
                    hidden_dims = []
                    if num_layers >= 1:
                        hidden_dims.append(hidden_units_layer1)
                    if num_layers >= 2:
                        hidden_dims.append(hidden_units_layer2)
                    if num_layers >= 3:
                        hidden_dims.append(hidden_units_layer3)
                    if num_layers >= 4:
                        hidden_dims.append(hidden_units_layer4)
                    # Truncate to the actual number of layers if needed
                    hidden_dims = hidden_dims[:num_layers]
                    
                    net = create_network(
                        model_type=model_type,
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        dropout=dropout
                    )
                    
                    # Create optimizer
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
                    
                    # Create model and move to device
                    if model_type.lower() == "deepsurv":
                        # Print target endpoint information for simple split model
                        if target_endpoint is not None:
                            print(f"Training simple split DeepSurv model for specific target endpoint: {target_endpoint}")
                        
                        model = CoxPH(net, optimizer=optimizer)
                    else:
                        model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
                    
                    # Move model to device
                    model.net = model.net.to(device)
                    
                    cv_splits = [(train_indices, val_indices)]
                    
                    for epoch in range(n_epochs):
                        # Train for one epoch
                        # Unpack the dataset tuple (X, durations, events)
                        x_train, durations_train, events_train = simple_train_ds
                        
                        # Convert input data to torch tensors with float32 dtype and move to device
                        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
                        durations_train_tensor = torch.tensor(durations_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if model_type.lower() == "deepsurv" and target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_train == target_endpoint).astype(float)
                            events_train_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_train_tensor = torch.tensor(events_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv (CoxPH), we need to pass the input and target separately
                        if model_type.lower() == "deepsurv":
                            model.fit(x_train_tensor, (durations_train_tensor, events_train_tensor), batch_size=batch_size, epochs=1, verbose=False)
                        else:
                            # For DeepHit, we can pass the dataset directly
                            model.fit(x_train_tensor, (durations_train_tensor, events_train_tensor), batch_size=batch_size, epochs=1, verbose=False)
                        
                        # Get training loss
                        train_loss = model.log.to_pandas().iloc[-1]['train_loss']
                        train_losses.append(train_loss)
                        
                        # Compute validation loss
                        x_val, durations_val, events_val = simple_val_ds
                        
                        # Convert validation data to torch tensors with float32 dtype and move to device
                        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
                        durations_val_tensor = torch.tensor(durations_val, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if model_type.lower() == "deepsurv" and target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_val == target_endpoint).astype(float)
                            events_val_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_val_tensor = torch.tensor(events_val, dtype=torch.float32).to(device)
                        
                        # Handle validation loss calculation differently for CoxPH and DeepHit
                        if model_type.lower() == "deepsurv":
                            # For CoxPH models, we need to use the loss function directly
                            # Get model predictions
                            with torch.no_grad():
                                phi = model.net(x_val_tensor)
                                # Calculate loss using the loss function
                                val_loss = model.loss(phi, durations_val_tensor, events_val_tensor)
                        else:
                            # For DeepHit, we can use compute_loss method
                            val_loss = model.compute_loss(x_val_tensor, (durations_val_tensor, events_val_tensor))
                        val_losses.append(val_loss)
                        
                        # Report intermediate values for pruning
                        trial.report(val_loss, epoch)
                        
                        # Check for early stopping
                        if early_stopping(val_loss, epoch):
                            break
                        
                        # Check if trial should be pruned
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    
                    # Get best validation metric
                    best_epoch = early_stopping.best_epoch
                    best_val_loss = early_stopping.best_val_loss
                    
                    # Calculate concordance index for DeepSurv models
                    if model_type.lower() == "deepsurv":
                        # For CoxPH models, we need to compute baseline hazards before predicting
                        x_train, durations_train, events_train = simple_train_ds
                        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
                        durations_train_tensor = torch.tensor(durations_train, dtype=torch.float32).to(device)
                        
                        # For DeepSurv with target_endpoint, convert multi-class events to binary
                        if target_endpoint is not None:
                            # Convert events to binary based on target_endpoint
                            binary_events = (events_train == target_endpoint).astype(float)
                            events_train_tensor = torch.tensor(binary_events, dtype=torch.float32).to(device)
                        else:
                            events_train_tensor = torch.tensor(events_train, dtype=torch.float32).to(device)
                        
                        # Set the training data for the model
                        model.training_data = (x_train_tensor, (durations_train_tensor, events_train_tensor))
                        
                        # Compute baseline hazards
                        _ = model.compute_baseline_hazards()
                        
                        # Get validation data
                        x_val, durations_val, events_val = simple_val_ds
                        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
                        
                        # Now we can predict survival probabilities
                        surv = model.predict_surv_df(x_val_tensor)
                        
                        # Create EvalSurv object
                        ev = EvalSurv(
                            surv,
                            durations_val,
                            events_val,
                            censor_surv='km'
                        )
                        
                        # Calculate concordance index
                        c_index = ev.concordance_td()
                        trial.set_user_attr("c_index", float(c_index))
                    
                    # Store training history
                    trial.set_user_attr("train_losses", train_losses)
                    trial.set_user_attr("val_losses", val_losses)
                    trial.set_user_attr("best_epoch", best_epoch)
                
                # Return the best validation loss
                return best_val_loss
            except Exception as e:
                print(f"Error in objective function: {e}")
                # Log the error to the trial
                trial.set_user_attr("error", str(e))
                # Return a high loss value to indicate failure
                return float('inf')
        
        # Create Optuna study
        endpoint_suffix = f"_endpoint{target_endpoint}" if target_endpoint is not None else ""
        study_name = f"{model_type}{endpoint_suffix}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        
        # Run optimization
        print(f"Running hyperparameter optimization with {max_trials} trials...")
        study.optimize(objective, n_trials=max_trials)
        
        # Get best trial
        best_trial = study.best_trial
        
        # Print best hyperparameters
        # Print model and endpoint information
        endpoint_info = f" for endpoint {target_endpoint}" if target_endpoint is not None else ""
        print(f"\n=== Best Hyperparameters for {model_type.upper()}{endpoint_info} ===")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
        
        # Print best metrics
        print(f"\nBest validation loss: {best_trial.value:.6f}")
        if "c_index" in best_trial.user_attrs:
            print(f"Best concordance index: {best_trial.user_attrs['c_index']:.6f}")
        print(f"Best epoch: {best_trial.user_attrs['best_epoch']}")
        
        # Save study
        endpoint_suffix = f"_endpoint{target_endpoint}" if target_endpoint is not None else ""
        study_path = os.path.join(output_dir, f"{model_type}{endpoint_suffix}_study.pkl")
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        print(f"Saved study to {study_path}")
        
        # Save best hyperparameters as JSON
        best_params = best_trial.params
        best_params["model_type"] = model_type
        best_params["input_dim"] = input_dim
        
        # Include target_endpoint in the parameters if specified
        if target_endpoint is not None:
            best_params["target_endpoint"] = target_endpoint
        
        # Ensure best_epoch is included in the parameters
        if "best_epoch" in best_trial.user_attrs:
            best_params["best_epoch"] = best_trial.user_attrs["best_epoch"]
            print(f"Using best_epoch={best_params['best_epoch']} from trial user attributes")
        else:
            # Default to 0 if best_epoch is not available
            print("Warning: best_epoch not found in trial user attributes, using default value 0")
            best_params["best_epoch"] = 0
        
        # Log all user attributes for debugging
        print("All trial user attributes:")
        for key, value in best_trial.user_attrs.items():
            print(f"  {key}: {value}")
            
        if model_type.lower() == "deephit" and time_grid is not None:
            best_params["time_grid"] = time_grid.tolist()
        
        endpoint_suffix = f"_endpoint{target_endpoint}" if target_endpoint is not None else ""
        params_path = os.path.join(output_dir, f"{model_type}{endpoint_suffix}_best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Saved best hyperparameters to {params_path}")
        
        # Return best hyperparameters
        return best_params
    
    except Exception as e:
        print(f"Error in hyperparameter optimization: {e}")
        # Return default hyperparameters if optimization fails
        print("Returning default hyperparameters due to optimization failure")
        default_params = {
            "model_type": model_type,
            "input_dim": input_dim,
            "learning_rate": 0.001,
            "num_layers": 3,
            "hidden_units_layer1": 128,
            "hidden_units_layer2": 64,
            "hidden_units_layer3": 32,
            "hidden_units_layer4": 16,
            "dropout": 0.2,
            "optimizer": "Adam",
            "batch_size": 64,
            "best_epoch": 0,
            "error": str(e)
        }
        
        # Include target_endpoint in default parameters if specified
        if target_endpoint is not None:
            default_params["target_endpoint"] = target_endpoint
            
        return default_params
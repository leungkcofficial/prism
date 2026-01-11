"""
Model Training Step for CKD Risk Prediction

This module contains the ZenML step for training deep learning survival analysis models
(DeepSurv and DeepHit) with the optimal hyperparameters.
"""

import os
import numpy as np
import pandas as pd
import torch
import mlflow
from zenml.steps import step
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import neural network architectures
from src.nn_architectures import create_network

# Import PyCox models
from pycox.models import CoxPH, DeepHit


@step
def model_train_surv(
    train_ds: Any,
    val_ds: Any,
    model_type: str,
    hyperparams: Dict[str, Any],
    input_dim: int,
    n_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    output_dir: str = "model_output",
    time_grid: Optional[List[int]] = None
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Train a survival model with the given hyperparameters.
    
    Args:
        train_ds: Training dataset (PyCox dataset)
        val_ds: Validation dataset (PyCox dataset)
        model_type: Type of model ("deepsurv" or "deephit")
        hyperparams: Hyperparameters from optimization
        input_dim: Number of input features
        n_epochs: Maximum number of epochs (default: 100)
        patience: Early stopping patience (default: 10)
        seed: Random seed (default: 42)
        output_dir: Directory to save model outputs (default: "model_output")
        time_grid: Time grid for DeepHit model (default: None)
        
    Returns:
        Tuple containing:
        - trained_model: Trained PyTorch model
        - best_params: Best hyperparameters
        - training_log: Training metrics log
    """
    try:
        print(f"\n=== Training {model_type.upper()} Model ===\n")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Extract hyperparameters
        lr = hyperparams.get("learning_rate", 0.001)
        num_layers = hyperparams.get("num_layers", 3)
        hidden_units = hyperparams.get("hidden_units", 128)
        dropout = hyperparams.get("dropout", 0.2)
        optimizer_name = hyperparams.get("optimizer", "Adam")
        batch_size = hyperparams.get("batch_size", 64)
        
        # Model-specific hyperparameters
        if model_type.lower() == "deephit":
            alpha = hyperparams.get("alpha", 0.2)
            sigma = hyperparams.get("sigma", 0.1)
            
            # Set default time grid for DeepHit if not provided
            if time_grid is None:
                # Get max duration from training data
                max_duration = train_ds[1].max()
                # Create time grid with 10 points
                time_grid = np.linspace(0, max_duration, 10)
                print(f"Using default time grid with 10 points up to {max_duration:.1f} days")
            
            output_dim = len(time_grid)
        else:
            alpha = None
            sigma = None
            output_dim = 1
        
        # Print training configuration
        print(f"Model type: {model_type}")
        print(f"Input dimension: {input_dim}")
        print(f"Output dimension: {output_dim}")
        print(f"Number of layers: {num_layers}")
        print(f"Hidden units: {hidden_units}")
        print(f"Dropout rate: {dropout}")
        print(f"Learning rate: {lr}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Batch size: {batch_size}")
        print(f"Maximum epochs: {n_epochs}")
        print(f"Early stopping patience: {patience}")
        if model_type.lower() == "deephit":
            print(f"Alpha: {alpha}")
            print(f"Sigma: {sigma}")
            print(f"Time grid: {time_grid}")
        
        # Build network architecture
        hidden_dims = [hidden_units] * num_layers
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
                self.best_state_dict = None
                
            def __call__(self, val_loss, model, epoch):
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.counter = 0
                    self.best_epoch = epoch
                    self.best_state_dict = {k: v.cpu().clone() for k, v in model.net.state_dict().items()}
                    return False
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        return True
                    return False
        
        early_stopping = EarlyStopping(patience=patience)
        
        # Train model
        print("\nStarting training...")
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Train for one epoch
            # Unpack the dataset tuple (X, durations, events)
            x, durations, events = train_ds
            
            # Convert input data to torch tensors with float32 dtype and move to device
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            durations_tensor = torch.tensor(durations, dtype=torch.float32).to(device)
            events_tensor = torch.tensor(events, dtype=torch.float32).to(device)
            
            # For DeepSurv (CoxPH), we need to pass the input and target separately
            if model_type.lower() == "deepsurv":
                model.fit(x_tensor, (durations_tensor, events_tensor), batch_size=batch_size, epochs=1, verbose=False)
            else:
                # For DeepHit, we can pass the dataset directly
                model.fit(x_tensor, (durations_tensor, events_tensor), batch_size=batch_size, epochs=1, verbose=False)
            
            # Get training loss
            train_loss = model.log.to_pandas().iloc[-1]['train_loss']
            train_losses.append(float(train_loss))  # Convert to float to ensure it's a scalar
            
            # Compute validation loss
            x_val, durations_val, events_val = val_ds
            
            # Convert validation data to torch tensors with float32 dtype and move to device
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            durations_val_tensor = torch.tensor(durations_val, dtype=torch.float32).to(device)
            events_val_tensor = torch.tensor(events_val, dtype=torch.float32).to(device)
            
            # Different models have different ways to compute loss
            if hasattr(model, 'compute_loss'):
                # Use compute_loss if available (DeepHit)
                val_loss = model.compute_loss(x_val_tensor, (durations_val_tensor, events_val_tensor))
            else:
                # For CoxPH, we need to use the loss function directly
                with torch.no_grad():
                    # Get predictions
                    phi = model.net(x_val_tensor)
                    # Compute loss using the loss function
                    val_loss = model.loss(phi, durations_val_tensor, events_val_tensor)
            
            val_losses.append(val_loss.item())  # Convert tensor to scalar
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Check for early stopping
            if early_stopping(val_loss, model, epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if early_stopping.best_state_dict is not None:
            net.load_state_dict(early_stopping.best_state_dict)
            if model_type.lower() == "deepsurv":
                model = CoxPH(net, optimizer=optimizer)
            else:
                model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
            
            # Move model to device
            model.net = model.net.to(device)
            print(f"Loaded best model from epoch {early_stopping.best_epoch+1}")
        
        # Create training log DataFrame
        training_log = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        
        # Save training log
        log_path = os.path.join(output_dir, f"{model_type}_training_log.csv")
        training_log.to_csv(log_path, index=False)
        print(f"Saved training log to {log_path}")
        
        # Create training curve plot
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(training_log['epoch'], training_log['train_loss'], label='Train Loss')
            plt.plot(training_log['epoch'], training_log['val_loss'], label='Validation Loss')
            plt.axvline(x=early_stopping.best_epoch+1, color='r', linestyle='--', 
                       label=f'Best Epoch ({early_stopping.best_epoch+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_type.upper()} Training Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(output_dir, f"{model_type}_training_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved training curve plot to {plot_path}")
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
        except Exception as e:
            print(f"Warning: Could not create training curve plot: {e}")
        
        # Calculate additional metrics for the best model
        if model_type.lower() == "deepsurv":
            from pycox.evaluation import EvalSurv
            
            # For CoxPH models, we need to compute baseline hazards before predicting
            # Use the training data to compute baseline hazards
            x_train = torch.tensor(train_ds[0], dtype=torch.float32).to(device)
            durations_train = torch.tensor(train_ds[1], dtype=torch.float32).to(device)
            events_train = torch.tensor(train_ds[2], dtype=torch.float32).to(device)
            
            # Set the training data for the model
            model.training_data = (x_train, (durations_train, events_train))
            
            # Compute baseline hazards
            _ = model.compute_baseline_hazards()
            
            # Get survival function predictions
            # Ensure validation data is on the correct device
            x_val = torch.tensor(val_ds[0], dtype=torch.float32).to(device)
            surv = model.predict_surv_df(x_val)
            
            # Create EvalSurv object
            ev = EvalSurv(
                surv,
                val_ds[1],
                val_ds[2],
                censor_surv='km'
            )
            
            # Calculate concordance index
            c_index = ev.concordance_td()
            print(f"Validation concordance index: {c_index:.6f}")
            
            # Log to MLflow
            mlflow.log_metric("val_c_index", float(c_index))
        
        # Log to MLflow
        mlflow.log_params(hyperparams)
        mlflow.log_metric("best_val_loss", float(early_stopping.best_val_loss))
        mlflow.log_metric("best_epoch", early_stopping.best_epoch + 1)
        mlflow.log_metric("num_epochs", len(train_losses))
        mlflow.log_artifact(log_path)
        
        # Save model hyperparameters
        final_hyperparams = {
            **hyperparams,
            "model_type": model_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "best_epoch": early_stopping.best_epoch + 1,
            "best_val_loss": float(early_stopping.best_val_loss),
            "time_grid": time_grid.tolist() if time_grid is not None else None,
        }
        
        if model_type.lower() == "deepsurv" and 'c_index' in locals():
            final_hyperparams["val_c_index"] = float(c_index)
        
        return model, final_hyperparams, training_log
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        raise
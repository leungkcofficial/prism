"""
Evaluation Module for CKD Risk Prediction Models

This module contains functions for evaluating survival models (DeepSurv and DeepHit)
using bootstrap to estimate confidence intervals for metrics and SHAP values.
It also includes functions for generating visualizations and saving results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from sklearn.metrics import roc_curve, auc
from pycox.evaluation import EvalSurv
from lifelines import KaplanMeierFitter, AalenJohansenFitter
import shap
from src.util import save_predictions_to_hdf5, load_predictions_from_hdf5


def evaluate_model(
    model: Any,
    datasets: Dict[str, Dict[str, Any]],
    n_bootstrap: int = 50,  # Reduced from 500 to 50 to save memory
    time_horizons: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    save_predictions: bool = True,
    calculate_shap: bool = False,  # Option to disable SHAP calculation
    process_sequentially: bool = True,  # Process datasets one by one
    monitor_memory: bool = False,  # Option to monitor memory usage
    batch_size: Optional[int] = None,  # Batch size for processing large datasets
    use_gpu_for_shap: bool = False  # Whether to use GPU for SHAP calculation if available
) -> Dict[str, Any]:
    """
    Evaluate a survival model using bootstrap to estimate confidence intervals.
    
    Args:
        model: The trained survival model (DeepSurv or DeepHit)
        datasets: Dictionary of datasets to evaluate (training, temporal_test, spatial_test)
                 Each dataset should contain 'x', 'durations', 'events', and 'feature_names'
        n_bootstrap: Number of bootstrap iterations (default: 50)
        time_horizons: List of time horizons for evaluation (default: [365, 730, 1095, 1460, 1825])
        output_dir: Directory to save results (default: None)
        save_predictions: Whether to save predictions to CSV files (default: True)
        calculate_shap: Whether to calculate SHAP values (default: False)
        process_sequentially: Whether to process datasets one by one (default: True)
        monitor_memory: Whether to monitor memory usage (default: False)
        batch_size: Batch size for processing large datasets (default: None)
        use_gpu_for_shap: Whether to use GPU for SHAP calculation if available (default: False)
        
    Returns:
        Dictionary of evaluation results for each dataset
    """
    # Set default time horizons if not provided
    if time_horizons is None:
        time_horizons = [365, 730, 1095, 1460, 1825]  # Default to years 1-5
    
    # Create output directory if not exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get model type
    model_type = "deepsurv" if hasattr(model, "compute_baseline_hazards") else "deephit"
    print(f"Evaluating {model_type.upper()} model")
    
    # Add memory monitoring
    if monitor_memory:
        import psutil
        # os is already imported at the top of the file
        
        def monitor_memory_usage(label: str = ""):
            """Monitor memory usage at different points in the code."""
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            
            print(f"Memory usage {label}: {memory_usage_mb:.2f} MB")
            
            return memory_usage_mb
        
        # Monitor initial memory usage
        monitor_memory_usage("at start of evaluation")
    
    # Dictionary to store results for all datasets
    all_results = {}
    
    # Device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import garbage collection
    import gc
    
    # Evaluate each dataset
    dataset_names = list(datasets.keys())
    for dataset_idx, dataset_name in enumerate(dataset_names):
        dataset = datasets[dataset_name]
        
        # Monitor memory before processing dataset
        if monitor_memory:
            monitor_memory_usage(f"before processing {dataset_name} dataset")
        print(f"\n=== Evaluating {dataset_name} dataset ===")
        
        # Extract data
        x = dataset['x']
        durations = dataset['durations']
        events = dataset['events']
        feature_names = dataset.get('feature_names', None)
        
        # Generate predictions if not already provided
        if 'predictions' not in dataset:
            print(f"Generating predictions for {dataset_name} dataset")
            
            # Convert to torch tensors if needed
            if not isinstance(x, torch.Tensor):
                # Use float32 instead of float64 to save memory
                x_np = x.astype(np.float32) if isinstance(x, np.ndarray) else x
                x = torch.tensor(x_np).float().to(device)
            
            # Process in batches if batch_size is provided and dataset is large
            if batch_size is not None and len(x) > batch_size:
                print(f"Processing {len(x)} samples in batches of {batch_size}")
                
                # Initialize an empty DataFrame for survival predictions
                all_surv_preds = None
                
                # Process in batches
                for i in range(0, len(x), batch_size):
                    # Get batch
                    end_idx = min(i + batch_size, len(x))
                    x_batch = x[i:end_idx]
                    
                    # Generate predictions for this batch
                    with torch.no_grad():
                        surv_batch = model.predict_surv_df(x_batch)
                    
                    # Concatenate with previous batches
                    if all_surv_preds is None:
                        all_surv_preds = surv_batch
                    else:
                        # Concatenate along columns (samples)
                        all_surv_preds = pd.concat([all_surv_preds, surv_batch], axis=1)
                    
                    # Clear memory
                    del x_batch
                    del surv_batch
                    gc.collect()
                    
                    print(f"Processed batch {i//batch_size + 1}/{(len(x) + batch_size - 1)//batch_size}")
                
                surv = all_surv_preds
                del all_surv_preds
                gc.collect()
            else:
                # Generate predictions for the entire dataset at once
                with torch.no_grad():
                    surv = model.predict_surv_df(x)
            
            print(f"Predictions shape: {surv.shape}")
        else:
            surv = dataset['predictions']
            print(f"Using provided predictions with shape: {surv.shape}")
        
        # Save predictions if requested
        if save_predictions and output_dir is not None:
            prediction_paths = save_predictions_to_files(
                surv=surv,
                durations=durations,
                events=events,
                dataset_name=dataset_name,
                output_dir=output_dir
            )
            print(f"Predictions saved to: {prediction_paths['predictions']}")
            print(f"Metadata saved to: {prediction_paths['metadata']}")
        
        # Calculate metrics with bootstrap
        print(f"Calculating metrics with {n_bootstrap} bootstrap iterations")
        metrics = calculate_metrics(
            surv=surv,
            durations=durations,
            events=events,
            time_horizons=time_horizons,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size  # Pass batch_size to calculate_metrics
        )
        
        # Calculate SHAP values with bootstrap (if enabled)
        if calculate_shap and feature_names is not None:
            print(f"Calculating SHAP values for {len(feature_names)} features")
            shap_values = calculate_shap_values(
                model=model,
                x=x,
                feature_names=feature_names,
                n_bootstrap=min(n_bootstrap, 10),  # Limit SHAP bootstrap to 50 max
                batch_size=batch_size,  # Pass batch_size to SHAP calculation
                use_gpu_for_shap=use_gpu_for_shap  # Pass GPU preference to SHAP calculation
            )
            
            # Save SHAP values
            if output_dir is not None:
                shap_paths = save_shap_values(
                    shap_values=shap_values,
                    dataset_name=dataset_name,
                    output_dir=output_dir
                )
                print(f"SHAP values saved to: {shap_paths['json']}")
                print(f"SHAP values CSV saved to: {shap_paths['csv']}")
        else:
            shap_values = None
            print("Feature names not provided, skipping SHAP value calculation")
        
        # Create visualizations
        if output_dir is not None:
            print("Creating visualizations")
            visualization_paths = create_visualizations(
                surv=surv,
                durations=durations,
                events=events,
                metrics=metrics,
                dataset_name=dataset_name,
                time_horizons=time_horizons,
                output_dir=output_dir,
                batch_size=batch_size  # Pass batch_size to visualizations
            )
            print(f"Visualizations saved to: {visualization_paths}")
        
        # Store results for this dataset
        all_results[dataset_name] = {
            'metrics': metrics,
            'shap_values': shap_values,
            'predictions': surv,
            'durations': durations,
            'events': events
        }
        
        # Clear memory if processing sequentially and not the last dataset
        if process_sequentially and dataset_idx < len(dataset_names) - 1:
            # Remove dataset from memory
            if dataset_name in datasets:
                del datasets[dataset_name]
            
            # Force garbage collection
            gc.collect()
            
            # Monitor memory after processing dataset
            if monitor_memory:
                monitor_memory_usage(f"after processing {dataset_name} dataset")
    
    # Create combined visualizations
    if output_dir is not None and len(all_results) > 0:
        print("\n=== Creating combined visualizations ===")
        
        # Create ROC curves
        roc_paths = create_roc_curve(
            datasets_results=all_results,
            model_type=model_type,
            time_horizons=time_horizons,
            output_dir=output_dir
        )
        print(f"ROC curves saved to: {output_dir}")
        
        # Create decision curve analysis
        dca_paths = create_decision_curve_analysis(
            datasets_results=all_results,
            model_type=model_type,
            time_horizons=time_horizons,
            output_dir=output_dir
        )
        print(f"Decision curve analysis saved to: {output_dir}")
        
        # Add paths to results
        all_results['visualization_paths'] = {
            'roc_curves': roc_paths,
            'decision_curves': dca_paths
        }
    
    # Save all results to JSON
    if output_dir is not None:
        # Create a serializable version of the results
        serializable_results = {}
        for dataset_name, results in all_results.items():
            if dataset_name == 'visualization_paths':
                serializable_results[dataset_name] = results
                continue
                
            serializable_results[dataset_name] = {
                'metrics': results['metrics'],
                'shap_values': results['shap_values']
            }
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"All results saved to: {results_path}")
    
    # Final garbage collection
    gc.collect()
    
    # Monitor final memory usage
    if monitor_memory:
        monitor_memory_usage("at end of evaluation")
    
    return all_results


def calculate_metrics(
    surv: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_horizons: List[int],
    n_bootstrap: int = 10,  # Already reduced to 10 in previous fix
    batch_size: Optional[int] = None  # Batch size for bootstrap processing
) -> Dict[str, Any]:
    """
    Calculate survival metrics with bootstrap confidence intervals.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    # Create EvalSurv object
    ev = EvalSurv(surv, durations, events, censor_surv='km')
    
    # Calculate concordance index
    c_index = ev.concordance_td()
    print(f"C-index: {c_index:.4f}")
    
    # Initialize arrays for bootstrap metrics
    bootstrap_c_index = np.zeros(n_bootstrap)
    bootstrap_integrated_brier_score = np.zeros(n_bootstrap)
    bootstrap_integrated_nbll = np.zeros(n_bootstrap)
    
    # Initialize dictionary for time-dependent metrics
    bootstrap_metrics_by_horizon = {horizon: {
        'brier_score': np.zeros(n_bootstrap)
    } for horizon in time_horizons}
    
    # Process bootstrap in batches if batch_size is provided
    if batch_size is not None and n_bootstrap > batch_size:
        print(f"Processing {n_bootstrap} bootstrap iterations in batches of {batch_size}")
        
        # Number of batches
        n_batches = (n_bootstrap + batch_size - 1) // batch_size
        
        # Process each batch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_bootstrap)
            batch_size_actual = end_idx - start_idx
            
            print(f"Processing bootstrap batch {batch_idx+1}/{n_batches} (iterations {start_idx}-{end_idx-1})")
            
            # Process bootstrap iterations in this batch
            for i in range(start_idx, end_idx):
                # Print progress for each bootstrap iteration
                print(f"Processing bootstrap iteration {i+1}/{n_bootstrap} after C-index calculation")
                
                # Sample with replacement
                indices = np.random.choice(len(durations), size=len(durations), replace=True)
                
                # Get bootstrap sample
                bootstrap_durations = durations[indices]
                bootstrap_events = events[indices]
                
                # Check if the DataFrame is transposed (more columns than rows)
                if surv.shape[1] > surv.shape[0]:
                    # If transposed, select columns instead of rows
                    bootstrap_surv = surv.iloc[:, indices]
                else:
                    # Original behavior - select rows
                    bootstrap_surv = surv.iloc[indices]
                
                # Create EvalSurv object for bootstrap sample
                bootstrap_ev = EvalSurv(bootstrap_surv, bootstrap_durations, bootstrap_events, censor_surv='km')
                
                # Calculate concordance index
                bootstrap_c_index[i] = bootstrap_ev.concordance_td()
                
                # Calculate other metrics (as in the original code)
                try:
                    # Create time grid for integrated metrics
                    time_grid = np.linspace(bootstrap_durations.min(), bootstrap_durations.max(), 10)
                    
                    # Calculate integrated Brier score
                    bootstrap_integrated_brier_score[i] = bootstrap_ev.integrated_brier_score(time_grid)
                    
                    # Calculate integrated negative log-likelihood
                    bootstrap_integrated_nbll[i] = bootstrap_ev.integrated_nbll(time_grid)
                except Exception as e:
                    print(f"Error calculating integrated metrics for bootstrap {i}: {e}")
                    bootstrap_integrated_brier_score[i] = np.nan
                    bootstrap_integrated_nbll[i] = np.nan
                
                # Calculate metrics at each time horizon
                for horizon in time_horizons:
                    try:
                        # Calculate Brier score - pass horizon as a numpy array with one element
                        brier_score = bootstrap_ev.brier_score(np.array([horizon]))
                        # Handle case where brier_score is a pandas Series
                        if hasattr(brier_score, 'iloc'):
                            brier_score = float(brier_score.iloc[0])
                        bootstrap_metrics_by_horizon[horizon]['brier_score'][i] = brier_score
                        
                        # Note: We're only calculating brier_score, not AUC
                    except Exception as e:
                        print(f"Error calculating metrics at {horizon} days for bootstrap {i}: {e}")
                        bootstrap_metrics_by_horizon[horizon]['brier_score'][i] = np.nan
                
                # Clear memory
                del bootstrap_surv
                del bootstrap_ev
            
            # Force garbage collection after each batch
            gc.collect()
            print(f"Completed bootstrap batch {batch_idx+1}/{n_batches}")
    else:
        # Original code for processing all bootstrap iterations at once
        for i in range(n_bootstrap):
            # Report progress every 10 iterations or at the beginning and end
            if i == 0 or i == n_bootstrap - 1 or (i + 1) % 10 == 0:
                print(f"Processing bootstrap iteration {i+1}/{n_bootstrap}")
            
            # Sample with replacement
            indices = np.random.choice(len(durations), size=len(durations), replace=True)
            
            # Get bootstrap sample
            bootstrap_durations = durations[indices]
            bootstrap_events = events[indices]
            
            # Check if the DataFrame is transposed (more columns than rows)
            if surv.shape[1] > surv.shape[0]:
                # If transposed, select columns instead of rows
                bootstrap_surv = surv.iloc[:, indices]
            else:
                # Original behavior - select rows
                bootstrap_surv = surv.iloc[indices]
            
            # Create EvalSurv object for bootstrap sample
            bootstrap_ev = EvalSurv(bootstrap_surv, bootstrap_durations, bootstrap_events, censor_surv='km')
            
            # Calculate concordance index
            bootstrap_c_index[i] = bootstrap_ev.concordance_td()
        
            # Calculate integrated metrics
            try:
                # Create time grid for integrated metrics
                time_grid = np.linspace(bootstrap_durations.min(), bootstrap_durations.max(), 10)
                
                # Calculate integrated Brier score
                bootstrap_integrated_brier_score[i] = bootstrap_ev.integrated_brier_score(time_grid)
                
                # Calculate integrated negative log-likelihood
                bootstrap_integrated_nbll[i] = bootstrap_ev.integrated_nbll(time_grid)
            except Exception as e:
                print(f"Error calculating integrated metrics for bootstrap {i}: {e}")
                bootstrap_integrated_brier_score[i] = np.nan
                bootstrap_integrated_nbll[i] = np.nan
            
            # Calculate metrics at each time horizon
            for horizon in time_horizons:
                try:
                    # Calculate Brier score - pass horizon as a numpy array with one element
                    brier_score = bootstrap_ev.brier_score(np.array([horizon]))
                    # Handle case where brier_score is a pandas Series
                    if hasattr(brier_score, 'iloc'):
                        brier_score = float(brier_score.iloc[0])
                    bootstrap_metrics_by_horizon[horizon]['brier_score'][i] = brier_score
                    
                    # Note: We're only calculating brier_score, not AUC
                except Exception as e:
                    print(f"Error calculating metrics at {horizon} days for bootstrap {i}: {e}")
                    bootstrap_metrics_by_horizon[horizon]['brier_score'][i] = np.nan
    
    # Calculate mean and confidence intervals for concordance index
    c_index_mean = np.nanmean(bootstrap_c_index)
    c_index_lower = np.nanpercentile(bootstrap_c_index, 2.5)
    c_index_upper = np.nanpercentile(bootstrap_c_index, 97.5)
    
    # Calculate mean and confidence intervals for integrated metrics
    integrated_brier_score_mean = np.nanmean(bootstrap_integrated_brier_score)
    integrated_brier_score_lower = np.nanpercentile(bootstrap_integrated_brier_score, 2.5)
    integrated_brier_score_upper = np.nanpercentile(bootstrap_integrated_brier_score, 97.5)
    
    integrated_nbll_mean = np.nanmean(bootstrap_integrated_nbll)
    integrated_nbll_lower = np.nanpercentile(bootstrap_integrated_nbll, 2.5)
    integrated_nbll_upper = np.nanpercentile(bootstrap_integrated_nbll, 97.5)
    
    # Calculate mean and confidence intervals for time-dependent metrics
    metrics_by_horizon = {}
    for horizon in time_horizons:
        brier_score_mean = np.nanmean(bootstrap_metrics_by_horizon[horizon]['brier_score'])
        brier_score_lower = np.nanpercentile(bootstrap_metrics_by_horizon[horizon]['brier_score'], 2.5)
        brier_score_upper = np.nanpercentile(bootstrap_metrics_by_horizon[horizon]['brier_score'], 97.5)
        
        metrics_by_horizon[horizon] = {
            'brier_score': {
                'mean': float(brier_score_mean),
                'lower': float(brier_score_lower),
                'upper': float(brier_score_upper)
            }
        }
    
    # Create results dictionary
    results = {
        'c_index': {
            'mean': float(c_index_mean),
            'lower': float(c_index_lower),
            'upper': float(c_index_upper),
            'original': float(c_index)
        },
        'integrated_brier_score': {
            'mean': float(integrated_brier_score_mean),
            'lower': float(integrated_brier_score_lower),
            'upper': float(integrated_brier_score_upper),
            'original': float(ev.integrated_brier_score(np.linspace(durations.min(), durations.max(), 10)))
        },
        'integrated_nbll': {
            'mean': float(integrated_nbll_mean),
            'lower': float(integrated_nbll_lower),
            'upper': float(integrated_nbll_upper),
            'original': float(ev.integrated_nbll(np.linspace(durations.min(), durations.max(), 10)))
        },
        'metrics_by_horizon': metrics_by_horizon
    }
    
    return results


def calculate_shap_values(
    model: Any,
    x: Union[np.ndarray, torch.Tensor],
    feature_names: List[str],
    n_bootstrap: int = 50,  # Reduced from 500 to 50
    batch_size: Optional[int] = None,  # Batch size for bootstrap processing
    use_gpu_for_shap: bool = False  # Whether to use GPU for SHAP calculation if available
) -> Dict[str, Any]:
    """
    Calculate SHAP values for feature importance with bootstrap.
    
    Args:
        model: The trained survival model
        x: Feature matrix
        feature_names: Names of features
        n_bootstrap: Number of bootstrap iterations
        batch_size: Batch size for bootstrap processing
        use_gpu_for_shap: Whether to use GPU for SHAP calculation if available
                         If True, data will be moved to GPU to match model
                         If False, model will be moved to CPU (safer for memory)
        
    Returns:
        Dictionary of SHAP values with confidence intervals
    """
    # Convert to numpy array if tensor
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    
    # Create a background dataset for SHAP
    try:
        # Sample a subset for background (for efficiency)
        n_background = min(50, len(x_np))  # Reduced from 100 to 50
        background_indices = np.random.choice(len(x_np), size=n_background, replace=False)
        
        # Try to get the device from the model, with a fallback to CPU if not available
        try:
            device = model.net.device
        except AttributeError:
            device = torch.device('cpu')
            print("Model does not have a device attribute, defaulting to CPU")
        
        # Determine which device to use for SHAP calculation
        if use_gpu_for_shap and torch.cuda.is_available() and device.type == 'cuda':
            # Use GPU for SHAP calculation
            print("Using GPU for SHAP calculation")
            # Keep model on GPU and move data to GPU
            shap_device = device
            model_for_shap = model.net  # Keep model on GPU
            # Create background tensor on GPU
            background = torch.tensor(x_np[background_indices]).float().to(shap_device)
        else:
            # Use CPU for SHAP calculation (safer for memory)
            print("Using CPU for SHAP calculation")
            # Move model to CPU and keep data on CPU
            shap_device = torch.device('cpu')
            # Save original device for restoring later if needed
            original_device = device
            model_for_shap = model.net.to(shap_device)
            # Create background tensor on CPU
            background = torch.tensor(x_np[background_indices]).float()
        
        # Create an explainer with model and background on the same device
        explainer = shap.DeepExplainer(model_for_shap, background)
        
        # Initialize array for bootstrap SHAP values
        all_shap_values = []
        
        # Process bootstrap in batches if batch_size is provided
        if batch_size is not None and n_bootstrap > batch_size:
            print(f"Processing {n_bootstrap} SHAP bootstrap iterations in batches of {batch_size}")
            
            # Number of batches
            n_batches = (n_bootstrap + batch_size - 1) // batch_size
            
            # Process each batch
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_bootstrap)
                batch_size_actual = end_idx - start_idx
                
                print(f"Processing SHAP bootstrap batch {batch_idx+1}/{n_batches} (iterations {start_idx}-{end_idx-1})")
                
                # Process bootstrap iterations in this batch
                for i in range(batch_size_actual):
                    # Sample with replacement
                    indices = np.random.choice(len(x_np), size=len(x_np), replace=True)
                    # Try to get the device from the model, with a fallback to CPU if not available
                    # Use the same device as determined for SHAP calculation
                    x_sample = torch.tensor(x_np[indices]).float().to(shap_device)
                    
                    # Calculate SHAP values
                    shap_values_raw = explainer.shap_values(x_sample)
                    
                    # Process SHAP values based on model type
                    if isinstance(shap_values_raw, list):
                        # For models with multiple outputs (e.g., DeepHit)
                        # Average across all outputs
                        shap_values_combined = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
                        feature_importance = np.mean(shap_values_combined, axis=0)
                    else:
                        # For models with single output (e.g., DeepSurv)
                        feature_importance = np.mean(np.abs(shap_values_raw), axis=0)
                    
                    all_shap_values.append(feature_importance)
                    
                    # Clear memory
                    del x_sample
                    del shap_values_raw
                    if isinstance(shap_values_raw, list):
                        del shap_values_combined
                
                # Force garbage collection after each batch
                gc.collect()
                print(f"Completed SHAP bootstrap batch {batch_idx+1}/{n_batches}")
        else:
            # Original code for processing all bootstrap iterations at once
            for i in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(x_np), size=len(x_np), replace=True)
                # Try to get the device from the model, with a fallback to CPU if not available
                # Use the same device as determined for SHAP calculation
                x_sample = torch.tensor(x_np[indices]).float().to(shap_device)
                
                # Calculate SHAP values
                shap_values_raw = explainer.shap_values(x_sample)
                
                # Process SHAP values based on model type
                if isinstance(shap_values_raw, list):
                    # For models with multiple outputs (e.g., DeepHit)
                    # Average across all outputs
                    shap_values_combined = np.mean([np.abs(sv) for sv in shap_values_raw], axis=0)
                    feature_importance = np.mean(shap_values_combined, axis=0)
                else:
                    # For models with single output (e.g., DeepSurv)
                    feature_importance = np.mean(np.abs(shap_values_raw), axis=0)
                
                all_shap_values.append(feature_importance)
        
        # Convert to numpy array
        all_shap_values = np.array(all_shap_values)
        
        # Calculate mean and confidence intervals
        mean_shap = np.mean(all_shap_values, axis=0)
        lower_ci = np.percentile(all_shap_values, 2.5, axis=0)
        upper_ci = np.percentile(all_shap_values, 97.5, axis=0)
        
        # Create result dictionary
        result = {
            'feature_names': feature_names,
            'mean_shap': mean_shap.tolist(),
            'lower_ci': lower_ci.tolist(),
            'upper_ci': upper_ci.tolist(),
            'all_bootstrap_values': all_shap_values.tolist()
        }
        
        return result
    
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty result
        return {
            'feature_names': feature_names,
            'mean_shap': [0] * len(feature_names),
            'lower_ci': [0] * len(feature_names),
            'upper_ci': [0] * len(feature_names),
            'all_bootstrap_values': [],
            'error': str(e)
        }


def create_visualizations(
    surv: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    metrics: Dict[str, Any],
    dataset_name: str,
    time_horizons: List[int],
    output_dir: str,
    batch_size: Optional[int] = None  # Batch size for bootstrap processing
) -> Dict[str, str]:
    """
    Create visualizations for model performance.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
        time_horizons: List of time horizons
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of paths to saved visualizations
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store paths to saved visualizations
    visualization_paths = {}
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create calibration plots
    calibration_plot_path = create_calibration_plots(
        surv=surv,
        durations=durations,
        events=events,
        time_horizons=time_horizons,
        dataset_name=dataset_name,
        output_dir=output_dir,
        timestamp=timestamp,
        batch_size=batch_size  # Pass batch_size to calibration plots
    )
    visualization_paths['calibration_plot'] = calibration_plot_path
    
    # Create metrics by time plot
    metrics_by_time_path = create_metrics_by_time_plot(
        metrics=metrics,
        time_horizons=time_horizons,
        dataset_name=dataset_name,
        output_dir=output_dir,
        timestamp=timestamp
    )
    visualization_paths['metrics_by_time'] = metrics_by_time_path
    
    return visualization_paths


def create_calibration_plots(
    surv: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    time_horizons: List[int],
    dataset_name: str,
    output_dir: str,
    timestamp: str,
    n_bootstrap: int = 100,  # Reduced from 1000 to 100
    n_quantiles: int = 5,    # Reduced from 10 to 5
    batch_size: Optional[int] = None  # Batch size for bootstrap processing
) -> str:
    """
    Create calibration plots for the model.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
        
    Returns:
        Path to saved calibration plot
    """
    # Define time horizons (years 1-5)
    horizon_labels = [f"Year {i+1}" for i in range(len(time_horizons))]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(time_horizons), figsize=(20, 6), sharey=True)
    
    # If only one time horizon, make axes iterable
    if len(time_horizons) == 1:
        axes = [axes]
    
    # Plot each time horizon
    for i, (horizon, label) in enumerate(zip(time_horizons, horizon_labels)):
        # Get predicted risks at this horizon
        try:
            # Debug information
            print(f"surv shape: {surv.shape}, surv columns type: {type(surv.columns)}")
            print(f"surv columns: {surv.columns[:5]}... (showing first 5)")
            print(f"horizon: {horizon}, type: {type(horizon)}")
            
            # Find the closest time point in the predictions
            closest_time = min(surv.columns, key=lambda x: abs(float(x) - horizon))
            print(f"closest_time: {closest_time}, type: {type(closest_time)}")
            
            pred_risks = 1 - surv[closest_time].values
            print(f"pred_risks shape: {pred_risks.shape}, type: {type(pred_risks)}")
            
            # Cut predicted risks into 10 quantiles
            quantiles = pd.qcut(pred_risks, n_quantiles, labels=False, duplicates='drop')
            unique_quantiles = np.unique(quantiles)
            n_quantiles = len(unique_quantiles)
            
            # Initialize arrays for observed risks and confidence intervals
            observed_risks = np.zeros(n_quantiles)
            observed_risks_lower = np.zeros(n_quantiles)
            observed_risks_upper = np.zeros(n_quantiles)
            mean_predicted_risks = np.zeros(n_quantiles)
            
            # Calculate observed risks for each quantile using bootstrap
            # n_bootstrap is now a parameter with default 100
            bootstrap_observed_risks = np.zeros((n_bootstrap, n_quantiles))
            
            for q_idx, q in enumerate(unique_quantiles):
                # Get indices for this quantile
                q_indices = np.where(quantiles == q)[0]
                
                # Calculate mean predicted risk for this quantile
                mean_predicted_risks[q_idx] = np.mean(pred_risks[q_indices])
                
                # Bootstrap to get confidence intervals
                # Process bootstrap in batches if batch_size is provided
                if batch_size is not None and n_bootstrap > batch_size:
                    # Number of batches
                    n_batches = (n_bootstrap + batch_size - 1) // batch_size
                    
                    for batch_idx in range(n_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, n_bootstrap)
                        
                        for b in range(start_idx, end_idx):
                            # Sample with replacement
                            bootstrap_indices = np.random.choice(q_indices, size=len(q_indices), replace=True)
                            
                            # Calculate observed risk using Kaplan-Meier estimator
                            kmf = KaplanMeierFitter()
                            kmf.fit(durations[bootstrap_indices], event_observed=events[bootstrap_indices])
                            
                            # Get survival probability at horizon
                            try:
                                # Ensure horizon is a scalar value
                                if isinstance(horizon, (list, np.ndarray)):
                                    h = float(horizon[0])
                                else:
                                    h = float(horizon)
                                surv_prob = kmf.predict(h)
                                bootstrap_observed_risks[b, q_idx] = 1 - surv_prob
                            except Exception as e:
                                print(f"Error in bootstrap prediction: {e}")
                                bootstrap_observed_risks[b, q_idx] = np.nan
                            
                            # Clear memory
                            del bootstrap_indices
                            del kmf
                        
                        # Force garbage collection after each batch
                        gc.collect()
                else:
                    # Original code for processing all bootstrap iterations at once
                    for b in range(n_bootstrap):
                        # Sample with replacement
                        bootstrap_indices = np.random.choice(q_indices, size=len(q_indices), replace=True)
                        
                        # Calculate observed risk using Kaplan-Meier estimator
                        kmf = KaplanMeierFitter()
                        kmf.fit(durations[bootstrap_indices], event_observed=events[bootstrap_indices])
                        
                        # Get survival probability at horizon
                        try:
                            # Ensure horizon is a scalar value
                            if isinstance(horizon, (list, np.ndarray)):
                                h = float(horizon[0])
                            else:
                                h = float(horizon)
                            surv_prob = kmf.predict(h)
                            bootstrap_observed_risks[b, q_idx] = 1 - surv_prob
                        except Exception as e:
                            print(f"Error in bootstrap prediction: {e}")
                            bootstrap_observed_risks[b, q_idx] = np.nan
                
                # Calculate mean and confidence intervals
                valid_bootstrap = bootstrap_observed_risks[:, q_idx][~np.isnan(bootstrap_observed_risks[:, q_idx])]
                if len(valid_bootstrap) > 0:
                    observed_risks[q_idx] = np.mean(valid_bootstrap)
                    observed_risks_lower[q_idx] = np.percentile(valid_bootstrap, 2.5)
                    observed_risks_upper[q_idx] = np.percentile(valid_bootstrap, 97.5)
                else:
                    observed_risks[q_idx] = np.nan
                    observed_risks_lower[q_idx] = np.nan
                    observed_risks_upper[q_idx] = np.nan
            
            # Convert to percentages
            mean_predicted_risks *= 100
            observed_risks *= 100
            observed_risks_lower *= 100
            observed_risks_upper *= 100
            
            # Set up bar positions
            bar_width = 0.35
            index = np.arange(n_quantiles)
            
            # Plot predicted risks
            axes[i].bar(index - bar_width/2, mean_predicted_risks, bar_width,
                        label='Predicted Risk', color='blue', alpha=0.7)
            
            # Plot observed risks with error bars
            # Ensure yerr is properly formatted as 1D arrays
            lower_err = observed_risks - observed_risks_lower
            upper_err = observed_risks_upper - observed_risks
            
            # Convert to 1D arrays and handle NaN values
            lower_err = np.array(lower_err, dtype=float).flatten()  # Ensure 1D array
            upper_err = np.array(upper_err, dtype=float).flatten()  # Ensure 1D array
            
            # Debug information
            print(f"lower_err shape: {lower_err.shape}, upper_err shape: {upper_err.shape}")
            print(f"observed_risks shape: {observed_risks.shape}")
            
            # Ensure yerr is in the correct format for matplotlib
            # For asymmetric error bars, yerr should be a 2xN array where N is the number of points
            yerr_array = np.array([lower_err, upper_err])
            print(f"yerr_array shape: {yerr_array.shape}")
            
            # Check if shapes match
            if yerr_array.shape[1] != len(observed_risks):
                print(f"Warning: yerr_array shape {yerr_array.shape} doesn't match observed_risks length {len(observed_risks)}")
                # Adjust shapes if needed
                min_len = min(yerr_array.shape[1], len(observed_risks))
                yerr_array = yerr_array[:, :min_len]
                observed_risks = observed_risks[:min_len]
                index = index[:min_len]
                print(f"Adjusted shapes - yerr_array: {yerr_array.shape}, observed_risks: {len(observed_risks)}")
            
            try:
                axes[i].bar(index + bar_width/2, observed_risks, bar_width,
                            label='Observed Risk', color='red', alpha=0.7,
                            yerr=yerr_array,
                            capsize=5)
            except Exception as e:
                print(f"Error in bar plot: {e}")
                # Try alternative format for yerr
                try:
                    print("Trying alternative yerr format...")
                    axes[i].bar(index + bar_width/2, observed_risks, bar_width,
                                label='Observed Risk', color='red', alpha=0.7,
                                yerr=None,  # Skip error bars if there's an issue
                                capsize=5)
                except Exception as e2:
                    print(f"Error in alternative bar plot: {e2}")
            
            # Set labels and title
            axes[i].set_xlabel('Risk Quantile')
            if i == 0:
                axes[i].set_ylabel('Risk (%)')
            axes[i].set_title(label)
            axes[i].set_xticks(index)
            axes[i].set_xticklabels([f'Q{q+1}' for q in range(n_quantiles)], rotation=45)
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
        except Exception as e:
            print(f"Error creating calibration plot for {horizon} days: {e}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[i].transAxes)
    
    # Add legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Add title to the figure
    fig.suptitle(f'Calibration Plot - {dataset_name.capitalize()} Dataset', fontsize=16)
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"{dataset_name}_calibration_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_metrics_by_time_plot(
    metrics: Dict[str, Any],
    time_horizons: List[int],
    dataset_name: str,
    output_dir: str,
    timestamp: str
) -> str:
    """
    Create plot of metrics by time horizon.
    
    Args:
        metrics: Dictionary of metrics
        time_horizons: List of time horizons
        dataset_name: Name of the dataset
        output_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
        
    Returns:
        Path to saved metrics by time plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract metrics by horizon
    brier_scores = []
    brier_scores_lower = []
    brier_scores_upper = []
    
    for horizon in time_horizons:
        if horizon in metrics['metrics_by_horizon']:
            # Brier score
            brier_scores.append(metrics['metrics_by_horizon'][horizon]['brier_score']['mean'])
            brier_scores_lower.append(metrics['metrics_by_horizon'][horizon]['brier_score']['lower'])
            brier_scores_upper.append(metrics['metrics_by_horizon'][horizon]['brier_score']['upper'])
            
    
    # Convert to numpy arrays
    brier_scores = np.array(brier_scores)
    brier_scores_lower = np.array(brier_scores_lower)
    brier_scores_upper = np.array(brier_scores_upper)
    
    # Plot Brier scores
    axes.plot(time_horizons, brier_scores, 'o-', color='blue', label='Brier Score')
    axes.fill_between(time_horizons, brier_scores_lower, brier_scores_upper, color='blue', alpha=0.2)
    
    # Set labels and title for Brier score
    axes.set_xlabel('Time Horizon (days)')
    axes.set_ylabel('Brier Score')
    axes.set_title('Brier Score by Time Horizon')
    axes.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(f'Metrics by Time Horizon - {dataset_name.capitalize()} Dataset', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"{dataset_name}_metrics_by_time_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_roc_curve(
    datasets_results: Dict[str, Dict[str, Any]],
    model_type: str,
    time_horizons: List[int],
    output_dir: str
) -> Dict[int, str]:
    """
    Create ROC curve for all datasets on the same plot.
    
    Args:
        datasets_results: Dictionary of results for all datasets
        model_type: Type of model ('deepsurv' or 'deephit')
        time_horizons: List of time horizons
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of paths to saved ROC curves
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store paths to saved ROC curves
    roc_paths = {}
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create ROC curves for each time horizon
    for horizon in time_horizons:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each dataset
        for dataset_name, results in datasets_results.items():
            if dataset_name == 'visualization_paths':
                continue
                
            try:
                # Get predictions and true values
                surv = results['predictions']
                durations = results['durations']
                events = results['events']
                
                # Find the closest time point in the predictions
                closest_time = min(surv.columns, key=lambda x: abs(float(x) - horizon))
                
                # Calculate predicted risks at this horizon
                pred_risks = 1 - surv[closest_time].values
                
                # Calculate true event status at this horizon
                event_at_horizon = (durations <= horizon) & events
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(event_at_horizon, pred_risks)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, lw=2, label=f'{dataset_name.capitalize()} (AUC = {roc_auc:.3f})')
                
            except Exception as e:
                print(f"Error creating ROC curve for {dataset_name} at {horizon} days: {e}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve at {horizon} Days ({horizon/365:.1f} Years)')
        
        # Add legend
        plt.legend(loc='lower right')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        roc_path = os.path.join(output_dir, f"roc_curve_{horizon}days_{timestamp}.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store path
        roc_paths[horizon] = roc_path
    
    return roc_paths

def create_decision_curve_analysis(
    datasets_results: Dict[str, Dict[str, Any]],
    model_type: str,
    time_horizons: List[int],
    output_dir: str
) -> Dict[int, str]:
    """
    Create decision curve analysis for all datasets on the same plot.
    
    Args:
        datasets_results: Dictionary of results for all datasets
        model_type: Type of model ('deepsurv' or 'deephit')
        time_horizons: List of time horizons
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of paths to saved decision curves
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store paths to saved decision curves
    dca_paths = {}
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create decision curves for each time horizon
    for horizon in time_horizons:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Define threshold values
        thresholds = np.linspace(0, 1, 100)
        
        # Plot treat-all and treat-none lines
        # Treat-none: Net benefit is always 0
        plt.plot(thresholds, np.zeros_like(thresholds), 'k--', label='Treat None')
        
        # Plot decision curves for each dataset
        for dataset_name, results in datasets_results.items():
            if dataset_name == 'visualization_paths':
                continue
                
            try:
                # Get predictions and true values
                surv = results['predictions']
                durations = results['durations']
                events = results['events']
                
                # Find the closest time point in the predictions
                closest_time = min(surv.columns, key=lambda x: abs(float(x) - horizon))
                
                # Calculate predicted risks at this horizon
                pred_risks = 1 - surv[closest_time].values
                
                # Calculate true event status at this horizon
                event_at_horizon = (durations <= horizon) & events
                
                # Calculate net benefit for each threshold
                net_benefits = []
                for threshold in thresholds:
                    # Skip extreme thresholds to avoid division by zero
                    if threshold == 0 or threshold == 1:
                        net_benefits.append(0)
                        continue
                        
                    # Calculate decisions based on threshold
                    decisions = pred_risks >= threshold
                    
                    # Calculate true positives and false positives
                    tp = np.sum(decisions & event_at_horizon)
                    fp = np.sum(decisions & ~event_at_horizon)
                    
                    # Calculate net benefit
                    n = len(decisions)
                    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
                    net_benefits.append(net_benefit)
                
                # Plot net benefit curve
                plt.plot(thresholds, net_benefits, lw=2, label=f'{dataset_name.capitalize()}')
                
                # Calculate treat-all net benefit for this dataset
                event_rate = np.mean(event_at_horizon)
                treat_all_net_benefits = []
                for threshold in thresholds:
                    # Skip extreme thresholds to avoid division by zero
                    if threshold == 0 or threshold == 1:
                        treat_all_net_benefits.append(0)
                        continue
                        
                    # Calculate net benefit for treating all
                    net_benefit = event_rate - (1 - event_rate) * (threshold / (1 - threshold))
                    treat_all_net_benefits.append(net_benefit)
                
                # Plot treat-all line for this dataset
                if dataset_name == list(datasets_results.keys())[0]:
                    plt.plot(thresholds, treat_all_net_benefits, 'r--', label='Treat All')
                
            except Exception as e:
                print(f"Error creating decision curve for {dataset_name} at {horizon} days: {e}")
        
        # Set labels and title
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis at {horizon} Days ({horizon/365:.1f} Years)')
        
        # Add legend
        plt.legend(loc='lower left')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x and y limits
        plt.xlim(0, 1)
        plt.ylim(bottom=-0.05)
        
        # Save the figure
        dca_path = os.path.join(output_dir, f"decision_curve_{horizon}days_{timestamp}.png")
        plt.savefig(dca_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store path
        dca_paths[horizon] = dca_path
    
    return dca_paths

def save_predictions_to_files(
    surv: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    dataset_name: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Save predictions to HDF5 file and metadata to CSV file.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        dataset_name: Name of the dataset
        output_dir: Directory to save files
        
    Returns:
        Dictionary of paths to saved files
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions to HDF5
    predictions_path = os.path.join(output_dir, f"{dataset_name}_predictions_{timestamp}.h5")
    
    # Add metadata for the HDF5 file
    metadata = {
        'dataset_name': dataset_name,
        'created_at': timestamp,
        'n_samples': len(durations),
        'n_events': int(np.sum(events))
    }
    
    # Save predictions using the utility function
    save_predictions_to_hdf5(surv, predictions_path, metadata=metadata)
    print(f"Predictions saved to HDF5 file: {predictions_path}")
    print(f"Prediction matrix shape: {surv.shape}")
    
    # Save metadata (durations and events)
    metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata_{timestamp}.csv")
    metadata_df = pd.DataFrame({
        'duration': durations,
        'event': events
    })
    metadata_df.to_csv(metadata_path, index=False)
    
    return {
        'predictions': predictions_path,
        'metadata': metadata_path
    }


def save_shap_values(
    shap_values: Dict[str, Any],
    dataset_name: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Save SHAP values to files.
    
    Args:
        shap_values: Dictionary of SHAP values
        dataset_name: Name of the dataset
        output_dir: Directory to save files
        
    Returns:
        Dictionary of paths to saved files
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save SHAP values to JSON
    json_path = os.path.join(output_dir, f"{dataset_name}_shap_values_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(shap_values, f, indent=2)
    
    # Save SHAP values to CSV for easier analysis
    csv_path = os.path.join(output_dir, f"{dataset_name}_shap_values_{timestamp}.csv")
    
    # Create DataFrame with feature names and mean SHAP values
    shap_df = pd.DataFrame({
        'feature': shap_values['feature_names'],
        'mean_shap': shap_values['mean_shap'],
        'lower_ci': shap_values['lower_ci'],
        'upper_ci': shap_values['upper_ci']
    })
    
    # Sort by absolute mean SHAP value
    shap_df['abs_mean_shap'] = np.abs(shap_df['mean_shap'])
    shap_df = shap_df.sort_values('abs_mean_shap', ascending=False)
    shap_df = shap_df.drop(columns=['abs_mean_shap'])
    
    # Save to CSV
    shap_df.to_csv(csv_path, index=False)
    
    return {
        'json': json_path,
        'csv': csv_path
    }
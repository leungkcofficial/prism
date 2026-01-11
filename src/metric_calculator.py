#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metric Calculator for CKD Risk Prediction Models

This script calculates evaluation metrics (C-index, Brier score, log likelihood)
and analyzes SHAP values from prediction files. It works with HDF5 prediction files
and CSV/JSON SHAP value files.

Usage:
    python metric_calculator.py --predictions <path_to_h5_file> [options]
    python metric_calculator.py --help

Options:
    --predictions PATH       Path to HDF5 file containing survival predictions
    --metadata PATH          Path to CSV file containing metadata (durations and events)
    --durations PATH         Path to CSV file containing event times (optional if included in metadata)
    --events PATH            Path to CSV file containing event indicators (optional if included in metadata)
    --shap PATH              Path to CSV or JSON file containing SHAP values (optional)
    --time-horizons DAYS     Comma-separated list of time horizons in days (default: 365,730,1095,1460,1825)
    --output-dir PATH        Directory to save output files (default: current directory)
    --n-bootstrap INT        Number of bootstrap iterations (default: 10)
    --visualize              Generate visualization plots (default: False)
    --verbose                Print detailed information (default: False)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import h5py
from pycox.evaluation import EvalSurv
from lifelines import KaplanMeierFitter
# Import scikit-survival for time-specific concordance index
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# Import utility functions
from src.util import load_predictions_from_hdf5


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics for CKD risk prediction models.')
    
    # Required arguments
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to HDF5 file containing survival predictions')
    
    # Optional arguments
    parser.add_argument('--metadata', type=str,
                        help='Path to CSV file containing metadata (durations and events)')
    parser.add_argument('--durations', type=str,
                        help='Path to CSV file containing event times')
    parser.add_argument('--events', type=str,
                        help='Path to CSV file containing event indicators')
    parser.add_argument('--shap', type=str,
                        help='Path to CSV or JSON file containing SHAP values')
    parser.add_argument('--time-horizons', type=str, default='365,730,1095,1460,1825',
                        help='Comma-separated list of time horizons in days (default: 365,730,1095,1460,1825)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--n-bootstrap', type=int, default=10,
                        help='Number of bootstrap iterations (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots (default: False)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information (default: False)')
    parser.add_argument('--model-type', type=str, default='deepsurv', choices=['deepsurv', 'deephit'],
                        help='Type of model: deepsurv or deephit (default: deepsurv)')
    
    return parser.parse_args()


def validate_predictions_file(file_path):
    """
    Validate the predictions HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Predictions file not found: {file_path}")
    
    # Check if file is a valid HDF5 file
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if required datasets exist
            if 'predictions' not in f:
                raise ValueError(f"Missing 'predictions' dataset in HDF5 file: {file_path}")
            
            # Check if predictions dataset has the correct shape
            predictions = f['predictions']
            # Allow 3D predictions for DeepHit competing risks format (2, 5, n_samples)
            if len(predictions.shape) == 3:
                # Check if this is DeepHit competing risks format
                if predictions.shape[0] == 2 and predictions.shape[1] <= 10:
                    print(f"Detected DeepHit competing risks format: {predictions.shape}")
                    # Keep as 3D array for proper competing risks processing
                    predictions = np.array(predictions)
                    print(f"Converted to numpy array: {predictions.shape}")
                else:
                    raise ValueError(f"Unsupported 3D predictions shape: {predictions.shape}")
            elif len(predictions.shape) != 2:
                raise ValueError(f"Predictions dataset must be 2-dimensional or valid 3D competing risks format, got shape {predictions.shape}")
            
            # Check if index and columns datasets exist
            # For competing risks format, index is optional
            has_competing_risks = len(predictions.shape) == 3 and predictions.shape[0] == 2
            if 'index' not in f and not has_competing_risks:
                raise ValueError(f"Missing 'index' dataset in HDF5 file: {file_path}")
            
            if 'columns' not in f:
                raise ValueError(f"Missing 'columns' dataset in HDF5 file: {file_path}")
            
            # Check if index and columns have the correct shape (skip for competing risks)
            if not has_competing_risks:
                index = f['index']
                columns = f['columns']
                
                if len(index) != predictions.shape[0]:
                    raise ValueError(f"Index length ({len(index)}) does not match predictions shape ({predictions.shape})")
                
                if len(columns) != predictions.shape[1]:
                    raise ValueError(f"Columns length ({len(columns)}) does not match predictions shape ({predictions.shape})")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Invalid HDF5 file: {file_path}. Error: {str(e)}")
    
    return True


def validate_durations_events_files(durations_path, events_path):
    """
    Validate the durations and events CSV files.
    
    Args:
        durations_path: Path to the durations CSV file
        events_path: Path to the events CSV file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if files exist
    if not os.path.exists(durations_path):
        raise FileNotFoundError(f"Durations file not found: {durations_path}")
    
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")
    
    # Check if files are valid CSV files
    try:
        durations_df = pd.read_csv(durations_path)
        events_df = pd.read_csv(events_path)
        
        # Check if files have at least one column
        if durations_df.shape[1] == 0:
            raise ValueError(f"Durations file has no columns: {durations_path}")
        
        if events_df.shape[1] == 0:
            raise ValueError(f"Events file has no columns: {events_path}")
        
        # Check if files have the same number of rows
        if durations_df.shape[0] != events_df.shape[0]:
            raise ValueError(f"Durations file ({durations_df.shape[0]} rows) and events file ({events_df.shape[0]} rows) have different numbers of rows")
        
        # Check if durations are numeric
        if not pd.to_numeric(durations_df.iloc[:, 0], errors='coerce').notna().all():
            raise ValueError(f"Durations file contains non-numeric values: {durations_path}")
        
        # Check if events are binary (0 or 1)
        events = events_df.iloc[:, 0]
        if not ((events == 0) | (events == 1)).all():
            raise ValueError(f"Events file contains non-binary values (must be 0 or 1): {events_path}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise ValueError(f"Invalid CSV file(s). Error: {str(e)}")
    
    return True


def validate_shap_file(file_path):
    """
    Validate the SHAP values file (CSV or JSON).
    
    Args:
        file_path: Path to the SHAP values file
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SHAP values file not found: {file_path}")
    
    # Check file extension
    if file_path.endswith('.json'):
        # Validate JSON file
        try:
            with open(file_path, 'r') as f:
                shap_values = json.load(f)
            
            # Check if required keys exist
            required_keys = ['feature_names', 'mean_shap']
            for key in required_keys:
                if key not in shap_values:
                    raise ValueError(f"Missing required key '{key}' in SHAP values JSON file: {file_path}")
            
            # Check if feature_names and mean_shap have the same length
            if len(shap_values['feature_names']) != len(shap_values['mean_shap']):
                raise ValueError(f"Length mismatch between feature_names ({len(shap_values['feature_names'])}) and mean_shap ({len(shap_values['mean_shap'])}) in SHAP values JSON file: {file_path}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Invalid JSON file: {file_path}. Error: {str(e)}")
    elif file_path.endswith('.csv'):
        # Validate CSV file
        try:
            shap_df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['feature', 'mean_shap']
            for column in required_columns:
                if column not in shap_df.columns:
                    raise ValueError(f"Missing required column '{column}' in SHAP values CSV file: {file_path}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Invalid CSV file: {file_path}. Error: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format for SHAP values: {file_path}. Must be .json or .csv")
    
    return True


def validate_data_consistency(predictions, durations, events):
    """
    Validate the consistency between predictions, durations, and events.
    
    Args:
        predictions: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if durations and events have the same length
    if len(durations) != len(events):
        raise ValueError(f"Length mismatch between durations ({len(durations)}) and events ({len(events)})")
    
    # Check if predictions has the correct shape
    if len(durations) != predictions.shape[1] and len(durations) != predictions.shape[0]:
        # If the number of durations doesn't match either dimension of predictions,
        # we need to check if the predictions matrix is transposed or if we need to filter it
        print(f"Warning: Dimension mismatch between durations ({len(durations)}) and predictions ({predictions.shape})")
        print("Attempting to handle the mismatch...")
        
        # If the number of durations is less than the number of rows in predictions,
        # we might need to filter the predictions matrix
        if len(durations) < predictions.shape[0]:
            print(f"Number of durations ({len(durations)}) is less than number of rows in predictions ({predictions.shape[0]})")
            print("Using the first {len(durations)} rows of predictions")
            
            # Use the first len(durations) rows of predictions
            predictions = predictions.iloc[:len(durations), :]
            print(f"New predictions shape: {predictions.shape}")
        elif len(durations) < predictions.shape[1]:
            print(f"Number of durations ({len(durations)}) is less than number of columns in predictions ({predictions.shape[1]})")
            print(f"Using the first {len(durations)} columns of predictions")
            
            # Use the first len(durations) columns of predictions
            predictions = predictions.iloc[:, :len(durations)]
            print(f"New predictions shape: {predictions.shape}")
        else:
            raise ValueError(f"Dimension mismatch between durations ({len(durations)}) and predictions ({predictions.shape})")
    
    # Check if durations are non-negative
    if (durations < 0).any():
        raise ValueError("Durations must be non-negative")
    
    # Check if events are binary (0 or 1)
    # Check if this is competing risks format (events can be 0, 1, 2, ...)
    unique_events = np.unique(events)
    is_competing_risks_events = len(unique_events) > 2 or (len(unique_events) == 3 and 2 in unique_events)
    
    if not is_competing_risks_events:
        # Standard binary events validation
        if not ((events == 0) | (events == 1)).all():
            raise ValueError("Events must be binary (0 or 1)")
    else:
        # Competing risks validation - events should be 0, 1, 2, ...
        if not all(event >= 0 and event == int(event) for event in unique_events):
            raise ValueError("Events must be non-negative integers for competing risks (0=censored, 1=cause1, 2=cause2, ...)")
        print(f"Detected competing risks events: {unique_events}")
    
    # Check if predictions are probabilities (between 0 and 1)
    if (predictions < 0).any().any() or (predictions > 1).any().any():
        raise ValueError("Predictions must be probabilities (between 0 and 1)")
    
    return True


def validate_time_horizons(time_horizons, durations):
    """
    Validate the time horizons.
    
    Args:
        time_horizons: List of time horizons
        durations: Event times
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if time horizons are positive
    if any(t <= 0 for t in time_horizons):
        raise ValueError("Time horizons must be positive")
    
    # Check if time horizons are in ascending order
    if not all(time_horizons[i] < time_horizons[i+1] for i in range(len(time_horizons)-1)):
        raise ValueError("Time horizons must be in ascending order")
    
    # Check if time horizons are within the range of durations
    if max(time_horizons) > max(durations) * 1.5:
        print(f"Warning: Maximum time horizon ({max(time_horizons)}) is greater than 1.5 times the maximum duration ({max(durations)})")
    
    return True


def validate_output_directory(output_dir):
    """
    Validate the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        True if valid, raises an exception otherwise
    """
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create output directory: {output_dir}. Error: {str(e)}")
    
    # Check if output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_dir}")
    
    return True


def load_data(args):
    """
    Load data from input files.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (predictions, durations, events, shap_values)
    """
    # Validate predictions file
    validate_predictions_file(args.predictions)
    
    # Load predictions from HDF5 file
    if args.verbose:
        print(f"Loading predictions from {args.predictions}")
    
    # Check if we need to return metadata
    return_metadata = args.durations is None or args.events is None
    
    try:
        if return_metadata:
            predictions, metadata = load_predictions_from_hdf5(args.predictions, return_metadata=True)
        else:
            predictions = load_predictions_from_hdf5(args.predictions, return_metadata=False)
            metadata = None
    except ValueError as e:
        if "too many values to unpack" in str(e):
            # Handle the case where the HDF5 file has a different structure
            if args.verbose:
                print("Detected different HDF5 file structure. Attempting to load directly...")
            
            with h5py.File(args.predictions, 'r') as f:
                # Load predictions
                predictions_array = f['predictions'][:]
                
                # Load index and columns (skip index for competing risks)
                if has_competing_risks:
                    # For competing risks, create default column values
                    column_values = np.arange(predictions.shape[-1])
                    index_values = None  # Not needed for competing risks
                else:
                    index_values = f['index'][:]
                    column_values = f['columns'][:]
                
                if args.verbose:
                    print(f"Predictions array shape: {predictions_array.shape}")
                    print(f"Index values shape: {index_values.shape}")
                    print(f"Column values shape: {column_values.shape}")
                
                # Check if the predictions array needs to be transposed
                # According to user, the expected shape is (1825, 42953)
                # where 1825 is the number of time points and 42953 is the number of patients
                if predictions_array.shape[0] > predictions_array.shape[1]:
                    if args.verbose:
                        print(f"Predictions array has more rows than columns. Transposing...")
                    
                    # Transpose the predictions array
                    predictions_array = predictions_array.T
                    
                    if args.verbose:
                        print(f"New predictions array shape after transposing: {predictions_array.shape}")
                
                # Convert index and column values to numeric if they are strings
                try:
                    if isinstance(column_values[0], bytes):
                        column_values = np.array([float(c.decode('utf-8')) for c in column_values])
                    if isinstance(index_values[0], bytes):
                        index_values = np.array([float(i.decode('utf-8')) for i in index_values])
                except Exception as e:
                    if args.verbose:
                        print(f"Error converting index/column values to numeric: {e}")
                
                # Sort the column values (time points) to ensure they are monotonically increasing
                if args.verbose:
                    print("Sorting column values to ensure monotonicity...")
                
                # Get the sorting indices
                sort_idx = np.argsort(column_values)
                
                # Sort the column values and the corresponding rows in the predictions array
                column_values = column_values[sort_idx]
                predictions_array = predictions_array[sort_idx, :]
                
                # Create DataFrame
                predictions = pd.DataFrame(
                    predictions_array,
                    index=column_values,
                    columns=index_values
                )
                
                if args.verbose:
                    print(f"Created predictions DataFrame with shape: {predictions.shape}")
                
                # Check if durations and events are directly in the file
                metadata = {}
                if 'durations' in f and 'events' in f:
                    metadata['durations'] = f['durations'][:]
                    metadata['events'] = f['events'][:]
                    if args.verbose:
                        print("Successfully loaded durations and events directly from HDF5 file")
                        print(f"Durations shape: {metadata['durations'].shape}")
                        print(f"Events shape: {metadata['events'].shape}")
        else:
            raise
    
    if args.verbose:
        print(f"Loaded predictions with shape {predictions.shape}")
    
    # Load durations and events
    if args.durations is not None and args.events is not None:
        # Validate durations and events files
        validate_durations_events_files(args.durations, args.events)
        
        # Load durations and events from CSV files
        if args.verbose:
            print(f"Loading durations from {args.durations}")
            print(f"Loading events from {args.events}")
        
        durations_df = pd.read_csv(args.durations)
        events_df = pd.read_csv(args.events)
        
        durations = durations_df.iloc[:, 0].values
        events = events_df.iloc[:, 0].values
    elif metadata is not None and 'durations' in metadata and 'events' in metadata:
        # Extract durations and events from metadata
        if args.verbose:
            print("Extracting durations and events from metadata")
        
        durations = metadata['durations']
        events = metadata['events']
    elif args.metadata is not None:
        # Load from user-specified metadata file
        if args.verbose:
            print(f"Loading durations and events from user-specified metadata file: {args.metadata}")
        
        if not os.path.exists(args.metadata):
            raise FileNotFoundError(f"User-specified metadata file not found: {args.metadata}")
        
        metadata_df = pd.read_csv(args.metadata)
        
        if 'duration' in metadata_df.columns and 'event' in metadata_df.columns:
            durations = metadata_df['duration'].values
            events = metadata_df['event'].values
        else:
            raise ValueError(f"Metadata file {args.metadata} does not contain 'duration' and 'event' columns")
    else:
        # Try to load from metadata CSV file
        metadata_path = args.predictions.replace('.h5', '_metadata.csv')
        if args.verbose:
            print(f"Checking for metadata file at: {metadata_path}")
        
        # If the standard metadata path doesn't exist, try alternative naming patterns
        if not os.path.exists(metadata_path):
            if args.verbose:
                print(f"Standard metadata path not found: {metadata_path}")
            
            # Try to find a metadata file with "training_metadata_" prefix in the same directory
            predictions_dir = os.path.dirname(args.predictions)
            predictions_filename = os.path.basename(args.predictions)
            timestamp = predictions_filename.split('_')[-1].replace('.h5', '')
            
            alt_metadata_path = os.path.join(predictions_dir, f"training_metadata_{timestamp}.csv")
            if args.verbose:
                print(f"Checking for alternative metadata file at: {alt_metadata_path}")
            
            if os.path.exists(alt_metadata_path):
                metadata_path = alt_metadata_path
                if args.verbose:
                    print(f"Found alternative metadata file: {metadata_path}")
        
        if os.path.exists(metadata_path):
            if args.verbose:
                print(f"Loading durations and events from metadata file: {metadata_path}")
            
            metadata_df = pd.read_csv(metadata_path)
            
            if 'duration' in metadata_df.columns and 'event' in metadata_df.columns:
                durations = metadata_df['duration'].values
                events = metadata_df['event'].values
            else:
                raise ValueError(f"Metadata file {metadata_path} does not contain 'duration' and 'event' columns")
        else:
            raise ValueError("Durations and events not provided and not found in metadata")
    
    if args.verbose:
        print(f"Loaded durations with shape {durations.shape}")
        print(f"Loaded events with shape {events.shape}")
    
    # Load SHAP values
    shap_values = None
    if args.shap is not None:
        # Validate SHAP file
        validate_shap_file(args.shap)
        
        # Load SHAP values from file
        if args.verbose:
            print(f"Loading SHAP values from {args.shap}")
        
        if args.shap.endswith('.json'):
            with open(args.shap, 'r') as f:
                shap_values = json.load(f)
        elif args.shap.endswith('.csv'):
            shap_df = pd.read_csv(args.shap)
            
            # Convert to dictionary format
            shap_values = {
                'feature_names': shap_df['feature'].tolist(),
                'mean_shap': shap_df['mean_shap'].tolist()
            }
            
            # Add confidence intervals if available
            if 'lower_ci' in shap_df.columns and 'upper_ci' in shap_df.columns:
                shap_values['lower_ci'] = shap_df['lower_ci'].tolist()
                shap_values['upper_ci'] = shap_df['upper_ci'].tolist()
        
        if args.verbose:
            print(f"Loaded SHAP values for {len(shap_values['feature_names'])} features")
    
    return predictions, durations, events, shap_values


def calculate_metrics(surv, durations, events, time_horizons, n_bootstrap=10, verbose=False):
    """
    Calculate survival metrics with bootstrap confidence intervals.
    
    Args:
        surv: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        n_bootstrap: Number of bootstrap iterations
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    if verbose:
        print("Calculating metrics...")
    
    # Ensure the index is monotonically increasing
    if not pd.Series(surv.index).is_monotonic_increasing:
        if verbose:
            print("Sorting survival DataFrame index to ensure monotonicity...")
        surv = surv.sort_index()
        
        if verbose:
            print(f"Survival DataFrame shape after sorting: {surv.shape}")
            print(f"Index is now monotonic: {pd.Series(surv.index).is_monotonic_increasing}")
    
    # Check if the survival DataFrame needs to be transposed
    # The EvalSurv class expects patients as columns and time points as rows
    if verbose:
        print(f"Survival DataFrame shape: {surv.shape}")
        print(f"Durations shape: {durations.shape}")
        print(f"Events shape: {events.shape}")
    
    # If the number of columns in surv doesn't match the number of patients (durations.shape[0]),
    # we need to transpose the DataFrame
    if surv.shape[1] != len(durations):
        if verbose:
            print("Transposing survival DataFrame to match expected dimensions...")
        
        # If we have more rows than columns, it's likely that rows are time points and columns are patients
        if surv.shape[0] > surv.shape[1]:
            if verbose:
                print("More rows than columns, transposing...")
            surv = surv.T
        # If we have more columns than rows, it's likely that columns are time points and rows are patients
        elif surv.shape[1] > surv.shape[0]:
            if verbose:
                print("More columns than rows, transposing...")
            surv = surv.T
        
        if verbose:
            print(f"New survival DataFrame shape after transposing: {surv.shape}")
    
    # Create EvalSurv object
    if verbose:
        print("Creating EvalSurv object...")
        print(f"Final shapes - surv: {surv.shape}, durations: {durations.shape}, events: {events.shape}")
    
    # Check if this is a DeepHit competing risks format (3D: 2 causes, time points, samples)
    is_competing_risks = len(surv.shape) == 3 and surv.shape[0] == 2
    
    if is_competing_risks:
        if verbose:
            print("Detected DeepHit competing risks format - using cause-specific evaluation")
            print(f"CIF predictions shape: {surv.shape}")
        
        # Extract CIF for each cause: surv[0] = cause 1, surv[1] = cause 2
        cif1 = surv[0]  # Shape: (time_points, n_samples)
        cif2 = surv[1]  # Shape: (time_points, n_samples)
        
        # Convert CIF to survival functions: survival = 1 - CIF
        # Create DataFrames as in PyCox example
        time_index = time_horizons if time_horizons is not None else np.array([365, 730, 1095, 1460, 1825])
        
        # Convert to pandas DataFrames with proper time index
        surv1_df = pd.DataFrame(1 - cif1, index=time_index)
        surv2_df = pd.DataFrame(1 - cif2, index=time_index)
        
        # Calculate C-index for each cause separately
        cause_c_indices = []
        
        # Cause 1 (RRT/eGFR<15)
        binary_events_1 = (events == 1).astype(int)
        if np.sum(binary_events_1) > 0:
            try:
                ev1 = EvalSurv(surv1_df, durations, binary_events_1, censor_surv='km')
                c_index_1 = ev1.concordance_td()
                cause_c_indices.append(c_index_1)
                if verbose:
                    print(f"Cause 1 (RRT/eGFR<15) C-index: {c_index_1:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Error calculating C-index for cause 1: {e}")
                cause_c_indices.append(0.0)
        else:
            if verbose:
                print("No events found for cause 1")
            cause_c_indices.append(0.0)
        
        # Cause 2 (Mortality)
        binary_events_2 = (events == 2).astype(int)
        if np.sum(binary_events_2) > 0:
            try:
                ev2 = EvalSurv(surv2_df, durations, binary_events_2, censor_surv='km')
                c_index_2 = ev2.concordance_td()
                cause_c_indices.append(c_index_2)
                if verbose:
                    print(f"Cause 2 (Mortality) C-index: {c_index_2:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Error calculating C-index for cause 2: {e}")
                cause_c_indices.append(0.0)
        else:
            if verbose:
                print("No events found for cause 2")
            cause_c_indices.append(0.0)
        
        # Use mean of cause-specific C-indices
        c_index = np.mean(cause_c_indices) if cause_c_indices else 0.0
        if verbose:
            print(f"Combined C-index (mean of causes): {c_index:.4f}")
        
        # For other metrics, use the first cause (primary endpoint)
        ev = EvalSurv(surv1_df, durations, binary_events_1, censor_surv='km')
        
    else:
        # Standard single-event survival analysis
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        c_index = ev.concordance_td()
        if verbose:
            print(f"C-index: {c_index:.4f}")
    
    # Initialize arrays for bootstrap metrics
    bootstrap_c_index = np.zeros(n_bootstrap)
    bootstrap_integrated_brier_score = np.zeros(n_bootstrap)
    bootstrap_integrated_nbll = np.zeros(n_bootstrap)
    
    # Initialize dictionary for time-dependent metrics
    bootstrap_metrics_by_horizon = {horizon: {
        'brier_score': np.zeros(n_bootstrap),
        'c_index_ipcw': np.zeros(n_bootstrap)
    } for horizon in time_horizons}
    
    # Perform bootstrap to estimate confidence intervals
    for i in range(n_bootstrap):
        if verbose and (i == 0 or i == n_bootstrap - 1 or (i + 1) % 10 == 0):
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
            if verbose:
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
                
                # Calculate time-specific concordance index using IPCW
                # Convert survival predictions to risk scores at the specific time horizon
                # Find the row index corresponding to the time horizon
                
                # First, check if the survival DataFrame has time points as index
                if verbose:
                    print(f"Bootstrap survival DataFrame shape: {bootstrap_surv.shape}")
                    print(f"Bootstrap survival DataFrame index: {bootstrap_surv.index[:5]}...")
                
                # Get a single row of survival probabilities at the specific time horizon
                # The survival DataFrame has time points as rows and patients as columns
                
                # Find the row index corresponding to the time horizon
                time_points = np.array(bootstrap_surv.index.astype(float))
                
                # Find the closest time point to the horizon
                if horizon <= time_points.min():
                    # If horizon is before the first time point, use the first time point
                    idx = 0
                elif horizon >= time_points.max():
                    # If horizon is after the last time point, use the last time point
                    idx = len(time_points) - 1
                else:
                    # Find the closest time point
                    idx = np.abs(time_points - horizon).argmin()
                
                # Get the survival probabilities at the specific time point
                # This should be a 1D array with one value per patient
                surv_probs = bootstrap_surv.iloc[idx].values
                
                # Convert to risk scores (higher score = higher risk)
                risk_scores = 1 - surv_probs
                
                # Ensure risk_scores is a 1D array
                if len(risk_scores.shape) > 1:
                    if verbose:
                        print(f"Warning: risk_scores has shape {risk_scores.shape}, flattening to 1D")
                    risk_scores = risk_scores.flatten()
                
                if verbose:
                    print(f"Risk scores shape: {risk_scores.shape}")
                    print(f"Risk scores first 5 values: {risk_scores[:5]}")
                
                # Create structured array for scikit-survival
                structured_events = np.zeros(len(bootstrap_events), dtype=[('event', bool), ('time', float)])
                structured_events['event'] = bootstrap_events.astype(bool)
                structured_events['time'] = bootstrap_durations
                
                # Calculate concordance index with IPCW
                # The function returns 5 values: cindex, concordant, discordant, tied_risk, tied_time
                # First argument is the training data (used to compute censoring distribution)
                # Second argument is the test data
                # Third argument is the risk scores (higher score = higher risk)
                # tau is the truncation time
                c_index_ipcw, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
                    Surv.from_arrays(events.astype(bool), durations),  # Use original data for censoring distribution
                    Surv.from_arrays(bootstrap_events.astype(bool), bootstrap_durations),  # Use bootstrap data for testing
                    risk_scores,
                    tau=horizon
                )
                
                bootstrap_metrics_by_horizon[horizon]['c_index_ipcw'][i] = c_index_ipcw
                
            except Exception as e:
                if verbose:
                    print(f"Error calculating metrics at {horizon} days for bootstrap {i}: {e}")
                bootstrap_metrics_by_horizon[horizon]['brier_score'][i] = np.nan
                bootstrap_metrics_by_horizon[horizon]['c_index_ipcw'][i] = np.nan
    
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
        
        c_index_ipcw_mean = np.nanmean(bootstrap_metrics_by_horizon[horizon]['c_index_ipcw'])
        c_index_ipcw_lower = np.nanpercentile(bootstrap_metrics_by_horizon[horizon]['c_index_ipcw'], 2.5)
        c_index_ipcw_upper = np.nanpercentile(bootstrap_metrics_by_horizon[horizon]['c_index_ipcw'], 97.5)
        
        metrics_by_horizon[horizon] = {
            'brier_score': {
                'mean': float(brier_score_mean),
                'lower': float(brier_score_lower),
                'upper': float(brier_score_upper)
            },
            'c_index_ipcw': {
                'mean': float(c_index_ipcw_mean),
                'lower': float(c_index_ipcw_lower),
                'upper': float(c_index_ipcw_upper)
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
    
    # Calculate time-specific concordance indices for the original (non-bootstrap) data
    # This is done separately to add to the 'original' field in the results
    for horizon in time_horizons:
        try:
            # Convert survival predictions to risk scores at the specific time horizon
            # Find the row index corresponding to the time horizon
            # The index of the survival DataFrame contains the time points
            time_points = np.array(surv.index.astype(float))
            
            # Find the closest time point to the horizon
            if horizon <= time_points.min():
                # If horizon is before the first time point, use the first time point
                idx = 0
                risk_scores = 1 - surv.iloc[idx].values
            elif horizon >= time_points.max():
                # If horizon is after the last time point, use the last time point
                idx = len(time_points) - 1
                risk_scores = 1 - surv.iloc[idx].values
            else:
                # Find the closest time point
                idx = np.abs(time_points - horizon).argmin()
                
                # If we need more precise interpolation:
                if np.abs(time_points[idx] - horizon) > 1:  # If more than 1 day difference
                    # Find the two closest time points for interpolation
                    if time_points[idx] > horizon and idx > 0:
                        idx_before = idx - 1
                        idx_after = idx
                    elif time_points[idx] < horizon and idx < len(time_points) - 1:
                        idx_before = idx
                        idx_after = idx + 1
                    else:
                        # Just use the closest point if we can't interpolate
                        risk_scores = 1 - surv.iloc[idx].values
                        
                    # Interpolate if we have before and after indices
                    if 'idx_before' in locals() and 'idx_after' in locals():
                        t1, t2 = time_points[idx_before], time_points[idx_after]
                        w1 = (t2 - horizon) / (t2 - t1)
                        w2 = (horizon - t1) / (t2 - t1)
                        # Interpolate survival probabilities
                        surv_probs = w1 * surv.iloc[idx_before].values + w2 * surv.iloc[idx_after].values
                        risk_scores = 1 - surv_probs
                else:
                    # Use the closest time point directly
                    risk_scores = 1 - surv.iloc[idx].values
            
            # Ensure risk_scores is a 1D array
            if len(risk_scores.shape) > 1:
                if verbose:
                    print(f"Warning: risk_scores has shape {risk_scores.shape}, flattening to 1D")
                risk_scores = risk_scores.flatten()
            
            # Create structured array for scikit-survival
            structured_events = np.zeros(len(events), dtype=[('event', bool), ('time', float)])
            structured_events['event'] = events.astype(bool)
            structured_events['time'] = durations
            
            # Calculate concordance index with IPCW
            # The function returns 5 values: cindex, concordant, discordant, tied_risk, tied_time
            # For the original (non-bootstrap) data, we use the same data for both arguments
            # since we don't have separate training and test data
            c_index_ipcw, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
                Surv.from_arrays(events.astype(bool), durations),
                Surv.from_arrays(events.astype(bool), durations),
                risk_scores,
                tau=horizon
            )
            
            # Add the original c_index_ipcw to the results
            results['metrics_by_horizon'][horizon]['c_index_ipcw']['original'] = float(c_index_ipcw)
            
        except Exception as e:
            if verbose:
                print(f"Error calculating original time-specific C-index at {horizon} days: {e}")
            results['metrics_by_horizon'][horizon]['c_index_ipcw']['original'] = np.nan
    
    return results


def analyze_shap_values(shap_values, verbose=False):
    """
    Analyze SHAP values to determine feature importance.
    
    Args:
        shap_values: Dictionary of SHAP values
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame of feature importance
    """
    if shap_values is None:
        return None
    
    if verbose:
        print("Analyzing SHAP values...")
    
    # Create DataFrame with feature names and mean SHAP values
    shap_df = pd.DataFrame({
        'feature': shap_values['feature_names'],
        'mean_shap': shap_values['mean_shap']
    })
    
    # Add confidence intervals if available
    if 'lower_ci' in shap_values and 'upper_ci' in shap_values:
        shap_df['lower_ci'] = shap_values['lower_ci']
        shap_df['upper_ci'] = shap_values['upper_ci']
    
    # Sort by absolute mean SHAP value
    shap_df['abs_mean_shap'] = np.abs(shap_df['mean_shap'])
    shap_df = shap_df.sort_values('abs_mean_shap', ascending=False)
    
    # Print top features
    if verbose:
        print("\nTop 10 features by SHAP value:")
        for i, row in shap_df.head(10).iterrows():
            ci_str = f" (95% CI: {row['lower_ci']:.4f} to {row['upper_ci']:.4f})" if 'lower_ci' in shap_df.columns else ""
            print(f"{row['feature']}: {row['mean_shap']:.4f}{ci_str}")
    
    # Remove abs_mean_shap column
    shap_df = shap_df.drop(columns=['abs_mean_shap'])
    
    return shap_df


def plot_metrics_by_time(metrics, time_horizons=None, output_path=None):
    """
    Create plot of metrics by time horizon, including Brier scores and time-specific concordance indices.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        time_horizons: List of time horizons (default: None, uses metrics_by_horizon keys)
        output_path: Path to save the plot (default: None, displays the plot)
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    # DEBUG: Print the metrics structure being received
    print(f"DEBUG plot_metrics_by_time: Received metrics keys: {list(metrics.keys())}")
    if 'integrated_brier_score' in metrics:
        print(f"DEBUG plot_metrics_by_time: integrated_brier_score = {metrics['integrated_brier_score']}")
    if 'integrated_nbll' in metrics:
        print(f"DEBUG plot_metrics_by_time: integrated_nbll = {metrics['integrated_nbll']}")
    # Extract metrics by horizon
    if time_horizons is None:
        time_horizons = sorted([int(h) for h in metrics['metrics_by_horizon'].keys()])
    
    # Extract Brier scores and C-indices
    brier_scores = []
    brier_scores_lower = []
    brier_scores_upper = []
    c_indices = []
    c_indices_lower = []
    c_indices_upper = []
    
    for horizon in time_horizons:
        if str(horizon) in metrics['metrics_by_horizon']:
            horizon_str = str(horizon)
        else:
            horizon_str = horizon
            
        # Brier score
        brier_scores.append(metrics['metrics_by_horizon'][horizon_str]['brier_score']['mean'])
        brier_scores_lower.append(metrics['metrics_by_horizon'][horizon_str]['brier_score']['lower'])
        brier_scores_upper.append(metrics['metrics_by_horizon'][horizon_str]['brier_score']['upper'])
        
        # C-index (try both c_index_ipcw and c_index for compatibility)
        if 'c_index_ipcw' in metrics['metrics_by_horizon'][horizon_str]:
            c_indices.append(metrics['metrics_by_horizon'][horizon_str]['c_index_ipcw']['mean'])
            c_indices_lower.append(metrics['metrics_by_horizon'][horizon_str]['c_index_ipcw']['lower'])
            c_indices_upper.append(metrics['metrics_by_horizon'][horizon_str]['c_index_ipcw']['upper'])
        elif 'c_index' in metrics['metrics_by_horizon'][horizon_str]:
            # Use c_index if c_index_ipcw is not available (for DeepHit competing risks)
            c_indices.append(metrics['metrics_by_horizon'][horizon_str]['c_index']['mean'])
            c_indices_lower.append(metrics['metrics_by_horizon'][horizon_str]['c_index']['lower'])
            c_indices_upper.append(metrics['metrics_by_horizon'][horizon_str]['c_index']['upper'])
        else:
            # If neither is available, use NaN
            c_indices.append(np.nan)
            c_indices_lower.append(np.nan)
            c_indices_upper.append(np.nan)
    
    # Convert to numpy arrays
    brier_scores = np.array(brier_scores)
    brier_scores_lower = np.array(brier_scores_lower)
    brier_scores_upper = np.array(brier_scores_upper)
    c_indices = np.array(c_indices)
    c_indices_lower = np.array(c_indices_lower)
    c_indices_upper = np.array(c_indices_upper)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Brier scores
    ax1.plot(time_horizons, brier_scores, 'o-', color='blue', label='Brier Score')
    ax1.fill_between(time_horizons, brier_scores_lower, brier_scores_upper, color='blue', alpha=0.2)
    ax1.set_ylabel('Brier Score')
    ax1.set_title('Brier Score by Time Horizon')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Plot C-indices
    if not np.isnan(c_indices).all():
        ax2.plot(time_horizons, c_indices, 'o-', color='green', label='C-index (IPCW)')
        ax2.fill_between(time_horizons, c_indices_lower, c_indices_upper, color='green', alpha=0.2)
    ax2.set_xlabel('Time Horizon (days)')
    ax2.set_ylabel('C-index (IPCW)')
    ax2.set_title('Time-specific Concordance Index by Time Horizon')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    # Add integrated metrics as text with proper error handling
    text_lines = []
    
    # C-index
    if 'c_index' in metrics and isinstance(metrics['c_index'], dict):
        text_lines.append(f"C-index: {metrics['c_index']['mean']:.4f} (95% CI: {metrics['c_index']['lower']:.4f}-{metrics['c_index']['upper']:.4f})")
    elif 'c_index' in metrics:
        text_lines.append(f"C-index: {metrics['c_index']:.4f}")
    
    # Integrated Brier Score
    if 'integrated_brier_score' in metrics:
        if isinstance(metrics['integrated_brier_score'], dict) and 'mean' in metrics['integrated_brier_score']:
            text_lines.append(f"Integrated Brier Score: {metrics['integrated_brier_score']['mean']:.4f} (95% CI: {metrics['integrated_brier_score']['lower']:.4f}-{metrics['integrated_brier_score']['upper']:.4f})")
        else:
            # Handle case where it's stored as a simple float
            brier_val = metrics['integrated_brier_score']
            if isinstance(brier_val, (int, float)):
                text_lines.append(f"Integrated Brier Score: {brier_val:.4f}")
            else:
                text_lines.append(f"Integrated Brier Score: {brier_val}")
    else:
        text_lines.append("Integrated Brier Score: N/A")
    
    # Integrated NBLL
    if 'integrated_nbll' in metrics:
        if isinstance(metrics['integrated_nbll'], dict) and 'mean' in metrics['integrated_nbll']:
            text_lines.append(f"Integrated NBLL: {metrics['integrated_nbll']['mean']:.4f} (95% CI: {metrics['integrated_nbll']['lower']:.4f}-{metrics['integrated_nbll']['upper']:.4f})")
        else:
            # Handle case where it's stored as a simple float
            nbll_val = metrics['integrated_nbll']
            if isinstance(nbll_val, (int, float)):
                text_lines.append(f"Integrated NBLL: {nbll_val:.4f}")
            else:
                text_lines.append(f"Integrated NBLL: {nbll_val}")
    else:
        text_lines.append("Integrated NBLL: N/A")
    
    textstr = '\n'.join(text_lines)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Position the text box in the upper right corner of the top subplot
    # This avoids overlapping with the plot
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save or display the plot
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_shap_values(shap_df, top_n=10, output_path=None):
    """
    Create visualization of SHAP values.
    
    Args:
        shap_df: DataFrame containing SHAP values
        top_n: Number of top features to display (default: 10)
        output_path: Path to save the plot (default: None, displays the plot)
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    if shap_df is None:
        return None
    
    # Sort features by absolute SHAP value
    shap_df = shap_df.copy()
    shap_df['abs_mean_shap'] = np.abs(shap_df['mean_shap'])
    shap_df = shap_df.sort_values('abs_mean_shap', ascending=False)
    
    # Select top N features
    top_features = shap_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(top_features['feature'], top_features['mean_shap'], color=['red' if x < 0 else 'blue' for x in top_features['mean_shap']])
    
    # Add error bars for confidence intervals if available
    if 'lower_ci' in top_features.columns and 'upper_ci' in top_features.columns:
        error_low = top_features['mean_shap'] - top_features['lower_ci']
        error_high = top_features['upper_ci'] - top_features['mean_shap']
        ax.errorbar(top_features['mean_shap'], top_features['feature'],
                   xerr=np.vstack([error_low, error_high]),
                   fmt='none', ecolor='black', capsize=5)
    
    # Set labels and title
    ax.set_xlabel('SHAP Value')
    ax.set_ylabel('Feature')
    ax.set_title('Top Features by SHAP Value')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save or display the plot
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_calibration(predictions, durations, events, time_horizons=None, output_path=None, model_type='deepsurv', event_of_interest=None):
    """
    Create calibration plot for survival predictions.
    
    Args:
        predictions: DataFrame containing survival predictions (for deepsurv) or numpy array for competing risks (for deephit)
        durations: Array of event times
        events: Array of event indicators
        time_horizons: List of time horizons (default: [365, 730, 1095, 1460, 1825])
        output_path: Path to save the plot (default: None, displays the plot)
        model_type: Type of model ('deepsurv' or 'deephit', default: 'deepsurv')
        
    Returns:
        Path to saved plot if output_path is provided, None otherwise
    """
    import traceback
    from lifelines import KaplanMeierFitter, AalenJohansenFitter
    
    # Set default time horizons if not provided
    if time_horizons is None:
        time_horizons = [365, 730, 1095, 1460, 1825]
    
    # Check if this is competing risks format (3D array or flattened 2D)
    is_competing_risks = False
    if isinstance(predictions, np.ndarray) and len(predictions.shape) == 3:
        is_competing_risks = True
        print("Detected 3D competing risks predictions")
    elif isinstance(predictions, pd.DataFrame) and predictions.shape[0] == 10 and model_type == 'deephit':
        is_competing_risks = True
        print("Detected flattened competing risks predictions")
    
    # Handle competing risks format
    if is_competing_risks and model_type == 'deephit':
        print("Using competing risks calibration for DeepHit model")
        return plot_competing_risks_calibration(predictions, durations, events, time_horizons, output_path)
    
    # Validate inputs for standard format
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError(f"Predictions must be a pandas DataFrame, got {type(predictions)}")
    
    if len(predictions.columns) == 0:
        raise ValueError("Predictions DataFrame has no columns")
    
    # Ensure durations and events are numpy arrays
    durations = np.asarray(durations)
    events = np.asarray(events)
    
    # Ensure durations and events are 1D arrays
    if durations.ndim > 1:
        print(f"Warning: Durations array has shape {durations.shape}, flattening to 1D array")
        durations = durations.flatten()
    
    if events.ndim > 1:
        print(f"Warning: Events array has shape {events.shape}, flattening to 1D array")
        events = events.flatten()
    
    # Ensure durations and events have the same length
    if len(durations) != len(events):
        raise ValueError(f"Durations and events must have the same length, got {len(durations)} and {len(events)}")
    
    # Print diagnostic information
    print(f"Predictions DataFrame shape: {predictions.shape}")
    print(f"Durations shape: {durations.shape}")
    print(f"Events shape: {events.shape}")
    
    # Check if the prediction format matches the number of patients
    # For DeepHit: rows are patients, columns are time points
    # For DeepSurv: columns are patients, rows are time points
    if predictions.shape[0] == len(durations):
        print(f"Detected DeepHit format: {predictions.shape} (patients, time_points)")
    elif predictions.shape[1] == len(durations):
        print(f"Detected DeepSurv format: {predictions.shape} (time_points, patients)")
    else:
        print(f"Warning: Predictions shape {predictions.shape} doesn't match expected format")
        print(f"Expected: ({len(durations)}, time_points) for DeepHit or (time_points, {len(durations)}) for DeepSurv")
        print("Proceeding with available data...")
    
    # Define time horizons labels
    horizon_labels = [f"Year {i+1}" for i in range(len(time_horizons))]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(time_horizons), figsize=(20, 6), sharey=True)
    
    # If only one time horizon, make axes iterable
    if len(time_horizons) == 1:
        axes = [axes]
    
    # Plot each time horizon
    for i, (horizon, label) in enumerate(zip(time_horizons, horizon_labels)):
        try:
            # Get predicted risks at this horizon
            # Check if this is DeepHit numpy array format or DeepSurv DataFrame format
            
            if isinstance(predictions, np.ndarray):
                # DeepHit format: numpy array where rows are time horizons [365, 730, 1095, 1460, 1825]
                # Find the index for this horizon in the time_horizons list
                try:
                    horizon_idx = time_horizons.index(horizon)
                    print(f"For horizon {horizon}, using row {horizon_idx} (DeepHit numpy array format)")
                    cif_values = predictions[horizon_idx]  # Get row for this time horizon
                    # For DeepHit competing risks, predictions are already CIF (risk) values, not survival probabilities
                    pred_risks = cif_values  # Use CIF directly as risk
                    print(f"DeepHit CIF values - Min: {pred_risks.min():.6f}, Max: {pred_risks.max():.6f}, Mean: {pred_risks.mean():.6f}")
                except ValueError:
                    print(f"Warning: Horizon {horizon} not found in time_horizons {time_horizons}")
                    continue
            elif hasattr(predictions, 'columns') and horizon in predictions.columns:
                # DeepHit DataFrame format: columns are time horizons [365, 730, 1095, 1460, 1825]
                print(f"For horizon {horizon}, using column {horizon} (DeepHit DataFrame format)")
                cif_values = predictions[horizon].values  # Get column for this time horizon
                # For DeepHit competing risks, predictions are already CIF (risk) values, not survival probabilities
                pred_risks = cif_values  # Use CIF directly as risk
                print(f"DeepHit CIF values - Min: {pred_risks.min():.6f}, Max: {pred_risks.max():.6f}, Mean: {pred_risks.mean():.6f}")
            else:
                # DeepSurv format: index is time points, columns are patients
                # Get the index (row) values as time points
                time_points = predictions.index.values
                
                # Find the closest time point in the predictions
                closest_time_idx = np.argmin(np.abs(time_points - horizon))
                closest_time = time_points[closest_time_idx]
                
                print(f"For horizon {horizon}, using time point {closest_time} (index {closest_time_idx}) (DeepSurv format)")
                
                # Get predicted risks (1 - survival probability) for all patients at this time point
                # Extract the row corresponding to the closest time point
                surv_probs = predictions.iloc[closest_time_idx].values
                pred_risks = 1 - surv_probs
            
            # Ensure pred_risks is a 1D array
            if pred_risks.ndim > 1:
                print(f"Warning: Predicted risks for time {horizon} have shape {pred_risks.shape}, flattening to 1D array")
                pred_risks = pred_risks.flatten()
            
            # Ensure pred_risks has the same length as durations and events
            if len(pred_risks) != len(durations):
                raise ValueError(f"Predicted risks length ({len(pred_risks)}) does not match durations length ({len(durations)})")
            
            # Create quantiles based on predicted risks
            n_quantiles = 10
            
            # Use pandas qcut to create quantiles
            try:
                quantiles = pd.qcut(pred_risks, n_quantiles, labels=False, duplicates='drop')
                quantiles = np.asarray(quantiles)
                unique_quantiles = np.unique(quantiles)
                n_quantiles = len(unique_quantiles)
            except Exception as e:
                print(f"Warning: Could not create quantiles: {e}")
                print(f"pred_risks min: {np.min(pred_risks)}, max: {np.max(pred_risks)}, mean: {np.mean(pred_risks)}")
                print(f"pred_risks has {np.sum(np.isnan(pred_risks))} NaN values")
                print(f"pred_risks has {np.sum(np.isinf(pred_risks))} Inf values")
                raise ValueError(f"Could not create quantiles: {e}")
            
            # Initialize arrays for observed risks
            observed_risks = np.zeros(n_quantiles)
            mean_predicted_risks = np.zeros(n_quantiles)
            
            # Process each quantile
            for q_idx, q in enumerate(unique_quantiles):
                # Get indices for this quantile
                q_indices = np.where(quantiles == q)[0]
                
                # Check if there are enough samples in this quantile
                if len(q_indices) < 5:
                    print(f"Warning: Quantile {q_idx+1} for time {horizon} has only {len(q_indices)} samples, which may be too few for reliable estimation")
                    observed_risks[q_idx] = np.nan
                    mean_predicted_risks[q_idx] = np.mean(pred_risks[q_indices]) if len(q_indices) > 0 else np.nan
                    continue
                
                # Ensure indices are within bounds
                valid_indices = q_indices[q_indices < len(durations)]
                if len(valid_indices) < len(q_indices):
                    print(f"Warning: {len(q_indices) - len(valid_indices)} indices were out of bounds and removed")
                
                if len(valid_indices) < 5:
                    print(f"Warning: After removing out-of-bounds indices, quantile {q_idx+1} has only {len(valid_indices)} samples")
                    observed_risks[q_idx] = np.nan
                    mean_predicted_risks[q_idx] = np.mean(pred_risks[valid_indices]) if len(valid_indices) > 0 else np.nan
                    continue
                
                # Calculate mean predicted risk for this quantile
                mean_predicted_risks[q_idx] = np.mean(pred_risks[valid_indices])
                
                # Calculate observed risk using appropriate estimator
                try:
                    if model_type.lower() == 'deepsurv':
                        # Use Kaplan-Meier estimator for binary outcomes
                        kmf = KaplanMeierFitter()
                        kmf.fit(durations[valid_indices], event_observed=events[valid_indices])
                        
                        # Get survival probability at horizon
                        try:
                            surv_prob = kmf.predict(horizon)
                            observed_risks[q_idx] = 1 - surv_prob
                        except Exception as e:
                            print(f"Warning: Could not predict survival probability at horizon {horizon} for quantile {q_idx+1}: {e}")
                            observed_risks[q_idx] = np.nan
                    
                    elif model_type.lower() == 'deephit':
                        # CORRECT DEEPHIT WORKFLOW: Use Aalen-Johansen estimator for competing risks
                        # Following the user's specified workflow:
                        # 1. Use transformed discrete durations/events
                        # 2. Fit Aalen-Johansen and extract cumulative_density_
                        # 3. Handle prediction shape (n_samples, 5) for each event
                        # 4. Use proper quantile cutting and observed risk calculation
                        
                        ajf = AalenJohansenFitter()
                        target_event = event_of_interest if event_of_interest is not None else 1
                        
                        print(f"DeepHit calibration workflow for quantile {q_idx+1}:")
                        print(f"  Using event_of_interest={target_event}")
                        print(f"  Valid indices count: {len(valid_indices)}")
                        print(f"  Duration range: [{durations[valid_indices].min():.1f}, {durations[valid_indices].max():.1f}]")
                        print(f"  Event range: [{events[valid_indices].min():.1f}, {events[valid_indices].max():.1f}]")
                        
                        # Step 2: Fit Aalen-Johansen with transformed discrete durations and events
                        ajf.fit(durations[valid_indices], events[valid_indices], event_of_interest=target_event)
                        
                        # Step 3: Calculate observed risk for this quantile using Aalen-Johansen
                        # The AJ estimator is fitted on the samples in this quantile
                        # We need to get the CIF at the specific horizon for this quantile
                        try:
                            # For discrete time, we need to map horizon to discrete time index
                            # horizon is in days (365, 730, etc.), but discrete durations are [0, 1, 2, 3, 4]
                            time_grid = [365, 730, 1095, 1460, 1825]  # Standard time grid
                            
                            if horizon in time_grid:
                                discrete_time_idx = time_grid.index(horizon)
                                print(f"  Horizon {horizon} maps to discrete time index {discrete_time_idx}")
                                
                                # Use the predict method to get CIF at the discrete time point
                                # For discrete time, we predict at the discrete time index
                                observed_cif = ajf.predict(discrete_time_idx)
                                observed_risks[q_idx] = observed_cif
                                print(f"  Quantile {q_idx+1}: Observed CIF = {observed_risks[q_idx]:.6f}")
                            else:
                                print(f"  Warning: Horizon {horizon} not in standard time grid")
                                observed_risks[q_idx] = np.nan
                                
                        except Exception as e:
                            print(f"  Warning: Could not predict CIF for quantile {q_idx+1}: {e}")
                            print(f"  Error details: {str(e)}")
                            observed_risks[q_idx] = np.nan
                    
                    else:
                        raise ValueError(f"Unknown model type: {model_type}. Must be 'deepsurv' or 'deephit'")
                
                except Exception as e:
                    print(f"Warning: Could not fit estimator for quantile {q_idx+1} at horizon {horizon}: {e}")
                    observed_risks[q_idx] = np.nan
            
            # Convert to percentages
            mean_predicted_risks *= 100
            observed_risks *= 100
            
            # Set up bar positions
            bar_width = 0.35
            index = np.arange(n_quantiles)
            
            # Plot predicted risks
            axes[i].bar(index - bar_width/2, mean_predicted_risks, bar_width,
                        label='Predicted Risk', color='blue', alpha=0.7)
            
            # Plot observed risks
            axes[i].bar(index + bar_width/2, observed_risks, bar_width,
                        label='Observed Risk', color='red', alpha=0.7)
            
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
            error_msg = f"Error creating calibration plot for {horizon} days: {e}"
            print(error_msg)
            print(traceback.format_exc())
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[i].transAxes)
    
    # Add legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Add title to the figure
    fig.suptitle(f'Calibration Plot ({model_type.capitalize()})', fontsize=16)
    
    # Save or display the plot
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, output_dir, verbose=False, model_type='deepsurv'):
    """
    Create visualizations for the evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        shap_df: DataFrame of SHAP values
        predictions: Survival predictions DataFrame
        durations: Event times
        events: Event indicators
        time_horizons: List of time horizons
        output_dir: Directory to save visualizations
        verbose: Whether to print detailed information
        model_type: Type of model ('deepsurv' or 'deephit', default: 'deepsurv')
        
    Returns:
        Dictionary of paths to saved visualizations
    """
    if verbose:
        print("Creating visualizations...")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dictionary to store paths to saved visualizations
    visualization_paths = {}
    
    # Create metrics by time plot
    metrics_plot_path = os.path.join(output_dir, f"metrics_by_time_{timestamp}.png")
    metrics_plot_path = plot_metrics_by_time(metrics, time_horizons, metrics_plot_path)
    visualization_paths['metrics_by_time'] = metrics_plot_path
    
    if verbose:
        print(f"Metrics by time plot saved to: {metrics_plot_path}")
    
    # Create SHAP values plot if available
    if shap_df is not None:
        shap_plot_path = os.path.join(output_dir, f"shap_values_{timestamp}.png")
        shap_plot_path = plot_shap_values(shap_df, top_n=10, output_path=shap_plot_path)
        visualization_paths['shap_values'] = shap_plot_path
        
        if verbose:
            print(f"SHAP values plot saved to: {shap_plot_path}")
    
    # Create calibration plot
    calibration_plot_path = os.path.join(output_dir, f"calibration_{timestamp}.png")
    calibration_plot_path = plot_calibration(predictions, durations, events, time_horizons, calibration_plot_path, model_type=model_type)
    visualization_paths['calibration'] = calibration_plot_path
    
    if verbose:
        print(f"Calibration plot saved to: {calibration_plot_path}")
    
    return visualization_paths


def save_results(metrics, shap_df, visualization_paths, output_dir, verbose=False):
    """
    Save results to files.
    
    Args:
        metrics: Dictionary of metrics
        shap_df: DataFrame of SHAP values
        visualization_paths: Dictionary of paths to saved visualizations
        output_dir: Directory to save results
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary of paths to saved files
    """
    if verbose:
        print("Saving results...")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if verbose:
        print(f"Metrics saved to: {metrics_path}")
    
    # Save SHAP values to CSV if available
    shap_path = None
    if shap_df is not None:
        shap_path = os.path.join(output_dir, f"shap_values_{timestamp}.csv")
        shap_df.to_csv(shap_path, index=False)
        
        if verbose:
            print(f"SHAP values saved to: {shap_path}")
    
    # Create results dictionary
    results_paths = {
        'metrics': metrics_path,
        'shap_values': shap_path,
        'visualizations': visualization_paths
    }
    
    # Save results paths to JSON
    results_paths_path = os.path.join(output_dir, f"results_paths_{timestamp}.json")
    with open(results_paths_path, 'w') as f:
        json.dump(results_paths, f, indent=2)
    
    if verbose:
        print(f"Results paths saved to: {results_paths_path}")
    
    return results_paths


def main():
    """Main function."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate output directory
        validate_output_directory(args.output_dir)
        
        # Load and validate data
        predictions, durations, events, shap_values = load_data(args)
        
        # Validate data consistency
        validate_data_consistency(predictions, durations, events)
        
        # Parse and validate time horizons
        time_horizons = [int(t) for t in args.time_horizons.split(',')]
        validate_time_horizons(time_horizons, durations)
        
        # Calculate metrics
        try:
            metrics = calculate_metrics(predictions, durations, events, time_horizons, args.n_bootstrap, args.verbose)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Analyze SHAP values
        shap_df = analyze_shap_values(shap_values, args.verbose)
        
        # Create visualizations if requested
        visualization_paths = {}
        if args.visualize:
            visualization_paths = create_visualizations(metrics, shap_df, predictions, durations, events, time_horizons, args.output_dir, args.verbose, model_type=args.model_type)
        
        # Save results
        results_paths = save_results(metrics, shap_df, visualization_paths, args.output_dir, args.verbose)
        
        # Print summary
        if args.verbose:
            print("\nSummary:")
            print(f"C-index: {metrics['c_index']['mean']:.4f} (95% CI: {metrics['c_index']['lower']:.4f}-{metrics['c_index']['upper']:.4f})")
            print(f"Integrated Brier Score: {metrics['integrated_brier_score']['mean']:.4f} (95% CI: {metrics['integrated_brier_score']['lower']:.4f}-{metrics['integrated_brier_score']['upper']:.4f})")
            print(f"Integrated NBLL: {metrics['integrated_nbll']['mean']:.4f} (95% CI: {metrics['integrated_nbll']['lower']:.4f}-{metrics['integrated_nbll']['upper']:.4f})")
            
            if shap_df is not None:
                print("\nTop 5 features by SHAP value:")
                for i, row in shap_df.head(5).iterrows():
                    ci_str = f" (95% CI: {row['lower_ci']:.4f} to {row['upper_ci']:.4f})" if 'lower_ci' in shap_df.columns else ""
                    print(f"{row['feature']}: {row['mean_shap']:.4f}{ci_str}")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def plot_competing_risks_calibration(predictions, durations, events, time_horizons=None, output_path=None):
    """
    Create calibration plots for competing risks predictions matching DeepSurv format.
    Creates 2 separate plots (Event 1 and Event 2), each with 5 subplots (Year 1-5).
    
    Args:
        predictions: Array or DataFrame containing competing risks predictions
        durations: Array of event times
        events: Array of event indicators (0=censored, 1=cause1, 2=cause2)
        time_horizons: List of time horizons (default: [365, 730, 1095, 1460, 1825])
        output_path: Base path to save the plots (will create 2 files)
        
    Returns:
        List of paths to saved plots if output_path is provided, None otherwise
    """
    from lifelines import AalenJohansenFitter
    
    # Set default time horizons if not provided
    if time_horizons is None:
        time_horizons = [365, 730, 1095, 1460, 1825]
    
    # Convert to numpy arrays
    durations = np.asarray(durations)
    events = np.asarray(events)
    
    # Handle different prediction formats
    if isinstance(predictions, pd.DataFrame):
        # Flattened format (10, n_samples) - reshape to (2, 5, n_samples)
        predictions_array = predictions.values
        if predictions_array.shape[0] == 10:
            n_samples = predictions_array.shape[1]
            predictions_array = predictions_array.reshape(2, 5, n_samples)
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions_array.shape}")
    else:
        # Already in 3D format
        predictions_array = np.asarray(predictions)
    
    print(f"Competing risks predictions shape: {predictions_array.shape}")
    
    # Extract CIF for each cause
    cif_cause1 = predictions_array[0]  # Shape: (5, n_samples)
    cif_cause2 = predictions_array[1]  # Shape: (5, n_samples)
    
    # Define cause information
    cause_info = [
        {'name': 'Event 1 (RRT/eGFR<15)', 'cif': cif_cause1, 'event_value': 1},
        {'name': 'Event 2 (Mortality)', 'cif': cif_cause2, 'event_value': 2}
    ]
    
    saved_paths = []
    
    # Create separate calibration plot for each cause (matching DeepSurv format)
    for cause_idx, cause in enumerate(cause_info):
        # Create figure with 5 subplots (one for each year)
        fig, axes = plt.subplots(1, len(time_horizons), figsize=(20, 6), sharey=True)
        
        # If only one time horizon, make axes iterable
        if len(time_horizons) == 1:
            axes = [axes]
        
        # Plot each time horizon
        for time_idx, horizon in enumerate(time_horizons):
            ax = axes[time_idx]
            
            try:
                # Get predicted risks (CIF values) for this cause and time horizon
                pred_risks = cause['cif'][time_idx]  # CIF values are already risk probabilities
                
                # Create quantiles based on predicted risks (same method as DeepSurv)
                n_quantiles = 10
                
                # Use pandas qcut to create quantiles
                quantile_labels = pd.qcut(pred_risks, n_quantiles, labels=False, duplicates='drop')
                unique_quantiles = np.unique(quantile_labels)
                
                observed_risks = []
                predicted_risks = []
                
                for q in unique_quantiles:
                    if pd.isna(q):
                        continue
                    
                    # Get patients in this quantile
                    mask = (quantile_labels == q)
                    
                    if np.sum(mask) == 0:
                        continue
                    
                    # Calculate observed risk using Aalen-Johansen estimator for competing risks
                    quantile_durations = durations[mask]
                    quantile_events = events[mask]  # Use original events (0, 1, 2)
                    
                    # Calculate observed risk at this time horizon using Aalen-Johansen
                    try:
                        # Fit Aalen-Johansen estimator for competing risks
                        aj = AalenJohansenFitter()
                        aj.fit(quantile_durations, quantile_events, event_of_interest=cause['event_value'])
                        
                        # Get cumulative incidence at horizon
                        if horizon in aj.cumulative_density_.index:
                            observed_risk = aj.cumulative_density_.loc[horizon].values[0]
                        else:
                            # Interpolate or use closest value
                            closest_time = aj.cumulative_density_.index[
                                np.argmin(np.abs(aj.cumulative_density_.index - horizon))
                            ]
                            observed_risk = aj.cumulative_density_.loc[closest_time].values[0]
                    except Exception as e:
                        print(f"Error with Aalen-Johansen for quantile {q}: {e}")
                        observed_risk = 0.0
                    
                    # Calculate mean predicted risk for this quantile
                    mean_pred_risk = np.mean(pred_risks[mask])
                    
                    observed_risks.append(observed_risk)
                    predicted_risks.append(mean_pred_risk)
                
                # Plot calibration (same style as DeepSurv)
                if len(observed_risks) > 0:
                    ax.scatter(predicted_risks, observed_risks, alpha=0.7, s=50)
                    
                    # Plot perfect calibration line
                    min_risk = min(min(predicted_risks), min(observed_risks))
                    max_risk = max(max(predicted_risks), max(observed_risks))
                    ax.plot([min_risk, max_risk], [min_risk, max_risk], 'r--', alpha=0.8, label='Perfect calibration')
                    
                    ax.set_xlabel('Predicted Risk')
                    if time_idx == 0:  # Only label y-axis on first subplot
                        ax.set_ylabel('Observed Risk')
                    ax.set_title(f'Year {time_idx + 1}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Set axis limits - fix for very low risk predictions
                    # Don't force minimum of 0.1 when risks are much smaller
                    if max_risk < 0.01:  # If max risk is less than 1%
                        # Use proportional scaling for very low risks
                        axis_limit = max_risk * 1.2
                    else:
                        # Use original logic for higher risks
                        axis_limit = max(max_risk * 1.1, 0.1)
                    
                    ax.set_xlim(0, axis_limit)
                    ax.set_ylim(0, axis_limit)
                else:
                    ax.text(0.5, 0.5, 'No events\nfor this cause',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Year {time_idx + 1}')
                
            except Exception as e:
                print(f"Error creating calibration plot for {cause['name']}, horizon {horizon}: {e}")
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Year {time_idx + 1}')
        
        # Set overall title for the figure
        fig.suptitle(f'Calibration Plot - {cause["name"]}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save plot
        if output_path:
            # Create separate file for each cause with proper filename sanitization
            sanitized_name = cause["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("<", "").replace(">", "")
            cause_output_path = output_path.replace('.png', f'_{sanitized_name}.png')
            plt.savefig(cause_output_path, dpi=300, bbox_inches='tight')
            saved_paths.append(cause_output_path)
            plt.close()
        else:
            plt.show()
    
    return saved_paths if output_path else None


if __name__ == "__main__":
    main()
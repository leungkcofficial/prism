"""
Data Splitting Step for CKD Risk Prediction

This module contains the ZenML step for splitting data into training, validation, and test sets.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import random
import os

@step
def split_data(
    raw_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    temporal_cutoff_date: Optional[str] = None,
    spatial_test_ratio: Optional[float] = None,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the merged dataframe into training, temporal test, and spatial test sets.
    
    The splitting has two components:
    1. Temporal isolation: Patients with any data after the cutoff date are isolated as the temporal test set
    2. Spatial isolation: From the remaining data, an additional percentage of patients are isolated as the spatial test set
    
    Args:
        prediction_df: The merged dataframe containing all patient data
        temporal_cutoff_date: The cutoff date for temporal isolation (default: "2022-01-01")
        spatial_test_ratio: The ratio of remaining patients to isolate for spatial testing (default: 0.10)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple containing:
        - train_df: DataFrame for training
        - temporal_test_df: DataFrame for temporal testing (patients with data after cutoff date)
        - spatial_test_df: DataFrame for spatial testing (additional patients isolated from training)
    """
    try:
        # remove any patient keys that only have one row in both prediction_df and raw_df
        print("\n=== Removing patients with only one row in both prediction_df and raw_df ===\n")
        prediction_patient_counts = prediction_df['key'].value_counts()
        raw_patient_counts = raw_df['key'].value_counts()
        single_row_patients = set(prediction_patient_counts[prediction_patient_counts == 1].index) & \
                              set(raw_patient_counts[raw_patient_counts == 1].index)
        print(f"Number of patients with only one row in both dataframes: {len(single_row_patients)}")
        prediction_df = prediction_df[~prediction_df['key'].isin(single_row_patients)].copy()
        raw_df = raw_df[~raw_df['key'].isin(single_row_patients)].copy()
        print(f"New prediction_df shape after removing single row patients: {prediction_df.shape}")
        print(f"New raw_df shape after removing single row patients: {raw_df.shape}")
        # Check if the prediction_df is empty after filtering
        if prediction_df.empty:
            print("Warning: prediction_df is empty after removing single row patients. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if raw_df.empty:
            print("Warning: raw_df is empty after removing single row patients. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Print initial shapes and unique patient counts
        print("Initial data shapes and unique patient counts:")
        print(f"Prediction DataFrame shape: {prediction_df.shape}, Unique patients: {prediction_df['key'].nunique()}")
        print(f"Raw DataFrame shape: {raw_df.shape}, Unique patients: {raw_df['key'].nunique()}")        
        print("\n=== Splitting data into training and test sets ===\n")
        print(f"Original dataframe shape: {prediction_df.shape}")
        print(f"Number of unique patients: {prediction_df['key'].nunique()}")
        
        # Get parameters from environment variables if not provided
        if temporal_cutoff_date is None:
            temporal_cutoff_date = os.getenv("TEMPORAL_CUTOFF_DATE", "2022-01-01")
            print(f"Using temporal cutoff date from environment: {temporal_cutoff_date}")
        
        if spatial_test_ratio is None:
            try:
                spatial_test_ratio = float(os.getenv("SPATIAL_TEST_RATIO", "0.10"))
            except (ValueError, TypeError):
                spatial_test_ratio = 0.10
            print(f"Using spatial test ratio from environment: {spatial_test_ratio}")
        
        if random_seed is None:
            try:
                random_seed = int(os.getenv("RANDOM_SEED", "42"))
            except (ValueError, TypeError):
                random_seed = 42
            print(f"Using random seed from environment: {random_seed}")
        
        # Ensure date column is in datetime format
        if 'date' in prediction_df.columns and not pd.api.types.is_datetime64_any_dtype(prediction_df['date']):
            prediction_df['date'] = pd.to_datetime(prediction_df['date'], errors='coerce')
        
        # Convert cutoff date to datetime
        cutoff_date = pd.to_datetime(temporal_cutoff_date)
        print(f"Temporal cutoff date: {cutoff_date}")
        
        # A. Temporal isolation
        # Identify patients whose earliest data is after the cutoff date
        if 'date' in prediction_df.columns:
            # Get the minimum date for each patient
            min_dates = prediction_df.groupby('key')['date'].min().reset_index()
            print(f"Found minimum dates for {len(min_dates)} patients")
            
            # Identify patients whose minimum date is after the cutoff date
            future_patients = set(min_dates[min_dates['date'] > cutoff_date]['key'].unique())
            print(f"Number of patients whose earliest data is after {temporal_cutoff_date}: {len(future_patients)}")
            
            # Split data into temporal test set and remaining data
            temporal_test_df = prediction_df[prediction_df['key'].isin(future_patients)].copy()
            remaining_df = prediction_df[~prediction_df['key'].isin(future_patients)].copy()
            
            print(f"Temporal test set shape: {temporal_test_df.shape}")
            print(f"Remaining data shape: {remaining_df.shape}")
        else:
            print("No 'date' column found in the dataframe. Skipping temporal isolation.")
            temporal_test_df = pd.DataFrame()
            remaining_df = prediction_df.copy()
        
        # B. Spatial isolation
        # Get unique patient keys from remaining data
        remaining_patients = list(remaining_df['key'].unique())
        print(f"Number of remaining patients: {len(remaining_patients)}")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Calculate number of patients for spatial test set
        n_spatial_test = int(len(remaining_patients) * spatial_test_ratio)
        print(f"Number of patients for spatial test set: {n_spatial_test}")
        
        # Randomly select patients for spatial test set
        spatial_test_patients = set(np.random.choice(remaining_patients, size=n_spatial_test, replace=False))
        print(f"Selected {len(spatial_test_patients)} patients for spatial test set")
        
        # Split remaining data into spatial test set and training set
        spatial_test_df = remaining_df[remaining_df['key'].isin(spatial_test_patients)].copy()
        train_df = remaining_df[~remaining_df['key'].isin(spatial_test_patients)].copy()
        
        print(f"Spatial test set shape: {spatial_test_df.shape}")
        print(f"Training set shape: {train_df.shape}")
        
        # Verify that the splits are disjoint
        train_patients = set(train_df['key'].unique())
        temporal_test_patients = set(temporal_test_df['key'].unique())
        spatial_test_patients = set(spatial_test_df['key'].unique())
        
        print("\n=== Verification of data splits ===")
        print(f"Number of patients in training set: {len(train_patients)}")
        print(f"Number of patients in temporal test set: {len(temporal_test_patients)}")
        print(f"Number of patients in spatial test set: {len(spatial_test_patients)}")
        
        # Check for overlaps
        temporal_train_overlap = len(temporal_test_patients.intersection(train_patients))
        spatial_train_overlap = len(spatial_test_patients.intersection(train_patients))
        temporal_spatial_overlap = len(temporal_test_patients.intersection(spatial_test_patients))
        
        print(f"Overlap between temporal test and training sets: {temporal_train_overlap}")
        print(f"Overlap between spatial test and training sets: {spatial_train_overlap}")
        print(f"Overlap between temporal and spatial test sets: {temporal_spatial_overlap}")
        
        # Calculate total percentage of data in each set
        total_rows = len(prediction_df)
        print(f"\nPercentage of data in training set: {len(train_df) / total_rows * 100:.2f}%")
        print(f"Percentage of data in temporal test set: {len(temporal_test_df) / total_rows * 100:.2f}%")
        print(f"Percentage of data in spatial test set: {len(spatial_test_df) / total_rows * 100:.2f}%")
        
        return train_df, temporal_test_df, spatial_test_df, raw_df
    
    except Exception as e:
        print(f"Error splitting data: {e}")
        # Return empty DataFrames if there's an error
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
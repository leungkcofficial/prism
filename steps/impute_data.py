"""
Data Imputing Step for CKD Risk Prediction

This module contains the ZenML step for imputing data in training, validation, and test sets.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import random
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import xgboost as xgb

@step
def impute_data(
    train_df: pd.DataFrame,
    temporal_test_df: Optional[pd.DataFrame] = None,
    spatial_test_df: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute missing data in training, temporal test, and spatial test sets using MICE and other strategies.
    for 'hard truth' in datasets such as dob, gender, endpoint, endpoint date ...etc, can use other rows' detail with same patient key
    for medical history related columns (e.g. CCI, hypertension related), can be forward filled within same patient key
    for laboratory investigation columns (e.g. 'creatinine', 'hemoglobin', 'a1c', 'albumin', 'phosphate', 'calcium' ...etc), use MICE to impute missing values.
    
    Args:
        train_df: DataFrame for training
        temporal_test_df: DataFrame for temporal testing (patients with data after cutoff date)
        spatial_test_df: DataFrame for spatial testing (additional patients isolated from training)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple containing:
        - train_df_imputed: DataFrame for training with imputed values
        - temporal_test_df_imputed: DataFrame for temporal testing with imputed values
        - spatial_test_df_imputed: DataFrame for spatial testing with imputed values
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    warnings.filterwarnings("ignore")
    
    print("\n=== Starting data imputation ===\n")
    
    # Initialize imputed dataframes
    train_df_imputed = train_df.copy() if train_df is not None else pd.DataFrame()
    temporal_test_df_imputed = temporal_test_df.copy() if temporal_test_df is not None else pd.DataFrame()
    spatial_test_df_imputed = spatial_test_df.copy() if spatial_test_df is not None else pd.DataFrame()
    
    # Check if dataframes are empty
    if train_df_imputed.empty:
        print("Warning: Training dataframe is empty. Skipping imputation.")
        return train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed
    
    # Get column categories from environment variables or use defaults
    # Read hard truth columns from environment
    hard_truth_env = os.getenv("HARD_TRUTH_COLUMNS", "dob,gender,endpoint,endpoint_date,endpoint_source,first_sub_60_date")
    hard_truth_columns = [col.strip() for col in hard_truth_env.split(",")]
    
    # Read medical history columns pattern from environment
    med_history_pattern = os.getenv("MED_HISTORY_PATTERN", "cci_")
    med_history_specific = os.getenv("MED_HISTORY_SPECIFIC", "myocardial_infarction, congestive_heart_failure, peripheral_vascular_disease, cerebrovascular_disease, dementia, chronic_pulmonary_disease, rheumatic_disease, peptic_ulcer_disease, mild_liver_disease, diabetes_wo_complication, renal_mild_moderate, diabetes_w_complication, hemiplegia_paraplegia, any_malignancy, liver_severe, renal_severe, hiv, metastatic_cancer, aids, cci_score_total, ht")
    med_history_specific_list = [col.strip() for col in med_history_specific.split(",")]
    
    # Combine pattern-based and specific medical history columns
    medical_history_columns = [col for col in train_df.columns if
                              any(col.startswith(pattern.strip()) for pattern in med_history_pattern.split(","))]
    medical_history_columns.extend([col for col in med_history_specific_list if col in train_df.columns])
    
    # Read laboratory investigation columns from environment
    lab_columns_env = os.getenv("LAB_COLUMNS", "creatinine,hemoglobin,a1c,albumin,phosphate,calcium,ca_adjusted,upcr,uacr,egfr,bicarbonate")
    lab_columns = [col.strip() for col in lab_columns_env.split(",")]
    
    # Filter to only include columns that exist in the dataframe
    hard_truth_columns = [col for col in hard_truth_columns if col in train_df.columns]
    medical_history_columns = [col for col in medical_history_columns if col in train_df.columns]
    lab_columns = [col for col in lab_columns if col in train_df.columns]
    
    print(f"Hard truth columns: {hard_truth_columns}")
    print(f"Medical history columns: {medical_history_columns}")
    print(f"Laboratory investigation columns: {lab_columns}")
    
    # Print environment variable information
    print("\n=== Environment Variable Configuration ===")
    print(f"HARD_TRUTH_COLUMNS: {os.getenv('HARD_TRUTH_COLUMNS', 'Not set, using default')}")
    print(f"MED_HISTORY_PATTERN: {os.getenv('MED_HISTORY_PATTERN', 'Not set, using default')}")
    print(f"MED_HISTORY_SPECIFIC: {os.getenv('MED_HISTORY_SPECIFIC', 'Not set, using default')}")
    print(f"LAB_COLUMNS: {os.getenv('LAB_COLUMNS', 'Not set, using default')}")
    print(f"STUDY_END_DATE: {os.getenv('STUDY_END_DATE', 'Not set, using default')}")
    
    # Initialize MICE imputer and scaler (will be fitted on training data)
    mice_imputer = None
    scaler = None
    impute_cols = None
    
    def impute_hard_truth_and_medical_history(df: pd.DataFrame) -> pd.DataFrame:
        """Impute hard truth and medical history columns in a DataFrame."""
        if df.empty:
            return df
        
        df_imputed = df.copy()
        
        # 1. Impute 'hard truth' data using other rows with the same patient key
        print("\n=== Imputing hard truth data ===\n")
        for col in hard_truth_columns:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                missing_before = df_imputed[col].isna().sum()
                
                # Group by patient key and fill missing values with the most common non-null value
                df_imputed[col] = df_imputed.groupby('key')[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
                )
                
                missing_after = df_imputed[col].isna().sum()
                print(f"  {col}: {missing_before - missing_after} values imputed, {missing_after} still missing")
        
        # Special case: Impute endpoint_date with study end date for records with endpoint=0
        if 'endpoint' in df_imputed.columns and 'endpoint_date' in df_imputed.columns:
            # Get study end date from environment variable
            study_end_date_str = os.getenv('STUDY_END_DATE', '2023-12-31')
            study_end_date = pd.to_datetime(study_end_date_str)
            
            # Count missing endpoint_date values for endpoint=0 records
            missing_endpoint_dates = ((df_imputed['endpoint'] == 0) & df_imputed['endpoint_date'].isna()).sum()
            
            if missing_endpoint_dates > 0:
                print(f"\n=== Imputing endpoint_date for censored records (endpoint=0) ===")
                print(f"  Found {missing_endpoint_dates} censored records with missing endpoint_date")
                print(f"  Using study end date: {study_end_date_str}")
                
                # Impute endpoint_date with study end date for records with endpoint=0
                df_imputed.loc[(df_imputed['endpoint'] == 0) & df_imputed['endpoint_date'].isna(), 'endpoint_date'] = study_end_date
                
                # Verify imputation
                remaining_missing = ((df_imputed['endpoint'] == 0) & df_imputed['endpoint_date'].isna()).sum()
                print(f"  Imputed {missing_endpoint_dates - remaining_missing} endpoint_date values, {remaining_missing} still missing")
        
        # 2. Forward fill medical history related columns within the same patient key
        print("\n=== Forward filling medical history data ===\n")
        for col in medical_history_columns:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                missing_before = df_imputed[col].isna().sum()
                
                # Sort by key and date, then forward fill within each patient group
                df_imputed = df_imputed.sort_values(['key', 'date'])
                df_imputed[col] = df_imputed.groupby('key')[col].ffill()
                
                # Also backward fill to handle cases where the first record has missing values
                df_imputed[col] = df_imputed.groupby('key')[col].bfill()
                
                missing_after = df_imputed[col].isna().sum()
                print(f"  {col}: {missing_before - missing_after} values imputed, {missing_after} still missing")
        
        return df_imputed
    
    def fit_mice_imputer(df: pd.DataFrame):
        """Fit MICE imputer and scaler on training data."""
        nonlocal mice_imputer, scaler, impute_cols
        
        print("\n=== Fitting MICE imputer on training data ===\n")
        
        # Check if there are any lab columns with missing values
        lab_cols_with_missing = [col for col in lab_columns
                                if col in df.columns and df[col].isna().any()]
        
        if not lab_cols_with_missing:
            print("No laboratory columns with missing values found in training data.")
            return None
        
        # Get numeric columns for MICE imputation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove non-lab columns that shouldn't be used for imputation
        exclude_cols = ['key', 'endpoint', 'endpoint_renal', 'endpoint_death', 'endpoint_censored']
        impute_cols = [col for col in numeric_cols if col not in exclude_cols and col in df.columns]
        
        # Create a subset of the dataframe with only the columns for imputation
        impute_df = df[impute_cols].copy()
        
        # Report missing values before imputation
        for col in lab_cols_with_missing:
            missing_before = impute_df[col].isna().sum()
            print(f"  {col}: {missing_before} missing values before imputation")
        
        # Initialize and fit the MICE imputer
        mice_imputer = IterativeImputer(
            estimator=xgb.XGBRegressor(tree_method='hist', device='cuda'),
            initial_strategy='mean',
            max_iter=10,
            random_state=random_seed,
            n_nearest_features=None,
            imputation_order='ascending',
            verbose=0
        )
        
        # Standardize the data before imputation
        scaler = StandardScaler()
        impute_df_scaled = pd.DataFrame(
            scaler.fit_transform(impute_df),
            columns=impute_df.columns
        )
        
        # Fit the MICE imputer on the training data
        mice_imputer.fit(impute_df_scaled)
        
        print("MICE imputer and scaler fitted on training data.")
        return impute_df
    
    def impute_lab_values(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Impute laboratory values using MICE."""
        nonlocal mice_imputer, scaler, impute_cols
        
        if df.empty or mice_imputer is None or scaler is None or impute_cols is None:
            return df
        
        df_imputed = df.copy()
        
        print("\n=== Using MICE to impute laboratory values ===\n")
        
        # Check if there are any lab columns with missing values
        lab_cols_with_missing = [col for col in lab_columns
                                if col in df_imputed.columns and df_imputed[col].isna().any()]
        
        if not lab_cols_with_missing:
            print("No laboratory columns with missing values found.")
            return df_imputed
        
        # Create a subset of the dataframe with only the columns for imputation
        impute_df = df_imputed[impute_cols].copy()
        
        # Report missing values before imputation
        for col in lab_cols_with_missing:
            missing_before = impute_df[col].isna().sum()
            print(f"  {col}: {missing_before} missing values before imputation")
        
        # Standardize the data using the fitted scaler
        impute_df_scaled = pd.DataFrame(
            scaler.transform(impute_df),
            columns=impute_df.columns,
            index=impute_df.index
        )
        
        # Perform MICE imputation using the fitted imputer
        if is_training:
            # For training data, use fit_transform to get the best imputation
            imputed_values = mice_imputer.fit_transform(impute_df_scaled)
        else:
            # For test data, use transform to avoid data leakage
            imputed_values = mice_imputer.transform(impute_df_scaled)
        
        # Inverse transform to get back to original scale
        imputed_values = scaler.inverse_transform(imputed_values)
        
        # Create a dataframe with the imputed values and preserve the original index
        imputed_df = pd.DataFrame(imputed_values, columns=impute_cols, index=impute_df.index)
        
        # Replace only the missing values in the original dataframe
        for col in lab_cols_with_missing:
            # Create a mask for missing values
            missing_mask = df_imputed[col].isna()
            
            # Get the indices where values are missing
            missing_indices = missing_mask[missing_mask].index
            
            # Get the positions of these indices in the imputed_df
            positions = [impute_df.index.get_loc(idx) for idx in missing_indices]
            
            # Extract the imputed values at these positions
            imputed_values_for_missing = imputed_df.iloc[positions][col].values
            
            # Replace only the missing values
            df_imputed.loc[missing_indices, col] = imputed_values_for_missing
            
            # Report missing values after imputation
            missing_after = df_imputed[col].isna().sum()
            print(f"  {col}: {missing_before - missing_after} values imputed, {missing_after} still missing")
        
        return df_imputed
    
    # Impute hard truth and medical history in each dataframe
    print("\n=== Imputing hard truth and medical history in training dataframe ===\n")
    train_df_imputed = impute_hard_truth_and_medical_history(train_df_imputed)
    
    if not temporal_test_df_imputed.empty:
        print("\n=== Imputing hard truth and medical history in temporal test dataframe ===\n")
        temporal_test_df_imputed = impute_hard_truth_and_medical_history(temporal_test_df_imputed)
    
    if not spatial_test_df_imputed.empty:
        print("\n=== Imputing hard truth and medical history in spatial test dataframe ===\n")
        spatial_test_df_imputed = impute_hard_truth_and_medical_history(spatial_test_df_imputed)
    
    # Fit MICE imputer on training data
    training_impute_df = fit_mice_imputer(train_df_imputed)
    
    # Impute laboratory values in each dataframe
    if mice_imputer is not None and scaler is not None:
        print("\n=== Imputing laboratory values in training dataframe ===\n")
        train_df_imputed = impute_lab_values(train_df_imputed, is_training=True)
        
        if not temporal_test_df_imputed.empty:
            print("\n=== Imputing laboratory values in temporal test dataframe ===\n")
            temporal_test_df_imputed = impute_lab_values(temporal_test_df_imputed, is_training=False)
        
        if not spatial_test_df_imputed.empty:
            print("\n=== Imputing laboratory values in spatial test dataframe ===\n")
            spatial_test_df_imputed = impute_lab_values(spatial_test_df_imputed, is_training=False)
    
    # Report final statistics
    print("\n=== Imputation complete ===\n")
    print(f"Training dataframe: {len(train_df_imputed)} rows, {train_df_imputed['key'].nunique()} unique patients")
    if not temporal_test_df_imputed.empty:
        print(f"Temporal test dataframe: {len(temporal_test_df_imputed)} rows, {temporal_test_df_imputed['key'].nunique()} unique patients")
    if not spatial_test_df_imputed.empty:
        print(f"Spatial test dataframe: {len(spatial_test_df_imputed)} rows, {spatial_test_df_imputed['key'].nunique()} unique patients")
    
    return train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed
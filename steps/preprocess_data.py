"""
Data Preprocessing Step for CKD Risk Prediction

This module contains the ZenML step for preprocess the imputed data in training, validation, and test sets.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats

@step
def preprocess_data(
    train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the imputed data in training, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training dataset.
        temporal_test_df (pd.DataFrame): Temporal test dataset.
        spatial_test_df (pd.DataFrame): Spatial test dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Preprocessed training, temporal test, and spatial test datasets.
    """
    print("\n=== Starting data preprocessing ===\n")
    
    # Initialize preprocessed dataframes
    train_df_processed = train_df.copy()
    temporal_test_df_processed = temporal_test_df.copy()
    spatial_test_df_processed = spatial_test_df.copy()
    
    # Get CCI column patterns from environment variables or use defaults
    cci_pattern = os.getenv("MED_HISTORY_PATTERN", "cci_")
    cci_specific = os.getenv("MED_HISTORY_SPECIFIC", "myocardial_infarction,congestive_heart_failure,peripheral_vascular_disease,cerebrovascular_disease,dementia,chronic_pulmonary_disease,rheumatic_disease,peptic_ulcer_disease,mild_liver_disease,diabetes_wo_complication,renal_mild_moderate,diabetes_w_complication,hemiplegia_paraplegia,any_malignancy,liver_severe,renal_severe,hiv,metastatic_cancer,aids,cci_score_total")
    cci_specific_list = [col.strip() for col in cci_specific.split(",")]
    
    # Combine pattern-based and specific CCI columns
    cci_columns = [col for col in train_df.columns if
                  any(col.startswith(pattern.strip()) for pattern in cci_pattern.split(","))]
    cci_columns.extend([col for col in cci_specific_list if col in train_df.columns])
    
    # Filter to only include columns that exist in the dataframe
    cci_columns = [col for col in cci_columns if col in train_df.columns]
    
    print(f"CCI columns to preprocess: {cci_columns}")
    
    # 1. For Charlson Comorbidity Index (CCI) related columns, only allow 1 and 0 values, change to categorical
    def preprocess_cci_columns(df):
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        print("\n=== Processing CCI columns ===\n")
        for col in cci_columns:
            if col in df_processed.columns:
                # Count values before processing
                value_counts_before = df_processed[col].value_counts().to_dict()
                
                # Replace any value > 0 with 1 (binary presence)
                df_processed[col] = df_processed[col].apply(lambda x: 1 if x > 0 else 0)
                
                # Convert to categorical
                df_processed[col] = df_processed[col].astype('category')
                
                # Count values after processing
                value_counts_after = df_processed[col].value_counts().to_dict()
                
                print(f"  {col}: Before={value_counts_before}, After={value_counts_after}")
        
        return df_processed
    
    # 2. Convert other categorical features to appropriate types
    def preprocess_categorical_features(df):
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        # Get categorical columns from environment variables or use defaults
        cat_columns_env = os.getenv("CATEGORICAL_COLUMNS", "gender,dm,ht,sprint,endpoint,endpoint_renal,endpoint_death,endpoint_censored")
        cat_columns = [col.strip() for col in cat_columns_env.split(",")]
        
        # Filter to only include columns that exist in the dataframe
        cat_columns = [col for col in cat_columns if col in df_processed.columns]
        
        print("\n=== Processing categorical features ===\n")
        for col in cat_columns:
            if col in df_processed.columns:
                # Count values before processing
                value_counts_before = df_processed[col].value_counts().to_dict()
                
                # For binary columns, ensure they are 0/1
                if df_processed[col].nunique() <= 2:
                    # Replace any non-zero value with 1
                    df_processed[col] = df_processed[col].apply(lambda x: 1 if x and x > 0 else 0)
                
                # Convert to categorical
                df_processed[col] = df_processed[col].astype('category')
                
                # Count values after processing
                value_counts_after = df_processed[col].value_counts().to_dict()
                
                print(f"  {col}: Before={value_counts_before}, After={value_counts_after}")
        
        return df_processed
    
    # 3. Apply log transformation to skewed features
    def transform_skewed_features(df, scaler_dict=None, is_training=True):
        if df.empty:
            return df, scaler_dict if is_training else df
        
        df_processed = df.copy()
        
        # Get numerical columns from environment variables or use defaults
        num_columns_env = os.getenv("NUMERICAL_COLUMNS", "creatinine,hemoglobin,a1c,albumin,phosphate,calcium,ca_adjusted,upcr,uacr,egfr,age,age_at_obs,bicarbonate,observation_period")
        num_columns = [col.strip() for col in num_columns_env.split(",")]
        
        # Filter to only include columns that exist in the dataframe
        num_columns = [col for col in num_columns if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]
        
        # Initialize scaler dictionary if in training mode
        if is_training and scaler_dict is None:
            scaler_dict = {'log_transformed': [], 'min_max_scaled': []}
        
        print("\n=== Applying log transformation to skewed features ===\n")
        for col in num_columns:
            if col in df_processed.columns:
                # Calculate skewness
                skewness = stats.skew(df_processed[col].dropna())
                
                # Calculate statistics before processing
                stats_before = {
                    'min': df_processed[col].min(),
                    'max': df_processed[col].max(),
                    'mean': df_processed[col].mean(),
                    'median': df_processed[col].median(),
                    'skewness': skewness
                }
                
                # Apply log transformation to highly skewed features (skewness > 1.0)
                # Make sure all values are positive before log transform
                if abs(skewness) > 1.0:
                    # Shift data to make all values positive if needed
                    min_val = df_processed[col].min()
                    shift = 0
                    if min_val <= 0:
                        shift = abs(min_val) + 1  # Add 1 to avoid log(0)
                    
                    # Apply log transformation
                    df_processed[f'{col}_log'] = np.log(df_processed[col] + shift)
                    
                    # Calculate statistics after processing
                    log_skewness = stats.skew(df_processed[f'{col}_log'].dropna())
                    stats_after = {
                        'min': df_processed[f'{col}_log'].min(),
                        'max': df_processed[f'{col}_log'].max(),
                        'mean': df_processed[f'{col}_log'].mean(),
                        'median': df_processed[f'{col}_log'].median(),
                        'skewness': log_skewness
                    }
                    
                    print(f"  {col} (log transformed): Before={stats_before}, After={stats_after}")
                    
                    # Store the column name and shift value for later use
                    if is_training:
                        scaler_dict['log_transformed'].append((col, shift))
                    
                    # Drop the original column
                    df_processed = df_processed.drop(columns=[col])
                    
                    # Rename the log-transformed column to the original name
                    df_processed = df_processed.rename(columns={f'{col}_log': col})
                else:
                    # If not highly skewed, add to min-max scaling list
                    if is_training:
                        scaler_dict['min_max_scaled'].append(col)
                    print(f"  {col} (not log transformed, will be min-max scaled): skewness={skewness}")
        
        if is_training:
            return df_processed, scaler_dict
        else:
            return df_processed
    
    # 4. Apply min-max scaling to remaining continuous features
    def apply_min_max_scaling(df, scaler_dict, is_training=True):
        if df.empty:
            return df, {} if is_training else df
        
        df_processed = df.copy()
        
        # Initialize scalers dictionary if in training mode
        if is_training:
            scalers = {}
        
        print("\n=== Applying min-max scaling to continuous features ===\n")
        
        # Get columns to scale
        columns_to_scale = scaler_dict['min_max_scaled']
        
        # Filter to only include columns that exist in the dataframe
        columns_to_scale = [col for col in columns_to_scale if col in df_processed.columns]
        
        if columns_to_scale:
            for col in columns_to_scale:
                # Calculate statistics before scaling
                stats_before = {
                    'min': df_processed[col].min(),
                    'max': df_processed[col].max(),
                    'mean': df_processed[col].mean(),
                    'median': df_processed[col].median()
                }
                
                # Create and fit scaler in training mode, or use existing scaler in test mode
                if is_training:
                    scaler = MinMaxScaler()
                    # Reshape for sklearn
                    df_processed[col] = scaler.fit_transform(df_processed[col].values.reshape(-1, 1)).flatten()
                    # Store the scaler for later use
                    scalers[col] = scaler
                else:
                    # Use the scaler from training
                    scaler = scaler_dict.get(col)
                    if scaler is not None:
                        df_processed[col] = scaler.transform(df_processed[col].values.reshape(-1, 1)).flatten()
                
                # Calculate statistics after scaling
                stats_after = {
                    'min': df_processed[col].min(),
                    'max': df_processed[col].max(),
                    'mean': df_processed[col].mean(),
                    'median': df_processed[col].median()
                }
                
                print(f"  {col} (min-max scaled): Before={stats_before}, After={stats_after}")
        else:
            print("  No columns to apply min-max scaling")
        
        if is_training:
            return df_processed, scalers
        else:
            return df_processed
    
    # 5. Calculate derived features
    def calculate_derived_features(df):
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        print("\n=== Calculating derived features ===\n")
        
        # Calculate observation_period if date columns are available
        if 'date' in df_processed.columns and 'first_sub_60_date' in df_processed.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df_processed['date']):
                df_processed['date'] = pd.to_datetime(df_processed['date'])
            if not pd.api.types.is_datetime64_dtype(df_processed['first_sub_60_date']):
                df_processed['first_sub_60_date'] = pd.to_datetime(df_processed['first_sub_60_date'])
            
            # Calculate observation period (days since recruitment)
            df_processed['observation_period'] = (df_processed['date'] - df_processed['first_sub_60_date']).dt.days
            
            # Ensure non-negative values
            df_processed['observation_period'] = df_processed['observation_period'].clip(lower=0)
            
            print(f"  Added 'observation_period' feature: min={df_processed['observation_period'].min()}, max={df_processed['observation_period'].max()}, mean={df_processed['observation_period'].mean():.2f}")
        
        # Calculate age at observation if DOB is available
        if 'dob' in df_processed.columns and 'date' in df_processed.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df_processed['dob']):
                df_processed['dob'] = pd.to_datetime(df_processed['dob'])
            if not pd.api.types.is_datetime64_dtype(df_processed['date']):
                df_processed['date'] = pd.to_datetime(df_processed['date'])
            
            # Calculate age at observation
            df_processed['age_at_obs'] = (df_processed['date'] - df_processed['dob']).dt.days / 365.25
            
            # Ensure reasonable age range
            df_processed['age_at_obs'] = df_processed['age_at_obs'].clip(lower=0, upper=120)
            
            print(f"  Added 'age_at_obs' feature: min={df_processed['age_at_obs'].min():.1f}, max={df_processed['age_at_obs'].max():.1f}, mean={df_processed['age_at_obs'].mean():.1f}")
        
        return df_processed
    
    # Apply preprocessing steps to each dataframe
    print("\n=== Preprocessing training dataframe ===\n")
    train_df_processed = preprocess_cci_columns(train_df_processed)
    train_df_processed = preprocess_categorical_features(train_df_processed)
    train_df_processed = calculate_derived_features(train_df_processed)
    
    # Apply log transformation to skewed features in training data
    train_df_processed, transform_info = transform_skewed_features(train_df_processed, is_training=True)
    
    # Apply min-max scaling to remaining continuous features in training data
    train_df_processed, scalers = apply_min_max_scaling(train_df_processed, transform_info, is_training=True)
    
    # Process test datasets if they exist
    if not temporal_test_df_processed.empty:
        print("\n=== Preprocessing temporal test dataframe ===\n")
        temporal_test_df_processed = preprocess_cci_columns(temporal_test_df_processed)
        temporal_test_df_processed = preprocess_categorical_features(temporal_test_df_processed)
        temporal_test_df_processed = calculate_derived_features(temporal_test_df_processed)
        
        # Apply the same transformations to test data using parameters from training
        temporal_test_df_processed = transform_skewed_features(temporal_test_df_processed, transform_info, is_training=False)
        temporal_test_df_processed = apply_min_max_scaling(temporal_test_df_processed, {'min_max_scaled': transform_info['min_max_scaled'], **scalers}, is_training=False)
    
    if not spatial_test_df_processed.empty:
        print("\n=== Preprocessing spatial test dataframe ===\n")
        spatial_test_df_processed = preprocess_cci_columns(spatial_test_df_processed)
        spatial_test_df_processed = preprocess_categorical_features(spatial_test_df_processed)
        spatial_test_df_processed = calculate_derived_features(spatial_test_df_processed)
        
        # Apply the same transformations to test data using parameters from training
        spatial_test_df_processed = transform_skewed_features(spatial_test_df_processed, transform_info, is_training=False)
        spatial_test_df_processed = apply_min_max_scaling(spatial_test_df_processed, {'min_max_scaled': transform_info['min_max_scaled'], **scalers}, is_training=False)
    
    # Report final statistics
    print("\n=== Preprocessing complete ===\n")
    print(f"Training dataframe: {len(train_df_processed)} rows, {train_df_processed['key'].nunique() if 'key' in train_df_processed.columns else 'N/A'} unique patients")
    if not temporal_test_df_processed.empty:
        print(f"Temporal test dataframe: {len(temporal_test_df_processed)} rows, {temporal_test_df_processed['key'].nunique() if 'key' in temporal_test_df_processed.columns else 'N/A'} unique patients")
    if not spatial_test_df_processed.empty:
        print(f"Spatial test dataframe: {len(spatial_test_df_processed)} rows, {spatial_test_df_processed['key'].nunique() if 'key' in spatial_test_df_processed.columns else 'N/A'} unique patients")
    
    return train_df_processed, temporal_test_df_processed, spatial_test_df_processed
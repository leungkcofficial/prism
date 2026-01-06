"""
CKD Preprocessor for Production Use

This module provides a preprocessor that can transform raw patient data
into the format expected by the CKD risk prediction models.
It includes all preprocessing steps from the training pipeline.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from typing import Dict, Any, Union, List, Optional, Tuple
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

warnings.filterwarnings("ignore")


class CKDPreprocessor:
    """
    Preprocessor for CKD risk prediction models.
    
    This class encapsulates all preprocessing steps including:
    - Imputation of missing values
    - Log transformation of skewed features
    - Min-max scaling
    - Feature engineering
    """
    
    def __init__(self):
        # Imputation components
        self.mice_imputer = None
        self.mice_scaler = None
        self.impute_cols = None
        self.hard_truth_values = {}
        self.medical_history_values = {}
        
        # Transformation components
        self.log_transform_params = []  # List of (column, shift) tuples
        self.minmax_scalers = {}
        self.columns_to_scale = []
        
        # Column configurations
        self.hard_truth_columns = []
        self.medical_history_columns = []
        self.lab_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.cci_columns = []
        
        # Feature engineering parameters
        self.feature_engineering_enabled = True
        self.study_end_date = pd.to_datetime('2023-12-31')
        
        # Expected input columns
        self.expected_columns = []
        
    def fit(self, train_df: pd.DataFrame, random_seed: int = 42):
        """
        Fit the preprocessor on training data.
        
        Args:
            train_df: Training dataframe
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        print("=== Fitting CKD Preprocessor ===\n")
        
        # Store expected columns
        self.expected_columns = train_df.columns.tolist()
        
        # Configure column types
        self._configure_columns(train_df)
        
        # Fit imputation components
        self._fit_imputation(train_df, random_seed)
        
        # Apply imputation to get imputed training data
        train_imputed = self._apply_imputation(train_df.copy())
        
        # Fit transformation components
        self._fit_transformations(train_imputed)
        
        print("\n=== Preprocessor fitting complete ===")
        
    def _configure_columns(self, df: pd.DataFrame):
        """Configure column types based on environment variables or defaults."""
        # Hard truth columns
        hard_truth_env = os.getenv("HARD_TRUTH_COLUMNS", 
                                   "dob,gender,endpoint,endpoint_date,endpoint_source,first_sub_60_date")
        self.hard_truth_columns = [col.strip() for col in hard_truth_env.split(",") 
                                  if col.strip() in df.columns]
        
        # Medical history columns
        med_history_pattern = os.getenv("MED_HISTORY_PATTERN", "cci_")
        med_history_specific = os.getenv("MED_HISTORY_SPECIFIC", 
                                        "myocardial_infarction,congestive_heart_failure,peripheral_vascular_disease,"
                                        "cerebrovascular_disease,dementia,chronic_pulmonary_disease,rheumatic_disease,"
                                        "peptic_ulcer_disease,mild_liver_disease,diabetes_wo_complication,"
                                        "renal_mild_moderate,diabetes_w_complication,hemiplegia_paraplegia,"
                                        "any_malignancy,liver_severe,renal_severe,hiv,metastatic_cancer,aids,"
                                        "cci_score_total,ht")
        
        med_history_specific_list = [col.strip() for col in med_history_specific.split(",")]
        self.medical_history_columns = [col for col in df.columns if
                                       any(col.startswith(pattern.strip()) 
                                          for pattern in med_history_pattern.split(","))]
        self.medical_history_columns.extend([col for col in med_history_specific_list 
                                           if col in df.columns and col not in self.medical_history_columns])
        
        # Laboratory columns
        lab_columns_env = os.getenv("LAB_COLUMNS", 
                                   "creatinine,hemoglobin,a1c,albumin,phosphate,calcium,"
                                   "ca_adjusted,upcr,uacr,egfr,bicarbonate")
        self.lab_columns = [col.strip() for col in lab_columns_env.split(",") 
                           if col.strip() in df.columns]
        
        # CCI columns (subset of medical history)
        self.cci_columns = [col for col in self.medical_history_columns 
                           if col.startswith('cci_') or col in med_history_specific_list]
        
        # Categorical columns
        cat_columns_env = os.getenv("CATEGORICAL_COLUMNS", 
                                   "gender,dm,ht,sprint,endpoint,endpoint_renal,endpoint_death,endpoint_censored")
        self.categorical_columns = [col.strip() for col in cat_columns_env.split(",") 
                                   if col.strip() in df.columns]
        
        # Numerical columns
        num_columns_env = os.getenv("NUMERICAL_COLUMNS", 
                                   "creatinine,hemoglobin,a1c,albumin,phosphate,calcium,ca_adjusted,"
                                   "upcr,uacr,egfr,age,age_at_obs,bicarbonate,observation_period")
        self.numerical_columns = [col.strip() for col in num_columns_env.split(",") 
                                 if col.strip() in df.columns and pd.api.types.is_numeric_dtype(df[col.strip()])]
        
        print(f"Configured columns:")
        print(f"  Hard truth: {len(self.hard_truth_columns)}")
        print(f"  Medical history: {len(self.medical_history_columns)}")
        print(f"  Laboratory: {len(self.lab_columns)}")
        print(f"  CCI: {len(self.cci_columns)}")
        print(f"  Categorical: {len(self.categorical_columns)}")
        print(f"  Numerical: {len(self.numerical_columns)}")
        
    def _fit_imputation(self, df: pd.DataFrame, random_seed: int):
        """Fit imputation components."""
        print("\n=== Fitting imputation components ===")
        
        # Calculate imputation values for hard truth columns
        print("\nCalculating hard truth imputation values...")
        for col in self.hard_truth_columns:
            if col in df.columns:
                # Use mode for categorical, median for numerical
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.hard_truth_values[col] = df.groupby('key')[col].median().median()
                else:
                    mode_series = df.groupby('key')[col].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                    self.hard_truth_values[col] = mode_series.mode()[0] if not mode_series.mode().empty else np.nan
                print(f"  {col}: {self.hard_truth_values[col]}")
        
        # Calculate imputation values for medical history columns
        print("\nCalculating medical history imputation values...")
        for col in self.medical_history_columns:
            if col in df.columns:
                # Use mode for these binary/categorical columns
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                self.medical_history_values[col] = mode_val
                print(f"  {col}: {self.medical_history_values[col]}")
        
        # Fit MICE imputer for laboratory columns
        print("\nFitting MICE imputer for laboratory columns...")
        lab_cols_with_missing = [col for col in self.lab_columns
                                if col in df.columns and df[col].isna().any()]
        
        if lab_cols_with_missing:
            # Get numeric columns for MICE imputation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['key', 'endpoint', 'endpoint_renal', 'endpoint_death', 'endpoint_censored']
            self.impute_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Create imputation dataframe
            impute_df = df[self.impute_cols].copy()
            
            # Initialize MICE imputer
            try:
                import xgboost as xgb
                estimator = xgb.XGBRegressor(tree_method='hist', device='cuda', random_state=random_seed)
            except:
                from sklearn.ensemble import ExtraTreesRegressor
                estimator = ExtraTreesRegressor(n_estimators=10, random_state=random_seed)
                print("  Note: XGBoost not available, using ExtraTreesRegressor")
            
            self.mice_imputer = IterativeImputer(
                estimator=estimator,
                initial_strategy='mean',
                max_iter=10,
                random_state=random_seed,
                n_nearest_features=None,
                imputation_order='ascending',
                verbose=0
            )
            
            # Fit scaler and imputer
            self.mice_scaler = StandardScaler()
            impute_df_scaled = pd.DataFrame(
                self.mice_scaler.fit_transform(impute_df),
                columns=impute_df.columns
            )
            
            print(f"  Fitting MICE on {len(self.impute_cols)} columns...")
            self.mice_imputer.fit(impute_df_scaled)
            print("  MICE imputer fitted successfully")
        else:
            print("  No laboratory columns with missing values found")
            
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to a dataframe."""
        df_imputed = df.copy()
        
        # Impute hard truth columns
        for col in self.hard_truth_columns:
            if col in df_imputed.columns and col in self.hard_truth_values:
                mask = df_imputed[col].isna()
                if mask.any():
                    df_imputed.loc[mask, col] = self.hard_truth_values[col]
        
        # Special handling for endpoint_date
        if 'endpoint' in df_imputed.columns and 'endpoint_date' in df_imputed.columns:
            mask = (df_imputed['endpoint'] == 0) & df_imputed['endpoint_date'].isna()
            if mask.any():
                df_imputed.loc[mask, 'endpoint_date'] = self.study_end_date
        
        # Impute medical history columns
        for col in self.medical_history_columns:
            if col in df_imputed.columns and col in self.medical_history_values:
                mask = df_imputed[col].isna()
                if mask.any():
                    df_imputed.loc[mask, col] = self.medical_history_values[col]
        
        # Apply MICE imputation if available
        if self.mice_imputer is not None and self.mice_scaler is not None and self.impute_cols is not None:
            # Check if any lab columns have missing values
            lab_cols_missing = [col for col in self.lab_columns 
                               if col in df_imputed.columns and df_imputed[col].isna().any()]
            
            if lab_cols_missing:
                # Create imputation dataframe
                impute_df = df_imputed[self.impute_cols].copy()
                
                # Scale data
                impute_df_scaled = pd.DataFrame(
                    self.mice_scaler.transform(impute_df),
                    columns=impute_df.columns,
                    index=impute_df.index
                )
                
                # Apply MICE
                imputed_values = self.mice_imputer.transform(impute_df_scaled)
                
                # Inverse transform
                imputed_values = self.mice_scaler.inverse_transform(imputed_values)
                
                # Create dataframe with imputed values
                imputed_df = pd.DataFrame(imputed_values, columns=self.impute_cols, index=impute_df.index)
                
                # Replace only missing values
                for col in lab_cols_missing:
                    if col in imputed_df.columns:
                        missing_mask = df_imputed[col].isna()
                        df_imputed.loc[missing_mask, col] = imputed_df.loc[missing_mask, col]
        
        return df_imputed
        
    def _fit_transformations(self, df: pd.DataFrame):
        """Fit transformation components on imputed data."""
        print("\n=== Fitting transformation components ===")
        
        # Process CCI columns (binarization happens during transform)
        print("\nIdentified CCI columns for binarization:")
        print(f"  {len(self.cci_columns)} columns")
        
        # Fit log transformations
        print("\nFitting log transformations...")
        for col in self.numerical_columns:
            if col in df.columns:
                skewness = stats.skew(df[col].dropna())
                
                if abs(skewness) > 1.0:
                    # Calculate shift for log transformation
                    min_val = df[col].min()
                    shift = 0
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                    
                    self.log_transform_params.append((col, shift))
                    print(f"  {col}: skewness={skewness:.2f}, shift={shift}")
                else:
                    # Add to scaling list
                    self.columns_to_scale.append(col)
        
        # Apply log transformations to get transformed data
        df_transformed = df.copy()
        for col, shift in self.log_transform_params:
            df_transformed[col] = np.log(df_transformed[col] + shift)
        
        # Fit MinMax scalers
        print("\nFitting MinMax scalers...")
        for col in self.columns_to_scale:
            if col in df_transformed.columns:
                scaler = MinMaxScaler()
                df_transformed[col] = scaler.fit_transform(df_transformed[col].values.reshape(-1, 1)).flatten()
                self.minmax_scalers[col] = scaler
                print(f"  {col}: fitted")
        
        print(f"\nTransformation summary:")
        print(f"  Log transformed: {len(self.log_transform_params)} columns")
        print(f"  MinMax scaled: {len(self.minmax_scalers)} columns")
        
    def transform(self, data: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform input data using fitted preprocessing pipeline.
        
        Args:
            data: Input data (DataFrame, Series, or dict)
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Add missing columns with NaN
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Apply imputation
        df = self._apply_imputation(df)
        
        # Process CCI columns (binarization)
        for col in self.cci_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
                df[col] = df[col].astype('category')
        
        # Process other categorical features
        for col in self.categorical_columns:
            if col in df.columns and col not in self.cci_columns:
                if df[col].nunique() <= 2:
                    df[col] = df[col].apply(lambda x: 1 if x and x > 0 else 0)
                df[col] = df[col].astype('category')
        
        # Calculate derived features
        if self.feature_engineering_enabled:
            df = self._calculate_derived_features(df)
        
        # Apply log transformations
        for col, shift in self.log_transform_params:
            if col in df.columns:
                df[col] = np.log(df[col] + shift)
        
        # Apply MinMax scaling
        for col, scaler in self.minmax_scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
        
        return df
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features."""
        # Calculate observation_period
        if 'date' in df.columns and 'first_sub_60_date' in df.columns:
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            if not pd.api.types.is_datetime64_dtype(df['first_sub_60_date']):
                df['first_sub_60_date'] = pd.to_datetime(df['first_sub_60_date'])
            
            df['observation_period'] = (df['date'] - df['first_sub_60_date']).dt.days
            df['observation_period'] = df['observation_period'].clip(lower=0)
        
        # Calculate age_at_obs
        if 'dob' in df.columns and 'date' in df.columns:
            if not pd.api.types.is_datetime64_dtype(df['dob']):
                df['dob'] = pd.to_datetime(df['dob'])
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            df['age_at_obs'] = (df['date'] - df['dob']).dt.days / 365.25
            df['age_at_obs'] = df['age_at_obs'].clip(lower=0, upper=120)
        
        return df
    
    def save(self, filepath: str):
        """Save the fitted preprocessor to file."""
        print(f"\nSaving preprocessor to {filepath}...")
        
        artifacts = {
            'mice_imputer': self.mice_imputer,
            'mice_scaler': self.mice_scaler,
            'impute_cols': self.impute_cols,
            'hard_truth_values': self.hard_truth_values,
            'medical_history_values': self.medical_history_values,
            'log_transform_params': self.log_transform_params,
            'minmax_scalers': self.minmax_scalers,
            'columns_to_scale': self.columns_to_scale,
            'hard_truth_columns': self.hard_truth_columns,
            'medical_history_columns': self.medical_history_columns,
            'lab_columns': self.lab_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'cci_columns': self.cci_columns,
            'feature_engineering_enabled': self.feature_engineering_enabled,
            'study_end_date': self.study_end_date,
            'expected_columns': self.expected_columns,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Calculate file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"Preprocessor saved successfully ({file_size:.2f} MB)")
        
    @classmethod
    def load(cls, filepath: str) -> 'CKDPreprocessor':
        """Load a fitted preprocessor from file."""
        print(f"Loading preprocessor from {filepath}...")
        
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        
        preprocessor = cls()
        
        # Restore all components
        preprocessor.mice_imputer = artifacts.get('mice_imputer')
        preprocessor.mice_scaler = artifacts.get('mice_scaler')
        preprocessor.impute_cols = artifacts.get('impute_cols')
        preprocessor.hard_truth_values = artifacts.get('hard_truth_values', {})
        preprocessor.medical_history_values = artifacts.get('medical_history_values', {})
        preprocessor.log_transform_params = artifacts.get('log_transform_params', [])
        preprocessor.minmax_scalers = artifacts.get('minmax_scalers', {})
        preprocessor.columns_to_scale = artifacts.get('columns_to_scale', [])
        preprocessor.hard_truth_columns = artifacts.get('hard_truth_columns', [])
        preprocessor.medical_history_columns = artifacts.get('medical_history_columns', [])
        preprocessor.lab_columns = artifacts.get('lab_columns', [])
        preprocessor.categorical_columns = artifacts.get('categorical_columns', [])
        preprocessor.numerical_columns = artifacts.get('numerical_columns', [])
        preprocessor.cci_columns = artifacts.get('cci_columns', [])
        preprocessor.feature_engineering_enabled = artifacts.get('feature_engineering_enabled', True)
        preprocessor.study_end_date = artifacts.get('study_end_date', pd.to_datetime('2023-12-31'))
        preprocessor.expected_columns = artifacts.get('expected_columns', [])
        
        version = artifacts.get('version', 'unknown')
        created_at = artifacts.get('created_at', 'unknown')
        print(f"Preprocessor loaded (version: {version}, created: {created_at})")
        
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """Get the list of features after preprocessing."""
        # This would be the expected columns after all transformations
        # For now, return the expected columns
        return self.expected_columns
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline."""
        info = {
            'version': '1.0.0',
            'n_features': len(self.expected_columns),
            'imputation': {
                'mice_fitted': self.mice_imputer is not None,
                'n_hard_truth_values': len(self.hard_truth_values),
                'n_medical_history_values': len(self.medical_history_values)
            },
            'transformations': {
                'n_log_transformed': len(self.log_transform_params),
                'n_minmax_scaled': len(self.minmax_scalers)
            },
            'column_types': {
                'hard_truth': len(self.hard_truth_columns),
                'medical_history': len(self.medical_history_columns),
                'laboratory': len(self.lab_columns),
                'cci': len(self.cci_columns),
                'categorical': len(self.categorical_columns),
                'numerical': len(self.numerical_columns)
            }
        }
        return info
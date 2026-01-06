"""
Lab Result Mapper module for CKD survival model development
This module handles the mapping of lab result data from CSV files to the appropriate format for the model
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

class LabResultMapper(ABC):
    """
    Abstract base class for lab result mappers
    """
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Process lab result data from a DataFrame.
        
        Args:
            data: DataFrame containing raw lab data
            
        Returns:
            Processed DataFrame or tuple of DataFrames
        """
        pass


class StandardLabResultMapper(LabResultMapper):
    """
    Mapper for standard lab result data (creatinine, hemoglobin, etc.)
    """
    
    def __init__(self, data_type: str, validation_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the standard lab result mapper with configuration.
        
        Args:
            data_type: Type of lab data (e.g., 'creatinine', 'hemoglobin')
            validation_rules: Dictionary with validation rules (min_value, max_value, unit)
        """
        self.data_type = data_type
        self.validation_rules = validation_rules or {}
        
        # Get column mapping from environment variables or use default
        default_mapping = {
            'use_column_index': False,
            'patient_id': 'Reference Key',
            'test_date': 'LIS Reference Datetime',
            'result_value': 'LIS Result (28 days) - LIS Result: Numeric Result',
            'case_number': 'LIS Case No.'
        }
        
        # Add demographic columns for creatinine data
        if data_type == 'creatinine':
            default_mapping.update({
                'dob': 'Date of Birth (yyyy-mm-dd)',
                'gender': 'Sex'
            })
        
        # Check for individual environment variables first (e.g., BICARBONATE_USE_COLUMN_INDEX)
        use_column_index = os.getenv(f'{data_type.upper()}_USE_COLUMN_INDEX')
        patient_id = os.getenv(f'{data_type.upper()}_PATIENT_ID_COL')
        test_date = os.getenv(f'{data_type.upper()}_TEST_DATE_COL')
        result_value = os.getenv(f'{data_type.upper()}_RESULT_VALUE_COL')
        case_number = os.getenv(f'{data_type.upper()}_CASE_NUMBER_COL')
        
        if use_column_index is not None:
            default_mapping['use_column_index'] = use_column_index.lower() == 'true'
        
        if patient_id is not None:
            default_mapping['patient_id'] = patient_id
        
        if test_date is not None:
            default_mapping['test_date'] = test_date
        
        if result_value is not None:
            default_mapping['result_value'] = result_value
        
        if case_number is not None:
            default_mapping['code'] = case_number
        
        # Also check for DOB and gender for creatinine
        if data_type == 'creatinine':
            dob = os.getenv('CREATININE_DOB_COL')
            gender = os.getenv('CREATININE_GENDER_COL')
            
            if dob is not None:
                default_mapping['dob'] = dob
            
            if gender is not None:
                default_mapping['gender'] = gender
        
        # Also try the old way with a JSON string
        env_mapping_str = os.getenv(f'{data_type.upper()}_COLUMN_MAPPING')
        if env_mapping_str:
            try:
                import json
                env_mapping = json.loads(env_mapping_str)
                default_mapping.update(env_mapping)
            except Exception as e:
                print(f"Error parsing column mapping from environment variable: {e}")
        
        self.column_mapping = default_mapping
        print(f"Using column mapping for {data_type}: {self.column_mapping}")
    
    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process standard lab result data from a DataFrame.
        
        Args:
            data: DataFrame containing raw lab data
            
        Returns:
            Tuple containing:
            - DataFrame with processed lab data
            - DataFrame with demographic data (if data_type is 'creatinine'), otherwise None
        """
        if data.empty:
            print(f"No {self.data_type} data provided")
            return pd.DataFrame(), None if self.data_type != 'creatinine' else pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Use column indices or column names based on configuration
        use_column_index = self.column_mapping.get('use_column_index', False)
        
        # Extract columns based on mapping
        lab_df = pd.DataFrame()
        
        if use_column_index:
            # Extract columns by index
            patient_id_idx = int(self.column_mapping.get('patient_id', 0))
            test_date_idx = int(self.column_mapping.get('test_date', 1))
            result_value_idx = int(self.column_mapping.get('result_value', 2))
            case_number_idx = int(self.column_mapping.get('case_number', 3))
            
            # Check if we have enough columns
            if df.shape[1] > max(patient_id_idx, test_date_idx, result_value_idx, case_number_idx):
                lab_df['key'] = df.iloc[:, patient_id_idx]
                lab_df['date'] = df.iloc[:, test_date_idx]
                lab_df['result_value'] = df.iloc[:, result_value_idx]
                lab_df['code'] = df.iloc[:, case_number_idx]
                
                # Extract demographic data for creatinine
                if self.data_type == 'creatinine':
                    dob_idx = int(self.column_mapping.get('dob', 4))
                    gender_idx = int(self.column_mapping.get('gender', 5))
                    
                    if df.shape[1] > max(dob_idx, gender_idx):
                        lab_df['dob'] = df.iloc[:, dob_idx]
                        lab_df['gender'] = df.iloc[:, gender_idx]
            else:
                print(f"Warning: {self.data_type} data has only {df.shape[1]} columns, but indices go up to {max(patient_id_idx, test_date_idx, result_value_idx, case_number_idx)}")
        else:
            # Extract columns by name
            patient_id_col = self.column_mapping.get('patient_id')
            test_date_col = self.column_mapping.get('test_date')
            result_value_col = self.column_mapping.get('result_value')
            case_number_col = self.column_mapping.get('case_number')
            
            # Check if all required columns exist
            required_cols = [patient_id_col, test_date_col, result_value_col]
            if all(col in df.columns for col in required_cols):
                lab_df['key'] = df[patient_id_col]
                lab_df['date'] = df[test_date_col]
                lab_df['result_value'] = df[result_value_col]
                
                if case_number_col in df.columns:
                    lab_df['code'] = df[case_number_col]
                
                # Extract demographic data for creatinine
                if self.data_type == 'creatinine':
                    dob_col = self.column_mapping.get('dob')
                    gender_col = self.column_mapping.get('gender')
                    
                    if dob_col in df.columns:
                        lab_df['dob'] = df[dob_col]
                    
                    if gender_col in df.columns:
                        lab_df['gender'] = df[gender_col]
            else:
                missing_cols = [col for col in required_cols if col not in df.columns]
                print(f"Warning: {self.data_type} data is missing required columns: {missing_cols}")
        
        # Add metadata columns
        if 'source_file' in df.columns:
            lab_df['source_file'] = df['source_file']
        if 'year' in df.columns:
            lab_df['year'] = df['year']
        if 'quarter' in df.columns:
            lab_df['quarter'] = df['quarter']
        
        # Add data type column
        lab_df['data_type'] = self.data_type
        
        # Convert key to numeric
        lab_df['key'] = pd.to_numeric(lab_df['key'], errors='coerce')
        
        # Convert date to datetime
        lab_df['date'] = pd.to_datetime(lab_df['date'], errors='coerce')
        
        # Convert result_value to numeric
        lab_df['result_value'] = pd.to_numeric(lab_df['result_value'], errors='coerce')
        
        # Apply validation rules if provided
        if self.validation_rules:
            min_value = self.validation_rules.get('min_value')
            max_value = self.validation_rules.get('max_value')
            
            if min_value is not None:
                below_min = lab_df['result_value'] < min_value
                if below_min.any():
                    print(f"Removing {below_min.sum()} values below minimum ({min_value}) for {self.data_type}")
                    lab_df = lab_df[~below_min]
            
            if max_value is not None:
                above_max = lab_df['result_value'] > max_value
                if above_max.any():
                    print(f"Removing {above_max.sum()} values above maximum ({max_value}) for {self.data_type}")
                    lab_df = lab_df[~above_max]
        
        # Create a separate DataFrame for demographic data if this is creatinine data
        demo_df = None
        if self.data_type == 'creatinine' and 'dob' in lab_df.columns and 'gender' in lab_df.columns:
            # Extract unique patient records
            demo_df = lab_df[['key', 'dob', 'gender']].drop_duplicates(subset=['key'])
            
            # Print gender distribution
            if 'gender' in demo_df.columns:
                gender_counts = demo_df['gender'].value_counts(dropna=False)
                print(f"Gender value counts for {self.data_type}: {gender_counts}")
        
        # Only set code column if it doesn't already exist
        if 'code' not in lab_df.columns:
            lab_df['code'] = self.data_type
        
        print(f"Final {self.data_type} dataset has {len(lab_df)} rows and {len(lab_df.columns)} columns")
        
        return lab_df, demo_df


class UrineLabResultMapper(LabResultMapper):
    """
    Mapper for urine lab result data (protein/creatinine ratio, albumin/creatinine ratio)
    """
    
    def __init__(self, data_type: str, validation_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the urine lab result mapper with configuration.
        
        Args:
            data_type: Type of lab data (e.g., 'urine_protein_creatinine_ratio', 'urine_albumin_creatinine_ratio')
            validation_rules: Dictionary with validation rules (min_value, max_value, unit)
        """
        self.data_type = data_type
        self.validation_rules = validation_rules or {}
        
        # Validate data_type
        valid_types = ['urine_protein_creatinine_ratio', 'urine_albumin_creatinine_ratio']
        if data_type not in valid_types and data_type != "":
            print(f"Warning: Unknown urine data type: {data_type}. Expected one of: {valid_types}")
        
        # Get column indices from environment variables
        self.key_col_idx = int(os.getenv('UPACR_KEY_COL_IDX', '0'))  # Reference Key is at index 0
        self.date_col_idx = int(os.getenv('UPACR_DATE_COL_IDX', '3'))  # LIS Reference Datetime is at index 3
        self.code_col_idx = int(os.getenv('UPACR_CODE_COL_IDX', '6'))  # Test Code is at index 6
        self.type_col_idx = int(os.getenv('UPACR_TYPE_COL_IDX', '7'))  # Test Name is at index 7
        self.value_col_idx = int(os.getenv('UPACR_VALUE_COL_IDX', '12'))  # Numeric Result is at index 12
        self.unit_col_idx = int(os.getenv('UPACR_UNIT_COL_IDX', '13'))  # Unit is at index 13
        
        # Get protein and albumin patterns from environment variables
        default_protein_patterns = 'ur. tp / cr ratio|msu_urine protein: creatinine ratio_mg/mmol|msu_urine protein: creatinine ratio_mg/mg|ur protein/creatinine|ur. prot/creat|spot t.prot/creat|urine protein / creatinine|protein creatinine ratio, urine|tp/cr ratio|ur tp/cr ratio|u prot/creat|ur. tp/cr ratio|ur prot/cr ratio|ur. prot/cr ratio|msu_urine protein: creatinine ratio_mg/mg cr|prot/creat ratio|ur prot/creat'
        default_albumin_patterns = 'ur. alb/cr ratio|msu_urine albumin: creatinine ratio_mg/mmol|ur alb/cr ratio|u albumin creatinine ratio|albumin creatinine ratio|microalbumin|ur. albumin/creatinine|urine albumin/creatinine ratio|microalbumin creatinine ratio, urine|albumin/creatinine|alb/creat ratio|msu_urine albumin: creatinine ratio_mg/mmol cr|ur alb/creat'
        
        protein_patterns_str = os.getenv('UPACR_PROTEIN_PATTERNS', default_protein_patterns)
        albumin_patterns_str = os.getenv('UPACR_ALBUMIN_PATTERNS', default_albumin_patterns)
        
        self.protein_patterns = protein_patterns_str.split('|')
        self.albumin_patterns = albumin_patterns_str.split('|')
        
        print(f"Using column indices: key={self.key_col_idx}, date={self.date_col_idx}, code={self.code_col_idx}, type={self.type_col_idx}, value={self.value_col_idx}, unit={self.unit_col_idx}")
        print(f"Using {len(self.protein_patterns)} protein patterns and {len(self.albumin_patterns)} albumin patterns")
    
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Process urine lab result data from a DataFrame.
        
        Args:
            data: DataFrame containing raw urine lab data
            
        Returns:
            If data_type is 'urine_protein_creatinine_ratio' or 'urine_albumin_creatinine_ratio':
                DataFrame with the specific processed data
            If data_type is not specified:
                Tuple containing:
                - DataFrame with processed urine protein/creatinine ratio data
                - DataFrame with processed urine albumin/creatinine ratio data
        """
        if data.empty:
            print("No urine lab data provided")
            return pd.DataFrame(), pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if we have enough columns
        max_col_idx = max(self.key_col_idx, self.date_col_idx, self.code_col_idx, self.type_col_idx, self.value_col_idx, self.unit_col_idx)
        
        if df.shape[1] <= max_col_idx:
            print(f"Warning: Urine lab data has only {df.shape[1]} columns, but max index is {max_col_idx}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract columns using the configured indices
        urine_df = pd.DataFrame()
        urine_df['key'] = df.iloc[:, self.key_col_idx]
        urine_df['date'] = df.iloc[:, self.date_col_idx]
        urine_df['code'] = df.iloc[:, self.code_col_idx]
        urine_df['type'] = df.iloc[:, self.type_col_idx]
        urine_df['value'] = df.iloc[:, self.value_col_idx]
        urine_df['unit'] = df.iloc[:, self.unit_col_idx]
        
        # Add metadata columns
        if 'source_file' in df.columns:
            urine_df['source_file'] = df['source_file']
        if 'year' in df.columns:
            urine_df['year'] = df['year']
        if 'quarter' in df.columns:
            urine_df['quarter'] = df['quarter']
        
        # Convert key to numeric
        urine_df['key'] = pd.to_numeric(urine_df['key'], errors='coerce')
        
        # Convert date to datetime
        urine_df['date'] = pd.to_datetime(urine_df['date'], errors='coerce')
        
        # Convert value to numeric
        urine_df['value'] = pd.to_numeric(urine_df['value'], errors='coerce')
        
        # Create masks for protein and albumin tests
        # Use re.escape to escape special characters in the patterns
        protein_mask = urine_df['type'].str.contains('|'.join([re.escape(p) for p in self.protein_patterns]), case=False, na=False)
        albumin_mask = urine_df['type'].str.contains('|'.join([re.escape(p) for p in self.albumin_patterns]), case=False, na=False)
        
        # Extract protein and albumin data
        upcr_df = urine_df[protein_mask].copy()
        uacr_df = urine_df[albumin_mask].copy()
        
        # Add data type column
        upcr_df['data_type'] = 'urine_protein_creatinine_ratio'
        uacr_df['data_type'] = 'urine_albumin_creatinine_ratio'
        
        # Apply validation rules if provided
        if self.validation_rules:
            min_value = self.validation_rules.get('min_value')
            max_value = self.validation_rules.get('max_value')
            
            if min_value is not None:
                # Apply to both dataframes
                upcr_df = upcr_df[upcr_df['value'] >= min_value]
                uacr_df = uacr_df[uacr_df['value'] >= min_value]
            
            if max_value is not None:
                # Apply to both dataframes
                upcr_df = upcr_df[upcr_df['value'] <= max_value]
                uacr_df = uacr_df[uacr_df['value'] <= max_value]
        
        print(f"Final urine protein creatinine ratio dataset has {len(upcr_df)} rows and {len(upcr_df.columns)} columns")
        print(f"Final urine albumin creatinine ratio dataset has {len(uacr_df)} rows and {len(uacr_df.columns)} columns")
        
        # Return the appropriate DataFrame based on the data_type
        if self.data_type == 'urine_protein_creatinine_ratio':
            return upcr_df
        elif self.data_type == 'urine_albumin_creatinine_ratio':
            return uacr_df
        else:
            # If no specific data_type is specified, return both DataFrames
            return upcr_df, uacr_df


class CalciumLabResultMapper(LabResultMapper):
    """
    Mapper for calcium lab result data (calcium, adjusted calcium)
    """
    
    def __init__(self, data_type: str, validation_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the calcium lab result mapper with configuration.
        
        Args:
            data_type: Type of lab data (e.g., 'calcium', 'calcium_adjusted')
            validation_rules: Dictionary with validation rules (min_value, max_value, unit)
        """
        self.data_type = data_type
        self.validation_rules = validation_rules or {}
        
        # Get column indices from environment variables
        self.key_col_idx = int(os.getenv('CALCIUM_KEY_COL_IDX', '0'))  # Reference Key is at index 0
        self.date_col_idx = int(os.getenv('CALCIUM_DATE_COL_IDX', '3'))  # LIS Reference Datetime is at index 3
        self.code_col_idx = int(os.getenv('CALCIUM_CODE_COL_IDX', '5'))  # Test Code is at index 5
        self.type_col_idx = int(os.getenv('CALCIUM_TYPE_COL_IDX', '6'))  # Test Name is at index 6
        self.value_col_idx = int(os.getenv('CALCIUM_VALUE_COL_IDX', '7'))  # Numeric Result is at index 7
        
        # Define patterns for adjusted calcium
        self.adj_patterns = [
            'Calcium, albumin adjusted',
            'Calcium (Alb adj)',
            'Alb.Adj.Calcium',
            'ALB Adjusted Calcium',
            'Calcium (adj.)',
            'Albumin Adjusted Calcium',
            'Calcium, Albumin Adjusted'
        ]
        
        print(f"Using column indices: key={self.key_col_idx}, date={self.date_col_idx}, code={self.code_col_idx}, type={self.type_col_idx}, value={self.value_col_idx}")
        print(f"Using adjusted calcium patterns: {self.adj_patterns}")
    
    def process(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process calcium lab result data from a DataFrame.
        
        Args:
            data: DataFrame containing raw calcium lab data
            
        Returns:
            Tuple containing:
            - DataFrame with processed calcium data
            - DataFrame with processed adjusted calcium data
        """
        if data.empty:
            print("No calcium lab data provided")
            return pd.DataFrame(), pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if we have enough columns
        max_col_idx = max(self.key_col_idx, self.date_col_idx, self.code_col_idx, self.type_col_idx, self.value_col_idx)
        
        if df.shape[1] <= max_col_idx:
            print(f"Warning: Calcium lab data has only {df.shape[1]} columns, but max index is {max_col_idx}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract columns using the configured indices
        calcium_df = pd.DataFrame()
        calcium_df['key'] = df.iloc[:, self.key_col_idx]
        calcium_df['date'] = df.iloc[:, self.date_col_idx]
        calcium_df['code'] = df.iloc[:, self.code_col_idx]
        calcium_df['type'] = df.iloc[:, self.type_col_idx]
        calcium_df['value'] = df.iloc[:, self.value_col_idx]
        
        # Add metadata columns
        if 'source_file' in df.columns:
            calcium_df['source_file'] = df['source_file']
        if 'year' in df.columns:
            calcium_df['year'] = df['year']
        if 'quarter' in df.columns:
            calcium_df['quarter'] = df['quarter']
        
        # Convert key to numeric
        calcium_df['key'] = pd.to_numeric(calcium_df['key'], errors='coerce')
        
        # Convert date to datetime
        calcium_df['date'] = pd.to_datetime(calcium_df['date'], errors='coerce')
        
        # Convert value to numeric
        calcium_df['value'] = pd.to_numeric(calcium_df['value'], errors='coerce')
        
        # Create a flag for adjusted calcium
        calcium_df['is_adjusted'] = calcium_df['type'].str.contains('|'.join([re.escape(p) for p in self.adj_patterns]), case=False, na=False)
        
        # Split into calcium and adjusted calcium
        ca_df = calcium_df[~calcium_df['is_adjusted']].copy()
        ca_adjusted_df = calcium_df[calcium_df['is_adjusted']].copy()
        
        # Add data type column
        ca_df['data_type'] = 'calcium'
        ca_adjusted_df['data_type'] = 'calcium_adjusted'
        
        # Apply validation rules if provided
        if self.validation_rules:
            min_value = self.validation_rules.get('min_value')
            max_value = self.validation_rules.get('max_value')
            
            if min_value is not None:
                # Apply to calcium dataframe
                below_min = ca_df['value'] < min_value
                if below_min.any():
                    print(f"Removing {below_min.sum()} values below minimum ({min_value}) for calcium")
                    ca_df = ca_df[~below_min]
            
            if max_value is not None:
                # Apply to calcium dataframe
                above_max = ca_df['value'] > max_value
                if above_max.any():
                    print(f"Removing {above_max.sum()} values above maximum ({max_value}) for calcium")
                    ca_df = ca_df[~above_max]
        
        # Apply validation rules for adjusted calcium if provided
        if self.validation_rules:
            min_value = self.validation_rules.get('min_value')
            max_value = self.validation_rules.get('max_value')
            
            if min_value is not None:
                # Apply to adjusted calcium dataframe
                below_min = ca_adjusted_df['value'] < min_value
                if below_min.any():
                    print(f"Removing {below_min.sum()} values below minimum ({min_value}) for adjusted calcium")
                    ca_adjusted_df = ca_adjusted_df[~below_min]
            
            if max_value is not None:
                # Apply to adjusted calcium dataframe
                above_max = ca_adjusted_df['value'] > max_value
                if above_max.any():
                    print(f"Removing {above_max.sum()} values above maximum ({max_value}) for adjusted calcium")
                    ca_adjusted_df = ca_adjusted_df[~above_max]
        
        print(f"Final calcium dataset has {len(ca_df)} rows and {len(ca_df.columns)} columns")
        print(f"Final adjusted calcium dataset has {len(ca_adjusted_df)} rows and {len(ca_adjusted_df.columns)} columns")
        
        return ca_df, ca_adjusted_df
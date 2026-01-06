"""Diagnosis ingestion module for CKD survival model development
This module handles the csv data and mapping information loaded from the pipeline, then output the dataframe
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from src.data_ingester import DataIngester, CSVDataIngester
from src.lab_result_ingester import LabResultIngester
from abc import ABC, abstractmethod

class ICD10Ingester(LabResultIngester):
    """Ingests ICD10 diagnosis data"""
    
    def __init__(self):
        """Initialize the ICD10 ingester with configuration"""
        # Get column names from environment variables
        self.key_col_name = os.getenv('ICD10_KEY_COL_NAME', 'Reference Key')
        self.date_col_name = os.getenv('ICD10_DATE_COL_NAME', 'Appointment Date (yyyy-mm-dd)')
        self.code_prefix = os.getenv('ICD10_CODE_PREFIX', 'Most Recent Diagnosis')
        self.code_suffix = os.getenv('ICD10_CODE_SUFFIX', 'Diagnosis Code (ICD10 v2010)')
        
        # For backward compatibility, also check for column indices
        self.use_indices = False
        if os.getenv('ICD10_KEY_COL_IDX') is not None:
            self.use_indices = True
            self.key_col_idx = int(os.getenv('ICD10_KEY_COL_IDX', '0'))
            self.date_col_idx = int(os.getenv('ICD10_DATE_COL_IDX', '1'))
            self.code_col_idx = int(os.getenv('ICD10_CODE_COL_IDX', '2'))
            self.desc_col_idx = int(os.getenv('ICD10_DESC_COL_IDX', '22'))
            print(f"Using column indices: key={self.key_col_idx}, date={self.date_col_idx}, code={self.code_col_idx}, desc={self.desc_col_idx}")
        else:
            print(f"Using column names: key={self.key_col_name}, date={self.date_col_name}")
            print(f"Looking for diagnosis codes with prefix '{self.code_prefix}' and suffix '{self.code_suffix}'")
        
        # Get CKD-related ICD10 codes from environment variables
        default_ckd_codes = 'N18|N18.1|N18.2|N18.3|N18.4|N18.5|N18.9'
        ckd_codes_str = os.getenv('ICD10_CKD_CODES', default_ckd_codes)
        self.ckd_codes = ckd_codes_str.split('|')
        
        print(f"Using CKD codes: {self.ckd_codes}")
    
    def process(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process ICD10 diagnosis data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw ICD10 data
            
        Returns:
            DataFrame with processed ICD10 data
        """
        if combined_df.empty:
            print("No ICD10 data provided")
            return pd.DataFrame()
        
        # Select relevant columns
        icd10_df = pd.DataFrame()
        
        # Check if columns are numeric (like '0', '1', '2', etc.)
        numeric_columns = all(col.isdigit() for col in combined_df.columns if isinstance(col, str) and col not in ['source_file', 'year', 'quarter'])
        
        # If columns are numeric or use_indices is True, use column indices
        if numeric_columns or self.use_indices:
            # Define default indices if not already set
            if not hasattr(self, 'key_col_idx'):
                self.key_col_idx = 0
                self.date_col_idx = 1
                self.code_col_idx = 2
                self.desc_col_idx = 22
                print(f"Using default column indices: key={self.key_col_idx}, date={self.date_col_idx}, code={self.code_col_idx}, desc={self.desc_col_idx}")
            
            # Use column indices approach
            max_col_idx = max(self.key_col_idx, self.date_col_idx, self.code_col_idx, self.desc_col_idx)
            
            if combined_df.shape[1] > max_col_idx:
                # Extract columns using the configured indices
                icd10_df['key'] = combined_df.iloc[:, self.key_col_idx]
                icd10_df['date'] = combined_df.iloc[:, self.date_col_idx]
                icd10_df['icd10_code'] = combined_df.iloc[:, self.code_col_idx]
                icd10_df['icd10_desc'] = combined_df.iloc[:, self.desc_col_idx]
            else:
                print(f"Warning: ICD10 data has only {combined_df.shape[1]} columns, but max index is {max_col_idx}")
                # Create an empty DataFrame with the right columns
                icd10_df = pd.DataFrame(columns=['key', 'date', 'icd10_code', 'icd10_desc'])
        else:
            # Use column names approach
            # Check if key and date columns exist
            if self.key_col_name in combined_df.columns and self.date_col_name in combined_df.columns:
                # Extract key and date columns
                icd10_df['key'] = combined_df[self.key_col_name]
                icd10_df['date'] = combined_df[self.date_col_name]
                
                # Find diagnosis code columns using prefix and suffix
                code_columns = [col for col in combined_df.columns
                               if self.code_prefix in col and self.code_suffix in col]
                
                if not code_columns:
                    # Try to find columns with just the suffix if no columns match both prefix and suffix
                    code_columns = [col for col in combined_df.columns if self.code_suffix in col]
                
                if code_columns:
                    print(f"Found {len(code_columns)} diagnosis code columns: {code_columns}")
                    
                    # Create a temporary DataFrame to hold all diagnosis codes
                    all_codes_df = pd.DataFrame()
                    all_codes_df['key'] = icd10_df['key']
                    all_codes_df['date'] = icd10_df['date']
                    
                    # Extract all diagnosis codes from all matching columns
                    for i, col in enumerate(code_columns):
                        temp_df = pd.DataFrame()
                        temp_df['key'] = icd10_df['key']
                        temp_df['date'] = icd10_df['date']
                        temp_df['icd10_code'] = combined_df[col]
                        
                        # Try to find corresponding description column
                        desc_col = col.replace(self.code_suffix, "Description")
                        if desc_col in combined_df.columns:
                            temp_df['icd10_desc'] = combined_df[desc_col]
                        else:
                            temp_df['icd10_desc'] = np.nan
                        
                        # Append to the result
                        all_codes_df = pd.concat([all_codes_df, temp_df], ignore_index=True)
                    
                    # Remove rows with missing or empty diagnosis codes
                    all_codes_df = all_codes_df.dropna(subset=['icd10_code'])
                    all_codes_df = all_codes_df[all_codes_df['icd10_code'].astype(str).str.strip() != '']
                    
                    # Use the processed DataFrame
                    icd10_df = all_codes_df
                else:
                    print(f"Warning: No diagnosis code columns found matching prefix '{self.code_prefix}' and suffix '{self.code_suffix}'")
                    # Create an empty DataFrame with the right columns
                    icd10_df = pd.DataFrame(columns=['key', 'date', 'icd10_code', 'icd10_desc'])
            else:
                missing_cols = []
                if self.key_col_name not in combined_df.columns:
                    missing_cols.append(self.key_col_name)
                if self.date_col_name not in combined_df.columns:
                    missing_cols.append(self.date_col_name)
                
                print(f"Warning: Required columns missing from ICD10 data: {missing_cols}")
                print(f"Available columns: {combined_df.columns.tolist()}")
                
                # If columns are missing but we have numeric columns, try using indices as a fallback
                if numeric_columns:
                    print("Falling back to using column indices since columns appear to be numeric")
                    # Define default indices
                    self.key_col_idx = 0
                    self.date_col_idx = 1
                    self.code_col_idx = 2
                    self.desc_col_idx = 22
                    
                    # Extract columns using indices
                    if combined_df.shape[1] > max(self.key_col_idx, self.date_col_idx, self.code_col_idx, self.desc_col_idx):
                        icd10_df['key'] = combined_df.iloc[:, self.key_col_idx]
                        icd10_df['date'] = combined_df.iloc[:, self.date_col_idx]
                        icd10_df['icd10_code'] = combined_df.iloc[:, self.code_col_idx]
                        icd10_df['icd10_desc'] = combined_df.iloc[:, self.desc_col_idx]
                    else:
                        # Create an empty DataFrame with the right columns
                        icd10_df = pd.DataFrame(columns=['key', 'date', 'icd10_code', 'icd10_desc'])
                else:
                    # Create an empty DataFrame with the right columns
                    icd10_df = pd.DataFrame(columns=['key', 'date', 'icd10_code', 'icd10_desc'])
        
        # Add metadata columns
        if 'source_file' in combined_df.columns:
            icd10_df['source_file'] = combined_df['source_file']
        if 'year' in combined_df.columns:
            icd10_df['year'] = combined_df['year']
        if 'quarter' in combined_df.columns:
            icd10_df['quarter'] = combined_df['quarter']
        
        # Drop rows with missing key, date, or icd10_code values
        icd10_df = icd10_df.dropna(subset=['key', 'date', 'icd10_code'])
        
        # Convert date to datetime
        # Using errors='coerce' to handle various date formats
        icd10_df['date'] = pd.to_datetime(icd10_df['date'], errors='coerce')
        
        # Convert key to numeric
        icd10_df['key'] = pd.to_numeric(icd10_df['key'], errors='coerce')
        
        # Add a column to indicate if the code is CKD-related
        icd10_df['is_ckd'] = icd10_df['icd10_code'].astype(str).str.startswith(tuple(self.ckd_codes))
        
        # Add data type column
        icd10_df['data_type'] = 'icd10'
        
        # Drop duplicates
        icd10_df = icd10_df.drop_duplicates(subset=['key', 'date', 'icd10_code'])
        
        # Rename icd10_code to icd10 to match expected column name in downstream processing
        if 'icd10_code' in icd10_df.columns and 'icd10' not in icd10_df.columns:
            icd10_df = icd10_df.rename(columns={'icd10_code': 'icd10'})
        
        print(f"Final ICD10 dataset has {len(icd10_df)} rows and {len(icd10_df.columns)} columns")
        print(f"CKD-related diagnoses: {icd10_df['is_ckd'].sum()}")
        
        return icd10_df
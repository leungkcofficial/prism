"""operation ingestion module for CKD survival model development
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

class OperationIngester(LabResultIngester):
    """Ingests operation data"""
    
    def __init__(self):
        """Initialize the operation ingester with configuration"""
        # Get column indices from environment variables
        self.key_col_idx = int(os.getenv('OPERATION_KEY_COL_IDX', '0'))  # Reference Key is at index 0
        self.date_col_idx = int(os.getenv('OPERATION_DATE_COL_IDX', '1'))  # OT Date is at index 1
        
        # Operation codes are in columns 3-17 (up to 15 operations per row)
        self.code_col_start_idx = int(os.getenv('OPERATION_CODE_START_COL_IDX', '3'))  # OT Procedure (1) is at index 3
        self.code_col_end_idx = int(os.getenv('OPERATION_CODE_END_COL_IDX', '17'))  # OT Procedure (15) is at index 17
        
        # Operation descriptions are in columns 18-32
        self.desc_col_start_idx = int(os.getenv('OPERATION_DESC_START_COL_IDX', '18'))  # OT Procedure (1) Description is at index 18
        self.desc_col_end_idx = int(os.getenv('OPERATION_DESC_END_COL_IDX', '32'))  # OT Procedure (15) Description is at index 32
        
        # Get dialysis-related operation codes from environment variables
        # Using ICD9 procedure codes for dialysis
        default_dialysis_codes = '54.93,39.27,38.95'
        dialysis_codes_str = os.getenv('OPERATION_DIALYSIS_CODES', default_dialysis_codes)
        self.dialysis_codes = dialysis_codes_str.split(',')
        
        # Transplant codes not needed for this project
        # default_transplant_codes = 'M0100|M0101|M0102|M0103|M0104|M0108|M0109'
        # transplant_codes_str = os.getenv('OPERATION_TRANSPLANT_CODES', default_transplant_codes)
        # self.transplant_codes = transplant_codes_str.split('|')
        self.transplant_codes = []
        
        print(f"Using column indices: key={self.key_col_idx}, date={self.date_col_idx}, code={self.code_col_start_idx}-{self.code_col_end_idx}, desc={self.desc_col_start_idx}-{self.desc_col_end_idx}")
        print(f"Using dialysis codes: {self.dialysis_codes}")
        print(f"Using transplant codes: {self.transplant_codes}")
    
    def process(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process operation data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw operation data
            
        Returns:
            DataFrame with processed operation data
        """
        if combined_df.empty:
            print("No operation data provided")
            return pd.DataFrame()
        
        # Select relevant columns based on configured indices
        operation_df = pd.DataFrame()
        
        # Check if we have enough columns
        max_col_idx = max(self.key_col_idx, self.date_col_idx, self.code_col_end_idx, self.desc_col_end_idx)
        
        if combined_df.shape[1] > max_col_idx:
            # Extract key and date columns
            operation_df['key'] = combined_df.iloc[:, self.key_col_idx]
            operation_df['date'] = combined_df.iloc[:, self.date_col_idx]
            
            # Extract all operation codes and descriptions
            # First, create a DataFrame with just the key and date
            base_df = pd.DataFrame({
                'key': combined_df.iloc[:, self.key_col_idx],
                'date': combined_df.iloc[:, self.date_col_idx]
            })
            
            # Create a list to store all operation records
            all_operations = []
            
            # Process each row
            for idx, row in combined_df.iterrows():
                # Get the key and date for this row
                key = row.iloc[self.key_col_idx]
                date = row.iloc[self.date_col_idx]
                
                # Process each operation code column
                for code_idx in range(self.code_col_start_idx, self.code_col_end_idx + 1):
                    # Skip if the code is empty
                    if code_idx >= len(row) or pd.isna(row.iloc[code_idx]) or row.iloc[code_idx] == '':
                        continue
                    
                    # Get the corresponding description index
                    desc_idx = self.desc_col_start_idx + (code_idx - self.code_col_start_idx)
                    
                    # Get the code and description
                    op_code = row.iloc[code_idx]
                    op_desc = row.iloc[desc_idx] if desc_idx < len(row) and not pd.isna(row.iloc[desc_idx]) else ''
                    
                    # Add this operation to the list
                    all_operations.append({
                        'key': key,
                        'date': date,
                        'op_code': op_code,
                        'op_desc': op_desc
                    })
            
            # Convert the list to a DataFrame
            if all_operations:
                operation_df = pd.DataFrame(all_operations)
            else:
                # Create an empty DataFrame with the right columns
                operation_df = pd.DataFrame(columns=['key', 'date', 'op_code', 'op_desc'])
        else:
            print(f"Warning: Operation data has only {combined_df.shape[1]} columns, but max index is {max_col_idx}")
            # Create an empty DataFrame with the right columns
            operation_df = pd.DataFrame(columns=['key', 'date', 'op_code', 'op_desc'])
        
        # Add metadata columns
        if 'source_file' in combined_df.columns:
            operation_df['source_file'] = combined_df['source_file']
        if 'year' in combined_df.columns:
            operation_df['year'] = combined_df['year']
        if 'quarter' in combined_df.columns:
            operation_df['quarter'] = combined_df['quarter']
        
        # Drop rows with missing key, date, or op_code values
        operation_df = operation_df.dropna(subset=['key', 'date', 'op_code'])
        
        # Convert date to datetime
        # Convert date to datetime with explicit format to avoid warnings
        # Using standard format '%Y-%m-%d' - adjust if your data uses a different format
        operation_df['date'] = pd.to_datetime(operation_df['date'], format='%Y-%m-%d', errors='coerce')
        
        # Convert key to numeric
        operation_df['key'] = pd.to_numeric(operation_df['key'], errors='coerce')
        
        # Add columns to indicate if the operation is dialysis or transplant
        # For ICD9 codes, we need to check for exact matches, not just startswith
        operation_df['is_dialysis'] = operation_df['op_code'].isin(self.dialysis_codes)
        operation_df['is_transplant'] = False  # Not using transplant codes in this project
        
        # Add data type column
        operation_df['data_type'] = 'operation'
        
        # Drop duplicates
        operation_df = operation_df.drop_duplicates(subset=['key', 'date', 'op_code'])
        
        print(f"Final operation dataset has {len(operation_df)} rows and {len(operation_df.columns)} columns")
        print(f"Dialysis operations: {operation_df['is_dialysis'].sum()}")
        print(f"Transplant operations: {operation_df['is_transplant'].sum()}")
        
        return operation_df
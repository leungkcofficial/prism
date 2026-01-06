"""Death ingestion module for CKD survival model development
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

class DeathIngester(LabResultIngester):
    """Ingests death data"""
    
    def __init__(self):
        """Initialize the death ingester with configuration"""
        # Get column indices from environment variables
        self.key_col_idx = int(os.getenv('DEATH_KEY_COL_IDX', '3'))  # Reference Key is at index 3
        self.date_col_idx = int(os.getenv('DEATH_DATE_COL_IDX', '8'))  # Death Date is at index 8
        
        print(f"Using column indices: key={self.key_col_idx}, date={self.date_col_idx}")
    
    def process(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process death data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw death data
            
        Returns:
            DataFrame with processed death data
        """
        if combined_df.empty:
            print("No death data provided")
            return pd.DataFrame()
        
        # Select relevant columns based on configured indices
        death_df = pd.DataFrame()
        
        # Check if we have enough columns
        max_col_idx = max(self.key_col_idx, self.date_col_idx)
        
        if combined_df.shape[1] > max_col_idx:
            # Extract columns using the configured indices
            death_df['key'] = combined_df.iloc[:, self.key_col_idx]
            death_df['death_date'] = combined_df.iloc[:, self.date_col_idx]
        else:
            print(f"Warning: Death data has only {combined_df.shape[1]} columns, but max index is {max_col_idx}")
            # Create an empty DataFrame with the right columns
            death_df = pd.DataFrame(columns=['key', 'death_date'])
        
        # Add metadata columns
        if 'source_file' in combined_df.columns:
            death_df['source_file'] = combined_df['source_file']
        if 'year' in combined_df.columns:
            death_df['year'] = combined_df['year']
        if 'quarter' in combined_df.columns:
            death_df['quarter'] = combined_df['quarter']
        
        # Drop rows with missing key or death_date values
        death_df = death_df.dropna(subset=['key', 'death_date'])
        
        # Convert death_date to datetime
        death_df['death_date'] = pd.to_datetime(death_df['death_date'], errors='coerce')
        
        # Convert key to numeric
        death_df['key'] = pd.to_numeric(death_df['key'], errors='coerce')
        
        # Add a death indicator column
        death_df['death'] = 1
        
        # Add data type column
        death_df['data_type'] = 'death'
        
        # Drop duplicates
        death_df = death_df.drop_duplicates(subset=['key'])
        
        print(f"Final death dataset has {len(death_df)} rows and {len(death_df.columns)} columns")
        
        return death_df

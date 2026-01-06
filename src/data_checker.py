"""
Data Checker module for CKD survival model development
This module handles the checking of dataframe structure to the appropriate format for the model
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

class DataChecker(ABC):
    """
    Abstract base class for data mappers
    """
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Process data from a DataFrame.
        
        Args:
            data: DataFrame containing raw data
            
        Returns:
            Processed DataFrame or tuple of DataFrames
        """
        pass


class DataCheckerFactory:
    """
    Factory class for creating data mappers
    """
    
    @staticmethod
    def create_checker(dataframe: str, column_structure) -> DataChecker:
        """
        Create a data checker for the specified data type.
        
        Args:
            dataframe: Type of data to map (e.g., 'creatinine', 'icd10')
            column_structure: Dictionary with expected column structure of the DataFrame
            
        Returns:
            DataChecker instance for the specified data type
        """
        return StandardDataChecker(dataframe, column_structure)


class StandardDataChecker(DataChecker):
    """
    Standard data checker for checking and fixing DataFrame column structure
    """
    
    def __init__(self, dataframe_name: str, expected_columns: List[str]):
        """
        Initialize the standard data checker with configuration.
        
        Args:
            dataframe_name: Name of the DataFrame (e.g., 'cr_df', 'hb_df')
            expected_columns: List of expected column names
        """
        self.dataframe_name = dataframe_name
        self.expected_columns = expected_columns
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame to ensure it has the expected column structure.
        
        Args:
            df: DataFrame to check and fix
            
        Returns:
            DataFrame with the expected column structure
        """
        if df.empty:
            return pd.DataFrame(columns=self.expected_columns)
        
        # Keep only the expected columns if they exist
        existing_columns = [col for col in self.expected_columns if col in df.columns]
        if existing_columns:
            df = df[existing_columns]
        
        # Add missing columns
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Reorder columns to match expected order
        df = df[self.expected_columns]
        
        return df


def load_column_structure(yaml_path: str) -> Dict[str, List[str]]:
    """
    Load column structure from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Dictionary with DataFrame names as keys and lists of column names as values
    """
    from src.util import load_yaml_file
    
    config = load_yaml_file(yaml_path)
    return config.get('expected_columns', {})


def check_dataframe(df: pd.DataFrame, dataframe_name: str, column_structure: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Check and fix a DataFrame's column structure.
    
    Args:
        df: DataFrame to check and fix
        dataframe_name: Name of the DataFrame (e.g., 'cr_df', 'hb_df')
        column_structure: Dictionary with DataFrame names as keys and lists of column names as values
        
    Returns:
        DataFrame with the expected column structure
    """
    if dataframe_name not in column_structure:
        print(f"Warning: No column structure defined for {dataframe_name}")
        return df
    
    expected_columns = column_structure[dataframe_name]
    checker = DataCheckerFactory.create_checker(dataframe_name, expected_columns)
    return checker.process(df)

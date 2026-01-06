"""Lab result ingestion module for CKD survival model development
This module handles the csv data and mapping information loaded from the pipeline, then output the dataframe
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod

# Import the new mapper classes
from src.lab_result_mapper import (
    LabResultMapper as BaseLabResultMapper,
    StandardLabResultMapper,
    UrineLabResultMapper,
    CalciumLabResultMapper
)

# For backward compatibility
class LabResultIngester(ABC):
    """Abstract base class for lab result ingester"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process lab result data and return a processed DataFrame or tuple of DataFrames"""
        pass

# Create adapter classes for backward compatibility
class UrineLabResultIngester(LabResultIngester):
    """Ingests urine lab result data"""
    
    def __init__(self):
        """Initialize the urine lab result ingester with configuration"""
        # Pass an empty string to get both protein and albumin data
        self.mapper = UrineLabResultMapper("")
    
    def process(self, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process urine protein/albumin data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw urine protein/albumin data
            
        Returns:
            Tuple containing:
            - DataFrame with urine protein creatinine ratio data (upcr_df)
            - DataFrame with urine albumin creatinine ratio data (uacr_df)
        """
        return self.mapper.process(combined_df)


class CalciumLabResultIngester(LabResultIngester):
    """Ingests calcium lab result data"""
    
    def __init__(self):
        """Initialize the calcium lab result ingester with configuration"""
        self.mapper = CalciumLabResultMapper("calcium")
    
    def process(self, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process calcium data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw calcium data
            
        Returns:
            Tuple containing:
            - DataFrame with non-adjusted calcium data
            - DataFrame with adjusted calcium data
        """
        return self.mapper.process(combined_df)


class StandardLabResultIngester(LabResultIngester):
    """Ingests standard lab result data (creatinine, hemoglobin, etc.)"""
    
    def __init__(self, data_type: str, validation_rules: Dict = None):
        """
        Initialize the standard lab result ingester with configuration.
        
        Args:
            data_type: Type of data to ingest (e.g., 'creatinine')
            validation_rules: Dictionary containing validation rules for this data type
        """
        self.mapper = StandardLabResultMapper(data_type, validation_rules)
        self.data_type = data_type
    
    def process(self, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process standard lab result data from a DataFrame.
        
        Args:
            combined_df: DataFrame containing raw lab result data
            
        Returns:
            Tuple containing:
            - DataFrame with lab data (key, date, code, value)
            - DataFrame with demographic data (key, dob, gender) if data_type is 'creatinine', otherwise None
        """
        return self.mapper.process(combined_df)


class LabResultIngesterFactory:
    """Factory class for creating lab result ingesters"""
    
    @staticmethod
    def create_ingester(data_type: str, validation_rules: Dict = None) -> LabResultIngester:
        """
        Create an appropriate lab result ingester based on the data type.
        
        Args:
            data_type: Type of data to ingest (e.g., 'creatinine', 'urine', 'calcium')
            validation_rules: Optional dictionary containing validation rules
            
        Returns:
            An instance of a LabResultIngester subclass
            
        Raises:
            ValueError: If the data type is not supported
        """
        # Map data types to ingester classes
        ingester_map = {
            'urine': UrineLabResultIngester,
            'urine_protein_creatinine_ratio': UrineLabResultIngester,
            'urine_albumin_creatinine_ratio': UrineLabResultIngester,
            'calcium': CalciumLabResultIngester,
            'calcium_adjusted': CalciumLabResultIngester,
            'creatinine': lambda: StandardLabResultIngester('creatinine', validation_rules),
            'hemoglobin': lambda: StandardLabResultIngester('hemoglobin', validation_rules),
            'hemoglobin_a1c': lambda: StandardLabResultIngester('hemoglobin_a1c', validation_rules),
            'albumin': lambda: StandardLabResultIngester('albumin', validation_rules),
            'phosphate': lambda: StandardLabResultIngester('phosphate', validation_rules),
            'icd10': None,  # Will be imported from dx_ingester.py
            'death': None,  # Will be imported from death_ingester.py
            'operation': None  # Will be imported from ot_ingester.py
        }
        
        # Import specialized ingesters if needed
        if data_type == 'icd10':
            from src.dx_ingester import ICD10Ingester
            return ICD10Ingester()
        elif data_type == 'death':
            from src.death_ingester import DeathIngester
            return DeathIngester()
        elif data_type == 'operation':
            from src.ot_ingester import OperationIngester
            return OperationIngester()
            
        # Check if the data type is supported
        if data_type not in ingester_map:
            raise ValueError(f"Unsupported data type: {data_type}. Supported types are: {list(ingester_map.keys())}")
        
        # Create and return the appropriate ingester
        ingester_class = ingester_map[data_type]
        
        # If the value is a function (for StandardLabResultIngester), call it
        if callable(ingester_class):
            return ingester_class()
        
        # Otherwise, instantiate the class
        return ingester_class()
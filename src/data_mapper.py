"""
Data Mapper module for CKD survival model development
This module handles the mapping of data from CSV files to the appropriate format for the model
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

class DataMapper(ABC):
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


class DataMapperFactory:
    """
    Factory class for creating data mappers
    """
    
    @staticmethod
    def create_mapper(data_type: str, validation_rules: Optional[Dict[str, Any]] = None) -> DataMapper:
        """
        Create a data mapper for the specified data type.
        
        Args:
            data_type: Type of data to map (e.g., 'creatinine', 'icd10')
            validation_rules: Dictionary with validation rules (min_value, max_value, unit)
            
        Returns:
            DataMapper instance for the specified data type
        """
        # Standard lab result data types
        standard_lab_types = [
            'creatinine', 'hemoglobin', 'hemoglobin_a1c', 'albumin', 'phosphate', 'bicarbonate'
        ]
        
        if data_type in standard_lab_types:
            from src.lab_result_mapper import StandardLabResultMapper
            return StandardLabResultMapper(data_type, validation_rules)
        elif data_type in ['urine_protein_creatinine_ratio', 'urine_albumin_creatinine_ratio']:
            from src.lab_result_mapper import UrineLabResultMapper
            return UrineLabResultMapper(data_type, validation_rules)
        elif data_type in ['calcium', 'calcium_adjusted']:
            from src.lab_result_mapper import CalciumLabResultMapper
            return CalciumLabResultMapper(data_type, validation_rules)
        elif data_type == 'icd10':
            from src.dx_ingester import ICD10Ingester
            return ICD10Ingester()
        elif data_type == 'death':
            from src.death_ingester import DeathIngester
            return DeathIngester()
        elif data_type == 'operation':
            from src.ot_ingester import OperationIngester
            return OperationIngester()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
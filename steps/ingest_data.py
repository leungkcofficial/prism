"""
Data Ingestion Step for CKD Risk Prediction

This module contains the ZenML step for ingesting lab result data.
"""

import os
import glob
import pandas as pd
import numpy as np
import re
import yaml
from zenml.steps import step
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
from pathlib import Path
from src.data_checker import load_column_structure, check_dataframe

from src.data_ingester import CSVDataIngester
from src.data_mapper import DataMapperFactory
from src.dx_ingester import ICD10Ingester
from src.death_ingester import DeathIngester
from src.ot_ingester import OperationIngester

# Load environment variables
load_dotenv()

# Load data types from YAML file
def load_yaml_file(file_path):
    """Load a YAML file and return its contents."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}

# Define paths to YAML files
data_types_yaml_path = Path("src/data_types.yml")
validation_rules_yaml_path = Path("src/default_data_validation_rules.yml")

# Load data types from YAML file
data_types_config = load_yaml_file(data_types_yaml_path)
DEFAULT_DATA_TYPES = data_types_config.get('DEFAULT_DATA_TYPES', {})

# Use data types directly from the YAML file
DATA_TYPES = DEFAULT_DATA_TYPES

# Extract lab data types for easier access
LAB_DATA_TYPES = DATA_TYPES.get('lab_data', {})
CLINICAL_DATA_TYPES = DATA_TYPES.get('clinical_data', {})

print(f"Loaded data types from {data_types_yaml_path}")

# Load validation rules from YAML file
validation_rules_config = load_yaml_file(validation_rules_yaml_path)
VALIDATION_RULES = validation_rules_config.get('DEFAULT_VALIDATION_RULES', {})

print(f"Loaded validation rules from {validation_rules_yaml_path}")

def load_lab_data(data_type: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load lab data for a specific data type using the appropriate ingester.
    
    Args:
        data_type: Type of data to load (e.g., 'creatinine')
        
    Returns:
        Tuple containing:
        - DataFrame with lab data
        - DataFrame with demographic data (key, dob, gender) if data_type is 'creatinine', otherwise None
    """
    data_dir = LAB_DATA_TYPES.get(data_type)
    if data_dir is None:
        print(f"Unknown lab data type: {data_type}")
        return pd.DataFrame(), None if data_type != 'creatinine' else pd.DataFrame()
    
    data_path = f"./data/{data_dir}"
    
    print(f"Loading {data_type} data from {data_path}")
    
    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"Directory not found: {data_path}")
        return pd.DataFrame(), None if data_type != 'creatinine' else pd.DataFrame()
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return pd.DataFrame(), None if data_type != 'creatinine' else pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files for {data_type}")
    
    # Create a CSV data ingester to load the data
    csv_ingester = CSVDataIngester(data_path)
    
    # Load the data
    combined_df = csv_ingester.ingest()
    
    if combined_df.empty:
        print(f"No data loaded for {data_type}")
        return pd.DataFrame(), None if data_type != 'creatinine' else pd.DataFrame()
    
    # Create the appropriate data mapper using the factory
    data_mapper = DataMapperFactory.create_mapper(data_type, VALIDATION_RULES.get(data_type))
    
    # Special handling for different data types
    if data_type in ['calcium', 'calcium_adjusted']:
        ca_df, ca_adjusted_df = data_mapper.process(combined_df)
        if data_type == 'calcium':
            return ca_df, None
        else:  # data_type == 'calcium_adjusted'
            return ca_adjusted_df, None
    
    elif data_type in ['urine_protein_creatinine_ratio', 'urine_albumin_creatinine_ratio']:
        result = data_mapper.process(combined_df)
        # The mapper now returns a single DataFrame for the specific data_type
        if isinstance(result, pd.DataFrame):
            return result, None
        # For backward compatibility, handle the case where it returns a tuple
        elif isinstance(result, tuple) and len(result) == 2:
            upcr_df, uacr_df = result
            if data_type == 'urine_protein_creatinine_ratio':
                return upcr_df, None
            else:  # data_type == 'urine_albumin_creatinine_ratio'
                return uacr_df, None
        else:
            print(f"Unexpected result type from urine data mapper: {type(result)}")
            return pd.DataFrame(), None
    
    elif data_type == 'creatinine':
        # For creatinine, we also get demographic data
        lab_df, demo_df = data_mapper.process(combined_df)
        return lab_df, demo_df
    
    else:
        # For other data types, we just get the lab data
        # StandardLabDataMapper.process always returns a tuple of (lab_df, demo_df)
        lab_df, _ = data_mapper.process(combined_df)
        return lab_df, None


def load_clinical_data(data_type: str) -> pd.DataFrame:
    """
    Load clinical data (ICD10, death, operation) for a specific data type using the appropriate ingester.
    
    Args:
        data_type: Type of data to load (e.g., 'icd10', 'death', 'operation')
        
    Returns:
        DataFrame with clinical data
    """
    data_dir = CLINICAL_DATA_TYPES.get(data_type)
    if data_dir is None:
        print(f"Unknown clinical data type: {data_type}")
        return pd.DataFrame()
    
    data_path = f"./data/{data_dir}"
    
    print(f"Loading {data_type} data from {data_path}")
    
    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"Directory not found: {data_path}")
        return pd.DataFrame()
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files for {data_type}")
    
    # Create a CSV data ingester to load the data
    # For ICD10 data, use header_row=1 to skip the first row with numeric indices
    if data_type == 'icd10':
        csv_ingester = CSVDataIngester(data_path, header_row=1)
    else:
        csv_ingester = CSVDataIngester(data_path)
    
    # Load the data
    combined_df = csv_ingester.ingest()
    
    if combined_df.empty:
        print(f"No data loaded for {data_type}")
        return pd.DataFrame()
    
    # Create the appropriate data mapper using the factory
    data_mapper = DataMapperFactory.create_mapper(data_type)
    
    # Process the data
    result_df = data_mapper.process(combined_df)
    
    return result_df


@step
def ingest_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                           pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                           pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ingest lab result and clinical data for multiple data types.
    
    Returns:
        Tuple containing:
        - DataFrame with creatinine data (cr_df)
        - DataFrame with hemoglobin data (hb_df)
        - DataFrame with hemoglobin A1c data (a1c_df)
        - DataFrame with albumin data (alb_df)
        - DataFrame with phosphate data (po4_df)
        - DataFrame with calcium data (ca_df)
        - DataFrame with adjusted calcium data (ca_adjusted_df)
        - DataFrame with bicarbonate data (hco3_df)
        - DataFrame with urine protein creatinine ratio data (upcr_df)
        - DataFrame with urine albumin creatinine ratio data (uacr_df)
        - DataFrame with demographic data (demo_df)
        - DataFrame with ICD10 diagnosis data (icd10_df)
        - DataFrame with death data (death_df)
        - DataFrame with operation data (operation_df)
    """
    try:
        # Initialize DataFrames
        cr_df = pd.DataFrame()
        hb_df = pd.DataFrame()
        a1c_df = pd.DataFrame()
        alb_df = pd.DataFrame()
        po4_df = pd.DataFrame()
        ca_df = pd.DataFrame()
        ca_adjusted_df = pd.DataFrame()
        hco3_df = pd.DataFrame()
        upcr_df = pd.DataFrame()
        uacr_df = pd.DataFrame()
        demo_df = pd.DataFrame()
        icd10_df = pd.DataFrame()
        death_df = pd.DataFrame()
        operation_df = pd.DataFrame()
        
        # Define the data types to load
        standard_data_types = ['creatinine', 'hemoglobin', 'hemoglobin_a1c', 'albumin', 'phosphate', 'bicarbonate']
        
        # Load standard lab data
        for data_type in standard_data_types:
            print(f"\n=== Loading {data_type} data ===\n")
            result = load_lab_data(data_type)
            
            # Check if result is a tuple or a DataFrame
            if isinstance(result, tuple):
                lab_df, current_demo_df = result
            else:
                lab_df = result
                current_demo_df = None
            
            # Now check if lab_df is a DataFrame and not empty
            if isinstance(lab_df, pd.DataFrame) and not lab_df.empty:
                # Store lab data in the appropriate DataFrame
                if data_type == 'creatinine':
                    cr_df = lab_df
                elif data_type == 'hemoglobin':
                    hb_df = lab_df
                elif data_type == 'hemoglobin_a1c':
                    a1c_df = lab_df
                elif data_type == 'albumin':
                    alb_df = lab_df
                elif data_type == 'phosphate':
                    po4_df = lab_df
                elif data_type == 'bicarbonate':
                    hco3_df = lab_df
            
            # Only use demographic data from creatinine
            if current_demo_df is not None and isinstance(current_demo_df, pd.DataFrame) and not current_demo_df.empty:
                demo_df = current_demo_df
        
        # Load calcium data
        print(f"\n=== Loading calcium data ===\n")
        result = load_lab_data('calcium')
        if isinstance(result, tuple):
            ca_df, _ = result
        else:
            ca_df = result
        
        # Load adjusted calcium data
        print(f"\n=== Loading adjusted calcium data ===\n")
        result = load_lab_data('calcium_adjusted')
        if isinstance(result, tuple):
            ca_adjusted_df, _ = result
        else:
            ca_adjusted_df = result
        
        # Load urine protein/albumin data
        print(f"\n=== Loading urine protein/albumin data ===\n")
        result = load_lab_data('urine_protein_creatinine_ratio')
        if isinstance(result, tuple):
            upcr_df, _ = result
        else:
            upcr_df = result
        
        # Load urine albumin/creatinine data
        print(f"\n=== Loading urine albumin/creatinine data ===\n")
        result = load_lab_data('urine_albumin_creatinine_ratio')
        if isinstance(result, tuple):
            uacr_df, _ = result
        else:
            uacr_df = result
        
        # Load ICD-10, operation, and death data
        print(f"\n=== Loading ICD10 diagnosis data ===\n")
        icd10_df = load_clinical_data('icd10')
        
        print(f"\n=== Loading death data ===\n")
        death_df = load_clinical_data('death')
        
        print(f"\n=== Loading operation data ===\n")
        operation_df = load_clinical_data('operation')
        
        # Print summary information
        print(f"\n=== Data Summary ===")
        print(f"Creatinine data: {len(cr_df) if isinstance(cr_df, pd.DataFrame) else 0} rows")
        print(f"Hemoglobin data: {len(hb_df) if isinstance(hb_df, pd.DataFrame) else 0} rows")
        print(f"Hemoglobin A1c data: {len(a1c_df) if isinstance(a1c_df, pd.DataFrame) else 0} rows")
        print(f"Albumin data: {len(alb_df) if isinstance(alb_df, pd.DataFrame) else 0} rows")
        print(f"Phosphate data: {len(po4_df) if isinstance(po4_df, pd.DataFrame) else 0} rows")
        print(f"Calcium data: {len(ca_df) if isinstance(ca_df, pd.DataFrame) else 0} rows")
        print(f"Adjusted Calcium data: {len(ca_adjusted_df) if isinstance(ca_adjusted_df, pd.DataFrame) else 0} rows")
        print(f"Bicarbonate data: {len(hco3_df) if isinstance(hco3_df, pd.DataFrame) else 0} rows")
        print(f"Urine Protein Creatinine Ratio data: {len(upcr_df) if isinstance(upcr_df, pd.DataFrame) else 0} rows")
        print(f"Urine Albumin Creatinine Ratio data: {len(uacr_df) if isinstance(uacr_df, pd.DataFrame) else 0} rows")
        print(f"Demographic data: {len(demo_df) if isinstance(demo_df, pd.DataFrame) else 0} rows")
        print(f"ICD10 diagnosis data: {len(icd10_df) if isinstance(icd10_df, pd.DataFrame) else 0} rows")
        print(f"Death data: {len(death_df) if isinstance(death_df, pd.DataFrame) else 0} rows")
        print(f"Operation data: {len(operation_df) if isinstance(operation_df, pd.DataFrame) else 0} rows")
        
        # Load the expected column structure from the YAML file
        output_structure_yaml_path = "src/default_ingest_data_output_dataframe structure.yml"
        column_structure = load_column_structure(output_structure_yaml_path)
        if not column_structure:
            print("Using default column structure")
            # Fallback to default structure if YAML file cannot be loaded
            column_structure = {
                'cr_df': ['key', 'date', 'code', 'cr'],
                'hb_df': ['key', 'date', 'code', 'hb'],
                'a1c_df': ['key', 'date', 'code', 'a1c'],
                'alb_df': ['key', 'date', 'code', 'alb'],
                'po4_df': ['key', 'date', 'code', 'po4'],
                'ca_df': ['key', 'date', 'code', 'ca'],
                'ca_adjusted_df': ['key', 'date', 'code', 'ca_adjusted'],
                'hco3_df': ['key', 'date', 'code', 'hco3'],
                'upcr_df': ['key', 'date', 'code', 'upacr'],
                'uacr_df': ['key', 'date', 'code', 'upacr'],
                'demo_df': ['key', 'dob', 'gender'],
                'icd10_df': ['key', 'date', 'icd10'],
                'death_df': ['key', 'date', 'cause'],
                'operation_df': ['key', 'date']
            }
        
        # Ensure all returned values are DataFrames and rename columns to match expected format
        # Create copies to avoid modifying the original dataframes
        cr_df_out = cr_df.copy() if isinstance(cr_df, pd.DataFrame) and not cr_df.empty else pd.DataFrame()
        hb_df_out = hb_df.copy() if isinstance(hb_df, pd.DataFrame) and not hb_df.empty else pd.DataFrame()
        a1c_df_out = a1c_df.copy() if isinstance(a1c_df, pd.DataFrame) and not a1c_df.empty else pd.DataFrame()
        alb_df_out = alb_df.copy() if isinstance(alb_df, pd.DataFrame) and not alb_df.empty else pd.DataFrame()
        po4_df_out = po4_df.copy() if isinstance(po4_df, pd.DataFrame) and not po4_df.empty else pd.DataFrame()
        ca_df_out = ca_df.copy() if isinstance(ca_df, pd.DataFrame) and not ca_df.empty else pd.DataFrame()
        ca_adjusted_df_out = ca_adjusted_df.copy() if isinstance(ca_adjusted_df, pd.DataFrame) and not ca_adjusted_df.empty else pd.DataFrame()
        hco3_df_out = hco3_df.copy() if isinstance(hco3_df, pd.DataFrame) and not hco3_df.empty else pd.DataFrame()
        upcr_df_out = upcr_df.copy() if isinstance(upcr_df, pd.DataFrame) and not upcr_df.empty else pd.DataFrame()
        uacr_df_out = uacr_df.copy() if isinstance(uacr_df, pd.DataFrame) and not uacr_df.empty else pd.DataFrame()
        demo_df_out = demo_df.copy() if isinstance(demo_df, pd.DataFrame) and not demo_df.empty else pd.DataFrame()
        icd10_df_out = icd10_df.copy() if isinstance(icd10_df, pd.DataFrame) and not icd10_df.empty else pd.DataFrame()
        death_df_out = death_df.copy() if isinstance(death_df, pd.DataFrame) and not death_df.empty else pd.DataFrame()
        operation_df_out = operation_df.copy() if isinstance(operation_df, pd.DataFrame) and not operation_df.empty else pd.DataFrame()
        
        # Rename columns to match expected format in clean_data.py
        # For lab result dataframes, rename result_value to the specific lab name
        if not cr_df_out.empty and 'result_value' in cr_df_out.columns:
            cr_df_out = cr_df_out.rename(columns={'result_value': 'cr'})
        
        if not hb_df_out.empty and 'result_value' in hb_df_out.columns:
            hb_df_out = hb_df_out.rename(columns={'result_value': 'hb'})
        
        if not a1c_df_out.empty and 'result_value' in a1c_df_out.columns:
            a1c_df_out = a1c_df_out.rename(columns={'result_value': 'a1c'})
        
        if not alb_df_out.empty and 'result_value' in alb_df_out.columns:
            alb_df_out = alb_df_out.rename(columns={'result_value': 'alb'})
        
        if not po4_df_out.empty and 'result_value' in po4_df_out.columns:
            po4_df_out = po4_df_out.rename(columns={'result_value': 'po4'})
        
        if not hco3_df_out.empty and 'result_value' in hco3_df_out.columns:
            hco3_df_out = hco3_df_out.rename(columns={'result_value': 'hco3'})
        
        if not ca_df_out.empty and 'value' in ca_df_out.columns:
            ca_df_out = ca_df_out.rename(columns={'value': 'ca'})
        
        if not ca_adjusted_df_out.empty and 'value' in ca_adjusted_df_out.columns:
            ca_adjusted_df_out = ca_adjusted_df_out.rename(columns={'value': 'ca_adjusted'})
        
        if not upcr_df_out.empty and 'value' in upcr_df_out.columns:
            upcr_df_out = upcr_df_out.rename(columns={'value': 'upacr'})
        
        if not uacr_df_out.empty and 'value' in uacr_df_out.columns:
            uacr_df_out = uacr_df_out.rename(columns={'value': 'upacr'})
        icd10_df.to_csv('/mnt/dump/yard/projects/tarot2/tests/pipeline_test_endpoints.csv')
        # For ICD10 data, ensure it has the correct columns and format
        if not icd10_df_out.empty:
            # Rename icd10_code to icd10 if needed
            if 'icd10_code' in icd10_df_out.columns and 'icd10' not in icd10_df_out.columns:
                icd10_df_out = icd10_df_out.rename(columns={'icd10_code': 'icd10'})
            
            # Make sure the key column is properly formatted
            if 'key' in icd10_df_out.columns:
                icd10_df_out['key'] = pd.to_numeric(icd10_df_out['key'], errors='coerce')
            
            # Make sure the date column is properly formatted
            if 'date' in icd10_df_out.columns:
                icd10_df_out['date'] = pd.to_datetime(icd10_df_out['date'], errors='coerce')
            
            # Make sure is_ckd column is preserved if it exists
            if 'is_ckd' in icd10_df.columns and 'is_ckd' not in icd10_df_out.columns:
                icd10_df_out['is_ckd'] = icd10_df['is_ckd']
            
            print(f"Prepared ICD10 data with {len(icd10_df_out)} rows")
        
        # For death data, ensure it has the correct columns and format
        if not death_df_out.empty:
            # Ensure it has a 'cause' column
            if 'cause' not in death_df_out.columns:
                # If there's no cause column, create one with NaN values
                death_df_out['cause'] = np.nan
            
            # Make sure the key column is properly formatted
            if 'key' in death_df_out.columns:
                death_df_out['key'] = pd.to_numeric(death_df_out['key'], errors='coerce')
            
            # Handle the date column - ComorbidityProcessor.process_endpoints expects 'date', not 'death_date'
            if 'death_date' in death_df_out.columns and 'date' not in death_df_out.columns:
                # Rename death_date to date
                death_df_out = death_df_out.rename(columns={'death_date': 'date'})
            
            # Make sure the date column is properly formatted
            if 'date' in death_df_out.columns:
                death_df_out['date'] = pd.to_datetime(death_df_out['date'], errors='coerce')
            
            print(f"Prepared death data with {len(death_df_out)} rows")
        
        # For operation data, ensure it has the correct columns and format
        if not operation_df_out.empty:
            # Keep only key and date columns if they exist
            keep_cols = ['key', 'date']
            if all(col in operation_df_out.columns for col in keep_cols):
                operation_df_out = operation_df_out[keep_cols]
            
            # Make sure the key column is properly formatted
            if 'key' in operation_df_out.columns:
                operation_df_out['key'] = pd.to_numeric(operation_df_out['key'], errors='coerce')
            
            # Make sure the date column is properly formatted
            if 'date' in operation_df_out.columns:
                operation_df_out['date'] = pd.to_datetime(operation_df_out['date'], errors='coerce')
            
            # Add is_dialysis column if it exists in the original dataframe
            if 'is_dialysis' in operation_df.columns:
                operation_df_out['is_dialysis'] = operation_df['is_dialysis']
            
            print(f"Prepared operation data with {len(operation_df_out)} rows")
            
        # Ensure each DataFrame has the expected column structure
        print("Checking and fixing column structure for all DataFrames")
        cr_df_out = check_dataframe(cr_df_out, 'cr_df', column_structure)
        hb_df_out = check_dataframe(hb_df_out, 'hb_df', column_structure)
        a1c_df_out = check_dataframe(a1c_df_out, 'a1c_df', column_structure)
        alb_df_out = check_dataframe(alb_df_out, 'alb_df', column_structure)
        po4_df_out = check_dataframe(po4_df_out, 'po4_df', column_structure)
        ca_df_out = check_dataframe(ca_df_out, 'ca_df', column_structure)
        ca_adjusted_df_out = check_dataframe(ca_adjusted_df_out, 'ca_adjusted_df', column_structure)
        hco3_df_out = check_dataframe(hco3_df_out, 'hco3_df', column_structure)
        upcr_df_out = check_dataframe(upcr_df_out, 'upcr_df', column_structure)
        uacr_df_out = check_dataframe(uacr_df_out, 'uacr_df', column_structure)
        demo_df_out = check_dataframe(demo_df_out, 'demo_df', column_structure)
        icd10_df_out = check_dataframe(icd10_df_out, 'icd10_df', column_structure)
        death_df_out = check_dataframe(death_df_out, 'death_df', column_structure)
        operation_df_out = check_dataframe(operation_df_out, 'operation_df', column_structure)
        
        # Return the dataframes in the expected order
        return (
            cr_df_out,
            hb_df_out,
            a1c_df_out,
            alb_df_out,
            po4_df_out,
            ca_df_out,
            ca_adjusted_df_out,
            hco3_df_out,
            upcr_df_out,
            uacr_df_out,
            demo_df_out,
            icd10_df_out,
            death_df_out,
            operation_df_out
        )
    
    except Exception as e:
        print(f"Error ingesting data: {e}")
        
        # Load the expected column structure from the YAML file if not already loaded
        if 'expected_columns' not in locals():
            output_structure_yaml_path = Path("src/default_ingest_data_output_dataframe structure.yml")
            try:
                output_structure_config = load_yaml_file(output_structure_yaml_path)
                expected_columns = output_structure_config.get('expected_columns', {})
                print(f"Loaded output structure from {output_structure_yaml_path}")
            except Exception as e:
                print(f"Error loading output structure from {output_structure_yaml_path}: {e}")
                print("Using default output structure")
                # Fallback to default structure if YAML file cannot be loaded
                expected_columns = {
                    'cr_df': ['key', 'date', 'code', 'cr'],
                    'hb_df': ['key', 'date', 'code', 'hb'],
                    'a1c_df': ['key', 'date', 'code', 'a1c'],
                    'alb_df': ['key', 'date', 'code', 'alb'],
                    'po4_df': ['key', 'date', 'code', 'po4'],
                    'ca_df': ['key', 'date', 'code', 'ca'],
                    'ca_adjusted_df': ['key', 'date', 'code', 'ca_adjusted'],
                    'hco3_df': ['key', 'date', 'code', 'hco3'],
                    'upcr_df': ['key', 'date', 'code', 'upacr'],
                    'uacr_df': ['key', 'date', 'code', 'upacr'],
                    'demo_df': ['key', 'dob', 'gender'],
                    'icd10_df': ['key', 'date', 'icd10'],
                    'death_df': ['key', 'date', 'cause'],
                    'operation_df': ['key', 'date']
                }
        
        # Return empty DataFrames if there's an error
        # Create empty dataframes with the correct column structure
        cr_df_empty = pd.DataFrame(columns=expected_columns['cr_df'])
        hb_df_empty = pd.DataFrame(columns=expected_columns['hb_df'])
        a1c_df_empty = pd.DataFrame(columns=expected_columns['a1c_df'])
        alb_df_empty = pd.DataFrame(columns=expected_columns['alb_df'])
        po4_df_empty = pd.DataFrame(columns=expected_columns['po4_df'])
        ca_df_empty = pd.DataFrame(columns=expected_columns['ca_df'])
        ca_adjusted_df_empty = pd.DataFrame(columns=expected_columns['ca_adjusted_df'])
        hco3_df_empty = pd.DataFrame(columns=expected_columns['hco3_df'])
        upcr_df_empty = pd.DataFrame(columns=expected_columns['upcr_df'])
        uacr_df_empty = pd.DataFrame(columns=expected_columns['uacr_df'])
        demo_df_empty = pd.DataFrame(columns=expected_columns['demo_df'])
        icd10_df_empty = pd.DataFrame(columns=expected_columns['icd10_df'])
        death_df_empty = pd.DataFrame(columns=expected_columns['death_df'])
        operation_df_empty = pd.DataFrame(columns=expected_columns['operation_df'])
        
        return (
            cr_df_empty,
            hb_df_empty,
            a1c_df_empty,
            alb_df_empty,
            po4_df_empty,
            ca_df_empty,
            ca_adjusted_df_empty,
            hco3_df_empty,
            upcr_df_empty,
            uacr_df_empty,
            demo_df_empty,
            icd10_df_empty,
            death_df_empty,
            operation_df_empty
        )
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def create_master_dataframe(
    icd10_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    lab_dfs: Dict[str, Tuple[pd.DataFrame, str]]
) -> pd.DataFrame:
    """
    Create a master dataframe with temporal data using ICD10 data as the primary dataframe.
    
    This function supports two formats for the ICD10 dataframe:
    1. Original format: Each row contains a single ICD10 code for a patient on a specific date
       (multiple rows per patient-date combination)
    2. Optimized format: Each row contains a list of ICD10 codes for a patient on a specific date
       (one row per patient-date combination)
    
    The optimized format significantly reduces memory usage and improves performance
    when merging with other dataframes.
    
    Args:
        icd10_df: DataFrame containing ICD10 diagnosis data (key, date, icd10)
                  The 'icd10' column can contain either individual codes (str) or
                  lists of codes (List[str]) in the optimized format
        patient_df: DataFrame containing patient demographic and endpoint data
        lab_dfs: Dictionary mapping lab names to tuples of (DataFrame, value_column)
        
    Returns:
        Master DataFrame with temporal data
    """
    print("\n=== Creating master dataframe with temporal data ===\n")
    
    # 1. Start with ICD10 dataframe as the primary dataframe
    if icd10_df.empty or 'key' not in icd10_df.columns:
        print("ICD10 DataFrame is empty or missing required columns, using a basic patient DataFrame instead")
        master_df = patient_df.copy()
        return master_df
    
    print(f"Using ICD10 DataFrame with {len(icd10_df)} rows as primary dataframe")
    print(f"ICD10 DataFrame columns: {icd10_df.columns.tolist()}")
    
    # Check if icd10 column contains lists (optimized format) or individual codes
    is_optimized = False
    if 'icd10' in icd10_df.columns and len(icd10_df) > 0:
        sample_value = icd10_df['icd10'].iloc[0]
        is_optimized = isinstance(sample_value, list)
        print(f"ICD10 data format: {'Optimized (list of codes per row)' if is_optimized else 'Original (one code per row)'}")
    
    # 2. Merge demographic and endpoint information from patient_df
    # Select only the demographic and endpoint columns from patient_df
    demo_cols = ['key', 'dob', 'gender', 'age',
                 'endpoint', 'endpoint_date', 'first_sub_60_date',
                ]
    
    # Only keep columns that exist in patient_df
    demo_cols = [col for col in demo_cols if col in patient_df.columns]
    patient_subset = patient_df[demo_cols]
    
    # Merge with ICD10 dataframe
    master_df = pd.merge(icd10_df, patient_subset, on='key', how='left')
    # master_df.drop_duplicates(inplace=True)
    print(f"Added demographic and endpoint information, now has {len(master_df)} rows")
    
    # 3. Merge lab data with temporal aspect
    for lab_name, (lab_df, value_col) in lab_dfs.items():
        if lab_df.empty or 'key' not in lab_df.columns or value_col not in lab_df.columns or 'date' not in lab_df.columns:
            print(f"Cannot merge {lab_name} data: missing required columns")
            continue
        
        print(f"Merging {lab_name} data with {len(lab_df)} rows")
        
        # Ensure date columns are datetime
        if 'date' in master_df.columns and not pd.api.types.is_datetime64_any_dtype(master_df['date']):
            master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
        
        if not pd.api.types.is_datetime64_any_dtype(lab_df['date']):
            lab_df['date'] = pd.to_datetime(lab_df['date'], errors='coerce')
        
        # Select only necessary columns from lab_df
        lab_subset = lab_df[['key', 'date', value_col]].copy()
        
        # Rename the value column to avoid conflicts
        lab_subset.rename(columns={value_col: f"{lab_name}_value"}, inplace=True)
        
        # Merge lab data with master dataframe
        merged_df = pd.merge(master_df, lab_subset, on='key', how='left', suffixes=('', f'_{lab_name}'))
        
        # Create a date difference column to determine which lab values to use
        merged_df[f'{lab_name}_date_diff'] = (merged_df['date'] - merged_df[f'date_{lab_name}']).dt.days
        
        # Only keep lab values that were measured before or on the same day as the diagnosis
        # (negative or zero date difference)
        merged_df[f'{lab_name}_value'] = np.where(
            merged_df[f'{lab_name}_date_diff'] >= 0,
            merged_df[f'{lab_name}_value'],
            np.nan
        )
        
        # Sort by key and date
        merged_df = merged_df.sort_values(['key', 'date'])
        
        # Forward fill lab values within each patient group
        merged_df[f'{lab_name}_value'] = merged_df.groupby('key')[f'{lab_name}_value'].ffill()
        
        # Drop the temporary columns
        merged_df = merged_df.drop(columns=[f'date_{lab_name}', f'{lab_name}_date_diff'])
        
        print(f"Added {lab_name} values, now has {merged_df[f'{lab_name}_value'].notna().sum()} non-null values")
        
        # Update master_df
        master_df = merged_df
    
    # 4. Remove duplicate rows
    master_df = master_df.drop_duplicates()
    print(f"Removed duplicates, final dataframe has {len(master_df)} rows")
    
    return master_df
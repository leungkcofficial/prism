"""
Data Merging Step for CKD Risk Prediction

This module contains the ZenML step for merging data from multiple sources.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple, List
import os
import re
from datetime import datetime

@step
def merge_data(
    patient_df: pd.DataFrame,
    icd10_df: pd.DataFrame,
    cr_df: pd.DataFrame,
    hb_df: pd.DataFrame,
    a1c_df: pd.DataFrame,
    alb_df: pd.DataFrame,
    po4_df: pd.DataFrame,
    ca_df: pd.DataFrame,
    ca_adjusted_df: pd.DataFrame,
    hco3_df: pd.DataFrame,
    upcr_df: pd.DataFrame,
    uacr_df: pd.DataFrame,
    operation_df: pd.DataFrame,
    death_df: pd.DataFrame,
    cci_df: pd.DataFrame,
    cci_score_df: pd.DataFrame,
    hypertension_df: pd.DataFrame,
    egfr_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge data from multiple sources into a single dataframe.
    
    Args:
        patient_df: DataFrame containing patient information
        icd10_df: DataFrame containing ICD-10 diagnosis codes
        cr_df: DataFrame containing creatinine lab results
        hb_df: DataFrame containing hemoglobin lab results
        a1c_df: DataFrame containing hemoglobin A1c lab results
        alb_df: DataFrame containing albumin lab results
        po4_df: DataFrame containing phosphate lab results
        ca_df: DataFrame containing calcium lab results
        ca_adjusted_df: DataFrame containing adjusted calcium lab results
        upcr_df: DataFrame containing urine protein-creatinine ratio lab results
        uacr_df: DataFrame containing urine albumin-creatinine ratio lab results
        operation_df: DataFrame containing operation information
        death_df: DataFrame containing death information
        cci_df: DataFrame containing Charlson Comorbidity Index information
        cci_score_df: DataFrame containing Charlson Comorbidity Index scores
        hypertension_df: DataFrame containing hypertension information
        egfr_df: DataFrame containing eGFR information
        
    Returns:
        Tuple containing:
        - final_df: DataFrame containing all merged data
        - prediction_df: DataFrame containing data for prediction
    """
    try:
        print("\n=== Starting data merging ===\n")
        
        # Get unique patient keys from patient_df
        patient_keys = patient_df['key'].unique()
        print(f"Found {len(patient_keys)} unique patient keys")
        
        # 1. Create a master dataframe with ICD10 data and patient information
        print("\n=== Creating master dataframe ===\n")
        if icd10_df is not None and 'key' in icd10_df.columns and 'date' in icd10_df.columns:
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(icd10_df['date']):
                icd10_df['date'] = pd.to_datetime(icd10_df['date'], errors='coerce')
            
            # Group by key and date, aggregating ICD10 codes
            grouped_icd10 = icd10_df.groupby(['key', 'date']).first().reset_index()
            grouped_icd10 = grouped_icd10[grouped_icd10['key'].isin(patient_keys)]
            
            # Select only the demographic and endpoint columns from patient_df
            demo_cols = ['key', 'dob', 'gender',
                    'endpoint', 'endpoint_date', 'first_sub_60_date'
                    ]
            # Only keep columns that exist in patient_df
            demo_cols = [col for col in demo_cols if col in patient_df.columns]
            patient_subset = patient_df[demo_cols]
            
            # Merge with ICD10 dataframe
            master_df = pd.merge(grouped_icd10, patient_subset, on='key', how='left')
            master_df.drop_duplicates(subset=['key', 'date'], inplace=True)
            master_df = master_df[master_df['key'].isin(patient_keys)]
            print(f"{master_df['key'].nunique()} unique keys in master_df")
            print(f"Added demographic and endpoint information, now has {len(master_df)} rows")
        else:
            print("ICD10 dataframe is missing required columns, using patient data as base")
            master_df = patient_df.copy()
            print(f"Using patient DataFrame with {len(master_df)} rows as base")
        
        # 2. Create a lab report dataframe by merging all lab dataframes
        print("\n=== Creating lab report dataframe ===\n")
        
        # Define lab dataframes and their value columns
        lab_data = {
            'creatinine': (cr_df, 'cr'),
            'hemoglobin': (hb_df, 'hb'),
            'a1c': (a1c_df, 'a1c'),
            'albumin': (alb_df, 'alb'),
            'phosphate': (po4_df, 'po4'),
            'calcium': (ca_df, 'ca'),
            'calcium_adjusted': (ca_adjusted_df, 'ca_adjusted'),
            'bicarbonate': (hco3_df, 'hco3'),
            'upcr': (upcr_df, 'upacr'),
            'uacr': (uacr_df, 'upacr'),
            'egfr': (egfr_df, 'egfr')  # Add eGFR to the lab data dictionary
        }
        
        # Initialize an empty lab report dataframe with just key and date columns
        # Collect all keys and dates from lab dataframes
        all_keys_dates = []
        # Process each lab dataframe to collect keys and dates
        for lab_name, (lab_df, value_col) in lab_data.items():
            if lab_df.empty or 'key' not in lab_df.columns or value_col not in lab_df.columns or 'date' not in lab_df.columns:
                print(f"Skipping {lab_name} data: missing required columns")
                continue
            
            # Filter to only include patients in patient_df
            lab_df_filtered = lab_df[lab_df['key'].isin(patient_keys)]
            
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(lab_df_filtered['date']):
                lab_df_filtered['date'] = pd.to_datetime(lab_df_filtered['date'], errors='coerce')
            
            # Add key-date pairs to the collection
            key_date_pairs = lab_df_filtered[['key', 'date']].drop_duplicates()
            all_keys_dates.append(key_date_pairs)
            
            # Free memory
            del lab_df_filtered
        
        # Combine all key-date pairs and remove duplicates
        if all_keys_dates:
            lab_report_df = pd.concat(all_keys_dates).drop_duplicates()
            print(f"Created lab report dataframe with {len(lab_report_df)} unique key-date pairs")
        else:
            print("No lab data available")
            lab_report_df = None
        
        # Process each lab dataframe to add lab values to the lab report dataframe
        if lab_report_df is not None:
            for lab_name, (lab_df, value_col) in lab_data.items():
                if lab_df.empty or 'key' not in lab_df.columns or value_col not in lab_df.columns or 'date' not in lab_df.columns:
                    print(f"Skipping {lab_name} data: missing required columns")
                    continue
                
                print(f"Processing {lab_name} data with {len(lab_df)} rows")
                
                # Filter to only include patients in patient_df
                lab_df_filtered = lab_df[lab_df['key'].isin(patient_keys)]
                print(f"Filtered {lab_name} data to {len(lab_df_filtered)} rows")
                
                # Ensure date is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(lab_df_filtered['date']):
                    lab_df_filtered['date'] = pd.to_datetime(lab_df_filtered['date'], errors='coerce')
                
                # Select necessary columns and rename value column
                if lab_name == 'uacr' and 'uacr_source' in lab_df_filtered.columns:
                    # For UACR, include the source column
                    lab_subset = lab_df_filtered[['key', 'date', value_col, 'uacr_source']].copy()
                    lab_subset.rename(columns={value_col: lab_name}, inplace=True)
                    print(f"Including uacr_source column for {lab_name} data")
                else:
                    # For other labs, just include the value column
                    lab_subset = lab_df_filtered[['key', 'date', value_col]].copy()
                    lab_subset.rename(columns={value_col: lab_name}, inplace=True)
                
                # Merge with lab report dataframe using outer join to preserve more data
                lab_report_df = pd.merge(lab_report_df, lab_subset, on=['key', 'date'], how='outer')
                
                # Free memory
                del lab_subset
                del lab_df_filtered
                print(f"Added {lab_name} values, now has {lab_report_df[lab_name].notna().sum()} non-null values")
            
            # Forward fill missing values within each patient group
            print("\n=== Forward filling missing lab values within patient groups ===\n")
            for column in lab_report_df.columns:
                if column not in ['key', 'date']:
                    lab_report_df[column] = lab_report_df.groupby('key')[column].ffill()
                    print(f"Forward filled {column}, now has {lab_report_df[column].notna().sum()} non-null values")
            
            # Remove rows with NaT dates
            if lab_report_df['date'].isna().any():
                before_count = len(lab_report_df)
                lab_report_df = lab_report_df.dropna(subset=['date'])
                print(f"Removed {before_count - len(lab_report_df)} rows with NaT dates")
            print(f"Final lab report dataframe has {len(lab_report_df)} rows and {len(lab_report_df.columns)} columns")
                    
        # 3. merge cci_df and cci_score_df dataframes
        print("\n=== Merging Charlson Comorbidity Index (CCI) data ===\n")
        if cci_df is not None and 'key' in cci_df.columns and 'date' in cci_df.columns:
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(cci_df['date']):
                cci_df['date'] = pd.to_datetime(cci_df['date'], errors='coerce')
            
            # Get CCI columns
            value_ls = [col for col in cci_df.columns if col not in ["key", "date"]]
            
            # Group by key and date, aggregating CCI values
            cci_detail = cci_df.groupby(['key', 'date'])[value_ls].sum().reset_index()
            cci_detail = cci_detail[cci_detail['key'].isin(patient_keys)]
            
            # Convert key to same data type in both dataframes
            lab_report_df['key'] = lab_report_df['key'].astype('int64')
            cci_detail['key'] = cci_detail['key'].astype('int64')
            
            # Ensure date formats match exactly
            lab_report_df['date'] = pd.to_datetime(lab_report_df['date']).dt.normalize()
            cci_detail['date'] = pd.to_datetime(cci_detail['date']).dt.normalize()
            
            # Check for overlapping key-date pairs
            lab_keys_dates = set(zip(lab_report_df['key'], lab_report_df['date']))
            cci_keys_dates = set(zip(cci_detail['key'], cci_detail['date']))
            overlap = lab_keys_dates.intersection(cci_keys_dates)
            print(f"Overlapping key-date pairs: {len(overlap)}")
            
            # Merge CCI data with lab report dataframe using outer join to preserve more data
            lab_report_df = pd.merge(lab_report_df, cci_detail, on=['key', 'date'], how='outer')
            for col in value_ls:
                print(f"  {col}: {lab_report_df[col].notna().sum()}")
        else:
            print("No CCI detail available, skipping merge")
            
        if cci_score_df is not None and 'key' in cci_score_df.columns and 'date' in cci_score_df.columns:
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(cci_score_df['date']):
                cci_score_df['date'] = pd.to_datetime(cci_score_df['date'], errors='coerce')
            
            # Get maximum CCI score for each key-date pair
            cci_score = cci_score_df.groupby(['key', 'date'])['cci_score_total'].max().reset_index()
            cci_score = cci_score[cci_score['key'].isin(patient_keys)]
            
            # Convert key to same data type in both dataframes
            if 'key' in lab_report_df.columns:
                lab_report_df['key'] = lab_report_df['key'].astype('int64')
            cci_score['key'] = cci_score['key'].astype('int64')
            
            # Ensure date formats match exactly
            if 'date' in lab_report_df.columns:
                lab_report_df['date'] = pd.to_datetime(lab_report_df['date']).dt.normalize()
            cci_score['date'] = pd.to_datetime(cci_score['date']).dt.normalize()
            
            # Check for overlapping key-date pairs
            if 'date' in lab_report_df.columns:
                lab_keys_dates = set(zip(lab_report_df['key'], lab_report_df['date']))
                cci_keys_dates = set(zip(cci_score['key'], cci_score['date']))
                overlap = lab_keys_dates.intersection(cci_keys_dates)
                print(f"Overlapping key-date pairs: {len(overlap)}")
            
            # Merge CCI score data with lab report dataframe using outer join to preserve more data
            lab_report_df = pd.merge(lab_report_df, cci_score, on=['key', 'date'], how='outer')
            
        else:
            print("No CCI score data available, skipping merge")
        
        print(f"Merged CCI data, lab report dataframe now has {len(lab_report_df)} rows and {len(lab_report_df.columns)} columns")
        
        # 4. Merge hypertension data
        print("\n=== Merging hypertension data ===\n")
        if hypertension_df is not None and 'key' in hypertension_df.columns and 'first_htn_date' in hypertension_df.columns:
            # Ensure date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(hypertension_df['first_htn_date']):
                hypertension_df['date'] = pd.to_datetime(hypertension_df['first_htn_date'], errors='coerce')
            
            # Filter to only include patients in patient_df
            hypertension_df_filtered = hypertension_df[hypertension_df['key'].isin(patient_keys)]
            
            # Add 'ht' column to lab_report_df initialized with 0
            lab_report_df['ht'] = 0
            
            # Ensure first_htn_date is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(hypertension_df_filtered['first_htn_date']):
                hypertension_df_filtered['first_htn_date'] = pd.to_datetime(hypertension_df_filtered['first_htn_date'], errors='coerce')
            
            # Create a dictionary mapping keys to their first_htn_date for faster lookup
            htn_dict = dict(zip(hypertension_df_filtered['key'], hypertension_df_filtered['first_htn_date']))
            
            # Function to determine if a date is after first_htn_date
            def set_ht_value(row):
                key = row['key']
                date = row['date']
                
                # If patient has hypertension data
                if key in htn_dict:
                    first_htn_date = htn_dict[key]
                    # If date is later than or equal to first_htn_date, set ht to 1
                    if date >= first_htn_date:
                        return 1
                return 0
            
            # Apply the function to set ht values
            lab_report_df['ht'] = lab_report_df.apply(set_ht_value, axis=1)
            
            print(f"Added hypertension indicator (ht), {lab_report_df['ht'].sum()} rows with ht=1")
        else:
            print("No hypertension data available, adding ht column with all zeros")
            lab_report_df['ht'] = 0
        
        # 4. Merge lab report dataframe with master dataframe
        if lab_report_df is not None:
            # Ensure date columns are in datetime format
            if 'date' in master_df.columns and not pd.api.types.is_datetime64_any_dtype(master_df['date']):
                master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')
            
            # Merge lab report dataframe with master dataframe
            if 'date' in master_df.columns:
                # If master_df has a date column, merge on both key and date
                # Use outer join to preserve all lab data, especially for UACR and UPCR
                final_df = pd.merge(master_df, lab_report_df, on=['key', 'date'], how='outer')
                print(f"Merged on key and date with outer join, final dataframe has {len(final_df)} rows")
            else:
                # If master_df doesn't have a date column (e.g., when using patient_df as base),
                # merge only on key
                final_df = pd.merge(master_df, lab_report_df, on='key', how='outer')
                print(f"Merged on key only with outer join, final dataframe has {len(final_df)} rows")
            
            # Sort by key and date if date column exists
            if 'date' in final_df.columns:
                final_df = final_df.sort_values(['key', 'date'])
            else:
                final_df = final_df.sort_values('key')
            
            # final_df: final dataframe contains only keys available in patient_df and most diagnoses and investigation data, use for data retrieval if necessary
            final_df = final_df[final_df['key'].isin(patient_keys)]
            prediction_df = final_df.copy()
            
            prediction_df = prediction_df.sort_values(['key', 'date'])
            for col in prediction_df.columns:
                if col not in ['key', 'date']:
                    prediction_df[col] = prediction_df.groupby('key')[col].ffill()
            prediction_df = prediction_df.drop_duplicates(subset=['key', 'date'], keep='first')
            prediction_df = prediction_df[prediction_df['key'].isin(patient_keys)]
            prediction_df = prediction_df[prediction_df['date'] >= prediction_df['first_sub_60_date']]
            
            # for each patient key group in prediction_df, if the first row egfr is >=60, remove it until the row with egfr <60 is reached, set the first_sub_60 day to the date of that row
            def adjust_first_sub_60_date(group):
                # Recursive function to ensure all rows have egfr < 60
                def filter_until_sub_60(df):
                    # If empty dataframe or no egfr column, return as is
                    if df.empty or 'egfr' not in df.columns:
                        return df
                    
                    # If first row already has egfr < 60, we're done
                    if df['egfr'].iloc[0] < 60:
                        return df
                    
                    # Find the first row where egfr < 60
                    sub_60_rows = df[df['egfr'] < 60]
                    if sub_60_rows.empty:
                        # No rows with egfr < 60, return empty dataframe or original based on preference
                        # Here we return the original to maintain patient data even if no egfr < 60
                        return df
                    
                    # Get the first row with egfr < 60
                    first_sub_60_idx = sub_60_rows.index.min()
                    first_sub_60_date = df.loc[first_sub_60_idx, 'date']
                    
                    # Update first_sub_60_date for all rows
                    df['first_sub_60_date'] = first_sub_60_date
                    
                    # Keep only rows from the first egfr < 60 onwards
                    filtered_df = df[df.index >= first_sub_60_idx]
                    
                    # Recursively apply the filter to ensure the first row now has egfr < 60
                    # This handles cases where there might be fluctuations in egfr values
                    return filter_until_sub_60(filtered_df)
                
                # Apply the recursive filtering
                filtered_group = filter_until_sub_60(group)
                
                # If filtering resulted in empty group, keep original first_sub_60_date
                if filtered_group.empty and not group.empty:
                    return group
                
                return filtered_group
            
            # Apply the function to each patient group
            before_count = len(prediction_df)
            prediction_df = prediction_df.groupby('key').apply(adjust_first_sub_60_date)
            # Reset index after grouping
            prediction_df.reset_index(drop=True, inplace=True)
            
            # Check if any patients still have first row with egfr >= 60
            patients_with_high_egfr = prediction_df.groupby('key').first()
            high_egfr_mask = patients_with_high_egfr['egfr'] >= 60
            high_egfr_count = high_egfr_mask.sum()
            
            if high_egfr_count > 0:
                print(f"Warning: {high_egfr_count} patients have never had eGFR < 60")
                
                # Get the list of patient keys that have never had eGFR < 60
                patients_to_remove = patients_with_high_egfr[high_egfr_mask].index.tolist()
                print(f"Removing {len(patients_to_remove)} patients from both dataframes")
                
                # Remove these patients from both prediction_df and final_df
                before_pred_count = len(prediction_df)
                before_final_count = len(final_df)
                
                prediction_df = prediction_df[~prediction_df['key'].isin(patients_to_remove)]
                final_df = final_df[~final_df['key'].isin(patients_to_remove)]
                
                print(f"Removed {before_pred_count - len(prediction_df)} rows from prediction_df")
                print(f"Removed {before_final_count - len(final_df)} rows from final_df")
            
            print(f"Removed {before_count - len(prediction_df)} rows with eGFR >= 60 before CKD onset")
            
            # Special case: Impute endpoint_date with study end date for records with endpoint=0
            if 'endpoint' in prediction_df.columns and 'endpoint_date' in prediction_df.columns:
                # Get study end date from environment variable
                study_end_date_str = os.getenv('STUDY_END_DATE', '2023-12-31')
                study_end_date = pd.to_datetime(study_end_date_str)
                
                # Count missing endpoint_date values for endpoint=0 records
                missing_endpoint_dates = ((prediction_df['endpoint'] == 0) & prediction_df['endpoint_date'].isna()).sum()
                
                if missing_endpoint_dates > 0:
                    print(f"\n=== Imputing endpoint_date for censored records (endpoint=0) ===")
                    print(f"  Found {missing_endpoint_dates} censored records with missing endpoint_date")
                    print(f"  Using study end date: {study_end_date_str}")
                    
                    # Impute endpoint_date with study end date for records with endpoint=0
                    prediction_df.loc[(prediction_df['endpoint'] == 0) & prediction_df['endpoint_date'].isna(), 'endpoint_date'] = study_end_date
                    
                    # Verify imputation
                    remaining_missing = ((prediction_df['endpoint'] == 0) & prediction_df['endpoint_date'].isna()).sum()
                    print(f"  Imputed {missing_endpoint_dates - remaining_missing} endpoint_date values, {remaining_missing} still missing")
            
            # Calculate duration and observation_period
            if 'endpoint' in prediction_df.columns and 'endpoint_date' in prediction_df.columns and 'date' in prediction_df.columns:
                print("\n=== Calculating duration and observation period ===")
                
                # Calculate duration in days (time to endpoint from observation date)
                prediction_df['duration'] = (pd.to_datetime(prediction_df['endpoint_date']) -
                                           pd.to_datetime(prediction_df['date'])).dt.days
                
                # Ensure duration is positive
                prediction_df['duration'] = prediction_df['duration'].clip(lower=1)
                
                # duration adjustment on each row: if 'duration column > 1825 days, set it to 1825 days, 'endpoint' becomes 0 (censored), 'endpoint_date' becomes 'date' + 1825 days
                long_duration_mask = prediction_df['duration'] > 1825
                if long_duration_mask.any():
                    long_duration_count = long_duration_mask.sum()
                    print(f"  Found {long_duration_count} rows with duration > 1825 days (5 years)")
                    
                    # Set duration to 1825 days for these rows
                    prediction_df.loc[long_duration_mask, 'duration'] = 1825
                    
                    # Change endpoint to 0 (censored) for these rows
                    prediction_df.loc[long_duration_mask, 'endpoint'] = 0
                    
                    # Set endpoint_date to date + 1825 days for these rows
                    prediction_df.loc[long_duration_mask, 'endpoint_date'] = pd.to_datetime(prediction_df.loc[long_duration_mask, 'date']) + pd.Timedelta(days=1825)
                    
                    print(f"  Adjusted {long_duration_count} rows: duration set to 1825 days, endpoint set to 0, endpoint_date adjusted")
                
                # Add observation_period feature (time since recruitment)
                if 'first_sub_60_date' in prediction_df.columns:
                    prediction_df['observation_period'] = (pd.to_datetime(prediction_df['date']) -
                                                        pd.to_datetime(prediction_df['first_sub_60_date'])).dt.days
                    prediction_df['observation_period'] = prediction_df['observation_period'].clip(lower=0)
                    
                    print(f"  Added 'observation_period' feature: min={prediction_df['observation_period'].min()}, max={prediction_df['observation_period'].max()}, mean={prediction_df['observation_period'].mean():.2f}")
                
                print(f"  Duration statistics: min={prediction_df['duration'].min()}, max={prediction_df['duration'].max()}, mean={prediction_df['duration'].mean():.2f}")
            
            # Remove rows with NaT dates
            if prediction_df['date'].isna().any():
                before_count = len(prediction_df)
                prediction_df = prediction_df.dropna(subset=['date'])
                print(f"Removed {before_count - len(prediction_df)} rows with NaT dates")
            
            # Calculate age on each row
            if not pd.api.types.is_datetime64_any_dtype(prediction_df['dob']):
                prediction_df['dob'] = pd.to_datetime(prediction_df['dob'], errors='coerce')
            
            if not pd.api.types.is_datetime64_any_dtype(prediction_df['date']):
                prediction_df['date'] = pd.to_datetime(prediction_df['date'], errors='coerce')
            prediction_df['age'] = (prediction_df['date'] - prediction_df['dob']).dt.days / 365.25
            
            # Remove rows with NaN values in key or date columns
            prediction_df = prediction_df.dropna(subset=['key', 'date'])
            
            # map gender column of prediction _df to M = 1 and F = 0
            prediction_df['gender'] = prediction_df['gender'].map({'M': 1, 'F': 0})
            
            print(f"Prediction dataframe has {len(prediction_df)} rows and {len(prediction_df.columns)} columns")
            print(prediction_df.info())
            print(f"Final dataframe has {len(final_df)} rows and {len(final_df.columns)} columns")
            print(final_df.info())
            return final_df, prediction_df
        else:
            print("No lab data available, returning master dataframe")
            return master_df
    
    except Exception as e:
        print(f"Error in merge_data: {e}")
        raise
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, List, Set, Any
from datetime import datetime

import pandas as pd
import numpy as np

# Import train test split strategy
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining a strategy for data handling 
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]: 
        pass
    

class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by handling missing values, feature engineering, and normalization.

        Args:
            data (pd.DataFrame): Input data to preprocess

        Returns:
            pd.DataFrame: Preprocessed data
        """
        
        try:
            data = data.drop_duplicates()
            # Add more preprocessing steps as needed
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            return pd.DataFrame()


def clean_creatinine_data(cr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean creatinine data by removing in-patient records and preparing for eGFR calculation.
    
    Args:
        cr_df: DataFrame containing creatinine data (key, date, code, cr)
        
    Returns:
        Tuple containing:
        - DataFrame with cleaned creatinine data for eGFR calculation
        - DataFrame with first RRT clinic dates for each patient
    """
    print(f"\n=== Cleaning creatinine data ===\n")
    
    if cr_df.empty:
        print("Creatinine DataFrame is empty, returning empty DataFrames")
        return pd.DataFrame(), pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    cr_df_clean = cr_df.copy()
    
    # 1. For the gender column, replace 'M' with 1 and 'F' with 0
    cr_df_clean['gender'] = cr_df_clean.map({'M': 1, 'F': 0})
    
    # 1. Remove '?' from code column
    if 'code' in cr_df_clean.columns:
        cr_df_clean['code'] = cr_df_clean['code'].str.replace('?', '', regex=False)
        print(f"Removed '?' from code column")
    
    # 2. Make a copy for eGFR calculation
    egfr_df = cr_df_clean.copy()
    
    # 3. Sort by key and date
    egfr_df = egfr_df.sort_values(by=['key', 'date'])
    print(f"Sorted data by key and date")
    
    # 4. Find first RRT clinic date for each patient
    endpoint_df = pd.DataFrame()
    
    if 'code' in egfr_df.columns:
        # Create a mask for rows where code starts with 'RRT'
        rrt_mask = egfr_df['code'].str.startswith('RRT', na=False)
        
        if rrt_mask.any():
            # Get the rows with RRT codes
            rrt_df = egfr_df[rrt_mask].copy()
            
            # Group by key and get the first date
            first_rrt_dates = rrt_df.groupby('key')['date'].min().reset_index()
            
            # Rename the date column
            first_rrt_dates.rename(columns={'date': 'first_rrt_clinic'}, inplace=True)
            
            # Set as endpoint_df
            endpoint_df = first_rrt_dates
            
            print(f"Found {len(endpoint_df)} patients with RRT clinic records")
    
    # 5. Keep only the first row with code starting with 'RRT' for each key
    if 'code' in egfr_df.columns:
        # Create a mask for rows where code starts with 'RRT'
        rrt_mask = egfr_df['code'].str.startswith('RRT', na=False)
        
        if rrt_mask.any():
            # Split the DataFrame into RRT and non-RRT rows
            rrt_df = egfr_df[rrt_mask].copy()
            non_rrt_df = egfr_df[~rrt_mask].copy()
            
            # Keep only the first RRT row for each key
            first_rrt_rows = rrt_df.sort_values(['key', 'date']).groupby('key').first().reset_index()
            
            # Combine the first RRT rows with non-RRT rows
            egfr_df = pd.concat([first_rrt_rows, non_rrt_df], ignore_index=True)
            
            print(f"Kept first 'RRT' row for each key ({len(first_rrt_rows)} rows), {len(egfr_df)} rows remaining")
        else:
            print("No rows with code starting with 'RRT' found")
    
    # 6. For each key, keep only the last row for each 'HN' code
    if 'code' in egfr_df.columns:
        # Create a mask for rows where code starts with 'HN'
        hn_mask = egfr_df['code'].str.startswith('HN', na=False)
        
        if hn_mask.any():
            # Split the DataFrame
            hn_df = egfr_df[hn_mask].copy()
            non_hn_df = egfr_df[~hn_mask].copy()
            
            # Group by key and code, and keep the last row
            hn_df = hn_df.sort_values(by=['key', 'code', 'date'])
            hn_df = hn_df.drop_duplicates(subset=['key', 'code'], keep='last')
            
            # Combine the DataFrames
            egfr_df = pd.concat([hn_df, non_hn_df], ignore_index=True)
            
            print(f"Kept only the last row for each 'HN' code, {len(egfr_df)} rows remaining")
    
    # Sort again by key and date
    egfr_df = egfr_df.sort_values(by=['key', 'date'])
    
    print(f"Final cleaned creatinine dataset has {len(egfr_df)} rows")
    print(f"Endpoint dataset has {len(endpoint_df)} rows")
    
    return egfr_df, endpoint_df


def find_persistent_low_egfr(egfr_df: pd.DataFrame, threshold: float = 60.0, min_days: int = 90) -> pd.DataFrame:
    """
    Find the first date when a patient's eGFR is persistently below a threshold for a specified period.
    
    Args:
        egfr_df: DataFrame containing eGFR data (key, date, egfr)
        threshold: eGFR threshold in ml/min/1.73m² (default: 60.0)
        min_days: Minimum number of days eGFR must remain below threshold (default: 90)
        
    Returns:
        DataFrame with patient keys and their first persistent low eGFR dates
    """
    print(f"\n=== Finding persistent low eGFR (< {threshold} ml/min/1.73m² for {min_days}+ days) ===\n")
    
    if egfr_df.empty or 'egfr' not in egfr_df.columns or 'date' not in egfr_df.columns:
        print("eGFR DataFrame is empty or missing required columns, returning empty DataFrame")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = egfr_df.copy()
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by key and date
    df = df.sort_values(by=['key', 'date'])
    
    # Initialize result DataFrame
    result = []
    
    # Process each patient
    for key, group in df.groupby('key'):
        # Skip if fewer than 2 measurements
        if len(group) < 2:
            continue
            
        # Reset index for easier iteration
        group = group.reset_index(drop=True)
        
        # Find the first date with eGFR below threshold
        for i in range(len(group)):
            # Skip if this measurement is above threshold
            if group.iloc[i]['egfr'] >= threshold:
                continue
                
            start_date = group.iloc[i]['date']
            start_egfr = group.iloc[i]['egfr']
            
            # Find all measurements within the persistence window
            end_date = start_date + pd.Timedelta(days=min_days)
            window_measurements = group[(group['date'] >= start_date) & (group['date'] <= end_date)]
            
            # If no additional measurements in window, check if next measurement (if any) is also below threshold
            if len(window_measurements) == 1 and i < len(group) - 1:
                next_measurement = group.iloc[i+1]
                if next_measurement['egfr'] < threshold:
                    # If time difference is greater than min_days, consider it persistent
                    if (next_measurement['date'] - start_date).days >= min_days:
                        result.append({
                            'key': key,
                            'starting_timepoint': start_date,
                            'starting_egfr': start_egfr,
                            'confirmation_date': next_measurement['date'],
                            'confirmation_egfr': next_measurement['egfr'],
                            'days_between': (next_measurement['date'] - start_date).days
                        })
                        break
            
            # If multiple measurements in window, check if all are below threshold
            elif len(window_measurements) > 1:
                # Check if all measurements in window are below threshold
                if all(measurement['egfr'] < threshold for _, measurement in window_measurements.iterrows()):
                    last_in_window = window_measurements.iloc[-1]
                    result.append({
                        'key': key,
                        'starting_timepoint': start_date,
                        'starting_egfr': start_egfr,
                        'confirmation_date': last_in_window['date'],
                        'confirmation_egfr': last_in_window['egfr'],
                        'days_between': (last_in_window['date'] - start_date).days
                    })
                    break
    
    # Create DataFrame from results
    result_df = pd.DataFrame(result)
    
    if not result_df.empty:
        print(f"Found {len(result_df)} patients with persistent low eGFR")
        # Keep only key and starting_timepoint columns
        result_df = result_df[['key', 'starting_timepoint']]
    else:
        print("No patients found with persistent low eGFR")
        result_df = pd.DataFrame(columns=['key', 'starting_timepoint'])
    
    return result_df


def calculate_egfr(cr_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate eGFR using the CKD-EPI equation based on creatinine, age, and gender.
    
    Args:
        cr_df: DataFrame containing creatinine data (key, date, code, cr)
        demo_df: DataFrame containing demographic data (key, dob, gender)
        
    Returns:
        DataFrame with eGFR values added
    """
    print(f"\n=== Calculating eGFR ===\n")
    
    if cr_df.empty or demo_df.empty:
        print("Creatinine or demographic DataFrame is empty, returning original DataFrame")
        return cr_df
    
    # Make a copy to avoid modifying the original
    egfr_df = cr_df.copy()
    
    # Ensure key is numeric in both DataFrames
    egfr_df['key'] = pd.to_numeric(egfr_df['key'], errors='coerce')
    demo_df['key'] = pd.to_numeric(demo_df['key'], errors='coerce')
    
    # Merge demographic data with creatinine data
    # Use 'dob' instead of 'age' since age needs to be calculated
    egfr_df = pd.merge(egfr_df, demo_df[['key', 'dob', 'gender']], on='key', how='left')
    
    # Calculate age from date of birth and creatinine measurement date
    if 'dob' in egfr_df.columns and 'date' in egfr_df.columns:
        # Ensure date columns are in datetime format
        egfr_df['dob'] = pd.to_datetime(egfr_df['dob'], errors='coerce')
        egfr_df['date'] = pd.to_datetime(egfr_df['date'], errors='coerce')
        
        # Calculate age in years
        egfr_df['age'] = (egfr_df['date'] - egfr_df['dob']).dt.days / 365.25
        
        print(f"Calculated age for {egfr_df['age'].notna().sum()} rows")
    else:
        print("Warning: Missing 'dob' or 'date' columns, cannot calculate age")
        egfr_df['age'] = np.nan
    
    # Check if 'cr' column exists, if not, try to find a column that might contain creatinine values
    if 'cr' not in egfr_df.columns:
        # Look for columns that might contain creatinine values
        potential_cr_columns = [col for col in egfr_df.columns if col.lower() in ['creatinine', 'cre', 'cr', 'scr']]
        
        if potential_cr_columns:
            # Use the first potential column
            cr_column = potential_cr_columns[0]
            print(f"'cr' column not found, using '{cr_column}' column instead")
            
            # Rename the column to 'cr'
            egfr_df['cr'] = egfr_df[cr_column]
        else:
            # If no potential column is found, print an error message
            print("Error: No column containing creatinine values found")
            return egfr_df
    
    # Convert creatinine from umol/L to mg/dL (if needed)
    # 1 mg/dL = 88.4 umol/L
    if 'cr' in egfr_df.columns:
        # Check if creatinine is likely in umol/L (values typically > 50)
        if egfr_df['cr'].median() > 50:
            egfr_df['cr_mgdl'] = egfr_df['cr'] / 88.4
            print(f"Converted creatinine from umol/L to mg/dL")
        else:
            egfr_df['cr_mgdl'] = egfr_df['cr']
            print(f"Creatinine appears to be already in mg/dL")
    else:
        print("Error: 'cr' column not found after attempting to rename")
        return egfr_df
    
    # Calculate eGFR using CKD-EPI equation
    # eGFR = 141 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^-1.209 × 0.993^Age × 1.018 [if female] × 1.159 [if Black]
    # where SCr is serum creatinine in mg/dL, κ is 0.7 for females and 0.9 for males, 
    # α is -0.329 for females and -0.411 for males
    
    # Initialize eGFR column
    egfr_df['egfr'] = np.nan
    
    # Calculate eGFR for females (gender = 0)
    female_mask = (egfr_df['gender'] == 0) & egfr_df['gender'].notna() & egfr_df['cr_mgdl'].notna() & egfr_df['age'].notna()
    if female_mask.any():
        k = 0.7
        alpha = -0.329
        egfr_df.loc[female_mask, 'egfr'] = 141 * \
            np.minimum(egfr_df.loc[female_mask, 'cr_mgdl'] / k, 1) ** alpha * \
            np.maximum(egfr_df.loc[female_mask, 'cr_mgdl'] / k, 1) ** -1.209 * \
            0.993 ** egfr_df.loc[female_mask, 'age'] * 1.018
    
    # Calculate eGFR for males (gender = 1)
    male_mask = (egfr_df['gender'] == 1) & egfr_df['gender'].notna() & egfr_df['cr_mgdl'].notna() & egfr_df['age'].notna()
    if male_mask.any():
        k = 0.9
        alpha = -0.411
        egfr_df.loc[male_mask, 'egfr'] = 141 * \
            np.minimum(egfr_df.loc[male_mask, 'cr_mgdl'] / k, 1) ** alpha * \
            np.maximum(egfr_df.loc[male_mask, 'cr_mgdl'] / k, 1) ** -1.209 * \
            0.993 ** egfr_df.loc[male_mask, 'age']
    
    # Create CKD stages based on eGFR
    egfr_df['ckd_stage'] = pd.cut(
        egfr_df['egfr'],
        bins=[0, 15, 30, 45, 60, 90, float('inf')],
        labels=['Stage 5', 'Stage 4', 'Stage 3b', 'Stage 3a', 'Stage 2', 'Stage 1']
    )
    
    print(f"eGFR calculated for {egfr_df['egfr'].notna().sum()} rows")
    print(f"CKD stage distribution: {egfr_df['ckd_stage'].value_counts(dropna=False)}")
    
    return egfr_df


class KidneyDataProcessor:
    """
    Class for processing kidney-related data, including creatinine cleaning, eGFR calculation,
    and CKD timepoint identification.
    """
    
    @staticmethod
    def clean_creatinine(cr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean creatinine data by removing in-patient records and preparing for eGFR calculation.
        
        Args:
            cr_df: DataFrame containing creatinine data (key, date, code, cr)
            
        Returns:
            Tuple containing:
            - DataFrame with cleaned creatinine data for eGFR calculation
            - DataFrame with first RRT clinic dates for each patient
        """
        print(f"\n=== Cleaning creatinine data ===\n")
        
        if cr_df.empty:
            print("Creatinine DataFrame is empty, returning empty DataFrames")
            return pd.DataFrame(), pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        cr_df_clean = cr_df.copy()
        
        # 1. Remove '?' from code column
        if 'code' in cr_df_clean.columns:
            cr_df_clean['code'] = cr_df_clean['code'].str.replace('?', '', regex=False)
            print(f"Removed '?' from code column")
        
        # 2. Make a copy for eGFR calculation
        egfr_df = cr_df_clean.copy()
        
        # 3. Sort by key and date
        egfr_df = egfr_df.sort_values(by=['key', 'date'])
        print(f"Sorted data by key and date")
        
        # 4. Find first RRT clinic date for each patient
        endpoint_df = pd.DataFrame()
        
        if 'code' in egfr_df.columns:
            # Create a mask for rows where code starts with 'RRT'
            rrt_mask = egfr_df['code'].str.startswith('RRT', na=False)
            
            if rrt_mask.any():
                # Get the rows with RRT codes
                rrt_df = egfr_df[rrt_mask].copy()
                
                # Group by key and get the first date
                first_rrt_dates = rrt_df.groupby('key')['date'].min().reset_index()
                
                # Rename the date column
                first_rrt_dates.rename(columns={'date': 'first_rrt_clinic'}, inplace=True)
                
                # Set as endpoint_df
                endpoint_df = first_rrt_dates
                
                print(f"Found {len(endpoint_df)} patients with RRT clinic records")
        
        # 5. Keep only the first row with code starting with 'RRT' for each key
        if 'code' in egfr_df.columns:
            # Create a mask for rows where code starts with 'RRT'
            rrt_mask = egfr_df['code'].str.startswith('RRT', na=False)
            
            if rrt_mask.any():
                # Split the DataFrame into RRT and non-RRT rows
                rrt_df = egfr_df[rrt_mask].copy()
                non_rrt_df = egfr_df[~rrt_mask].copy()
                
                # Keep only the first RRT row for each key
                first_rrt_rows = rrt_df.sort_values(['key', 'date']).groupby('key').first().reset_index()
                
                # Combine the first RRT rows with non-RRT rows
                egfr_df = pd.concat([first_rrt_rows, non_rrt_df], ignore_index=True)
                
                print(f"Kept first 'RRT' row for each key ({len(first_rrt_rows)} rows), {len(egfr_df)} rows remaining")
            else:
                print("No rows with code starting with 'RRT' found")
        
        # 6. For each key, keep only the last row for each 'HN' code
        if 'code' in egfr_df.columns:
            # Create a mask for rows where code starts with 'HN'
            hn_mask = egfr_df['code'].str.startswith('HN', na=False)
            
            if hn_mask.any():
                # Split the DataFrame
                hn_df = egfr_df[hn_mask].copy()
                non_hn_df = egfr_df[~hn_mask].copy()
                
                # Group by key and code, and keep the last row
                hn_df = hn_df.sort_values(by=['key', 'code', 'date'])
                hn_df = hn_df.drop_duplicates(subset=['key', 'code'], keep='last')
                
                # Combine the DataFrames
                egfr_df = pd.concat([hn_df, non_hn_df], ignore_index=True)
                
                print(f"Kept only the last row for each 'HN' code, {len(egfr_df)} rows remaining")
        
        # Sort again by key and date
        egfr_df = egfr_df.sort_values(by=['key', 'date'])
        
        print(f"Final cleaned creatinine dataset has {len(egfr_df)} rows")
        print(f"Endpoint dataset has {len(endpoint_df)} rows")
        
        return egfr_df, endpoint_df
    
    @staticmethod
    def calculate_egfr(cr_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate eGFR using the CKD-EPI equation based on creatinine, age, and gender.
        
        Args:
            cr_df: DataFrame containing creatinine data (key, date, code, cr)
            demo_df: DataFrame containing demographic data (key, dob, gender)
            
        Returns:
            DataFrame with eGFR values added
        """
        print(f"\n=== Calculating eGFR ===\n")
        
        if cr_df.empty or demo_df.empty:
            print("Creatinine or demographic DataFrame is empty, returning original DataFrame")
            return cr_df
        
        # Make a copy to avoid modifying the original
        egfr_df = cr_df.copy()
        
        # Ensure key is numeric in both DataFrames
        egfr_df['key'] = pd.to_numeric(egfr_df['key'], errors='coerce')
        demo_df['key'] = pd.to_numeric(demo_df['key'], errors='coerce')
        
        # Merge demographic data with creatinine data
        # Use 'dob' instead of 'age' since age needs to be calculated
        egfr_df = pd.merge(egfr_df, demo_df[['key', 'dob', 'gender']], on='key', how='left')
        
        # Calculate age from date of birth and creatinine measurement date
        if 'dob' in egfr_df.columns and 'date' in egfr_df.columns:
            # Ensure date columns are in datetime format
            egfr_df['dob'] = pd.to_datetime(egfr_df['dob'], errors='coerce')
            egfr_df['date'] = pd.to_datetime(egfr_df['date'], errors='coerce')
            
            # Calculate age in years
            egfr_df['age'] = (egfr_df['date'] - egfr_df['dob']).dt.days / 365.25
            
            print(f"Calculated age for {egfr_df['age'].notna().sum()} rows")
        else:
            print("Warning: Missing 'dob' or 'date' columns, cannot calculate age")
            egfr_df['age'] = np.nan
        
        # Check if 'cr' column exists, if not, try to find a column that might contain creatinine values
        if 'cr' not in egfr_df.columns:
            # Look for columns that might contain creatinine values
            potential_cr_columns = [col for col in egfr_df.columns if col.lower() in ['creatinine', 'cre', 'cr', 'scr']]
            
            if potential_cr_columns:
                # Use the first potential column
                cr_column = potential_cr_columns[0]
                print(f"'cr' column not found, using '{cr_column}' column instead")
                
                # Rename the column to 'cr'
                egfr_df['cr'] = egfr_df[cr_column]
            else:
                # If no potential column is found, print an error message
                print("Error: No column containing creatinine values found")
                return egfr_df
        
        # Convert creatinine from umol/L to mg/dL (if needed)
        # 1 mg/dL = 88.4 umol/L
        egfr_df['gender'] = egfr_df['gender'].map({'M': 1, 'F': 0})
        if 'cr' in egfr_df.columns:
            # Check if creatinine is likely in umol/L (values typically > 50)
            if egfr_df['cr'].median() > 50:
                egfr_df['cr_mgdl'] = egfr_df['cr'] / 88.4
                print(f"Converted creatinine from umol/L to mg/dL")
            else:
                egfr_df['cr_mgdl'] = egfr_df['cr']
                print(f"Creatinine appears to be already in mg/dL")
        else:
            print("Error: 'cr' column not found after attempting to rename")
            return egfr_df
        
        # Calculate eGFR using CKD-EPI equation
        # eGFR = 141 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^-1.209 × 0.993^Age × 1.018 [if female] × 1.159 [if Black]
        # where SCr is serum creatinine in mg/dL, κ is 0.7 for females and 0.9 for males,
        # α is -0.329 for females and -0.411 for males
        
        # Initialize eGFR column
        egfr_df['egfr'] = np.nan
        
        # Calculate eGFR for females (gender = 0)
        female_mask = (egfr_df['gender'] == 0) & egfr_df['gender'].notna() & egfr_df['cr_mgdl'].notna() & egfr_df['age'].notna()
        if female_mask.any():
            k = 0.7
            alpha = -0.329
            egfr_df.loc[female_mask, 'egfr'] = 141 * \
                np.minimum(egfr_df.loc[female_mask, 'cr_mgdl'] / k, 1) ** alpha * \
                np.maximum(egfr_df.loc[female_mask, 'cr_mgdl'] / k, 1) ** -1.209 * \
                0.993 ** egfr_df.loc[female_mask, 'age'] * 1.018
        
        # Calculate eGFR for males (gender = 1)
        male_mask = (egfr_df['gender'] == 1) & egfr_df['gender'].notna() & egfr_df['cr_mgdl'].notna() & egfr_df['age'].notna()
        if male_mask.any():
            k = 0.9
            alpha = -0.411
            egfr_df.loc[male_mask, 'egfr'] = 141 * \
                np.minimum(egfr_df.loc[male_mask, 'cr_mgdl'] / k, 1) ** alpha * \
                np.maximum(egfr_df.loc[male_mask, 'cr_mgdl'] / k, 1) ** -1.209 * \
                0.993 ** egfr_df.loc[male_mask, 'age']
        
        # Create CKD stages based on eGFR
        egfr_df['ckd_stage'] = pd.cut(
            egfr_df['egfr'],
            bins=[0, 15, 30, 45, 60, 90, float('inf')],
            labels=['Stage 5', 'Stage 4', 'Stage 3b', 'Stage 3a', 'Stage 2', 'Stage 1']
        )
        
        print(f"eGFR calculated for {egfr_df['egfr'].notna().sum()} rows")
        print(f"CKD stage distribution: {egfr_df['ckd_stage'].value_counts(dropna=False)}")
        
        return egfr_df
    
    @staticmethod
    def find_ckd_timepoint(egfr_df: pd.DataFrame, threshold: float = 60.0, min_days: int = 90) -> pd.DataFrame:
        """
        Find the first date when a patient's eGFR is persistently below a threshold for a specified period.
        
        Args:
            egfr_df: DataFrame containing eGFR data (key, date, egfr)
            threshold: eGFR threshold in ml/min/1.73m² (default: 60.0)
            min_days: Minimum number of days eGFR must remain below threshold (default: 90)
            
        Returns:
            DataFrame with patient keys and their first persistent low eGFR dates
        """
        print(f"\n=== Finding persistent low eGFR (< {threshold} ml/min/1.73m² for {min_days}+ days) ===\n")
        
        if egfr_df.empty or 'egfr' not in egfr_df.columns or 'date' not in egfr_df.columns:
            print("eGFR DataFrame is empty or missing required columns, returning empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = egfr_df.copy()
        
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by key and date
        df = df.sort_values(by=['key', 'date'])
        
        # Initialize result DataFrame
        result = []
        
        # Process each patient
        for key, group in df.groupby('key'):
            # Skip if fewer than 2 measurements
            if len(group) < 2:
                continue
                
            # Reset index for easier iteration
            group = group.reset_index(drop=True)
            
            # Find the first date with eGFR below threshold
            for i in range(len(group)):
                # Skip if this measurement is above threshold
                if group.iloc[i]['egfr'] >= threshold:
                    continue
                    
                start_date = group.iloc[i]['date']
                start_egfr = group.iloc[i]['egfr']
                
                # Find the next measurement that's at least min_days later
                for j in range(i + 1, len(group)):
                    days_diff = (group.iloc[j]['date'] - start_date).days
                    
                    if days_diff >= min_days:
                        # If eGFR is still below threshold, we've found a persistent low eGFR
                        if group.iloc[j]['egfr'] < threshold:
                            result.append({
                                'key': key,
                                'starting_timepoint': start_date,
                                'starting_egfr': start_egfr,
                                'confirmation_timepoint': group.iloc[j]['date'],
                                'confirmation_egfr': group.iloc[j]['egfr'],
                                'days_between': days_diff
                            })
                            break
                
                # If we found a result, no need to check later measurements
                if result and result[-1]['key'] == key:
                    break
        
        # Convert result to DataFrame
        result_df = pd.DataFrame(result)
        
        if result_df.empty:
            print(f"No patients found with persistent eGFR < {threshold} for {min_days}+ days")
            return pd.DataFrame()
        
        print(f"Found {len(result_df)} patients with persistent eGFR < {threshold} for {min_days}+ days")
        
        return result_df
    
    @staticmethod
    def process_endpoints(endpoint_df: pd.DataFrame,
                         operation_df: pd.DataFrame = None,
                         death_df: pd.DataFrame = None,
                         egfr_df: pd.DataFrame = None,
                         ckd_start_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process endpoint data to determine patient outcomes.
        
        Args:
            endpoint_df: DataFrame with first RRT clinic dates
            operation_df: DataFrame with operation data
            death_df: DataFrame with death data
            egfr_df: DataFrame with eGFR data
            ckd_start_df: DataFrame with CKD starting timepoints (eGFR < 60)
            
        Returns:
            DataFrame with processed endpoint data
        """
        # Forward the call to ComorbidityProcessor's process_endpoints method
        return ComorbidityProcessor.process_endpoints(
            endpoint_df=endpoint_df,
            operation_df=operation_df,
            death_df=death_df,
            egfr_df=egfr_df,
            ckd_start_df=ckd_start_df
        )


class ComorbidityProcessor:
    """
    Class for processing ICD-10 diagnosis codes to calculate the Charlson Comorbidity Index (CCI).
    """
    
    # Dictionary of CCI conditions with their corresponding ICD-10 codes
    cci_conditions = {
        "myocardial_infarction": {"prefix": ["I21", "I22", "I25.2"]},
        "congestive_heart_failure": {"exact": ["I11.0", "I13.0", "I13.2", "I25.5", "I42.0", "I42.5", "I42.6", "I42.7", "I42.8", "I42.9", "P29.0"], "prefix": ["I43", "I50"]},
        "peripheral_vascular_disease": {"exact": ["I73.1", "I73.8", "I73.9", "I77.1", "I79.0", "I79.1", "I79.8", "K55.1", "K55.8", "K55.9", "Z95.8", "Z95.9"], "prefix": ["I70", "I71"]},
        "cerebrovascular_disease": {"prefix": ["G45", "G46", "H34.0", "H34.1", "H34.2", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68"]},
        "dementia": {"exact": ["F04", "F05", "F06.1", "F06.8", "G13.2", "G13.8", "G31.1", "G31.2", "G91.4", "G94", "R41.81", "R54"], "prefix": ["F01", "F02", "F03", "G30", "G31.0"]},
        "chronic_pulmonary_disease": {"exact": ["J68.4", "J70.1", "J70.3"], "prefix": ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"]},
        "rheumatic_disease": {"exact": ["M31.5", "M35.1", "M35.3", "M36.0"], "prefix": ["M05", "M06", "M32", "M33", "M34"]},
        "peptic_ulcer_disease": {"prefix": ["K25", "K26", "K27", "K28"]},
        "mild_liver_disease": {"exact": ["K70.0", "K70.1", "K70.2", "K70.3", "K70.9", "K71.3", "K71.4", "K71.5", "K71.7", "K76.0", "K76.2", "K76.3", "K76.4", "K76.8", "K76.9", "Z94.4"], "prefix": ["B18", "K73", "K74"]},
        "diabetes_wo_complication": {"prefix": ["E08", "E09", "E10", "E11", "E13"], "subcode_in": [".0", ".1", ".6", ".8", ".9"]},
        "renal_mild_moderate": {"exact": ["I12.9", "I13.0", "I13.10", "N18.1", "N18.2", "N18.3", "N18.4", "N18.9", "Z94.0"], "prefix": ["N03", "N05"]},
        "diabetes_w_complication": {"prefix": ["E08", "E09", "E10", "E11", "E13"], "subcode_in": [".2", ".3", ".4", ".5"]},
        "hemiplegia_paraplegia": {"exact": ["G04.1", "G11.4", "G80.0", "G80.1", "G80.2"], "prefix": ["G81", "G82", "G83"]},
        "any_malignancy": {"exact": ["C43", "C50", "C76", "C80.1"], "prefix": ["C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C45", "C46", "C47", "C48", "C49", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C81", "C82", "C83", "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"]},
        "liver_severe": {"exact": ["I86.4", "K76.5", "K76.6", "K76.7"], "prefix": ["I85.0", "K70.4", "K71.1", "K72.1", "K72.9"]},
        "renal_severe": {"exact": ["I12.0", "I13.11", "I13.2", "N18.5", "N18.6", "N25.0", "Z99.2"], "prefix": ["N19", "Z49"]},
        "hiv": {"prefix": ["B20"]},
        "metastatic_cancer": {"exact": ["C80.0", "C80.2"], "prefix": ["C77", "C78", "C79"]},
        "aids": {"exact": ["A07.2", "A07.3", "A02.1", "A81.2", "B59", "Z87.01", "R64", "B00", "B58"], "prefix": ["B37", "C53", "B38", "B45", "B25", "G93.4", "B39", "C46", "A31", "B58"], "ranges": [(("C81", "C96")), (("A15", "A19"))]}
    }
    
    # Charlson Comorbidity Index weights
    cci_weights = {
        'myocardial_infarction': 1,
        'congestive_heart_failure': 1,
        'peripheral_vascular_disease': 1,
        'cerebrovascular_disease': 1,
        'dementia': 1,
        'chronic_pulmonary_disease': 1,
        'rheumatic_disease': 1,
        'peptic_ulcer_disease': 1,
        'mild_liver_disease': 1,
        'diabetes_wo_complication': 1,
        'diabetes_w_complication': 2,
        'hemiplegia_paraplegia': 2,
        'renal_mild_moderate': 1,
        'renal_severe': 3,
        'any_malignancy': 2,
        'metastatic_cancer': 6,
        'liver_severe': 3,
        'hiv': 3,
        'aids': 6
    }
    
    @staticmethod
    def match_code(code: str, exact: List[str] = None, prefix: List[str] = None, ranges: List[Tuple[str, str]] = None) -> bool:
        """
        Match an ICD-10 code to a condition based on exact match, prefix match, or range match.
        
        Args:
            code: The ICD-10 code to match
            exact: List of exact codes to match
            prefix: List of code prefixes to match
            ranges: List of code ranges to match
            
        Returns:
            True if the code matches any of the criteria, False otherwise
        """
        if exact and code in exact:
            return True
        if prefix and any(code.startswith(p) for p in prefix):
            return True
        if ranges:
            root = re.match(r"[A-Z]+\\d+", code)
            if root:
                root = root.group(0)
                for start, end in ranges:
                    if start <= root <= end:
                        return True
        return False
    
    @staticmethod
    def match_diabetes_code(code: str, main_prefixes: List[str], subcodes: List[str]) -> bool:
        """
        Special matching function for diabetes codes that need subcode matching.
        
        Args:
            code: The ICD-10 code to match
            main_prefixes: List of main code prefixes (e.g., E10, E11)
            subcodes: List of subcodes to match (e.g., .0, .1)
            
        Returns:
            True if the code matches the criteria, False otherwise
        """
        if any(code.startswith(m) for m in main_prefixes):
            try:
                return f".{code.split('.')[1][:1]}" in subcodes
            except IndexError:
                return False
        return False
    
    @staticmethod
    def process_icd10_data(icd10_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process ICD-10 diagnosis data to calculate the Charlson Comorbidity Index.
        
        Args:
            icd10_df: DataFrame containing ICD-10 diagnosis data (key, date, icd10)
            
        Returns:
            DataFrame with patient keys, dates, and CCI scores
        """
        print(f"\n=== Processing ICD-10 data for Charlson Comorbidity Index ===\n")
        
        if icd10_df.empty or 'icd10' not in icd10_df.columns:
            print("ICD-10 DataFrame is empty or missing required columns, returning empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = icd10_df.copy()
        
        # Ensure date is in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Create a DataFrame with patient keys and dates
        cci_df = df[['key', 'date']].copy()
        
        # Initialize columns for each CCI condition
        for condition in ComorbidityProcessor.cci_conditions.keys():
            cci_df[condition] = 0.0
        
        # Process each row to identify CCI conditions
        for idx, row in df.iterrows():
            if pd.isna(row['icd10']):
                continue
                
            code = str(row['icd10']).upper()
            
            # Check each condition
            for condition, rules in ComorbidityProcessor.cci_conditions.items():
                # Special case for diabetes
                if condition in ["diabetes_wo_complication", "diabetes_w_complication"]:
                    if ComorbidityProcessor.match_diabetes_code(code, rules["prefix"], rules["subcode_in"]):
                        cci_df.at[idx, condition] = 1.0
                else:
                    if ComorbidityProcessor.match_code(code, rules.get("exact"), rules.get("prefix"), rules.get("ranges")):
                        cci_df.at[idx, condition] = 1.0
        
        # Sort by patient and date
        cci_df = cci_df.sort_values(['key', 'date'])
        
        # Forward fill within each patient group to carry forward diagnoses
        cci_df = cci_df.groupby('key').apply(lambda group: group.ffill()).reset_index(drop=True)
        
        print(f"Processed {len(df)} ICD-10 records into CCI data")
        return cci_df
    
    @staticmethod
    def process_endpoints(endpoint_df: pd.DataFrame,
                         operation_df: pd.DataFrame = None,
                         death_df: pd.DataFrame = None,
                         egfr_df: pd.DataFrame = None,
                         ckd_start_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process endpoint data to determine patient outcomes.
        
        Args:
            endpoint_df: DataFrame with first RRT clinic dates
            operation_df: DataFrame with operation data
            death_df: DataFrame with death data
            egfr_df: DataFrame with eGFR data
            ckd_start_df: DataFrame with CKD starting timepoints (eGFR < 60)
            
        Returns:
            DataFrame with processed endpoint data
        """
        print(f"\n=== Processing endpoint data ===\n")
        
        # Make a copy to avoid modifying the original
        result_df = endpoint_df.copy() if not endpoint_df.empty else pd.DataFrame(columns=['key'])
        
        # Ensure key column exists
        if 'key' not in result_df.columns and result_df.empty:
            print("Creating empty endpoint DataFrame with key column")
            result_df = pd.DataFrame(columns=['key'])
        
        # Rename first_rrt_clinic to first_clinic_date if it exists
        if 'first_rrt_clinic' in result_df.columns:
            result_df = result_df.rename(columns={'first_rrt_clinic': 'first_clinic_date'})
        
        # 1. Extract first dialysis access operation date from operation_df
        if operation_df is not None and not operation_df.empty and 'date' in operation_df.columns:
            print("Extracting first dialysis access operation date")
            
            # Get the first operation date for each patient
            first_op_dates = operation_df.sort_values('date').groupby('key')['date'].first().reset_index()
            first_op_dates = first_op_dates.rename(columns={'date': 'first_ot_date'})
            
            # Merge with result_df
            if not first_op_dates.empty:
                if result_df.empty:
                    result_df = first_op_dates
                else:
                    result_df = pd.merge(result_df, first_op_dates, on='key', how='outer')
                
                print(f"Added first operation date for {len(first_op_dates)} patients")
        
        # 2. Extract death date from death_df
        if death_df is not None and not death_df.empty and 'date' in death_df.columns:
            print("Extracting death date")
            
            # Get the death date for each patient
            death_dates = death_df[['key', 'date']].copy()
            death_dates = death_dates.rename(columns={'date': 'death_date'})
            
            # Merge with result_df
            if not death_dates.empty:
                if result_df.empty:
                    result_df = death_dates
                else:
                    result_df = pd.merge(result_df, death_dates, on='key', how='outer')
                
                print(f"Added death date for {len(death_dates)} patients")
        
        # 3. Find patients with eGFR < 10 persistently for 90+ days
        if egfr_df is not None and not egfr_df.empty and 'egfr' in egfr_df.columns:
            print("Finding patients with eGFR < 10 persistently for 90+ days")
            
            # Use find_persistent_low_egfr with threshold=10.0
            sub10_df = find_persistent_low_egfr(egfr_df, threshold=10.0, min_days=90)
            
            if not sub10_df.empty:
                # Rename column to first_sub_10_date
                sub10_df = sub10_df.rename(columns={'starting_timepoint': 'first_sub_10_date'})
                
                # Merge with result_df
                if result_df.empty:
                    result_df = sub10_df
                else:
                    result_df = pd.merge(result_df, sub10_df, on='key', how='outer')
                
                print(f"Added first eGFR < 10 date for {len(sub10_df)} patients")
        
        # 4. Add CKD starting timepoint (eGFR < 60)
        if ckd_start_df is not None and not ckd_start_df.empty:
            print("Adding CKD starting timepoint (eGFR < 60)")
            
            # Rename column to first_sub_60_date
            ckd_start_df_renamed = ckd_start_df.copy()
            ckd_start_df_renamed = ckd_start_df_renamed.rename(columns={'starting_timepoint': 'first_sub_60_date'})
            
            # Merge with result_df
            if result_df.empty:
                result_df = ckd_start_df_renamed
            else:
                result_df = pd.merge(result_df, ckd_start_df_renamed, on='key', how='outer')
            
            print(f"Added first eGFR < 60 date for {len(ckd_start_df_renamed)} patients")
        
        # 5. Process endpoints
        if not result_df.empty:
            print("Processing endpoints")
            
            # Exclude patients without first_sub_60_date
            if 'first_sub_60_date' in result_df.columns:
                valid_mask = result_df['first_sub_60_date'].notna()
                result_df = result_df[valid_mask].copy()
                print(f"Excluded {sum(~valid_mask)} patients without persistent eGFR < 60")
            
            # Initialize endpoint column
            result_df['endpoint'] = 0  # Default: censored
            
            # Create a DataFrame to store the earliest date for each endpoint type
            # Explicitly list only the date columns we want to compare for endpoints
            date_cols = ['first_clinic_date', 'first_ot_date', 'first_sub_10_date', 'death_date']
            # Filter to only include columns that exist in the dataframe
            date_cols = [col for col in date_cols if col in result_df.columns]
            
            if date_cols:
                # Find the earliest date and its source for each patient
                for idx, row in result_df.iterrows():
                    dates = {col: row[col] for col in date_cols if pd.notna(row[col])}
                    
                    if dates:
                        # Find the earliest date and its column name
                        earliest_date = min(dates.values())
                        earliest_col = [col for col, date in dates.items() if date == earliest_date][0]
                        
                        # Set endpoint based on the source of the earliest date
                        if earliest_col in ['first_clinic_date', 'first_ot_date', 'first_sub_10_date']:
                            result_df.at[idx, 'endpoint'] = 1  # Renal endpoint
                        elif earliest_col == 'death_date':
                            result_df.at[idx, 'endpoint'] = 2  # Death
                        # If only first_sub_60_date exists, endpoint remains 0 (censored)
                        
                        # Store the earliest date and its source
                        result_df.at[idx, 'endpoint_date'] = earliest_date
                        result_df.at[idx, 'endpoint_source'] = earliest_col
            # Count endpoints
            if 'endpoint' in result_df.columns:
                endpoint_counts = result_df['endpoint'].value_counts()
                print(f"Endpoint distribution: {endpoint_counts}")
        
        return result_df
    
    @staticmethod
    def identify_hypertension(icd10_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify patients with hypertension diagnoses and their first diagnosis dates.
        
        Args:
            icd10_df: DataFrame containing ICD-10 diagnosis data (key, date, icd10)
            
        Returns:
            DataFrame with patient keys and their first hypertension diagnosis dates
        """
        print(f"\n=== Identifying hypertension diagnoses ===\n")
        
        if icd10_df.empty or 'icd10' not in icd10_df.columns:
            print("ICD-10 DataFrame is empty or missing required columns, returning empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = icd10_df.copy()
        
        # Ensure date is in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Define hypertension-related ICD-10 codes
        hypertension_codes = [
            'I10',    # Essential (primary) hypertension
            'I11',    # Hypertensive heart disease
            'I11.9',  # Hypertensive heart disease without heart failure
            'I12',    # Hypertensive chronic kidney disease
            'I12.9',  # Hypertensive chronic kidney disease without renal failure
            'I13.0',  # Hypertensive heart and chronic kidney disease with heart failure
            'I13.1',  # Hypertensive heart and chronic kidney disease without heart failure
            'I13.2',  # Hypertensive heart and chronic kidney disease with both heart and renal failure
            'I13.9',  # Hypertensive heart and chronic kidney disease, unspecified
            'I15.0',  # Renovascular hypertension
            'I15.1',  # Hypertension secondary to other renal disorders
            'I15.9',  # Secondary hypertension, unspecified
            'I67.4'   # Hypertensive encephalopathy
        ]
        
        # Create a mask for rows with hypertension diagnoses
        hypertension_mask = df['icd10'].astype(str).apply(
            lambda code: any(code.startswith(htn_code) for htn_code in hypertension_codes)
        )
        
        # Filter to only hypertension diagnoses
        hypertension_df = df[hypertension_mask].copy()
        
        if hypertension_df.empty:
            print("No hypertension diagnoses found")
            return pd.DataFrame(columns=['key', 'first_htn_date'])
        
        # Get the first diagnosis date for each patient
        first_htn_dates = hypertension_df.sort_values('date').groupby('key')['date'].first().reset_index()
        first_htn_dates.rename(columns={'date': 'first_htn_date'}, inplace=True)
        
        print(f"Found {len(first_htn_dates)} patients with hypertension diagnoses")
        
        return first_htn_dates
    
    @staticmethod
    def calculate_cci_score(cci_df: pd.DataFrame, demo_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate the Charlson Comorbidity Index score from the processed ICD-10 data.
        
        Args:
            cci_df: DataFrame with patient keys, dates, and CCI conditions
            demo_df: Optional DataFrame with demographic data (key, dob)
            
        Returns:
            DataFrame with patient keys, dates, and CCI scores
        """
        print(f"\n=== Calculating Charlson Comorbidity Index scores ===\n")
        
        if cci_df.empty:
            print("CCI DataFrame is empty, returning empty DataFrame")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = cci_df.copy()
        
        # Fill NAs with 0 for all condition columns
        for condition in ComorbidityProcessor.cci_conditions.keys():
            if condition in df.columns:
                df[condition] = df[condition].fillna(0)
        
        # Calculate the base CCI score (sum of weighted conditions)
        df['cci_score'] = 0
        for condition, weight in ComorbidityProcessor.cci_weights.items():
            if condition in df.columns:
                df['cci_score'] += df[condition] * weight
        
        # Add age adjustment if demographic data is available
        if demo_df is not None and 'dob' in demo_df.columns:
            print("Adding age adjustment to CCI score")
            
            # Merge with demographic data
            df = pd.merge(df, demo_df[['key', 'dob']], on='key', how='left')
            
            # Calculate age at each date
            df['dob'] = pd.to_datetime(df['dob'], format='mixed', errors='coerce')
            
            try:
                # Calculate age without forcing conversion to int
                age_days = (df['date'] - df['dob']).dt.days
                df['age'] = (age_days / 365.25)
                
                # Create age_points column with NaN values initially
                df['age_points'] = np.nan
                
                # Only assign age points where age is not NaN
                valid_age_mask = df['age'].notna()
                
                # Assign age points based on age ranges only for valid ages
                df.loc[valid_age_mask & (df['age'] < 50), 'age_points'] = 0
                df.loc[valid_age_mask & (df['age'] >= 50) & (df['age'] < 60), 'age_points'] = 1
                df.loc[valid_age_mask & (df['age'] >= 60) & (df['age'] < 70), 'age_points'] = 2
                df.loc[valid_age_mask & (df['age'] >= 70) & (df['age'] < 80), 'age_points'] = 3
                df.loc[valid_age_mask & (df['age'] >= 80) & (df['age'] < 90), 'age_points'] = 4
                df.loc[valid_age_mask & (df['age'] >= 90), 'age_points'] = 5
                
                print(f"Age points calculated for {valid_age_mask.sum()} patients")
                print(f"Missing age data for {(~valid_age_mask).sum()} patients")
                
            except Exception as e:
                print(f"Error calculating age adjustment: {e}")
                print("Age points will remain as NaN for later imputation")
                df['age'] = np.nan
                df['age_points'] = np.nan
            
            # Add age points to CCI score where age_points is not NaN
            # For rows with NaN age_points, cci_score_total will also be NaN
            df['cci_score_total'] = df['cci_score'] + df['age_points']
        else:
            # If no demographic data, total score is the same as base score
            df['cci_score_total'] = df['cci_score']
        
        # Ensure score columns are numeric but preserve NaN values for later imputation
        score_columns = ['cci_score', 'cci_score_total']
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Do NOT fill NaN values here - they will be imputed later with MICE
        
        # Create a simplified output DataFrame
        result_df = df[['key', 'date', 'cci_score', 'cci_score_total']].copy()
        
        print(f"Calculated CCI scores for {len(df['key'].unique())} patients")
        
        return result_df


class UrineDataProcessor:
    """
    Class for processing urine-related data, including filtering protein data,
    converting units, and predicting albumin-creatinine ratio from protein-creatinine ratio.
    """
    
    @staticmethod
    def filter_protein_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters protein data to retain only one row per group of 'key', 'code', and 'date',
        prioritizing those with a unit of 'mg/mmol Cr'.
        
        Args:
            df: DataFrame containing protein data
            
        Returns:
            DataFrame with filtered protein data
        """
        if df.empty or 'unit' not in df.columns:
            print("Protein DataFrame is empty or missing unit column, returning original DataFrame")
            return df
        
        print(f"\n=== Filtering protein data ===\n")
        
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Group by 'key', 'code', and 'date'
        def choose_row(group):
            # Combined pattern for multiple matches
            combined_pattern = 'mg/mmol Cr|mg/mmol'
            # Prioritize row with 'mg/mmol Cr'
            mmcr_rows = group[group['unit'].str.contains(combined_pattern, case=False, na=False)]
            if not mmcr_rows.empty:
                return mmcr_rows.iloc[0]  # Return the first matching row
            # Otherwise, return the first row in the group
            return group.iloc[0]

        # Apply the function to each group
        try:
            filtered_df = filtered_df.groupby(['key', 'code', 'date']).apply(choose_row).reset_index(drop=True)
            print(f"Filtered protein data from {len(df)} to {len(filtered_df)} rows")
        except Exception as e:
            print(f"Error filtering protein data: {e}")
            return df

        return filtered_df
    
    @staticmethod
    def convert_protein_units(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts 'mg/mg Cr' units to 'mg/mmol' in the protein data.
        
        Args:
            df: DataFrame containing protein data
            
        Returns:
            DataFrame with converted units
        """
        if df.empty or 'unit' not in df.columns or 'upacr' not in df.columns:
            print("Protein DataFrame is empty or missing required columns, returning original DataFrame")
            return df
        
        print(f"\n=== Converting protein units ===\n")
        
        # Make a copy to avoid modifying the original
        converted_df = df.copy()
        
        # Get conversion factor from environment variable
        conversion_factor = float(os.getenv('URINE_MGMG_TO_MGMMOL_FACTOR', '113'))
        print(f"Using conversion factor of {conversion_factor} for mg/mg Cr to mg/mmol")
        
        # Identify rows with 'mg/mg Cr' unit
        mgmg_rows = converted_df['unit'].str.contains('mg/mg Cr', case=False, na=False)
        
        if mgmg_rows.any():
            # Apply the conversion factor
            converted_df.loc[mgmg_rows, 'upacr'] = converted_df.loc[mgmg_rows, 'upacr'] * conversion_factor
            
            # Update the unit column
            converted_df.loc[mgmg_rows, 'unit'] = 'mg/mmol'
            
            print(f"Converted {mgmg_rows.sum()} rows from 'mg/mg Cr' to 'mg/mmol'")
        else:
            print("No rows with 'mg/mg Cr' unit found")
        
        return converted_df
    
    @staticmethod
    def predict_acr(df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the albumin-creatinine ratio (ACR) from protein-creatinine ratio (PCR).
        
        Args:
            df: DataFrame containing protein data with 'upacr' column in mg/mmol
            
        Returns:
            DataFrame with predicted ACR values
        """
        if df.empty or 'upacr' not in df.columns:
            print("Protein DataFrame is empty or missing upacr column, returning original DataFrame")
            return df
        
        print(f"\n=== Predicting albumin-creatinine ratio ===\n")
        
        # Make a copy to avoid modifying the original
        predicted_df = df.copy()
        
        # Get conversion factor and formula coefficients from environment variables
        mgmmol_to_mgg_factor = float(os.getenv('URINE_MGMMOL_TO_MGG_FACTOR', '8.84'))
        intercept = float(os.getenv('ACR_PREDICTION_INTERCEPT', '5.3920'))
        coef1 = float(os.getenv('ACR_PREDICTION_COEF1', '0.3072'))
        coef2 = float(os.getenv('ACR_PREDICTION_COEF2', '1.5793'))
        coef3 = float(os.getenv('ACR_PREDICTION_COEF3', '1.1266'))
        
        print(f"Using conversion factor of {mgmmol_to_mgg_factor} for mg/mmol to mg/g")
        print(f"Using ACR prediction formula coefficients: intercept={intercept}, coef1={coef1}, coef2={coef2}, coef3={coef3}")
        
        # Convert PCR from mg/mmol to mg/g
        pcr_series_mg_g = predicted_df['upacr'] * mgmmol_to_mgg_factor
        
        # Vectorized computation of min and max values
        log_min_pcr50 = np.log(np.minimum(pcr_series_mg_g / 50, 1))
        log_max_pcr500_01 = np.log(np.maximum(np.minimum(pcr_series_mg_g / 500, 1), 0.1))
        log_max_pcr500 = np.log(np.maximum(pcr_series_mg_g / 500, 1))
        
        # Compute the ACR using the specified formula
        acr_values = np.exp(intercept + coef1 * log_min_pcr50 +
                            coef2 * log_max_pcr500_01 +
                            coef3 * log_max_pcr500)
        
        # Add predicted ACR to the DataFrame
        predicted_df['predicted_uacr'] = acr_values / mgmmol_to_mgg_factor  # Convert back to mg/mmol
        
        print(f"Added predicted albumin-creatinine ratio for {len(predicted_df)} rows")
        
        return predicted_df
    
    @staticmethod
    def process_urine_data(upcr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process urine protein data by filtering, converting units, and predicting ACR.
        
        Args:
            upcr_df: DataFrame containing urine protein data
            
        Returns:
            DataFrame with processed urine data
        """
        if upcr_df.empty:
            print("Urine protein DataFrame is empty, returning empty DataFrame")
            return pd.DataFrame()
        
        print(f"\n=== Processing urine protein data ===\n")
        
        # Filter protein data
        filtered_df = UrineDataProcessor.filter_protein_data(upcr_df)
        
        # Convert protein units
        converted_df = UrineDataProcessor.convert_protein_units(filtered_df)
        
        # Predict ACR
        predicted_df = UrineDataProcessor.predict_acr(converted_df)
        
        # Drop rows with missing key, date, or upacr
        processed_df = predicted_df.dropna(subset=['key', 'date', 'upacr'])
        
        print(f"Final processed urine data has {len(processed_df)} rows")
        
        return processed_df
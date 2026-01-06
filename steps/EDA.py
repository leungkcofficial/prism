"""
Exploratory Data Analysis Step for CKD Risk Prediction

This module contains the ZenML step for performing exploratory data analysis
and generating comparative statistics between training and test datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zenml.steps import step
from typing import Dict, Any, Optional, Tuple, List
import os
from datetime import datetime
import scipy.stats as stats
from pathlib import Path
from src.util import load_yaml_file
from lifelines import AalenJohansenFitter

@step
def perform_eda(
    train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform exploratory data analysis on the training and test datasets.
    
    Args:
        train_df: DataFrame containing the training data
        temporal_test_df: DataFrame containing the temporal test data
        spatial_test_df: DataFrame containing the spatial test data
        output_path: Path to save the EDA results
        
    Returns:
        Dictionary containing EDA results and paths to generated visualizations
    """
    try:
        print("\n=== Performing Exploratory Data Analysis ===\n")
        
        # Get output path from environment variable if not provided
        if output_path is None:
            output_path = os.getenv("EDA_OUTPUT_PATH", "results")
        
        # Create absolute path if relative
        output_path = Path(output_path)
        if not output_path.is_absolute():
            # Assuming the current working directory is the project root
            
            output_path = Path(os.getcwd()) / output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        print(f"EDA results will be saved to: {output_path}")
        
        # Get first occurrence for each patient (for baseline characteristics)
        train_first = get_first_occurrence(train_df)
        temporal_first = get_first_occurrence(temporal_test_df)
        spatial_first = get_first_occurrence(spatial_test_df)
        
        # Get all diagnoses for each patient (including those developed during study)
        train_all_diagnoses = get_all_diagnoses(train_df)
        temporal_all_diagnoses = get_all_diagnoses(temporal_test_df)
        spatial_all_diagnoses = get_all_diagnoses(spatial_test_df)
        
        print(f"Training set: {len(train_first)} patients")
        print(f"Temporal test set: {len(temporal_first)} patients")
        print(f"Spatial test set: {len(spatial_first)} patients")
        
        # Generate comparative statistics table using both first occurrence and all diagnoses
        stats_table = generate_comparative_stats(
            train_first, temporal_first, spatial_first,
            train_all_diagnoses, temporal_all_diagnoses, spatial_all_diagnoses
        )
        
        # Save the statistics table to CSV
        stats_path = os.path.join(output_path, "comparative_statistics.csv")
        stats_table.to_csv(stats_path, index=True)
        print(f"Saved comparative statistics to {stats_path}")
        
        # Generate visualizations
        visualization_paths = generate_visualizations(train_first, temporal_first, spatial_first, output_path)
        
        # Compare event risks between datasets
        event_risk_results = compare_event_risks(train_df, temporal_test_df, spatial_test_df, output_path)
        visualization_paths.extend(event_risk_results["visualization_paths"])
        
        # Return results
        results = {
            "stats_table_path": stats_path,
            "visualization_paths": visualization_paths,
            "dialysis_summary_csv_path": event_risk_results["dialysis_summary_csv_path"],
            "mortality_summary_csv_path": event_risk_results["mortality_summary_csv_path"],
            "dialysis_summary_html_path": event_risk_results["dialysis_summary_html_path"],
            "mortality_summary_html_path": event_risk_results["mortality_summary_html_path"],
            "num_patients": {
                "train": len(train_first),
                "temporal_test": len(temporal_first),
                "spatial_test": len(spatial_first)
            }
        }
        
        return results
    
    except Exception as e:
        print(f"Error performing EDA: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def get_first_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the first occurrence (earliest date) for each patient.
    
    Args:
        df: DataFrame containing patient data with 'key' and 'date' columns
        
    Returns:
        DataFrame with one row per patient, representing their first occurrence
    """
    if df.empty or 'key' not in df.columns or 'date' not in df.columns:
        print("Warning: DataFrame is empty or missing required columns")
        return pd.DataFrame()
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by key and date
    df_sorted = df.sort_values(['key', 'date'])
    
    # Get the first occurrence for each patient
    first_occurrence = df_sorted.groupby('key').first().reset_index()
    
    return first_occurrence


def get_all_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate diagnoses across all occurrences for each patient.
    This ensures we count all diagnoses, including those that developed during the study period.
    
    Args:
        df: DataFrame containing patient data with 'key' and Charlson-related columns
        
    Returns:
        DataFrame with one row per patient, with aggregated diagnosis information
    """
    if df.empty or 'key' not in df.columns:
        print("Warning: DataFrame is empty or missing required columns")
        return pd.DataFrame()
    
    # Get all Charlson-related columns
    charlson_cols = [col for col in df.columns if col in [
        'myocardial_infarction', 'congestive_heart_failure', 'peripheral_vascular_disease',
        'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease', 'rheumatic_disease',
        'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_wo_complication',
        'diabetes_w_complication', 'hemiplegia_paraplegia', 'renal_mild_moderate',
        'renal_severe', 'any_malignancy', 'metastatic_cancer', 'liver_severe',
        'hiv', 'aids', 'cci_score_total', 'ht'
    ]]
    
    if not charlson_cols:
        print("Warning: No Charlson-related columns found")
        return pd.DataFrame()
    
    # Group by key and aggregate diagnoses
    # For each diagnosis, take the maximum value (0 or 1) to indicate if the patient ever had the diagnosis
    agg_dict = {col: 'max' for col in charlson_cols}
    
    # For cci_score_total, take the maximum score
    if 'cci_score_total' in charlson_cols:
        agg_dict['cci_score_total'] = 'max'
    
    # Add key to the columns to keep
    all_cols = ['key'] + charlson_cols
    
    # Select only the columns we need
    df_subset = df[all_cols].copy()
    
    # Group by key and aggregate
    aggregated = df_subset.groupby('key').agg(agg_dict).reset_index()
    
    print(f"Aggregated diagnoses for {len(aggregated)} patients")
    
    return aggregated


def generate_comparative_stats(
    train_df: pd.DataFrame, temporal_test_df: pd.DataFrame, spatial_test_df: pd.DataFrame,
    train_all_diagnoses: pd.DataFrame = None, temporal_all_diagnoses: pd.DataFrame = None,
    spatial_all_diagnoses: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate a comparative statistics table for the training and test datasets.
    
    Args:
        train_df: DataFrame containing the first occurrence for each patient in the training set
        temporal_test_df: DataFrame containing the first occurrence for each patient in the temporal test set
        spatial_test_df: DataFrame containing the first occurrence for each patient in the spatial test set
        train_all_diagnoses: DataFrame containing all diagnoses for each patient in the training set
        temporal_all_diagnoses: DataFrame containing all diagnoses for each patient in the temporal test set
        spatial_all_diagnoses: DataFrame containing all diagnoses for each patient in the spatial test set
        
    Returns:
        DataFrame containing comparative statistics
    """
    # Get stats rows from environment variable or use default
    default_rows = [
        "No. of participants",
        "Age (years), median (IQR)",
        "Age < 65 years, n(%)",
        "Age 65-74 years, n(%)",
        "Age 75-84 years, n(%)",
        "Age > 85 years, n(%)",
        "Male sex, n(%)",
        "Follow up time in months, median (IQR)",
        "CKD-EPI 2021 formula - Baseline (IQR)",
        "Baseline eGFR 45-59, n (%)",
        "Baseline eGFR 30-44, n (%)",
        "Baseline eGFR 15-29, n (%)",
        "Baseline eGFR <15, n (%)",
        "Baseline Albuminuria (mg/g), median (IQR)*",
        "A1 (<30 mg/g), n (%)",
        "A2 (30-300 mg/g), n (%)",
        "A3 (>300 mg/g), n (%)",
        "Albumin-to-creatinine ratio type - Measured, n (%)",
        "Albumin-to-creatinine ratio type - Protein-to-creatinine ratio calculated, n (%)",
        "Diabetes, n (%)",
        "Cardiovascular disease, n (%)",
        "Myocardial infarction, n (%)",
        "Heart failure, n (%)",
        "Stroke or TIA, n (%)",
        "Peripheral vascular disease, n (%)",
        "Chronic pulmonary disease, n (%)",
        "Cancer, n (%)",
        "Charlson Comorbidity Index (mean + SD)",
        "Event 1 by year 5(%)",
        "Event 2 by year 5(%)"
    ]
    
    env_rows = os.getenv("EDA_STATS_ROWS")
    if env_rows:
        stats_rows = env_rows.split(',')
        print(f"Using {len(stats_rows)} rows from environment variable")
    else:
        stats_rows = default_rows
        print(f"Using {len(stats_rows)} default rows")
    
    # Initialize the statistics table
    stats_table = pd.DataFrame(index=stats_rows)
    
    # Add columns for each dataset
    stats_table["Training Set"] = ""
    stats_table["Temporal Test Set"] = ""
    stats_table["Spatial Test Set"] = ""
    
    # Calculate total number of patients across all datasets
    total_patients = len(train_df) + len(temporal_test_df) + len(spatial_test_df)
    
    # Function to calculate statistics for a dataset
    def calculate_stats(df: pd.DataFrame, all_diagnoses_df: pd.DataFrame = None) -> List[str]:
        if df.empty:
            return ["N/A"] * len(stats_table.index)
        
        stats = []
        
        # Use all_diagnoses_df for comorbidity statistics if available
        # Otherwise, fall back to the first occurrence dataframe
        diagnoses_df = all_diagnoses_df if all_diagnoses_df is not None else df
        
        # Number of participants with percentage of total
        patient_count = len(df)
        percentage = (patient_count / total_patients) * 100 if total_patients > 0 else 0
        stats.append(f"{patient_count} ({percentage:.1f}%)")
        
        # Age statistics - calculate from dob and date if available
        if 'dob' in df.columns and 'date' in df.columns:
            # Create a copy of the dataframe to avoid warnings
            df_age = df.copy()
            
            # Ensure date columns are datetime
            if not pd.api.types.is_datetime64_any_dtype(df_age['dob']):
                df_age['dob'] = pd.to_datetime(df_age['dob'], errors='coerce')
            
            if not pd.api.types.is_datetime64_any_dtype(df_age['date']):
                df_age['date'] = pd.to_datetime(df_age['date'], errors='coerce')
            
            # Calculate age at first occurrence
            df_age.loc[:, 'calculated_age'] = (df_age['date'] - df_age['dob']).dt.days / 365.25
            
            # Use calculated age or existing age column
            age_series = df_age['calculated_age'].dropna()
            
            if len(age_series) > 0:
                age_median = age_series.median()
                age_q1 = age_series.quantile(0.25)
                age_q3 = age_series.quantile(0.75)
                stats.append(f"{age_median:.1f} ({age_q1:.1f}-{age_q3:.1f})")
                
                # Age groups
                age_under_65 = (age_series < 65).sum()
                age_65_74 = ((age_series >= 65) & (age_series < 75)).sum()
                age_75_84 = ((age_series >= 75) & (age_series < 85)).sum()
                age_over_85 = (age_series >= 85).sum()
                
                stats.append(f"{age_under_65} ({age_under_65/len(df)*100:.1f}%)")
                stats.append(f"{age_65_74} ({age_65_74/len(df)*100:.1f}%)")
                stats.append(f"{age_75_84} ({age_75_84/len(df)*100:.1f}%)")
                stats.append(f"{age_over_85} ({age_over_85/len(df)*100:.1f}%)")
            else:
                stats.extend(["N/A"] * 5)  # Age median and 4 age groups
        elif 'age' in df.columns:
            # Use existing age column if available
            age_series = df['age'].dropna()
            
            if len(age_series) > 0:
                age_median = age_series.median()
                age_q1 = age_series.quantile(0.25)
                age_q3 = age_series.quantile(0.75)
                stats.append(f"{age_median:.1f} ({age_q1:.1f}-{age_q3:.1f})")
                
                # Age groups
                age_under_65 = (age_series < 65).sum()
                age_65_74 = ((age_series >= 65) & (age_series < 75)).sum()
                age_75_84 = ((age_series >= 75) & (age_series < 85)).sum()
                age_over_85 = (age_series >= 85).sum()
                
                stats.append(f"{age_under_65} ({age_under_65/len(df)*100:.1f}%)")
                stats.append(f"{age_65_74} ({age_65_74/len(df)*100:.1f}%)")
                stats.append(f"{age_75_84} ({age_75_84/len(df)*100:.1f}%)")
                stats.append(f"{age_over_85} ({age_over_85/len(df)*100:.1f}%)")
            else:
                stats.extend(["N/A"] * 5)  # Age median and 4 age groups
        else:
            stats.extend(["N/A"] * 5)  # Age median and 4 age groups
        
        # Gender statistics
        if 'gender' in df.columns:
            male_count = (df['gender'] == 1).sum()
            stats.append(f"{male_count} ({male_count/len(df)*100:.1f}%)")
        else:
            stats.append("N/A")
        
        # Follow-up time
        if 'endpoint_date' in df.columns and 'date' in df.columns:
            # Calculate follow-up time in months
            df_with_dates = df.dropna(subset=['date', 'endpoint_date']).copy()  # Create an explicit copy
            if len(df_with_dates) > 0:
                # Use .loc to avoid SettingWithCopyWarning
                df_with_dates.loc[:, 'follow_up_months'] = (df_with_dates['endpoint_date'] - df_with_dates['date']).dt.days / 30.44  # Average days per month
                follow_up_median = df_with_dates['follow_up_months'].median()
                follow_up_q1 = df_with_dates['follow_up_months'].quantile(0.25)
                follow_up_q3 = df_with_dates['follow_up_months'].quantile(0.75)
                stats.append(f"{follow_up_median:.1f} ({follow_up_q1:.1f}-{follow_up_q3:.1f})")
            else:
                stats.append("N/A")
        else:
            stats.append("N/A")
        
        # eGFR statistics
        if 'egfr' in df.columns:
            # Use the eGFR values directly
            egfr_series = df['egfr'].dropna()
            
            if len(egfr_series) > 0:
                egfr_median = egfr_series.median()
                egfr_q1 = egfr_series.quantile(0.25)
                egfr_q3 = egfr_series.quantile(0.75)
                stats.append(f"{egfr_median:.1f} ({egfr_q1:.1f}-{egfr_q3:.1f})")
                
                # eGFR groups
                egfr_45_59 = ((egfr_series >= 45) & (egfr_series < 60)).sum()
                egfr_30_44 = ((egfr_series >= 30) & (egfr_series < 45)).sum()
                egfr_15_29 = ((egfr_series >= 15) & (egfr_series < 30)).sum()
                egfr_below_15 = (egfr_series < 15).sum()
                
                stats.append(f"{egfr_45_59} ({egfr_45_59/len(df)*100:.1f}%)")
                stats.append(f"{egfr_30_44} ({egfr_30_44/len(df)*100:.1f}%)")
                stats.append(f"{egfr_15_29} ({egfr_15_29/len(df)*100:.1f}%)")
                stats.append(f"{egfr_below_15} ({egfr_below_15/len(df)*100:.1f}%)")
            else:
                stats.extend(["N/A"] * 5)  # eGFR median and 4 eGFR groups
        else:
            stats.extend(["N/A"] * 5)  # eGFR median and 4 eGFR groups
        
        # Albuminuria statistics
        if 'uacr' in df.columns:
            # Albuminuria median
            alb_median = df['uacr'].median()
            alb_q1 = df['uacr'].quantile(0.25)
            alb_q3 = df['uacr'].quantile(0.75)
            stats.append(f"{alb_median:.1f} ({alb_q1:.1f}-{alb_q3:.1f})")
            
            # Albuminuria groups
            alb_a1 = (df['uacr'] < 30).sum()
            alb_a2 = ((df['uacr'] >= 30) & (df['uacr'] <= 300)).sum()
            alb_a3 = (df['uacr'] > 300).sum()
            
            stats.append(f"{alb_a1} ({alb_a1/len(df)*100:.1f}%)")
            stats.append(f"{alb_a2} ({alb_a2/len(df)*100:.1f}%)")
            stats.append(f"{alb_a3} ({alb_a3/len(df)*100:.1f}%)")
        else:
            stats.extend(["N/A"] * 4)  # Albuminuria median and 3 albuminuria groups
        
        # Albumin-to-creatinine ratio type
        if 'uacr_source' in df.columns:
            original_count = (df['uacr_source'] == 'original').sum()
            predicted_count = (df['uacr_source'] == 'predicted').sum()
            stats.append(f"{original_count} ({original_count/len(df)*100:.1f}%)")
            stats.append(f"{predicted_count} ({predicted_count/len(df)*100:.1f}%)")
        else:
            stats.extend(["N/A", "N/A"])
        
        # Comorbidities - using diagnoses_df which includes all diagnoses throughout the study
        # Diabetes
        if 'diabetes_wo_complication' in diagnoses_df.columns or 'diabetes_w_complication' in diagnoses_df.columns:
            diabetes_count = 0
            if 'diabetes_wo_complication' in diagnoses_df.columns:
                diabetes_count += (diagnoses_df['diabetes_wo_complication'] > 0).sum()
            if 'diabetes_w_complication' in diagnoses_df.columns:
                diabetes_count += (diagnoses_df['diabetes_w_complication'] > 0).sum()
            # Remove duplicates (patients with both types)
            if 'diabetes_wo_complication' in diagnoses_df.columns and 'diabetes_w_complication' in diagnoses_df.columns:
                diabetes_count = ((diagnoses_df['diabetes_wo_complication'] > 0) | (diagnoses_df['diabetes_w_complication'] > 0)).sum()
            stats.append(f"{diabetes_count} ({diabetes_count/len(df)*100:.1f}%)")
        else:
            stats.append("N/A")
        
        # Cardiovascular disease (composite) - using diagnoses_df
        cv_columns = ['myocardial_infarction', 'congestive_heart_failure', 'cerebrovascular_disease', 'peripheral_vascular_disease']
        cv_present = False
        for col in cv_columns:
            if col in diagnoses_df.columns:
                cv_present = True
                break
        
        if cv_present:
            # Create a mask for any CV disease
            cv_mask = pd.Series(False, index=diagnoses_df.index)
            for col in cv_columns:
                if col in diagnoses_df.columns:
                    cv_mask = cv_mask | (diagnoses_df[col] > 0)
            cv_count = cv_mask.sum()
            stats.append(f"{cv_count} ({cv_count/len(df)*100:.1f}%)")
            
            # Individual CV diseases
            for col in ['myocardial_infarction', 'congestive_heart_failure', 'cerebrovascular_disease', 'peripheral_vascular_disease']:
                if col in diagnoses_df.columns:
                    count = (diagnoses_df[col] > 0).sum()
                    stats.append(f"{count} ({count/len(df)*100:.1f}%)")
                else:
                    stats.append("N/A")
        else:
            stats.extend(["N/A"] * 5)  # CV composite + 4 individual CV diseases
        
        # Chronic pulmonary disease - using diagnoses_df
        if 'chronic_pulmonary_disease' in diagnoses_df.columns:
            count = (diagnoses_df['chronic_pulmonary_disease'] > 0).sum()
            stats.append(f"{count} ({count/len(df)*100:.1f}%)")
        else:
            stats.append("N/A")
        
        # Cancer - using diagnoses_df
        cancer_columns = ['any_malignancy', 'metastatic_cancer']
        cancer_present = False
        for col in cancer_columns:
            if col in diagnoses_df.columns:
                cancer_present = True
                break
        
        if cancer_present:
            # Create a mask for any cancer
            cancer_mask = pd.Series(False, index=diagnoses_df.index)
            for col in cancer_columns:
                if col in diagnoses_df.columns:
                    cancer_mask = cancer_mask | (diagnoses_df[col] > 0)
            cancer_count = cancer_mask.sum()
            stats.append(f"{cancer_count} ({cancer_count/len(df)*100:.1f}%)")
        else:
            stats.append("N/A")
        
        # Charlson Comorbidity Index - using diagnoses_df
        if 'cci_score_total' in diagnoses_df.columns:
            cci_mean = diagnoses_df['cci_score_total'].mean()
            cci_sd = diagnoses_df['cci_score_total'].std()
            stats.append(f"{cci_mean:.2f} ± {cci_sd:.2f}")
        else:
            stats.append("N/A")
        
        # Events by year 5 - count endpoints
        if 'endpoint' in df.columns:
            # Count occurrences of each endpoint value (0, 1, 2)
            endpoint_counts = df['endpoint'].value_counts().to_dict()
            
            # Calculate percentages for endpoint=1 (Event 1)
            event1_count = endpoint_counts.get(1, 0)
            event1_pct = (event1_count / len(df)) * 100
            stats.append(f"{event1_count} ({event1_pct:.1f}%)")
            
            # Calculate percentages for endpoint=2 (Event 2)
            event2_count = endpoint_counts.get(2, 0)
            event2_pct = (event2_count / len(df)) * 100
            stats.append(f"{event2_count} ({event2_pct:.1f}%)")
        else:
            stats.append("N/A")
            stats.append("N/A")
        
        return stats
    
    # Calculate statistics for each dataset, using all diagnoses for comorbidity statistics
    train_stats = calculate_stats(train_df, train_all_diagnoses)
    temporal_stats = calculate_stats(temporal_test_df, temporal_all_diagnoses)
    spatial_stats = calculate_stats(spatial_test_df, spatial_all_diagnoses)
    
    # Populate the table
    stats_table["Training Set"] = train_stats
    stats_table["Temporal Test Set"] = temporal_stats
    stats_table["Spatial Test Set"] = spatial_stats
    
    return stats_table


def generate_visualizations(train_df: pd.DataFrame, temporal_test_df: pd.DataFrame, spatial_test_df: pd.DataFrame, output_path: str) -> List[str]:
    """
    Generate visualizations comparing the training and test datasets.
    
    Args:
        train_df: DataFrame containing the first occurrence for each patient in the training set
        temporal_test_df: DataFrame containing the first occurrence for each patient in the temporal test set
        spatial_test_df: DataFrame containing the first occurrence for each patient in the spatial test set
        output_path: Path to save the visualizations
        
    Returns:
        List of paths to the generated visualizations
    """
    visualization_paths = []
    
    # Set the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Age distribution
    # Check if we can calculate age from dob and date, or if age column exists
    if (('dob' in train_df.columns and 'date' in train_df.columns) or
        ('age' in train_df.columns)):
        plt.figure(figsize=(12, 8))
        
        # Create a DataFrame for plotting
        age_data = []
        
        for df, label in [(train_df, 'Training Set'),
                          (temporal_test_df, 'Temporal Test Set'),
                          (spatial_test_df, 'Spatial Test Set')]:
            if not df.empty:
                # Calculate age from dob and date if available
                if 'dob' in df.columns and 'date' in df.columns:
                    # Create a copy of the dataframe to avoid warnings
                    df_age = df.copy()
                    
                    # Ensure date columns are datetime
                    if not pd.api.types.is_datetime64_any_dtype(df_age['dob']):
                        df_age['dob'] = pd.to_datetime(df_age['dob'], errors='coerce')
                    
                    if not pd.api.types.is_datetime64_any_dtype(df_age['date']):
                        df_age['date'] = pd.to_datetime(df_age['date'], errors='coerce')
                    
                    # Calculate age
                    df_age.loc[:, 'calculated_age'] = (df_age['date'] - df_age['dob']).dt.days / 365.25
                    
                    # Add calculated ages to the plotting data
                    for age in df_age['calculated_age'].dropna():
                        age_data.append({'Age': age, 'Dataset': label})
                
                # Fallback to age column if it exists
                elif 'age' in df.columns:
                    for age in df['age'].dropna():
                        age_data.append({'Age': age, 'Dataset': label})
        
        age_df = pd.DataFrame(age_data)
        
        if not age_df.empty:
            # Create violin plot
            ax = sns.violinplot(x='Dataset', y='Age', data=age_df, inner='quartile')
            
            # Add boxplot inside the violin plot
            # sns.boxplot(x='Dataset', y='Age', data=age_df, width=0.2, color='white', ax=ax)
            
            # Add individual points with jitter
            # sns.stripplot(x='Dataset', y='Age', data=age_df, size=3, color='black', alpha=0.3, jitter=True)
            
            plt.title('Age Distribution Across Datasets')
            plt.tight_layout()
            
            # Save the figure
            age_path = os.path.join(output_path, 'age_distribution.png')
            plt.savefig(age_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(age_path)
            print(f"Saved age distribution visualization to {age_path}")
    
    # 2. Gender distribution
    if 'gender' in train_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Calculate gender percentages
        gender_data = []
        
        for df, label in [(train_df, 'Training Set'),
                          (temporal_test_df, 'Temporal Test Set'),
                          (spatial_test_df, 'Spatial Test Set')]:
            if not df.empty and 'gender' in df.columns:
                # Get value counts and calculate percentages
                gender_counts = df['gender'].value_counts(dropna=False)
                total_count = len(df)
                
                # Only include known gender values (0=Female, 1=Male)
                if 1 in gender_counts:
                    male_count = gender_counts[1]
                    male_pct = (male_count / total_count) * 100
                    gender_data.append({'Dataset': label, 'Gender': 'Male', 'Percentage': male_pct})
                
                if 0 in gender_counts:
                    female_count = gender_counts[0]
                    female_pct = (female_count / total_count) * 100
                    gender_data.append({'Dataset': label, 'Gender': 'Female', 'Percentage': female_pct})
        
        gender_df = pd.DataFrame(gender_data)
        
        if not gender_df.empty:
            # Create grouped bar chart
            ax = sns.barplot(x='Dataset', y='Percentage', hue='Gender', data=gender_df)
            
            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=10)
            
            plt.title('Gender Distribution Across Datasets')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            plt.tight_layout()
            
            # Save the figure
            gender_path = os.path.join(output_path, 'gender_distribution.png')
            plt.savefig(gender_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(gender_path)
            print(f"Saved gender distribution visualization to {gender_path}")
    
    # 3. eGFR distribution
    if 'egfr' in train_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a DataFrame for plotting
        egfr_data = []
        
        for df, label in [(train_df, 'Training Set'), 
                          (temporal_test_df, 'Temporal Test Set'), 
                          (spatial_test_df, 'Spatial Test Set')]:
            if not df.empty and 'egfr' in df.columns:
                for egfr in df['egfr'].dropna():
                    egfr_data.append({'eGFR': egfr, 'Dataset': label})
        
        egfr_df = pd.DataFrame(egfr_data)
        
        if not egfr_df.empty:
            # Create violin plot
            ax = sns.violinplot(x='Dataset', y='eGFR', data=egfr_df, inner='quartile')
            
            # Add boxplot inside the violin plot
            # sns.boxplot(x='Dataset', y='eGFR', data=egfr_df, width=0.2, color='white', ax=ax)
            
            # Add individual points with jitter
            # sns.stripplot(x='Dataset', y='eGFR', data=egfr_df, size=3, color='black', alpha=0.3, jitter=True)
            
            # Add horizontal lines for CKD stages
            plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Stage 1 (≥90)')
            plt.axhline(y=60, color='blue', linestyle='--', alpha=0.7, label='Stage 2 (60-89)')
            plt.axhline(y=45, color='orange', linestyle='--', alpha=0.7, label='Stage 3a (45-59)')
            plt.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Stage 3b (30-44)')
            plt.axhline(y=15, color='purple', linestyle='--', alpha=0.7, label='Stage 4 (15-29)')
            
            plt.title('eGFR Distribution Across Datasets')
            plt.ylabel('eGFR (mL/min/1.73m²)')
            plt.legend(title='CKD Stages', loc='upper right')
            plt.tight_layout()
            
            # Save the figure
            egfr_path = os.path.join(output_path, 'egfr_distribution.png')
            plt.savefig(egfr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(egfr_path)
            print(f"Saved eGFR distribution visualization to {egfr_path}")
    
    # 4. Comorbidity prevalence
    comorbidity_columns = [
        'diabetes_wo_complication', 'diabetes_w_complication',
        'myocardial_infarction', 'congestive_heart_failure',
        'cerebrovascular_disease', 'peripheral_vascular_disease',
        'chronic_pulmonary_disease', 'any_malignancy', 'metastatic_cancer'
    ]
    
    comorbidity_names = {
        'diabetes_wo_complication': 'Diabetes without complications',
        'diabetes_w_complication': 'Diabetes with complications',
        'myocardial_infarction': 'Myocardial infarction',
        'congestive_heart_failure': 'Heart failure',
        'cerebrovascular_disease': 'Stroke or TIA',
        'peripheral_vascular_disease': 'Peripheral vascular disease',
        'chronic_pulmonary_disease': 'Chronic pulmonary disease',
        'any_malignancy': 'Any malignancy',
        'metastatic_cancer': 'Metastatic cancer'
    }
    
    # Check if any comorbidity columns exist
    comorbidity_exists = False
    for col in comorbidity_columns:
        if col in train_df.columns or col in temporal_test_df.columns or col in spatial_test_df.columns:
            comorbidity_exists = True
            break
    
    if comorbidity_exists:
        plt.figure(figsize=(14, 10))
        
        # Calculate comorbidity prevalence
        comorbidity_data = []
        
        for df, label in [(train_df, 'Training Set'), 
                          (temporal_test_df, 'Temporal Test Set'), 
                          (spatial_test_df, 'Spatial Test Set')]:
            if not df.empty:
                for col in comorbidity_columns:
                    if col in df.columns:
                        prevalence = (df[col] > 0).mean() * 100
                        comorbidity_data.append({
                            'Dataset': label,
                            'Comorbidity': comorbidity_names.get(col, col),
                            'Prevalence': prevalence
                        })
        
        comorbidity_df = pd.DataFrame(comorbidity_data)
        
        if not comorbidity_df.empty:
            # Create grouped bar chart
            ax = sns.barplot(x='Comorbidity', y='Prevalence', hue='Dataset', data=comorbidity_df)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            plt.title('Comorbidity Prevalence Across Datasets')
            plt.ylabel('Prevalence (%)')
            plt.tight_layout()
            
            # Save the figure
            comorbidity_path = os.path.join(output_path, 'comorbidity_prevalence.png')
            plt.savefig(comorbidity_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(comorbidity_path)
            print(f"Saved comorbidity prevalence visualization to {comorbidity_path}")
    
    # 5. Charlson Comorbidity Index distribution
    if 'cci_score_total' in train_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a DataFrame for plotting
        cci_data = []
        
        for df, label in [(train_df, 'Training Set'), 
                          (temporal_test_df, 'Temporal Test Set'), 
                          (spatial_test_df, 'Spatial Test Set')]:
            if not df.empty and 'cci_score_total' in df.columns:
                for cci in df['cci_score_total'].dropna():
                    cci_data.append({'CCI Score': cci, 'Dataset': label})
        
        cci_df = pd.DataFrame(cci_data)
        
        if not cci_df.empty:
            # Create violin plot
            ax = sns.violinplot(x='Dataset', y='CCI Score', data=cci_df, inner='quartile')
            
            # Add boxplot inside the violin plot
            # sns.boxplot(x='Dataset', y='CCI Score', data=cci_df, width=0.2, color='white', ax=ax)
            
            # Add individual points with jitter
            # sns.stripplot(x='Dataset', y='CCI Score', data=cci_df, size=3, color='black', alpha=0.3, jitter=True)
            
            plt.title('Charlson Comorbidity Index Distribution Across Datasets')
            plt.tight_layout()
            
            # Save the figure
            cci_path = os.path.join(output_path, 'cci_distribution.png')
            plt.savefig(cci_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(cci_path)
            print(f"Saved CCI distribution visualization to {cci_path}")
    
    # 7. Laboratory investigations distributions
    lab_investigations = {
        'creatinine': 'Creatinine',
        'hemoglobin': 'Hemoglobin',
        'a1c': 'HbA1c',
        'albumin': 'Albumin',
        'phosphate': 'Phosphate',
        'calcium': 'Calcium',
        'adjusted_calcium': 'Adjusted Calcium',
        'bicarbonate': 'Bicarbonate',
        'upcr': 'UPCR',
        'uacr': 'UACR'
    }
    
    for lab_col, lab_name in lab_investigations.items():
        # Check if the column exists in any of the datasets
        if (lab_col in train_df.columns or
            lab_col in temporal_test_df.columns or
            lab_col in spatial_test_df.columns):
            
            plt.figure(figsize=(12, 8))
            
            # Create a DataFrame for plotting
            lab_data = []
            
            for df, label, color in [
                (train_df, 'Training Set', 'blue'),
                (temporal_test_df, 'Temporal Test Set', 'orange'),
                (spatial_test_df, 'Spatial Test Set', 'green')
            ]:
                if not df.empty and lab_col in df.columns:
                    # Get values and drop NaNs
                    values = df[lab_col].dropna()
                    
                    # Add to plotting data
                    for value in values:
                        lab_data.append({
                            'Value': value,
                            'Dataset': label
                        })
            
            lab_df = pd.DataFrame(lab_data)
            
            if not lab_df.empty:
                # Create histogram with KDE
                # Note: Figure is already created above, no need to create it again
                
                # Plot histograms for each dataset
                for dataset, color in [
                    ('Training Set', 'blue'),
                    ('Temporal Test Set', 'orange'),
                    ('Spatial Test Set', 'green')
                ]:
                    subset = lab_df[lab_df['Dataset'] == dataset]
                    if not subset.empty:
                        sns.histplot(
                            data=subset,
                            x='Value',
                            color=color,
                            label=dataset,
                            kde=True,
                            alpha=0.5
                        )
                
                plt.title(f'{lab_name} Distribution Across Datasets')
                plt.xlabel(f'{lab_name} Value')
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
                
                # Save the figure
                lab_path = os.path.join(output_path, f'{lab_col}_distribution.png')
                plt.savefig(lab_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_paths.append(lab_path)
                print(f"Saved {lab_name} distribution visualization to {lab_path}")
    
    return visualization_paths


def compare_event_risks(
    train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    output_path: str
) -> Dict[str, Any]:
    """
    Compare the risk of dialysis (event 1) and all-cause mortality (event 2) between datasets.
    
    This function:
    1. Classifies patients based on UACR (A1, A2, A3) and eGFR (G3a, G3b, G4, G5) values
    2. Calculates cumulative incidence of events using AalenJohansenFitter
    3. Generates plots comparing risks across datasets (both discrete time points and continuous curves)
    4. Creates summary tables with 5-year risk percentages in both CSV and HTML formats
    
    Args:
        train_df: DataFrame containing the training data
        temporal_test_df: DataFrame containing the temporal test data
        spatial_test_df: DataFrame containing the spatial test data
        output_path: Path to save the visualizations and tables
        
    Returns:
        Dictionary containing paths to generated visualizations and tables
    """
    print("\n=== Comparing Event Risks Between Datasets ===\n")
    
    # Make copies of dataframes to avoid modifying originals
    train_copy = train_df.copy()
    spatial_test_copy = spatial_test_df.copy()
    temporal_test_copy = temporal_test_df.copy()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize list to store visualization paths
    visualization_paths = []
    
    # Initialize dictionaries to store risk percentages for summary tables
    dialysis_risk_summary = {
        'UACR Stage': [],
        'eGFR Stage': [],
        'Training Set (n)': [],
        'Training Set (%)': [],
        'Spatial Test Set (n)': [],
        'Spatial Test Set (%)': [],
        'Temporal Test Set (n)': [],
        'Temporal Test Set (%)': []
    }
    
    mortality_risk_summary = {
        'UACR Stage': [],
        'eGFR Stage': [],
        'Training Set (n)': [],
        'Training Set (%)': [],
        'Spatial Test Set (n)': [],
        'Spatial Test Set (%)': [],
        'Temporal Test Set (n)': [],
        'Temporal Test Set (%)': []
    }
    
    # Function to classify UACR values
    def classify_uacr(df):
        df['uacr_stage'] = pd.NA
        df.loc[df['uacr'] < 30, 'uacr_stage'] = 'A1'
        df.loc[(df['uacr'] >= 30) & (df['uacr'] <= 300), 'uacr_stage'] = 'A2'
        df.loc[df['uacr'] > 300, 'uacr_stage'] = 'A3'
        return df
    
    # Function to classify eGFR values
    def classify_egfr(df):
        df['egfr_stage'] = pd.NA
        df.loc[(df['egfr'] >= 45) & (df['egfr'] < 60), 'egfr_stage'] = 'G3a'
        df.loc[(df['egfr'] >= 30) & (df['egfr'] < 45), 'egfr_stage'] = 'G3b'
        df.loc[(df['egfr'] >= 15) & (df['egfr'] < 30), 'egfr_stage'] = 'G4'
        df.loc[df['egfr'] < 15, 'egfr_stage'] = 'G5'
        return df
    
    # Apply classifications to all datasets
    for df in [train_copy, spatial_test_copy, temporal_test_copy]:
        classify_uacr(df)
        classify_egfr(df)
    
    print("Classified patients by UACR and eGFR stages")
    
    # Define time points for evaluation (in days)
    time_points = [365, 730, 1095, 1460, 1825]  # 1-5 years
    years = [1, 2, 3, 4, 5]  # For x-axis labels
    
    # Define UACR and eGFR stages
    uacr_stages = ['A1', 'A2', 'A3']
    egfr_stages = ['G3a', 'G3b', 'G4', 'G5']
    
    # Create figures for discrete time points
    # Dialysis risk (event 1)
    fig_dialysis_discrete, axes_dialysis_discrete = plt.subplots(4, 3, figsize=(20, 20), sharex=True, sharey=True)
    fig_dialysis_discrete.suptitle('Risk of Dialysis (Event 1) by UACR and eGFR Stage - Yearly Time Points', fontsize=16)
    
    # Mortality risk (event 2)
    fig_mortality_discrete, axes_mortality_discrete = plt.subplots(4, 3, figsize=(20, 20), sharex=True, sharey=True)
    fig_mortality_discrete.suptitle('Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage - Yearly Time Points', fontsize=16)
    
    # Create figures for bar charts
    # Dialysis risk (event 1)
    fig_dialysis_bar, axes_dialysis_bar = plt.subplots(4, 3, figsize=(20, 20), sharex=True, sharey=True)
    fig_dialysis_bar.suptitle('Risk of Dialysis (Event 1) by UACR and eGFR Stage - Bar Chart', fontsize=16)
    
    # Mortality risk (event 2)
    fig_mortality_bar, axes_mortality_bar = plt.subplots(4, 3, figsize=(20, 20), sharex=True, sharey=True)
    fig_mortality_bar.suptitle('Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage - Bar Chart', fontsize=16)
    
    # Set up row and column titles for all plots
    for j, uacr_stage in enumerate(uacr_stages):
        for i, egfr_stage in enumerate(egfr_stages):
            # Discrete plots
            axes_dialysis_discrete[i, j].set_title(f'UACR {uacr_stage} / eGFR {egfr_stage}', fontsize=12)
            axes_mortality_discrete[i, j].set_title(f'UACR {uacr_stage} / eGFR {egfr_stage}', fontsize=12)
            
            # Bar plots
            axes_dialysis_bar[i, j].set_title(f'UACR {uacr_stage} / eGFR {egfr_stage}', fontsize=12)
            axes_mortality_bar[i, j].set_title(f'UACR {uacr_stage} / eGFR {egfr_stage}', fontsize=12)
            
            # Set x-axis labels for bottom row
            if i == 3:  # Last row
                axes_dialysis_discrete[i, j].set_xlabel('Follow-up (years)', fontsize=12)
                axes_mortality_discrete[i, j].set_xlabel('Follow-up (years)', fontsize=12)
                axes_dialysis_bar[i, j].set_xlabel('Follow-up (years)', fontsize=12)
                axes_mortality_bar[i, j].set_xlabel('Follow-up (years)', fontsize=12)
            
            # Set y-axis labels for leftmost column
            if j == 0:  # First column
                axes_dialysis_discrete[i, j].set_ylabel(f'eGFR {egfr_stage}\nCumulative Incidence (%)', fontsize=12)
                axes_mortality_discrete[i, j].set_ylabel(f'eGFR {egfr_stage}\nCumulative Incidence (%)', fontsize=12)
                axes_dialysis_bar[i, j].set_ylabel(f'eGFR {egfr_stage}\nCumulative Incidence (%)', fontsize=12)
                axes_mortality_bar[i, j].set_ylabel(f'eGFR {egfr_stage}\nCumulative Incidence (%)', fontsize=12)
    
    # Function to calculate cumulative incidence using AalenJohansenFitter
    def calculate_cumulative_incidence(df, event_of_interest, time_points):
            if len(df) == 0:
                return [np.nan] * len(time_points), None
            
            # Debug: Print information about the dataframe
            print(f"DataFrame size: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Duration stats: min={df['duration'].min()}, max={df['duration'].max()}, mean={df['duration'].mean()}")
            print(f"Endpoint values: {df['endpoint'].value_counts().to_dict()}")
            
            # Check if there are any events of the specified type
            event_count = (df['endpoint'] == event_of_interest).sum()
            print(f"Number of events of type {event_of_interest}: {event_count}")
            
            if event_count == 0:
                print(f"No events of type {event_of_interest} found in this subset")
                return [np.nan] * len(time_points), None
                
            durations = df['duration'].values
            events = df['endpoint'].values
            
            try:
                # Debug: Print some values from durations and events
                print(f"First 5 durations: {durations[:5]}")
                print(f"First 5 events: {events[:5]}")
                
                ajf = AalenJohansenFitter()
                ajf.fit(durations, events, event_of_interest=event_of_interest)
                
                # Calculate risk at specific time points (1-5 years)
                risk_values = []
                
                for days in time_points:
                    try:
                        # Check if the time point is within the range of the fitted model
                        if days <= max(durations):
                            # Find the closest time point in the model's timeline
                            timeline = ajf.cumulative_density_.index.values
                            closest_time_idx = np.abs(timeline - days).argmin()
                            closest_time = timeline[closest_time_idx]
                            
                            # Get the risk at this time point (as percentage)
                            risk = ajf.cumulative_density_.iloc[closest_time_idx].values[0] * 100
                            risk_values.append(risk)
                            print(f"Risk at {days} days (using closest time {closest_time}): {risk}%")
                        else:
                            print(f"Time point {days} is beyond the maximum duration in the data ({max(durations)})")
                            risk_values.append(np.nan)
                    except Exception as e:
                        print(f"Error calculating risk at {days} days: {str(e)}")
                        risk_values.append(np.nan)
                        
                return risk_values, ajf
            except Exception as e:
                print(f"Error calculating cumulative incidence: {str(e)}")
                return [np.nan] * len(time_points), None
    
    # Function to plot discrete time points
    def plot_discrete_points(ax, years, risk_values, dataset_name, color):
        if all(np.isnan(risk_values)):
            return
            
        ax.plot(years, risk_values, marker='o', linestyle='-', label=dataset_name, color=color)
    
    # Function to plot bar chart
    def plot_bar_chart(ax, years, risk_values, dataset_name, color, width=0.25, position=0):
        if all(np.isnan(risk_values)):
            return
        
        # Replace NaN values with zeros for plotting
        risk_values_clean = np.nan_to_num(risk_values, nan=0.0)
        
        # Calculate bar positions based on the dataset
        positions = np.array(years) + (position * width)
        
        # Plot the bars
        ax.bar(positions, risk_values_clean, width=width, label=dataset_name, color=color, alpha=0.7)
        
        # Set the x-ticks to be at the years
        ax.set_xticks(years)
        ax.set_xticklabels([str(year) for year in years])
        
        # Add value labels on top of bars
        for i, (pos, val) in enumerate(zip(positions, risk_values)):
            if not np.isnan(val) and val > 0:
                ax.text(pos, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    print("Calculating cumulative incidence and generating plots...")
    
    # Initialize data for HTML tables
    dialysis_html_data = []
    mortality_html_data = []
    
    # Loop through each combination of UACR and eGFR stage
    for j, uacr_stage in enumerate(uacr_stages):
        for i, egfr_stage in enumerate(egfr_stages):
            # Get subsets for each dataset with the current UACR and eGFR stage
            train_subset = train_copy[(train_copy['uacr_stage'] == uacr_stage) &
                                     (train_copy['egfr_stage'] == egfr_stage)]
            
            spatial_subset = spatial_test_copy[(spatial_test_copy['uacr_stage'] == uacr_stage) &
                                              (spatial_test_copy['egfr_stage'] == egfr_stage)]
            
            temporal_subset = temporal_test_copy[(temporal_test_copy['uacr_stage'] == uacr_stage) &
                                                (temporal_test_copy['egfr_stage'] == egfr_stage)]
            
            # Calculate dialysis risk (event 1)
            train_dialysis_risk, train_dialysis_ajf = calculate_cumulative_incidence(train_subset, 1, time_points)
            spatial_dialysis_risk, spatial_dialysis_ajf = calculate_cumulative_incidence(spatial_subset, 1, time_points)
            temporal_dialysis_risk, temporal_dialysis_ajf = calculate_cumulative_incidence(temporal_subset, 1, time_points)
            
            # Calculate mortality risk (event 2)
            train_mortality_risk, train_mortality_ajf = calculate_cumulative_incidence(train_subset, 2, time_points)
            spatial_mortality_risk, spatial_mortality_ajf = calculate_cumulative_incidence(spatial_subset, 2, time_points)
            temporal_mortality_risk, temporal_mortality_ajf = calculate_cumulative_incidence(temporal_subset, 2, time_points)
            
            # Plot discrete time points for dialysis risk
            plot_discrete_points(axes_dialysis_discrete[i, j], years, train_dialysis_risk, 'Training Set', 'blue')
            plot_discrete_points(axes_dialysis_discrete[i, j], years, spatial_dialysis_risk, 'Spatial Test Set', 'green')
            plot_discrete_points(axes_dialysis_discrete[i, j], years, temporal_dialysis_risk, 'Temporal Test Set', 'orange')
            
            # Plot discrete time points for mortality risk
            plot_discrete_points(axes_mortality_discrete[i, j], years, train_mortality_risk, 'Training Set', 'blue')
            plot_discrete_points(axes_mortality_discrete[i, j], years, spatial_mortality_risk, 'Spatial Test Set', 'green')
            plot_discrete_points(axes_mortality_discrete[i, j], years, temporal_mortality_risk, 'Temporal Test Set', 'orange')
            
            # Plot bar charts for dialysis risk
            plot_bar_chart(axes_dialysis_bar[i, j], years, train_dialysis_risk, 'Training Set', 'blue', width=0.25, position=-1)
            plot_bar_chart(axes_dialysis_bar[i, j], years, spatial_dialysis_risk, 'Spatial Test Set', 'green', width=0.25, position=0)
            plot_bar_chart(axes_dialysis_bar[i, j], years, temporal_dialysis_risk, 'Temporal Test Set', 'orange', width=0.25, position=1)
            
            # Plot bar charts for mortality risk
            plot_bar_chart(axes_mortality_bar[i, j], years, train_mortality_risk, 'Training Set', 'blue', width=0.25, position=-1)
            plot_bar_chart(axes_mortality_bar[i, j], years, spatial_mortality_risk, 'Spatial Test Set', 'green', width=0.25, position=0)
            plot_bar_chart(axes_mortality_bar[i, j], years, temporal_mortality_risk, 'Temporal Test Set', 'orange', width=0.25, position=1)
            
            # Add grid and set y-axis limit for all plots
            # Discrete plots
            axes_dialysis_discrete[i, j].grid(True, alpha=0.3)
            axes_mortality_discrete[i, j].grid(True, alpha=0.3)
            axes_dialysis_discrete[i, j].set_ylim(0, 100)
            axes_mortality_discrete[i, j].set_ylim(0, 100)
            
            # Bar plots
            axes_dialysis_bar[i, j].grid(True, alpha=0.3)
            axes_mortality_bar[i, j].grid(True, alpha=0.3)
            axes_dialysis_bar[i, j].set_ylim(0, 100)
            axes_mortality_bar[i, j].set_ylim(0, 100)
            
            # Add sample size annotation to all plots
            sample_text = f"n={len(train_subset)}/{len(spatial_subset)}/{len(temporal_subset)}"
            
            # Discrete plots
            axes_dialysis_discrete[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction',
                                                ha='center', fontsize=8)
            axes_mortality_discrete[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction',
                                                 ha='center', fontsize=8)
            
            # Bar plots
            axes_dialysis_bar[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction',
                                           ha='center', fontsize=8)
            axes_mortality_bar[i, j].annotate(sample_text, xy=(0.5, 0.95), xycoords='axes fraction',
                                            ha='center', fontsize=8)
            
            # Format 5-year risk values for display
            train_dialysis_5yr = train_dialysis_risk[-1] if not np.isnan(train_dialysis_risk[-1]) else np.nan
            spatial_dialysis_5yr = spatial_dialysis_risk[-1] if not np.isnan(spatial_dialysis_risk[-1]) else np.nan
            temporal_dialysis_5yr = temporal_dialysis_risk[-1] if not np.isnan(temporal_dialysis_risk[-1]) else np.nan
            
            train_mortality_5yr = train_mortality_risk[-1] if not np.isnan(train_mortality_risk[-1]) else np.nan
            spatial_mortality_5yr = spatial_mortality_risk[-1] if not np.isnan(spatial_mortality_risk[-1]) else np.nan
            temporal_mortality_5yr = temporal_mortality_risk[-1] if not np.isnan(temporal_mortality_risk[-1]) else np.nan
            
            # Add to CSV summary tables (5-year risk)
            dialysis_risk_summary['UACR Stage'].append(uacr_stage)
            dialysis_risk_summary['eGFR Stage'].append(egfr_stage)
            dialysis_risk_summary['Training Set (n)'].append(len(train_subset))
            dialysis_risk_summary['Training Set (%)'].append(train_dialysis_5yr)
            dialysis_risk_summary['Spatial Test Set (n)'].append(len(spatial_subset))
            dialysis_risk_summary['Spatial Test Set (%)'].append(spatial_dialysis_5yr)
            dialysis_risk_summary['Temporal Test Set (n)'].append(len(temporal_subset))
            dialysis_risk_summary['Temporal Test Set (%)'].append(temporal_dialysis_5yr)
            
            mortality_risk_summary['UACR Stage'].append(uacr_stage)
            mortality_risk_summary['eGFR Stage'].append(egfr_stage)
            mortality_risk_summary['Training Set (n)'].append(len(train_subset))
            mortality_risk_summary['Training Set (%)'].append(train_mortality_5yr)
            mortality_risk_summary['Spatial Test Set (n)'].append(len(spatial_subset))
            mortality_risk_summary['Spatial Test Set (%)'].append(spatial_mortality_5yr)
            mortality_risk_summary['Temporal Test Set (n)'].append(len(temporal_subset))
            mortality_risk_summary['Temporal Test Set (%)'].append(temporal_mortality_5yr)
            
            # Add to HTML table data
            dialysis_html_data.append({
                'uacr_stage': uacr_stage,
                'egfr_stage': egfr_stage,
                'train_n': len(train_subset),
                'train_pct': train_dialysis_5yr,
                'spatial_n': len(spatial_subset),
                'spatial_pct': spatial_dialysis_5yr,
                'temporal_n': len(temporal_subset),
                'temporal_pct': temporal_dialysis_5yr
            })
            
            mortality_html_data.append({
                'uacr_stage': uacr_stage,
                'egfr_stage': egfr_stage,
                'train_n': len(train_subset),
                'train_pct': train_mortality_5yr,
                'spatial_n': len(spatial_subset),
                'spatial_pct': spatial_mortality_5yr,
                'temporal_n': len(temporal_subset),
                'temporal_pct': temporal_mortality_5yr
            })
    
    # Add legends to all plots
    # Discrete plots
    handles_discrete, labels_discrete = axes_dialysis_discrete[0, 0].get_legend_handles_labels()
    if handles_discrete:
        fig_dialysis_discrete.legend(handles_discrete, labels_discrete, loc='upper center',
                                   bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
        fig_mortality_discrete.legend(handles_discrete, labels_discrete, loc='upper center',
                                    bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
    
    # Bar plots
    handles_bar, labels_bar = axes_dialysis_bar[0, 0].get_legend_handles_labels()
    if handles_bar:
        fig_dialysis_bar.legend(handles_bar, labels_bar, loc='upper center',
                              bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
        fig_mortality_bar.legend(handles_bar, labels_bar, loc='upper center',
                               bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=3)
    
    # Save all figures
    # Discrete plots
    dialysis_discrete_path = os.path.join(output_path, 'dialysis_risk_yearly_points.png')
    mortality_discrete_path = os.path.join(output_path, 'mortality_risk_yearly_points.png')
    
    # Bar plots
    dialysis_bar_path = os.path.join(output_path, 'dialysis_risk_bar_chart.png')
    mortality_bar_path = os.path.join(output_path, 'mortality_risk_bar_chart.png')
    
    # Apply tight layout to all figures
    fig_dialysis_discrete.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_mortality_discrete.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_dialysis_bar.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_mortality_bar.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save all figures
    plt.figure(fig_dialysis_discrete.number)
    plt.savefig(dialysis_discrete_path, dpi=300, bbox_inches='tight')
    
    plt.figure(fig_mortality_discrete.number)
    plt.savefig(mortality_discrete_path, dpi=300, bbox_inches='tight')
    
    plt.figure(fig_dialysis_bar.number)
    plt.savefig(dialysis_bar_path, dpi=300, bbox_inches='tight')
    
    plt.figure(fig_mortality_bar.number)
    plt.savefig(mortality_bar_path, dpi=300, bbox_inches='tight')
    
    # Close all figures
    plt.close(fig_dialysis_discrete)
    plt.close(fig_mortality_discrete)
    plt.close(fig_dialysis_bar)
    plt.close(fig_mortality_bar)
    
    print(f"Saved risk visualizations to {output_path}")
    
    visualization_paths.extend([
        dialysis_discrete_path,
        mortality_discrete_path,
        dialysis_bar_path,
        mortality_bar_path
    ])
    
    # Create CSV summary DataFrames with all time points
    # Initialize new dictionaries for the expanded summary tables
    dialysis_expanded_summary = {
        'UACR Stage': [],
        'eGFR Stage': [],
        'timepoints': [],
        'Training Set (n)': [],
        'Training Set (%)': [],
        'Spatial Test Set (n)': [],
        'Spatial Test Set (%)': [],
        'Temporal Test Set (n)': [],
        'Temporal Test Set (%)': []
    }
    
    mortality_expanded_summary = {
        'UACR Stage': [],
        'eGFR Stage': [],
        'timepoints': [],
        'Training Set (n)': [],
        'Training Set (%)': [],
        'Spatial Test Set (n)': [],
        'Spatial Test Set (%)': [],
        'Temporal Test Set (n)': [],
        'Temporal Test Set (%)': []
    }
    
    # Loop through each combination of UACR and eGFR stage again to calculate risks at all time points
    for j, uacr_stage in enumerate(uacr_stages):
        for i, egfr_stage in enumerate(egfr_stages):
            # Get subsets for each dataset with the current UACR and eGFR stage
            train_subset = train_copy[(train_copy['uacr_stage'] == uacr_stage) &
                                     (train_copy['egfr_stage'] == egfr_stage)]
            
            spatial_subset = spatial_test_copy[(spatial_test_copy['uacr_stage'] == uacr_stage) &
                                              (spatial_test_copy['egfr_stage'] == egfr_stage)]
            
            temporal_subset = temporal_test_copy[(temporal_test_copy['uacr_stage'] == uacr_stage) &
                                                (temporal_test_copy['egfr_stage'] == egfr_stage)]
            
            # Calculate dialysis risk (event 1) for all time points
            train_dialysis_risk, train_dialysis_ajf = calculate_cumulative_incidence(train_subset, 1, time_points)
            spatial_dialysis_risk, spatial_dialysis_ajf = calculate_cumulative_incidence(spatial_subset, 1, time_points)
            temporal_dialysis_risk, temporal_dialysis_ajf = calculate_cumulative_incidence(temporal_subset, 1, time_points)
            
            # Calculate mortality risk (event 2) for all time points
            train_mortality_risk, train_mortality_ajf = calculate_cumulative_incidence(train_subset, 2, time_points)
            spatial_mortality_risk, spatial_mortality_ajf = calculate_cumulative_incidence(spatial_subset, 2, time_points)
            temporal_mortality_risk, temporal_mortality_ajf = calculate_cumulative_incidence(temporal_subset, 2, time_points)
            
            # Print some debug information
            print(f"UACR {uacr_stage}, eGFR {egfr_stage}:")
            print(f"  Train dialysis risk: {train_dialysis_risk}")
            print(f"  Spatial dialysis risk: {spatial_dialysis_risk}")
            print(f"  Temporal dialysis risk: {temporal_dialysis_risk}")
            
            # Add to expanded summary tables for each time point
            for t, timepoint in enumerate(time_points):
                # Get risk values for this time point, handling potential issues
                # Dialysis risk
                train_dialysis_val = float(train_dialysis_risk[t]) if t < len(train_dialysis_risk) and not (np.isnan(train_dialysis_risk[t]) if isinstance(train_dialysis_risk[t], (float, int)) else False) else None
                spatial_dialysis_val = float(spatial_dialysis_risk[t]) if t < len(spatial_dialysis_risk) and not (np.isnan(spatial_dialysis_risk[t]) if isinstance(spatial_dialysis_risk[t], (float, int)) else False) else None
                temporal_dialysis_val = float(temporal_dialysis_risk[t]) if t < len(temporal_dialysis_risk) and not (np.isnan(temporal_dialysis_risk[t]) if isinstance(temporal_dialysis_risk[t], (float, int)) else False) else None
                
                # Mortality risk
                train_mortality_val = float(train_mortality_risk[t]) if t < len(train_mortality_risk) and not (np.isnan(train_mortality_risk[t]) if isinstance(train_mortality_risk[t], (float, int)) else False) else None
                spatial_mortality_val = float(spatial_mortality_risk[t]) if t < len(spatial_mortality_risk) and not (np.isnan(spatial_mortality_risk[t]) if isinstance(spatial_mortality_risk[t], (float, int)) else False) else None
                temporal_mortality_val = float(temporal_mortality_risk[t]) if t < len(temporal_mortality_risk) and not (np.isnan(temporal_mortality_risk[t]) if isinstance(temporal_mortality_risk[t], (float, int)) else False) else None
                
                # Print debug info for the first time point
                if t == 0:
                    print(f"  Time point {timepoint} - Train dialysis: {train_dialysis_val}, Spatial dialysis: {spatial_dialysis_val}, Temporal dialysis: {temporal_dialysis_val}")
                
                # Dialysis risk
                dialysis_expanded_summary['UACR Stage'].append(uacr_stage)
                dialysis_expanded_summary['eGFR Stage'].append(egfr_stage)
                dialysis_expanded_summary['timepoints'].append(timepoint)
                dialysis_expanded_summary['Training Set (n)'].append(len(train_subset))
                dialysis_expanded_summary['Training Set (%)'].append(train_dialysis_val)
                dialysis_expanded_summary['Spatial Test Set (n)'].append(len(spatial_subset))
                dialysis_expanded_summary['Spatial Test Set (%)'].append(spatial_dialysis_val)
                dialysis_expanded_summary['Temporal Test Set (n)'].append(len(temporal_subset))
                dialysis_expanded_summary['Temporal Test Set (%)'].append(temporal_dialysis_val)
                
                # Mortality risk
                mortality_expanded_summary['UACR Stage'].append(uacr_stage)
                mortality_expanded_summary['eGFR Stage'].append(egfr_stage)
                mortality_expanded_summary['timepoints'].append(timepoint)
                mortality_expanded_summary['Training Set (n)'].append(len(train_subset))
                mortality_expanded_summary['Training Set (%)'].append(train_mortality_val)
                mortality_expanded_summary['Spatial Test Set (n)'].append(len(spatial_subset))
                mortality_expanded_summary['Spatial Test Set (%)'].append(spatial_mortality_val)
                mortality_expanded_summary['Temporal Test Set (n)'].append(len(temporal_subset))
                mortality_expanded_summary['Temporal Test Set (%)'].append(temporal_mortality_val)
    
    # Create DataFrames from the expanded summary dictionaries
    dialysis_expanded_df = pd.DataFrame(dialysis_expanded_summary)
    mortality_expanded_df = pd.DataFrame(mortality_expanded_summary)
    
    # Save CSV summary tables
    dialysis_csv_path = os.path.join(output_path, 'dialysis_risk_summary.csv')
    mortality_csv_path = os.path.join(output_path, 'mortality_risk_summary.csv')
    
    dialysis_expanded_df.to_csv(dialysis_csv_path, index=False)
    mortality_expanded_df.to_csv(mortality_csv_path, index=False)
    
    print(f"Saved CSV summary tables to {dialysis_csv_path} and {mortality_csv_path}")
    
    visualization_paths.extend([dialysis_csv_path, mortality_csv_path])
    
    # Create HTML tables
    # Function to create HTML table
    def create_html_table(data, title, description):
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Create pivot table with UACR stages as rows and eGFR stages as columns
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                p {{ color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; color: #333; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .dataset {{ font-weight: bold; }}
                .train {{ color: #3498db; }}
                .spatial {{ color: #2ecc71; }}
                .temporal {{ color: #e67e22; }}
                .small-sample {{ color: #999; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>{description}</p>
            <table>
                <tr>
                    <th>UACR Stage / eGFR Stage</th>
        """
        
        # Add column headers (eGFR stages)
        for egfr_stage in egfr_stages:
            html_content += f"<th>{egfr_stage}</th>"
        
        html_content += "</tr>"
        
        # Add rows for each UACR stage
        for uacr_stage in uacr_stages:
            html_content += f"<tr><th>{uacr_stage}</th>"
            
            # Add cells for each eGFR stage
            for egfr_stage in egfr_stages:
                # Find the data for this combination
                cell_data = [item for item in data if item['uacr_stage'] == uacr_stage and item['egfr_stage'] == egfr_stage]
                
                if cell_data:
                    cell = cell_data[0]
                    
                    # Format the cell content with all three datasets
                    train_class = "small-sample" if cell['train_n'] < 10 else ""
                    spatial_class = "small-sample" if cell['spatial_n'] < 10 else ""
                    temporal_class = "small-sample" if cell['temporal_n'] < 10 else ""
                    
                    train_pct = f"{cell['train_pct']:.1f}%" if not pd.isna(cell['train_pct']) else "N/A"
                    spatial_pct = f"{cell['spatial_pct']:.1f}%" if not pd.isna(cell['spatial_pct']) else "N/A"
                    temporal_pct = f"{cell['temporal_pct']:.1f}%" if not pd.isna(cell['temporal_pct']) else "N/A"
                    
                    html_content += f"""
                    <td>
                        <div class="dataset train {train_class}">Train: {train_pct} (n={cell['train_n']})</div>
                        <div class="dataset spatial {spatial_class}">Spatial: {spatial_pct} (n={cell['spatial_n']})</div>
                        <div class="dataset temporal {temporal_class}">Temporal: {temporal_pct} (n={cell['temporal_n']})</div>
                    </td>
                    """
                else:
                    html_content += "<td>No data</td>"
            
            html_content += "</tr>"
        
        html_content += """
            </table>
            <p><em>Note: Values in lighter text indicate groups with fewer than 10 patients.</em></p>
        </body>
        </html>
        """
        
        return html_content
    
    # Create and save HTML tables
    dialysis_html = create_html_table(
        dialysis_html_data,
        "5-Year Risk of Dialysis (Event 1) by UACR and eGFR Stage",
        "Cumulative incidence (%) of dialysis initiation at 5 years across different UACR and eGFR stages."
    )
    
    mortality_html = create_html_table(
        mortality_html_data,
        "5-Year Risk of All-Cause Mortality (Event 2) by UACR and eGFR Stage",
        "Cumulative incidence (%) of all-cause mortality at 5 years across different UACR and eGFR stages."
    )
    
    # Save HTML tables
    dialysis_html_path = os.path.join(output_path, 'dialysis_risk_summary.html')
    mortality_html_path = os.path.join(output_path, 'mortality_risk_summary.html')
    
    with open(dialysis_html_path, 'w') as f:
        f.write(dialysis_html)
    
    with open(mortality_html_path, 'w') as f:
        f.write(mortality_html)
    
    print(f"Saved HTML summary tables to {dialysis_html_path} and {mortality_html_path}")
    
    visualization_paths.extend([dialysis_html_path, mortality_html_path])
    
    # Return dictionary with paths
    return {
        "visualization_paths": visualization_paths,
        "dialysis_summary_csv_path": dialysis_csv_path,
        "mortality_summary_csv_path": mortality_csv_path,
        "dialysis_summary_html_path": dialysis_html_path,
        "mortality_summary_html_path": mortality_html_path
    }

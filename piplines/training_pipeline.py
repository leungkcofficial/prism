"""
Training Pipeline for CKD Risk Prediction

This module defines the ZenML pipeline for training the CKD risk prediction model.
"""

import os
import numpy as np
import pandas as pd
from zenml.pipelines import pipeline
from typing import Dict, Any, Optional
import glob

def get_latest_file(directory, prefix):
    """Get the most recently created file with the given prefix in the directory.
    
    Args:
        directory: Directory to search in
        prefix: Prefix of the filename to search for
        
    Returns:
        Path to the most recently created file, or None if no files found
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return None
    
    # Get all files with the given prefix
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No files found matching {pattern}")
        return None
    
    # Return the most recently created file
    latest_file = max(files, key=os.path.getctime)
    print(f"Found latest file: {latest_file}")
    return latest_file


@pipeline(enable_cache=True)
def train_pipeline():
    """
    Pipeline for training the CKD risk prediction model.
    
    This pipeline connects the following steps:
    1. ingest_data: Loads lab data and demographic data for multiple lab tests
       - Creatinine, hemoglobin, hemoglobin A1c, albumin, phosphate
       - Calcium and adjusted calcium
       - Urine protein creatinine ratio and urine albumin creatinine ratio
    2. clean_data: Cleans and preprocesses the data
    3. merge_data: Merges all data sources into a master dataframe
    4. split_data: Splits data into training and test sets
    5. perform_eda: Performs exploratory data analysis
    6. kfre_eval: Evaluates KFRE as a baseline
    7. impute_data: Imputes missing values
    8. preprocess_data: Preprocesses data for model training
    9. feature_selection: Selects important features
    10. train_model: Trains the model with hyperparameter optimization
    
    Returns:
        Model details and optimization metrics
    """
    # Import utility function for loading YAML files
    from src.util import load_yaml_file
    
    # Load hyperparameter configuration from YAML file
    hyperparameter_config = load_yaml_file("src/hyperparameter_config.yml")
    
    # Get model type from configuration
    model_type = hyperparameter_config.get("model_type", "deepsurv")
    print(f"Using model type from configuration: {model_type}")
    
    # Validate model type
    if model_type.lower() not in ["deepsurv", "deephit"]:
        raise ValueError(f"Unsupported model type in configuration: {model_type}. Supported types are 'deepsurv' and 'deephit'.")
    
    # Import steps here to avoid circular imports
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.EDA import perform_eda
    from steps.kfre_eval import kfre_eval
    from steps.feature_selection import feature_selection
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    from steps.model_train import train_model
    from steps.model_deploy import deploy_model
    from steps.model_eval import eval_model
    # Separate evaluation step for more flexibility
    
    # Define the pipeline steps
    # ingest_data now returns multiple DataFrames as separate outputs
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    # Pass all DataFrames to clean_data, which returns a tuple of cleaned dataframes
    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df, cci_score_df, hypertension_df, egfr_df = clean_data(
        cr_df=cr_df,
        hb_df=hb_df,
        a1c_df=a1c_df,
        alb_df=alb_df,
        po4_df=po4_df,
        ca_df=ca_df,
        ca_adjusted_df=ca_adjusted_df,
        hco3_df=hco3_df,
        upcr_df=upcr_df,
        uacr_df=uacr_df,
        icd10_df=icd10_df,
        operation_df=operation_df,
        death_df=death_df,
        demo_df=demo_df
    )
    
    # Merge the cleaned dataframes into a master dataframe
    final_df, prediction_df = merge_data(
        patient_df=patient_df,
        icd10_df=icd10_df_clean,
        cr_df=cr_df_clean,
        hb_df=hb_df_clean,
        a1c_df=a1c_df_clean,
        alb_df=alb_df_clean,
        po4_df=po4_df_clean,
        ca_df=ca_df_clean,
        ca_adjusted_df=ca_adjusted_df_clean,
        hco3_df=hco3_df_clean,
        upcr_df=upcr_df_clean,
        uacr_df=uacr_df_clean,
        operation_df=operation_df_clean,
        death_df=death_df_clean,
        cci_df=cci_df,
        cci_score_df=cci_score_df,
        hypertension_df=hypertension_df,
        egfr_df=egfr_df
    )
    
    # Split the merged data into training, temporal test, and spatial test sets
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )
    
    # Perform exploratory data analysis on the datasets
    eda_results = perform_eda(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # Calculate KFRE for all model types (as baseline)
    kfre_results = kfre_eval(
        train_df=train_df,
        spatial_test_df=spatial_test_df,
        temporal_test_df=temporal_test_df
    )
    
    # Impute missing data via MICE
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    # Preprocess data for model training
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # Perform feature selection on training set only
    feature_selection_results = feature_selection(train_df_preprocessed)
    
    # Define the path to the master dataframe mapping
    master_df_mapping_path = "src/default_master_df_mapping.yml"
    print(f"Using master dataframe mapping from: {master_df_mapping_path}")
    
    # Train model with hyperparameter optimization
    # model_details, optimization_metrics = train_model(
    #     train_df_preprocessed=train_df_preprocessed,
    #     master_df_mapping_path="src/default_master_df_mapping.yml",
    #     hyperparameter_config_path="src/hyperparameter_config.yml"
    # )
    
    # # Deploy model using the file paths
    # print(f"Deploying model using files:\n- Model details: {model_details_path}\n- Optimization metrics: {optimization_metrics_path}")
    deployed_model_details = deploy_model(
        # model_metadata=model_details,
        model_metadata="/mnt/dump/yard/projects/tarot2/results/model_details/model26_details_20250803_014235.json",
        # optimization_metrics=optimization_metrics,
        optimization_metrics="/mnt/dump/yard/projects/tarot2/results/model_details/model26_optimization_metrics_20250803_014235.json",
        model_name=f"CKD_{model_type.upper()}",
        model_stage="Development",
        register_model=True,
        train_df_preprocessed=train_df_preprocessed,  # Pass the preprocessed training data
        temporal_test_df_preprocessed=temporal_test_df_preprocessed,  # Pass the preprocessed temporal test data
        spatial_test_df_preprocessed=spatial_test_df_preprocessed  # Pass the preprocessed spatial test data
    )
    
    # Evaluate model using the prediction files generated by deploy_model
    # The prediction file paths are extracted from deployed_model_details inside the eval_model function
    evaluation_results = eval_model(
        deployed_model_details=deployed_model_details,
        # Pass the dataframes for additional context
        train_df=train_df_preprocessed,
        temporal_test_df=temporal_test_df_preprocessed,
        spatial_test_df=spatial_test_df_preprocessed,
        # Additional parameters
        n_bootstrap=10,  # Reduced from default for faster execution
        visualize=True,
        output_dir="results/model_evaluation"
    )
    
    return {
        "model_type": model_type,
        # "model_details": model_details,
        # "optimization_metrics": optimization_metrics,
        "deployed_model_details": deployed_model_details,
        "evaluation_results": evaluation_results,
        "feature_selection_results": feature_selection_results,
        "kfre_results": kfre_results,
        "eda_results": eda_results
    }
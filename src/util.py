"""Utility module of CKD Risk Prediction model
contain functions commonly used in different steps of the pipeline"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import h5py
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Try to import CUDA libraries
try:
    import cupy as cp
    import numba.cuda
    CUDA_AVAILABLE = True
except ImportError:
    logger.warning("CUDA libraries not found. GPU acceleration will not be available.")
    CUDA_AVAILABLE = False

# Initialize imblearn availability flag
IMBLEARN_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

def load_yaml_file(file_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents.
    
    Args:
        file_path: Path to the YAML file (can be string or Path object)
        default: Default value to return if the file cannot be loaded (default: empty dict)
        
    Returns:
        Dictionary containing the YAML file contents, or the default value if loading fails
    """
    if default is None:
        default = {}
        
    # Convert to Path object for better path handling
    path = Path(file_path)
    
    try:
        # Check if file exists
        if not path.exists():
            logger.warning(f"YAML file not found: {path}")
            return default
            
        # Open and load the file
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
            
        # Check if data is None (empty file)
        if data is None:
            logger.warning(f"YAML file is empty: {path}")
            return default
            
        return data
    except Exception as e:
        logger.error(f"Error loading YAML file {path}: {e}")
        return default

def save_yaml_file(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Save data to a YAML file.
    
    Args:
        file_path: Path to the YAML file (can be string or Path object)
        data: Dictionary containing the data to save
        
    Returns:
        bool: True if the file was saved successfully, False otherwise
    """
    # Convert to Path object for better path handling
    path = Path(file_path)
    
    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open and save the file
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
            
        logger.info(f"YAML file saved successfully: {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving YAML file {path}: {e}")
        return False

def df_event_focus(df, event_col, event_focus=1):
    """
    Creates a copy of the DataFrame where the event column is set to 1 for rows matching the event_focus value, else 0.

    Args:
        df (pd.DataFrame): Input DataFrame.
        event_col (str): Column representing the event.
        event_focus (int, optional): Event value to focus on. Defaults to 1.

    Returns:
        pd.DataFrame: Modified DataFrame with updated event column.
    """
    df2 = df.copy()
    df2[event_col] = np.where(df2[event_col] == event_focus, 1, 0)
    logger.info(f"Event column '{event_col}' updated with focus on event value {event_focus}.")
    return df2

def setup_imblearn() -> bool:
    """
    Check if imblearn package is available and install it if not.
    
    Returns:
        bool: True if imblearn is available (either already installed or successfully installed), False otherwise
    """
    global IMBLEARN_AVAILABLE
    
    if IMBLEARN_AVAILABLE:
        return True
        
    try:
        # Try to import imblearn
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.under_sampling import NearMiss
        IMBLEARN_AVAILABLE = True
        return True
    except ImportError:
        logger.warning("imblearn package not found. Attempting to install...")
        try:
            # Attempt to install imblearn
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
            logger.info("Successfully installed imbalanced-learn package.")
            
            # Try importing again after installation
            from imblearn.combine import SMOTEENN, SMOTETomek
            from imblearn.under_sampling import NearMiss
            IMBLEARN_AVAILABLE = True
            return True
        except Exception as e:
            logger.error(f"Failed to install imbalanced-learn: {e}")
            logger.warning("SMOTEENN, SMOTETomek, and NearMiss methods will not be available.")
            IMBLEARN_AVAILABLE = False
            return False

def setup_cuda_environment() -> bool:
    """
    Set up the CUDA environment for GPU-accelerated operations.
    Handles memory management and device initialization.
    
    Returns:
        bool: True if CUDA environment was successfully initialized, False otherwise
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA libraries not available. Using CPU fallback.")
        return False
    
    try:
        # Clear any existing CUDA memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Check if CUDA device is available
        if numba.cuda.is_available():
            logger.info("CUDA environment initialized successfully.")
            return True
        else:
            logger.warning("CUDA device not available. Using CPU fallback.")
            return False
    except Exception as e:
        logger.error(f"Error initializing CUDA environment: {e}")
        logger.warning("Using CPU fallback.")
        return False
    
def observed_risk_calculator(df: pd.DataFrame,
                             duration: str,
                             event: str,
                             time_points: List[float] = None,
                             model='KM',
                             output_col='observed_risk',
                             mapping_file: str = 'src/default_master_df_mapping.yml') -> pd.DataFrame:
    """
    Calculate observed risk using survival analysis and add it to the dataframe.
    
    Args:
        df: DataFrame containing patient data
        duration: Column name for duration/time
        event: Column name for event indicator
        time_points: List of time points to calculate risk at (if None, uses unique durations)
        model: Model to use for risk calculation ('KM' for Kaplan-Meier, 'AJ' for Aalen-Johansen)
        output_col: Name of the output column
        mapping_file: Path to the YAML file containing column mappings
        
    Returns:
        DataFrame with added observed risk column
    """
    try:
        # 1. Get the mapping of columns
        mapping = load_yaml_file(mapping_file)
        if not mapping:
            logger.error(f"Failed to load mapping from {mapping_file}")
            return df
            
        # Make a copy of the input DataFrame
        result_df = df.copy()
        
        # Check if required columns exist
        if duration not in df.columns:
            logger.error(f"Duration column '{duration}' not found in dataframe")
            return result_df
            
        if event not in df.columns:
            logger.error(f"Event column '{event}' not found in dataframe")
            return result_df
            
        # Extract duration and event data
        # Ensure data is numeric
        try:
            durations = pd.to_numeric(df[duration], errors='coerce').values
            events = pd.to_numeric(df[event], errors='coerce').values
            
            # Check for NaN values after conversion
            valid_mask = ~np.isnan(durations) & ~np.isnan(events)
            if not np.all(valid_mask):
                logger.warning(f"Removed {(~valid_mask).sum()} non-numeric values from durations/events")
                durations = durations[valid_mask]
                events = events[valid_mask]
                
                # Create a filtered dataframe for further processing
                filtered_df = df.iloc[valid_mask].copy()
            else:
                filtered_df = df.copy()
                
            # If no valid data after filtering, return original dataframe
            if len(durations) == 0:
                logger.error("No valid numeric data for risk calculation")
                return df
        except Exception as e:
            logger.error(f"Error converting duration/event data to numeric: {e}")
            return df
        
        # If no time points provided, use unique durations
        if time_points is None:
            time_points = sorted(filtered_df[duration].unique())
        
        # 2. Use lifelines to calculate risk
        from lifelines import KaplanMeierFitter, AalenJohansenFitter
        
        if model.upper() == 'KM':
            # Use Kaplan-Meier for binary events
            fitter = KaplanMeierFitter()
            fitter.fit(durations, events)
            
            # Calculate cumulative incidence (1 - survival)
            cumulative_incidence = 1 - fitter.survival_function_
            
            # Interpolate risk at each time point
            risk_at_times = {}
            for t in time_points:
                if t in cumulative_incidence.index:
                    risk_at_times[t] = cumulative_incidence.loc[t].values[0]
                else:
                    # Find closest time point
                    idx = cumulative_incidence.index.get_indexer([t], method='nearest')[0]
                    closest_t = cumulative_incidence.index[idx]
                    risk_at_times[t] = cumulative_incidence.loc[closest_t].values[0]
                    
        elif model.upper() == 'AJ':
            # Use Aalen-Johansen for competing risks
            fitter = AalenJohansenFitter()
            fitter.fit(durations, events, event_of_interest=1)  # Assuming 1 is the event of interest
            
            # Get cumulative incidence
            cumulative_incidence = fitter.cumulative_density_
            
            # Interpolate risk at each time point
            risk_at_times = {}
            for t in time_points:
                if t in cumulative_incidence.index:
                    risk_at_times[t] = cumulative_incidence.loc[t].values[0]
                else:
                    # Find closest time point
                    idx = cumulative_incidence.index.get_indexer([t], method='nearest')[0]
                    closest_t = cumulative_incidence.index[idx]
                    risk_at_times[t] = cumulative_incidence.loc[closest_t].values[0]
        else:
            logger.error(f"Unknown model '{model}'. Use 'KM' or 'AJ'.")
            return result_df
            
        # 3. Store results in new column
        # Create a new column for each time point
        for t in time_points:
            col_name = f"{output_col}_{t}" if len(time_points) > 1 else output_col
            result_df[col_name] = risk_at_times[t]
            
        logger.info(f"Successfully calculated observed risk using {model} model")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating observed risk: {e}")
        return df


def access_zenml_artifact_data(artifact: Any, flatten_nested_tuples: bool = True) -> Any:
    """
    Directly access the internal data of a ZenML StepArtifact.
    
    This function attempts to access the internal data structure of a ZenML StepArtifact
    by inspecting its attributes and trying to access the raw data directly.
    
    Args:
        artifact: The ZenML StepArtifact object
        flatten_nested_tuples: If True, flatten single-item nested tuples (default: True)
        
    Returns:
        The raw data contained in the artifact, or None if it cannot be accessed
    """
    if artifact is None:
        print("Artifact is None")
        return None
        
    print(f"Accessing ZenML artifact data (type: {type(artifact).__name__})")
    
    # If the artifact is already a basic type, return it directly
    if isinstance(artifact, (tuple, list, dict, pd.DataFrame, np.ndarray)):
        print(f"Artifact is already a basic type: {type(artifact).__name__}")
        return artifact
    
    # Special handling for torch tensors - convert to numpy
    if 'torch' in str(type(artifact)):
        try:
            import torch
            if isinstance(artifact, torch.Tensor):
                print("Converting torch tensor to numpy array")
                print(f"Tensor shape: {artifact.shape}, dtype: {artifact.dtype}")
                print(f"Tensor device: {artifact.device if hasattr(artifact, 'device') else 'unknown'}")
                
                # Move tensor to CPU if it's on GPU
                if hasattr(artifact, 'is_cuda') and artifact.is_cuda:
                    print(f"Moving tensor from GPU ({artifact.device}) to CPU")
                    try:
                        artifact = artifact.cpu()
                        print("Successfully moved tensor to CPU")
                    except Exception as e:
                        print(f"Error moving tensor to CPU: {e}")
                        print(f"Tensor properties: shape={artifact.shape}, dtype={artifact.dtype}, device={getattr(artifact, 'device', 'unknown')}")
                        # Try a different approach - create a new tensor on CPU
                        try:
                            print("Attempting alternative CPU conversion")
                            artifact = torch.tensor(artifact.detach().cpu().numpy())
                            print("Successfully created new CPU tensor")
                        except Exception as e2:
                            print(f"Alternative CPU conversion also failed: {e2}")
                            raise
                
                # Check for NaN or Inf values
                try:
                    if torch.isnan(artifact).any():
                        print("WARNING: Tensor contains NaN values!")
                        nan_count = torch.isnan(artifact).sum().item()
                        print(f"NaN count: {nan_count}")
                    if torch.isinf(artifact).any():
                        print("WARNING: Tensor contains Inf values!")
                        inf_count = torch.isinf(artifact).sum().item()
                        print(f"Inf count: {inf_count}")
                except Exception as e:
                    print(f"Error checking for NaN/Inf values: {e}")
                
                # Convert to numpy
                try:
                    # Make sure tensor is on CPU and detached from computation graph
                    if hasattr(artifact, 'is_cuda') and artifact.is_cuda:
                        artifact = artifact.cpu()
                    if hasattr(artifact, 'detach'):
                        artifact = artifact.detach()
                    
                    # Convert to numpy
                    numpy_array = artifact.numpy()
                    print(f"Successfully converted to numpy array with shape {numpy_array.shape}")
                    return numpy_array
                except Exception as e:
                    print(f"Error in final tensor to numpy conversion: {e}")
                    # Try an alternative approach
                    try:
                        print("Attempting alternative numpy conversion")
                        numpy_array = artifact.cpu().detach().numpy()
                        print(f"Alternative conversion successful, shape: {numpy_array.shape}")
                        return numpy_array
                    except Exception as e2:
                        print(f"Alternative numpy conversion also failed: {e2}")
                        raise
        except Exception as e:
            print(f"Error converting torch tensor to numpy: {e}")
            print(f"Tensor type: {type(artifact).__name__}, attributes: {dir(artifact)[:10]}...")
            print(f"Error details: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    # Print all attributes of the artifact
    all_attrs = dir(artifact)
    print(f"All attributes: {all_attrs}")
    
    # Try to access internal data structures
    for attr_name in ['_data', '_value', 'value', 'data', '_artifact_data', 'artifact_data', '_content', 'content']:
        if hasattr(artifact, attr_name):
            try:
                data = getattr(artifact, attr_name)
                print(f"Found data in '{attr_name}' attribute: {type(data).__name__}")
                
                # If data is a torch tensor, convert to numpy
                if 'torch' in str(type(data)):
                    try:
                        import torch
                        if isinstance(data, torch.Tensor):
                            print("Converting nested torch tensor to numpy array")
                            print(f"Nested tensor shape: {data.shape}, dtype: {data.dtype}")
                            print(f"Nested tensor device: {data.device}")
                            
                            # Move tensor to CPU if it's on GPU
                            if data.is_cuda:
                                print(f"Moving nested tensor from GPU ({data.device}) to CPU")
                                try:
                                    data = data.cpu()
                                    print("Successfully moved nested tensor to CPU")
                                except Exception as e:
                                    print(f"Error moving nested tensor to CPU: {e}")
                                    print(f"Nested tensor properties: shape={data.shape}, dtype={data.dtype}, device={data.device}")
                                    raise
                            
                            # Check for NaN or Inf values
                            if torch.isnan(data).any():
                                print("WARNING: Nested tensor contains NaN values!")
                            if torch.isinf(data).any():
                                print("WARNING: Nested tensor contains Inf values!")
                            
                            # Convert to numpy
                            try:
                                numpy_array = data.detach().numpy()
                                print(f"Successfully converted nested tensor to numpy array with shape {numpy_array.shape}")
                                return numpy_array
                            except Exception as e:
                                print(f"Error in final nested tensor to numpy conversion: {e}")
                                raise
                    except Exception as e:
                        print(f"Error converting nested torch tensor to numpy: {e}")
                        print(f"Nested tensor type: {type(data).__name__}, attributes: {dir(data)[:10]}...")
                        print(f"Error details: {str(e)}")
                
                # Handle nested tuples if requested
                if flatten_nested_tuples and isinstance(data, tuple) and len(data) == 1:
                    inner_item = data[0]
                    if isinstance(inner_item, tuple):
                        print(f"Flattening nested tuple: outer len={len(data)}, inner len={len(inner_item)}")
                        return inner_item
                return data
            except Exception as e:
                print(f"Error accessing '{attr_name}' attribute: {e}")
    
    # Try to access the artifact as a descriptor
    try:
        if hasattr(artifact, '__get__'):
            print("Trying to access artifact as descriptor")
            data = artifact.__get__(None, type(None))
            print(f"Descriptor access returned: {type(data).__name__}")
            
            # If data is a torch tensor, convert to numpy
            if 'torch' in str(type(data)):
                try:
                    import torch
                    if isinstance(data, torch.Tensor):
                        print("Converting torch tensor to numpy array")
                        # Move tensor to CPU if it's on GPU
                        if data.is_cuda:
                            print("Moving tensor from GPU to CPU")
                            data = data.cpu()
                        return data.detach().numpy()
                except Exception as e:
                    print(f"Error converting torch tensor to numpy: {e}")
                    print(f"Tensor type: {type(data).__name__}, attributes: {dir(data)[:10]}...")
            
            return data
    except Exception as e:
        print(f"Error accessing artifact as descriptor: {e}")
    
    # Try to call the artifact as a function
    try:
        if callable(artifact):
            print("Trying to call artifact as function")
            data = artifact()
            print(f"Function call returned: {type(data).__name__}")
            
            # If data is a torch tensor, convert to numpy
            if 'torch' in str(type(data)):
                try:
                    import torch
                    if isinstance(data, torch.Tensor):
                        print("Converting torch tensor to numpy array")
                        # Move tensor to CPU if it's on GPU
                        if data.is_cuda:
                            print("Moving tensor from GPU to CPU")
                            data = data.cpu()
                        return data.detach().numpy()
                except Exception as e:
                    print(f"Error converting torch tensor to numpy: {e}")
                    print(f"Tensor type: {type(data).__name__}, attributes: {dir(data)[:10]}...")
            
            return data
    except Exception as e:
        print(f"Error calling artifact as function: {e}")
    
    # Try to access the artifact as a materialized object
    try:
        if hasattr(artifact, 'materialize'):
            print("Trying to materialize artifact")
            data = artifact.materialize()
            print(f"Materialization returned: {type(data).__name__}")
            
            # If data is a torch tensor, convert to numpy
            if 'torch' in str(type(data)):
                try:
                    import torch
                    if isinstance(data, torch.Tensor):
                        print("Converting torch tensor to numpy array")
                        # Move tensor to CPU if it's on GPU
                        if data.is_cuda:
                            print("Moving tensor from GPU to CPU")
                            data = data.cpu()
                        return data.detach().numpy()
                except Exception as e:
                    print(f"Error converting torch tensor to numpy: {e}")
                    print(f"Tensor type: {type(data).__name__}, attributes: {dir(data)[:10]}...")
            
            return data
    except Exception as e:
        print(f"Error materializing artifact: {e}")
    
    # Try to access the artifact as a ZenML artifact
    try:
        if 'zenml' in str(type(artifact)):
            print("Trying to access ZenML artifact data")
            
            # Try to access the data attribute
            if hasattr(artifact, 'data'):
                print("Accessing 'data' attribute of ZenML artifact")
                data = artifact.data
                print(f"Data attribute returned: {type(data).__name__}")
                return data
            
            # Try to access the value attribute
            if hasattr(artifact, 'value'):
                print("Accessing 'value' attribute of ZenML artifact")
                data = artifact.value
                print(f"Value attribute returned: {type(data).__name__}")
                return data
    except Exception as e:
        print(f"Error accessing ZenML artifact data: {e}")
    
    # If all else fails, return the artifact itself
    print("All methods failed, returning the artifact itself")
    return artifact

def extract_from_step_artifact(artifact: Any, expected_type: Optional[type] = None,
                              artifact_name: str = "artifact", flatten_nested_tuples: bool = True,
                              default_value: Optional[Any] = None) -> Any:
    """
    Extract data from a ZenML step artifact with better error handling and debugging.
    
    This function attempts to extract the data from a ZenML step artifact using various methods,
    with detailed error reporting and fallback mechanisms.
    
    Args:
        artifact: The ZenML step artifact
        expected_type: The expected type of the extracted data (optional)
        artifact_name: Name of the artifact for logging purposes (default: "artifact")
        flatten_nested_tuples: If True, flatten single-item nested tuples (default: True)
        default_value: Value to return if extraction fails (default: None)
        
    Returns:
        The extracted data, or default_value if extraction fails
    """
    try:
        print(f"Extracting data from {artifact_name} (type: {type(artifact).__name__})")
        
        # Store raw data from direct access for fallback
        raw_data = None
        
        # Method 1: If the artifact is None, return default value
        if artifact is None:
            print(f"{artifact_name} is None")
            return default_value if default_value is not None else None
        
        # Method 2: If the artifact is already the expected type, return it directly
        if expected_type is not None and isinstance(artifact, expected_type):
            print(f"{artifact_name} is already the expected type ({expected_type.__name__})")
            return artifact
        
        # Method 3: If the artifact is a StepArtifact, try to extract its data
        if 'StepArtifact' in str(type(artifact)):
            print(f"{artifact_name} is a StepArtifact")
            
            # Try to access the data attribute
            if hasattr(artifact, 'data'):
                try:
                    data = artifact.data
                    print(f"Accessed 'data' attribute: {type(data).__name__}")
                    raw_data = data
                    
                    # Check if data matches expected type
                    if expected_type is not None and not isinstance(data, expected_type):
                        print(f"Raw data type {type(data).__name__} doesn't match expected type {expected_type.__name__}")
                        
                        # Try to convert to expected type
                        try:
                            if expected_type == pd.DataFrame and isinstance(data, np.ndarray):
                                print(f"Converting numpy array to DataFrame")
                                return pd.DataFrame(data)
                            elif expected_type == np.ndarray and isinstance(data, pd.DataFrame):
                                print(f"Converting DataFrame to numpy array")
                                return data.values
                            elif expected_type == list and isinstance(data, np.ndarray):
                                print(f"Converting numpy array to list")
                                return data.tolist()
                            elif expected_type == list and isinstance(data, pd.DataFrame):
                                print(f"Converting DataFrame to list")
                                return data.values.tolist()
                        except Exception as e:
                            print(f"Error converting data to expected type: {e}")
                    
                    return data
                except Exception as e:
                    print(f"Error accessing 'data' attribute: {e}")
            
            # Try to materialize the artifact
            if hasattr(artifact, 'materialize'):
                try:
                    print(f"Attempting to materialize {artifact_name}")
                    data = artifact.materialize()
                    print(f"Materialized data type: {type(data).__name__}")
                    return data
                except Exception as e:
                    print(f"Error materializing artifact: {e}")
        
        # Method 4: Try to access the artifact directly using ZenML's access_artifact_data
        try:
            from zenml.utils.artifact_utils import access_artifact_data
            print(f"Trying to access artifact data using ZenML's access_artifact_data")
            data = access_artifact_data(artifact)
            print(f"ZenML access_artifact_data returned: {type(data).__name__}")
            raw_data = data
            return data
        except Exception as e:
            print(f"Error using ZenML's access_artifact_data: {e}")
        
        # Method 5: Try to access the artifact using our custom function
        try:
            print(f"Trying to access artifact data using custom access_zenml_artifact_data")
            data = access_zenml_artifact_data(artifact, flatten_nested_tuples)
            print(f"Custom access_zenml_artifact_data returned: {type(data).__name__}")
            raw_data = data
            return data
        except Exception as e:
            print(f"Error using custom access_zenml_artifact_data: {e}")
        
        # Method 6: If the artifact is a numpy array, return it directly
        if isinstance(artifact, np.ndarray):
            print(f"{artifact_name} is a numpy array with shape {artifact.shape}")
            return artifact
        
        # Method 7: If the artifact is a pandas DataFrame, return it directly
        if isinstance(artifact, pd.DataFrame):
            print(f"{artifact_name} is a pandas DataFrame with shape {artifact.shape}")
            return artifact
        
        # Method 7.5: If the artifact is a dictionary, return it directly
        if isinstance(artifact, dict):
            print(f"{artifact_name} is already a dictionary")
            return artifact
            
        # Method 8: For dataset-like objects, try to access the first element
        if hasattr(artifact, '__getitem__'):
            print(f"{artifact_name} has __getitem__ method, treating as dataset")
            try:
                # Try to access the first element to verify it's a dataset
                _ = artifact[0]
                return artifact
            except Exception as e:
                print(f"Error accessing first element of {artifact_name}: {e}")
                
                # It might be a dict-like object with string keys
                try:
                    if hasattr(artifact, 'keys'):
                        keys = list(artifact.keys())
                        print(f"{artifact_name} has keys method, keys: {keys}")
                        
                        # Check for common hyperparameter keys
                        for key in ['best_params', 'params', 'hyperparams', 'best_hyperparams']:
                            if key in keys:
                                print(f"Found key '{key}' in {artifact_name}")
                                return artifact[key]
                except Exception as e2:
                    print(f"Error checking keys of {artifact_name}: {e2}")
        
        # Method 9: Try to convert to a pandas DataFrame
        try:
            import pandas as pd
            print(f"Attempting to convert {artifact_name} to DataFrame")
            df = pd.DataFrame(artifact)
            print(f"Successfully converted {artifact_name} to DataFrame")
            return df
        except Exception as e:
            print(f"Error converting {artifact_name} to DataFrame: {e}")
        
        # Method 10: Try to access the artifact as a tuple or list
        if isinstance(artifact, (tuple, list)):
            print(f"{artifact_name} is a {type(artifact).__name__}, returning directly")
            return artifact
        
        # Method 11: Try to convert the artifact to the expected type
        if expected_type is not None:
            try:
                print(f"Attempting to convert {artifact_name} to {expected_type.__name__}")
                converted = expected_type(artifact)
                print(f"Successfully converted {artifact_name} to {expected_type.__name__}")
                return converted
            except Exception as e:
                print(f"Error converting {artifact_name} to {expected_type.__name__}: {e}")
            
        # If we have raw_data from direct access, return it as a last resort
        if raw_data is not None:
            print(f"Using raw data from direct access as fallback for {artifact_name}")
            return raw_data
            
        # Special case: If the artifact itself is a simple type, return it directly
        if isinstance(artifact, (int, float, str, bool, list, tuple, dict, np.ndarray)) or artifact is None:
            print(f"{artifact_name} is a simple type ({type(artifact).__name__}), returning directly")
            
            # If it's a tuple-like object with a single item and expected_type is tuple
            if expected_type == tuple and hasattr(artifact, '__getitem__') and hasattr(artifact, '__len__') and len(artifact) == 1:
                item = artifact[0]
                if isinstance(item, tuple):
                    print(f"Converting single-item tuple to tuple: {item}")
                    return item
            
            # Return the artifact itself
            return artifact
            
    except Exception as e:
        print(f"Error extracting data from {artifact_name}: {e}")
        return default_value if default_value is not None else None


def save_predictions_to_hdf5(
    predictions: pd.DataFrame,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = 'gzip',
    compression_opts: int = 9
) -> None:
    """
    Save prediction matrix to HDF5 file format.
    
    This function efficiently stores large prediction matrices in HDF5 format,
    which is more memory-efficient than CSV for large numerical data.
    
    Args:
        predictions: DataFrame containing survival predictions
        file_path: Path to save the HDF5 file
        metadata: Optional dictionary of metadata to store with the predictions
        compression: Compression filter to use (default: 'gzip')
        compression_opts: Compression level (default: 9, highest compression)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert predictions to numpy array if it's a DataFrame
        if isinstance(predictions, pd.DataFrame):
            # Save the index and columns separately for reconstruction
            index_values = predictions.index.values
            column_values = predictions.columns.values
            predictions_array = predictions.values
        else:
            # If it's already a numpy array, just use it directly
            predictions_array = predictions
            index_values = np.arange(predictions_array.shape[0])
            column_values = np.arange(predictions_array.shape[1])
        
        # Open HDF5 file
        with h5py.File(file_path, 'w') as f:
            # Create a dataset for the predictions with compression
            f.create_dataset(
                'predictions', 
                data=predictions_array,
                compression=compression,
                compression_opts=compression_opts
            )
            
            # Save index and columns
            f.create_dataset('index', data=index_values)
            f.create_dataset('columns', data=column_values)
            
            # Save metadata if provided
            if metadata is not None:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    # Handle different types of metadata
                    if isinstance(value, (int, float, bool)):
                        # Store simple numeric types directly
                        metadata_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        # Store numpy arrays as datasets
                        print(f"Saving numpy array '{key}' with shape {value.shape} as dataset")
                        metadata_group.create_dataset(key, data=value)
                    else:
                        # Convert other types to strings
                        metadata_group.attrs[key] = str(value)
        
        # Calculate file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Predictions saved to HDF5 file: {file_path} (Size: {file_size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Error saving predictions to HDF5: {e}")
        raise


def load_predictions_from_hdf5(
    file_path: str,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Load prediction matrix from HDF5 file format.
    
    This function efficiently loads large prediction matrices from HDF5 format.
    
    Args:
        file_path: Path to the HDF5 file
        return_metadata: Whether to return metadata along with predictions
        
    Returns:
        If return_metadata is False: DataFrame containing survival predictions
        If return_metadata is True: Tuple of (DataFrame, metadata dictionary)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        # Open HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Load predictions
            predictions_array = f['predictions'][:]
            
            # Check if this is the new competing risks format or old format
            if 'index' in f:
                # Old format (DeepSurv or legacy DeepHit)
                index_values = f['index'][:]
                column_values = f['columns'][:]
            else:
                # New competing risks format (DeepHit)
                # Check if we have time_grid and competing risks structure
                if 'time_grid' in f and len(predictions_array.shape) == 3:
                    # This is the new (2, 5, n_samples) format
                    time_grid = f['time_grid'][:]
                    num_causes, num_time_points, num_samples = predictions_array.shape
                    
                    print(f"Detected competing risks format: {predictions_array.shape}")
                    print(f"Causes: {num_causes}, Time points: {num_time_points}, Samples: {num_samples}")
                    
                    # For DeepHit competing risks, keep the 3D format (2, 5, n_samples)
                    # Don't flatten - the evaluation should handle each cause separately
                    # This preserves the (5, n_samples) shape for each event as requested
                    
                    # Create metadata for the 3D format
                    column_values = f['columns'][:] if 'columns' in f else np.arange(num_samples)
                    
                    # Create index that reflects the 3D structure
                    index_values = [f"cause_{i+1}" for i in range(num_causes)]
                    index_values = np.array(index_values, dtype='S')
                    
                    print(f"Preserved 3D competing risks format: {predictions_array.shape}")
                    print(f"Each cause has shape: ({num_time_points}, {num_samples})")
                    
                    # Skip all the padding logic for 3D arrays - return directly
                    predictions_df = predictions_array  # Return as numpy array for 3D format
                    
                    if return_metadata:
                        metadata = {
                            'time_grid': time_grid,
                            'num_causes': num_causes,
                            'num_time_points': num_time_points,
                            'num_samples': num_samples
                        }
                        return predictions_df, metadata
                    else:
                        return predictions_df
                else:
                    # Fallback: create default index and columns
                    column_values = np.arange(predictions_array.shape[-1])
                    if len(predictions_array.shape) == 2:
                        index_values = np.arange(predictions_array.shape[0])
                    else:
                        # Flatten if needed
                        predictions_array = predictions_array.reshape(-1, predictions_array.shape[-1])
                        index_values = np.arange(predictions_array.shape[0])
            
            # Check if the number of rows in predictions matches the metadata
            if 'metadata' in f and return_metadata:
                metadata_group = f['metadata']
                if 'durations' in metadata_group.keys() and 'events' in metadata_group.keys():
                    durations = metadata_group['durations'][:]
                    events = metadata_group['events'][:]
                    
                    n_samples = len(durations)
                    
                    # Determine expected time points based on model type
                    # Check if this is competing risks format (DeepHit) or standard format (DeepSurv)
                    if 'time_grid' in f and len(f['time_grid'][:]) <= 10:
                        # DeepHit competing risks format - use actual time grid
                        time_grid = f['time_grid'][:]
                        num_causes = 2  # Assuming 2 competing causes
                        n_time_points = len(time_grid) * num_causes  # e.g., 5 * 2 = 10
                        print(f"Detected DeepHit competing risks format with {len(time_grid)} time points and {num_causes} causes")
                        print(f"Expected time points: {n_time_points} (flattened from {num_causes} causes × {len(time_grid)} time points)")
                    else:
                        # DeepSurv standard format - use full time grid (1825 time points)
                        n_time_points = 1825
                        print(f"Detected DeepSurv standard format with {n_time_points} time points")
                    
                    print(f"Predictions array shape: {predictions_array.shape}")
                    print(f"Expected shape: ({n_time_points}, {n_samples})")
                    
                    # Check if the shape matches the expected format
                    if predictions_array.shape[0] == n_time_points and predictions_array.shape[1] == n_samples:
                        print("Predictions have the correct shape")
                    else:
                        print(f"INFO: Predictions shape {predictions_array.shape} does not match expected shape ({n_time_points}, {n_samples})")
                        
                        # Check if we need to transpose
                        if predictions_array.shape[0] == n_samples and predictions_array.shape[1] == n_time_points:
                            print("Transposing predictions to match expected format")
                            predictions_array = predictions_array.T
                            print(f"Transposed predictions shape: {predictions_array.shape}")
                        
                        # Check if the number of columns (samples) matches the durations length
                        if predictions_array.shape[1] != n_samples:
                            print(f"Adjusting predictions to match {n_samples} samples")
                            if predictions_array.shape[1] > n_samples:
                                # Truncate columns
                                predictions_array = predictions_array[:, :n_samples]
                                print(f"Truncated predictions to {predictions_array.shape}")
                            else:
                                # Pad columns with zeros
                                pad_cols = n_samples - predictions_array.shape[1]
                                pad_array = np.zeros((predictions_array.shape[0], pad_cols))
                                predictions_array = np.hstack([predictions_array, pad_array])
                                print(f"Padded predictions to {predictions_array.shape}")
                        
                        # For DeepHit competing risks, don't pad time points - the shape is correct as-is
                        if predictions_array.shape[0] != n_time_points:
                            if 'time_grid' in f and len(f['time_grid'][:]) <= 10:
                                print(f"INFO: DeepHit predictions have {predictions_array.shape[0]} time points, which is correct for competing risks format")
                                print("No padding needed for DeepHit competing risks predictions")
                            else:
                                print(f"WARNING: DeepSurv predictions have {predictions_array.shape[0]} time points but expected {n_time_points}")
                                # Only pad for DeepSurv format
                                if predictions_array.shape[0] > n_time_points:
                                    # Truncate rows
                                    predictions_array = predictions_array[:n_time_points, :]
                                    print(f"Truncated predictions to {predictions_array.shape}")
                                else:
                                    # Pad rows with zeros
                                    pad_rows = n_time_points - predictions_array.shape[0]
                                    pad_array = np.zeros((pad_rows, predictions_array.shape[1]))
                                    predictions_array = np.vstack([predictions_array, pad_array])
                                    print(f"Padded predictions to {predictions_array.shape}")
            
            # Create DataFrame
            predictions_df = pd.DataFrame(
                predictions_array,
                columns=column_values
            )
            
            # Load metadata if requested
            if return_metadata and 'metadata' in f:
                metadata = dict(f['metadata'].attrs)
                
                # Check for numpy array datasets in metadata group
                if 'metadata' in f:
                    metadata_group = f['metadata']
                    # Add datasets to metadata dictionary
                    for key in metadata_group.keys():
                        print(f"Loading '{key}' from metadata")
                        dataset = metadata_group[key]
                        
                        # Handle scalar vs array datasets
                        if dataset.shape == ():  # Scalar dataset
                            metadata[key] = dataset[()]  # Use [()] for scalar
                            print(f"Loaded scalar '{key}' with value {metadata[key]}")
                        else:  # Array dataset
                            metadata[key] = dataset[:]
                            print(f"Loaded array '{key}' with shape {metadata[key].shape}")
                
                return predictions_df, metadata
            
            return predictions_df
            
    except Exception as e:
        logger.error(f"Error loading predictions from HDF5: {e}")
        raise


# =============================================================================
# LSTM Sequence Utility Functions
# =============================================================================

def create_sequences_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    cluster_col: str = 'key',
    date_col: str = 'date',
    feature_cols: List[str] = None,
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None,
    pad_value: float = 0.0,
    min_sequence_length: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from a dataframe for LSTM training.
    
    Args:
        df: Input dataframe with patient data
        sequence_length: Desired sequence length
        cluster_col: Column name for patient/cluster ID (default: 'key')
        date_col: Column name for date/time (default: 'date')
        feature_cols: List of feature columns to include
        duration_col: Column name for survival duration (default: 'duration')
        event_col: Column name for event indicator (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        pad_value: Value to use for padding short sequences (default: 0.0)
        min_sequence_length: Minimum sequence length to include (default: 1)
        
    Returns:
        Tuple of (X_sequences, durations, events, patient_ids)
        - X_sequences: (n_samples, sequence_length, n_features)
        - durations: (n_samples,)
        - events: (n_samples,)
        - patient_ids: (n_samples,) - patient IDs for each sequence
    """
    print(f"\n=== Creating sequences from dataframe ===")
    print(f"Input dataframe shape: {df.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"Cluster column: {cluster_col}")
    print(f"Date column: {date_col}")
    
    # Validate required columns
    required_cols = [cluster_col, date_col, duration_col, event_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare feature columns
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = [cluster_col, date_col, duration_col, event_col,
                       'patient_id', 'endpoint_date', 'first_sub_60_date', 'dob', 'icd10']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    # Convert categorical columns to numeric
    df_processed = df.copy()
    for col in feature_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            print(f"Converting categorical column to numeric: {col}")
            if df_processed[col].nunique() <= 2:
                # Binary categorical
                most_common = df_processed[col].mode()[0]
                df_processed[col] = (df_processed[col] != most_common).astype(float)
            else:
                # Multi-category - use label encoding for simplicity
                df_processed[col] = pd.Categorical(df_processed[col]).codes.astype(float)
    
    # Sort by patient and date
    df_sorted = df_processed.sort_values([cluster_col, date_col]).reset_index(drop=True)
    print(f"Sorted dataframe by {cluster_col} and {date_col}")
    
    # Group by patient
    patient_groups = df_sorted.groupby(cluster_col)
    print(f"Found {len(patient_groups)} unique patients")
    
    sequences = []
    durations = []
    events = []
    patient_ids = []
    
    patients_with_sufficient_data = 0
    patients_padded = 0
    
    for patient_id, group in patient_groups:
        group_sorted = group.sort_values(date_col).reset_index(drop=True)
        n_observations = len(group_sorted)
        
        if n_observations < min_sequence_length:
            continue
        
        # Extract features, duration, and event from the last observation
        last_obs = group_sorted.iloc[-1]
        duration = last_obs[duration_col]
        event = last_obs[event_col]
        
        # Apply target endpoint filtering if specified
        if target_endpoint is not None:
            event = 1.0 if event == target_endpoint else 0.0
        
        # Extract feature matrix for this patient
        feature_matrix = group_sorted[feature_cols].values.astype(float)
        
        if n_observations >= sequence_length:
            # Use the last sequence_length observations
            sequence = feature_matrix[-sequence_length:]
            patients_with_sufficient_data += 1
        else:
            # Pad with zeros at the beginning (older timestamps)
            padding_needed = sequence_length - n_observations
            padding = np.full((padding_needed, len(feature_cols)), pad_value, dtype=float)
            sequence = np.vstack([padding, feature_matrix])
            patients_padded += 1
        
        sequences.append(sequence)
        durations.append(duration)
        events.append(event)
        patient_ids.append(patient_id)
    
    # Convert to numpy arrays
    X_sequences = np.array(sequences, dtype=float)
    durations_array = np.array(durations, dtype=float)
    events_array = np.array(events, dtype=float)
    patient_ids_array = np.array(patient_ids)
    
    print(f"\n=== Sequence generation summary ===")
    print(f"Total patients processed: {len(patient_groups)}")
    print(f"Patients with sufficient data (>= {sequence_length} obs): {patients_with_sufficient_data}")
    print(f"Patients requiring padding: {patients_padded}")
    print(f"Final sequences shape: {X_sequences.shape}")
    print(f"Durations shape: {durations_array.shape}")
    print(f"Events shape: {events_array.shape}")
    print(f"Event rate: {events_array.mean():.2%}")
    
    return X_sequences, durations_array, events_array, patient_ids_array


def validate_sequences(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    sequence_length: int,
    n_features: int
) -> None:
    """
    Validate sequence data for LSTM training.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        patient_ids: Patient ID array
        sequence_length: Expected sequence length
        n_features: Expected number of features
    """
    print(f"\n=== Validating sequence data ===")
    
    # Check shapes
    expected_shape = (len(durations), sequence_length, n_features)
    if X_sequences.shape != expected_shape:
        raise ValueError(f"X_sequences shape {X_sequences.shape} != expected {expected_shape}")
    
    if len(durations) != len(events) or len(durations) != len(patient_ids):
        raise ValueError("Inconsistent array lengths")
    
    # Check for NaN values
    if np.isnan(X_sequences).any():
        nan_count = np.isnan(X_sequences).sum()
        warnings.warn(f"Found {nan_count} NaN values in X_sequences")
    
    if np.isnan(durations).any():
        raise ValueError("Found NaN values in durations")
    
    if np.isnan(events).any():
        raise ValueError("Found NaN values in events")
    
    # Check value ranges
    if (durations < 0).any():
        raise ValueError("Found negative durations")
    
    if not np.all(np.isin(events, [0, 1])):
        unique_events = np.unique(events)
        warnings.warn(f"Events contain non-binary values: {unique_events}")
    
    # Check for infinite values
    if np.isinf(X_sequences).any():
        inf_count = np.isinf(X_sequences).sum()
        warnings.warn(f"Found {inf_count} infinite values in X_sequences")
    
    print(f"Validation passed for {len(durations)} sequences")
    print(f"Sequence shape: {X_sequences.shape}")
    print(f"Duration range: [{durations.min():.1f}, {durations.max():.1f}]")
    print(f"Event distribution: {np.bincount(events.astype(int))}")


def prepare_lstm_survival_dataset(
    df: pd.DataFrame,
    sequence_length: int,
    feature_cols: List[str] = None,
    cluster_col: str = 'key',
    date_col: str = 'date',
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a dataset for LSTM-based survival analysis.
    
    This is a convenience function that combines sequence creation and validation.
    
    Args:
        df: Input dataframe
        sequence_length: Desired sequence length
        feature_cols: List of feature columns (optional)
        cluster_col: Patient ID column (default: 'key')
        date_col: Date column (default: 'date')
        duration_col: Duration column (default: 'duration')
        event_col: Event column (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        
    Returns:
        Tuple of (X_sequences, durations, events)
    """
    print(f"\n=== Preparing LSTM survival dataset ===")
    
    # Create sequences
    X_sequences, durations, events, patient_ids = create_sequences_from_dataframe(
        df=df,
        sequence_length=sequence_length,
        cluster_col=cluster_col,
        date_col=date_col,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        target_endpoint=target_endpoint
    )
    
    # Validate sequences
    n_features = len(feature_cols) if feature_cols else X_sequences.shape[2]
    validate_sequences(
        X_sequences=X_sequences,
        durations=durations,
        events=events,
        patient_ids=patient_ids,
        sequence_length=sequence_length,
        n_features=n_features
    )
    
    return X_sequences, durations, events


def get_sequence_statistics(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray
) -> Dict[str, Any]:
    """
    Get statistics about sequence data.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_samples': len(durations),
        'sequence_length': X_sequences.shape[1],
        'n_features': X_sequences.shape[2],
        'event_rate': events.mean(),
        'duration_stats': {
            'mean': durations.mean(),
            'std': durations.std(),
            'min': durations.min(),
            'max': durations.max(),
            'median': np.median(durations)
        },
        'feature_stats': {
            'mean': X_sequences.mean(axis=(0, 1)),
            'std': X_sequences.std(axis=(0, 1)),
            'min': X_sequences.min(axis=(0, 1)),
            'max': X_sequences.max(axis=(0, 1))
        }
    }
    
    return stats


def print_sequence_summary(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    name: str = "Dataset"
) -> None:
    """
    Print a summary of sequence data.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        name: Name for the dataset
    """
    stats = get_sequence_statistics(X_sequences, durations, events)
    
    print(f"\n=== {name} Summary ===")
    print(f"Samples: {stats['n_samples']}")
    print(f"Sequence length: {stats['sequence_length']}")
    print(f"Features: {stats['n_features']}")
    print(f"Event rate: {stats['event_rate']:.2%}")
    print(f"Duration - Mean: {stats['duration_stats']['mean']:.1f}, "
          f"Std: {stats['duration_stats']['std']:.1f}, "
          f"Range: [{stats['duration_stats']['min']:.1f}, {stats['duration_stats']['max']:.1f}]")
    print(f"Feature means (first 5): {stats['feature_stats']['mean'][:5]}")
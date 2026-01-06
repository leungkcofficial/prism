"""
Duration Mapper for CKD Risk Prediction

This module contains functions for mapping continuous duration values (in days) to
discrete time intervals (years) for use with survival models like DeepHit.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Optional, Dict, Any
from dotenv import load_dotenv

try:
    from pycox.models import DeepHitSingle
except ImportError:
    print("Warning: pycox is not installed. Please install it using: pip install pycox")


def map_duration_to_intervals(
    durations: Union[np.ndarray, pd.Series], 
    events: Union[np.ndarray, pd.Series],
    cuts: Optional[np.ndarray] = None,
    return_cuts: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Map continuous duration values (in days) to discrete time intervals for DeepHit model.
    
    Args:
        durations: Array-like of duration values in days
        events: Array-like of event indicators (1=event, 0=censored)
        cuts: Optional custom cut points in days. If None, uses yearly intervals up to 5 years
              (365, 730, 1095, 1460, 1825 days)
        return_cuts: If True, also return the cut points used
        
    Returns:
        Tuple containing:
        - discrete_time_indices: Array of discretized time indices
        - events: Array of event indicators
        - cuts: Array of cut points (only if return_cuts=True)
    """
    # Convert inputs to numpy arrays with explicit data types
    if isinstance(durations, pd.Series):
        durations = durations.values
    if isinstance(events, pd.Series):
        events = events.values
    
    # Ensure correct data types
    durations = np.array(durations, dtype=np.float64)
    events = np.array(events, dtype=np.int32)
        
    # Define default cut points if not provided (yearly intervals up to 5 years)
    if cuts is None:
        cuts = np.array([365, 730, 1095, 1460, 1825], dtype=np.float64)
    else:
        cuts = np.array(cuts, dtype=np.float64)
    
    try:
        # Use pycox's label_transform to discretize the durations
        labtrans = DeepHitSingle.label_transform(cuts=cuts)
        discrete_time_indices, events_transformed = labtrans.fit_transform(durations, events)
        
        if return_cuts:
            return discrete_time_indices, events_transformed, cuts
        else:
            return discrete_time_indices, events_transformed
            
    except (NameError, TypeError, ImportError) as e:
        # If pycox is not installed or there's a type error, implement a simple version of the transformation
        print(f"Using fallback implementation due to: {str(e)}")
        
        # Create discrete time indices based on the cut points
        discrete_time_indices = np.digitize(durations, cuts)
        
        # For censored observations that fall beyond the last cut point,
        # we need to adjust the time index
        censored_beyond_last_cut = (events == 0) & (durations > cuts[-1])
        discrete_time_indices[censored_beyond_last_cut] = len(cuts)
        
        if return_cuts:
            return discrete_time_indices, events, cuts
        else:
            return discrete_time_indices, events


def prepare_deephit_data(
    df: pd.DataFrame,
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    custom_cuts: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Prepare data for DeepHit model by mapping durations to discrete time intervals.
    
    Args:
        df: DataFrame containing duration and event columns
        duration_col: Name of the column containing duration values in days
        event_col: Name of the column containing event indicators (1=event, 0=censored)
        custom_cuts: Optional custom cut points in days. If None, uses yearly intervals up to 5 years
        
    Returns:
        Dictionary containing:
        - discrete_durations: Array of discretized time indices
        - events: Array of event indicators
        - cuts: Array of cut points used
        - num_intervals: Number of discrete time intervals
    """
    # Extract durations and events from the DataFrame
    durations = df[duration_col].values
    events = df[event_col].values
    
    # Convert events to binary (0 or 1)
    # In some datasets, events might be coded differently (e.g., 1, 2 for different event types)
    # Here we assume any non-zero value is an event
    binary_events = (events > 0).astype(int)
    
    # Define cut points (yearly intervals up to 5 years by default)
    cuts = np.array(custom_cuts) if custom_cuts is not None else np.array([365, 730, 1095, 1460, 1825])
    
    # Map durations to discrete time intervals
    discrete_durations, mapped_events, used_cuts = map_duration_to_intervals(
        durations, binary_events, cuts=cuts, return_cuts=True
    )
    
    # Return the prepared data
    return {
        'discrete_durations': discrete_durations,
        'events': mapped_events,
        'cuts': used_cuts,
        'num_intervals': len(used_cuts) + 1  # Number of intervals is one more than number of cut points
    }


def get_interval_labels(cuts: np.ndarray) -> List[str]:
    """
    Generate human-readable labels for time intervals based on cut points.
    
    Args:
        cuts: Array of cut points in days
        
    Returns:
        List of interval labels (e.g., "0-1 year", "1-2 years", etc.)
    """
    # Convert cut points from days to years (approximate)
    years_cuts = cuts / 365.25
    
    # Generate interval labels
    labels = []
    
    # First interval: 0 to first cut point
    labels.append(f"0-{years_cuts[0]:.1f} years")
    
    # Middle intervals
    for i in range(len(years_cuts) - 1):
        labels.append(f"{years_cuts[i]:.1f}-{years_cuts[i+1]:.1f} years")
    
    # Last interval: last cut point to infinity
    labels.append(f"{years_cuts[-1]:.1f}+ years")
    
    return labels


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Load environment variables
    load_dotenv()
    
    # Get random seed from environment variables, default to 42 if not found
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    np.random.seed(RANDOM_SEED)
    print(f"Using RANDOM_SEED={RANDOM_SEED}")
    
    # Sample data: durations in days and event indicators
    durations = np.array([100, 400, 800, 1200, 1600, 2000])
    events = np.array([0, 1, 0, 1, 0, 1])
    
    # Map durations to discrete time intervals
    discrete_durations, mapped_events, cuts = map_duration_to_intervals(
        durations, events, return_cuts=True
    )
    
    # Print results
    print("Original durations (days):", durations)
    print("Original events:", events)
    print("Cut points (days):", cuts)
    print("Discrete time indices:", discrete_durations)
    print("Mapped events:", mapped_events)
    
    # Get interval labels
    interval_labels = get_interval_labels(cuts)
    print("Interval labels:", interval_labels)
    
    # For each sample, print its interval
    for i, (duration, discrete_idx) in enumerate(zip(durations, discrete_durations)):
        interval = interval_labels[discrete_idx]
        event_status = "event" if events[i] == 1 else "censored"
        print(f"Sample {i+1}: {duration} days -> {interval} ({event_status})")

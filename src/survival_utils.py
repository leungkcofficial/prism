"""
Utility Functions for Survival Analysis

This module provides utility functions for survival analysis, including data preprocessing,
metrics calculation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from pycox.evaluation import EvalSurv
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc


def prepare_survival_data(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for survival analysis.
    
    Args:
        df: DataFrame containing the data
        duration_col: Name of the duration column
        event_col: Name of the event column
        feature_cols: List of feature column names
        categorical_cols: List of categorical column names (default: None)
        
    Returns:
        Tuple containing:
        - X: Feature matrix
        - durations: Array of durations
        - events: Array of event indicators
    """
    # Extract features
    X = df[feature_cols].copy()
    
    # Handle categorical features
    if categorical_cols is not None and len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Extract durations and events
    durations = df[duration_col].values
    events = df[event_col].values
    
    return X.values, durations, events


def create_pycox_dataset(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Create a PyCox dataset from a pandas DataFrame.
    
    Args:
        df: DataFrame containing the data
        duration_col: Name of the duration column
        event_col: Name of the event column
        feature_cols: List of feature column names
        categorical_cols: List of categorical column names (default: None)
        scaler: StandardScaler to use for feature scaling (default: None)
        fit_scaler: Whether to fit the scaler on this data (default: False)
        
    Returns:
        Tuple containing:
        - dataset: PyCox dataset
        - scaler: Fitted StandardScaler (if fit_scaler=True)
    """
    # Prepare data
    X, durations, events = prepare_survival_data(
        df, duration_col, event_col, feature_cols, categorical_cols
    )
    
    # Scale features
    if scaler is None and fit_scaler:
        scaler = StandardScaler().fit(X)
    
    if scaler is not None:
        X = scaler.transform(X)
    
    # Convert to float32 to avoid dtype mismatch with model weights
    X = X.astype(np.float32)
    durations = durations.astype(np.float32)
    
    # Create PyCox dataset (manually since from_pandas is not available)
    # In PyCox, a dataset is typically a tuple of (x, durations, events)
    dataset = (X, durations, events)
    
    if fit_scaler:
        return dataset, scaler
    else:
        return dataset, None


def calculate_c_index(
    model: Any,
    dataset: Any,
    model_type: str
) -> float:
    """
    Calculate concordance index for a survival model.
    
    Args:
        model: Trained survival model (PyCox model)
        dataset: PyCox dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        
    Returns:
        Concordance index
    """
    # Get survival function predictions
    surv = model.predict_surv_df(dataset[0])
    
    # Create EvalSurv object
    ev = EvalSurv(
        surv,
        dataset[1],
        dataset[2],
        censor_surv='km'
    )
    
    # Calculate concordance index
    return ev.concordance_td()


def calculate_integrated_brier_score(
    model: Any,
    dataset: Any,
    model_type: str,
    time_points: List[int]
) -> float:
    """
    Calculate integrated Brier score for a survival model.
    
    Args:
        model: Trained survival model (PyCox model)
        dataset: PyCox dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        time_points: List of time points for evaluation
        
    Returns:
        Integrated Brier score
    """
    # Get survival function predictions
    surv = model.predict_surv_df(dataset[0])
    
    # Create EvalSurv object
    ev = EvalSurv(
        surv,
        dataset[1],
        dataset[2],
        censor_surv='km'
    )
    
    # Calculate integrated Brier score
    return ev.integrated_brier_score(time_points)


def calculate_time_dependent_auc(
    model: Any,
    dataset: Any,
    model_type: str,
    time_points: List[int]
) -> Dict[int, float]:
    """
    Calculate time-dependent AUC for a survival model.
    
    Args:
        model: Trained survival model (PyCox model)
        dataset: PyCox dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        time_points: List of time points for evaluation
        
    Returns:
        Dictionary mapping time points to AUC values
    """
    # Get survival function predictions
    surv = model.predict_surv_df(dataset[0])
    
    # Create EvalSurv object
    ev = EvalSurv(
        surv,
        dataset[1],
        dataset[2],
        censor_surv='km'
    )
    
    # Calculate time-dependent AUC for each time point
    auc_dict = {}
    for t in time_points:
        auc_dict[t] = ev.concordance_td(t)
    
    return auc_dict


def plot_survival_curves(
    model: Any,
    dataset: Any,
    model_type: str,
    num_samples: int = 10,
    time_points: Optional[List[int]] = None,
    title: str = "Survival Curves"
) -> plt.Figure:
    """
    Plot survival curves for a subset of samples.
    
    Args:
        model: Trained survival model (PyCox model)
        dataset: PyCox dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        num_samples: Number of samples to plot (default: 10)
        time_points: List of time points for evaluation (default: None)
        title: Plot title (default: "Survival Curves")
        
    Returns:
        Matplotlib figure
    """
    # Get survival function predictions
    X = dataset[0]
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X_subset = X[indices]
    else:
        X_subset = X
        
    surv = model.predict_surv_df(X_subset)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot survival curves
    for i, s in enumerate(surv.transpose().values):
        ax.step(surv.index, s, where="post", label=f"Sample {i+1}")
    
    # Add time points if provided
    if time_points is not None:
        for t in time_points:
            if t <= surv.index.max():
                ax.axvline(t, color='red', linestyle='--', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend for a subset of samples if there are many
    if num_samples > 10:
        ax.legend(loc='lower left', ncol=2)
    else:
        ax.legend(loc='best')
    
    return fig


def plot_calibration_curve(
    model: Any,
    dataset: Any,
    model_type: str,
    time_points: List[int],
    num_quantiles: int = 10,
    title: str = "Calibration Curve"
) -> plt.Figure:
    """
    Plot calibration curve for a survival model.
    
    Args:
        model: Trained survival model (PyCox model)
        dataset: PyCox dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        time_points: List of time points for evaluation
        num_quantiles: Number of quantiles for binning (default: 10)
        title: Plot title (default: "Calibration Curve")
        
    Returns:
        Matplotlib figure
    """
    # Get survival function predictions
    surv = model.predict_surv_df(dataset[0])
    
    # Create figure
    fig, axes = plt.subplots(1, len(time_points), figsize=(5*len(time_points), 5))
    if len(time_points) == 1:
        axes = [axes]
    
    # For each time point
    for i, t in enumerate(time_points):
        if t > surv.index.max():
            continue
            
        # Get survival probabilities at time t
        t_idx = surv.index.get_indexer([t], method='nearest')[0]
        surv_t = surv.iloc[t_idx].values
        
        # Create quantiles
        quantiles = pd.qcut(surv_t, num_quantiles, duplicates='drop')
        
        # Calculate observed survival for each quantile
        kmf = KaplanMeierFitter()
        observed_surv = []
        predicted_surv = []
        
        for q in range(len(quantiles.categories)):
            mask = (quantiles.codes == q)
            if sum(mask) > 0:
                kmf.fit(dataset[1][mask], dataset[2][mask])
                if t in kmf.survival_function_.index:
                    obs_surv = kmf.survival_function_.loc[t].values[0]
                else:
                    # Find closest time point
                    closest_idx = kmf.survival_function_.index.get_indexer([t], method='nearest')[0]
                    closest_time = kmf.survival_function_.index[closest_idx]
                    obs_surv = kmf.survival_function_.loc[closest_time].values[0]
                
                pred_surv = surv_t[mask].mean()
                
                observed_surv.append(obs_surv)
                predicted_surv.append(pred_surv)
        
        # Plot calibration curve
        ax = axes[i]
        ax.scatter(predicted_surv, observed_surv, s=50, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Predicted Survival Probability")
        ax.set_ylabel("Observed Survival Probability")
        ax.set_title(f"Calibration at t={t}")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(title, y=1.05, fontsize=16)
    
    return fig


def create_evaluation_report(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    model_type: str,
    time_points: List[int],
    output_path: str
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report for a survival model.
    
    Args:
        model: Trained survival model (PyCox model)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model_type: Type of model ('deepsurv' or 'deephit')
        time_points: List of time points for evaluation
        output_path: Path to save the report
        
    Returns:
        Dictionary containing evaluation metrics
    """
    import os
    from datetime import datetime
    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize metrics dictionary
    metrics = {
        'model_type': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'time_points': time_points,
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Evaluate on each dataset
    for name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        if dataset is None:
            continue
            
        # Calculate metrics
        c_index = calculate_c_index(model, dataset, model_type)
        ibs = calculate_integrated_brier_score(model, dataset, model_type, time_points)
        time_auc = calculate_time_dependent_auc(model, dataset, model_type, time_points)
        
        # Store metrics
        metrics[name]['c_index'] = float(c_index)
        metrics[name]['integrated_brier_score'] = float(ibs)
        metrics[name]['time_auc'] = {str(t): float(auc) for t, auc in time_auc.items()}
        
        # Create plots
        surv_fig = plot_survival_curves(
            model, dataset, model_type, num_samples=10, time_points=time_points,
            title=f"Survival Curves - {name.capitalize()} Set"
        )
        surv_fig.savefig(os.path.join(output_path, f"{name}_survival_curves.png"), dpi=300, bbox_inches='tight')
        plt.close(surv_fig)
        
        cal_fig = plot_calibration_curve(
            model, dataset, model_type, time_points,
            title=f"Calibration Curves - {name.capitalize()} Set"
        )
        cal_fig.savefig(os.path.join(output_path, f"{name}_calibration_curves.png"), dpi=300, bbox_inches='tight')
        plt.close(cal_fig)
    
    # Save metrics to JSON
    with open(os.path.join(output_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create HTML report
    html_report = create_html_report(metrics, output_path)
    with open(os.path.join(output_path, "evaluation_report.html"), 'w') as f:
        f.write(html_report)
    
    return metrics


def create_html_report(metrics: Dict[str, Any], output_path: str) -> str:
    """
    Create an HTML report from evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path where plot images are saved
        
    Returns:
        HTML report as a string
    """
    import os
    
    # Create HTML header
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Survival Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-section {{ margin-bottom: 30px; }}
            .plot-section {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
            .plot-container {{ max-width: 100%; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Survival Model Evaluation Report</h1>
        <p><strong>Model Type:</strong> {metrics['model_type']}</p>
        <p><strong>Timestamp:</strong> {metrics['timestamp']}</p>
        <p><strong>Time Points:</strong> {', '.join(map(str, metrics['time_points']))}</p>
    """
    
    # Add metrics tables for each dataset
    for dataset in ['train', 'val', 'test']:
        if dataset not in metrics or not metrics[dataset]:
            continue
            
        html += f"""
        <div class="metric-section">
            <h2>{dataset.capitalize()} Set Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Concordance Index</td>
                    <td>{metrics[dataset]['c_index']:.4f}</td>
                </tr>
                <tr>
                    <td>Integrated Brier Score</td>
                    <td>{metrics[dataset]['integrated_brier_score']:.4f}</td>
                </tr>
        """
        
        # Add time-dependent AUC
        for t, auc in metrics[dataset]['time_auc'].items():
            html += f"""
                <tr>
                    <td>AUC at t={t}</td>
                    <td>{auc:.4f}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        """
        
        # Add plots
        html += f"""
        <div class="plot-section">
            <div class="plot-container">
                <h3>Survival Curves</h3>
                <img src="{dataset}_survival_curves.png" alt="Survival Curves">
            </div>
            <div class="plot-container">
                <h3>Calibration Curves</h3>
                <img src="{dataset}_calibration_curves.png" alt="Calibration Curves">
            </div>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html


def compare_models(
    kfre_results: Dict[str, Any],
    deepsurv_metrics: Optional[Dict[str, Any]] = None,
    deephit_metrics: Optional[Dict[str, Any]] = None,
    time_points: List[int] = [365, 730, 1095, 1460, 1825],
    output_path: str = "comparison"
) -> str:
    """
    Create a comparison report for KFRE, DeepSurv, and DeepHit models.
    
    Args:
        kfre_results: Results from KFRE evaluation
        deepsurv_metrics: Metrics from DeepSurv evaluation (default: None)
        deephit_metrics: Metrics from DeepHit evaluation (default: None)
        time_points: Time points for comparison (default: [365, 730, 1095, 1460, 1825])
        output_path: Path to save the comparison report (default: "comparison")
        
    Returns:
        Path to the HTML comparison report
    """
    import os
    from datetime import datetime
    import json
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize comparison dictionary
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'time_points': time_points,
        'models': []
    }
    
    # Add KFRE results
    if kfre_results:
        kfre_model = {
            'name': 'KFRE',
            'metrics': {}
        }
        
        # Extract metrics from KFRE results
        # This will depend on the structure of kfre_results
        # For now, we'll assume it has a similar structure to our metrics
        
        comparison['models'].append(kfre_model)
    
    # Add DeepSurv metrics
    if deepsurv_metrics:
        deepsurv_model = {
            'name': 'DeepSurv',
            'metrics': {
                'train': deepsurv_metrics['train'],
                'val': deepsurv_metrics['val'],
                'test': deepsurv_metrics['test']
            }
        }
        
        comparison['models'].append(deepsurv_model)
    
    # Add DeepHit metrics
    if deephit_metrics:
        deephit_model = {
            'name': 'DeepHit',
            'metrics': {
                'train': deephit_metrics['train'],
                'val': deephit_metrics['val'],
                'test': deephit_metrics['test']
            }
        }
        
        comparison['models'].append(deephit_model)
    
    # Save comparison to JSON
    with open(os.path.join(output_path, "comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(comparison, output_path)
    
    # Create HTML report
    html_report_path = os.path.join(output_path, "comparison_report.html")
    html_report = create_comparison_html_report(comparison, output_path)
    with open(html_report_path, 'w') as f:
        f.write(html_report)
    
    return html_report_path


def create_comparison_plots(comparison: Dict[str, Any], output_path: str) -> None:
    """
    Create comparison plots for multiple models.
    
    Args:
        comparison: Dictionary containing comparison data
        output_path: Path to save the plots
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create C-index comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract model names and C-indices
    model_names = []
    train_c_indices = []
    val_c_indices = []
    test_c_indices = []
    
    for model in comparison['models']:
        model_names.append(model['name'])
        
        if 'metrics' in model and 'train' in model['metrics'] and 'c_index' in model['metrics']['train']:
            train_c_indices.append(model['metrics']['train']['c_index'])
        else:
            train_c_indices.append(np.nan)
            
        if 'metrics' in model and 'val' in model['metrics'] and 'c_index' in model['metrics']['val']:
            val_c_indices.append(model['metrics']['val']['c_index'])
        else:
            val_c_indices.append(np.nan)
            
        if 'metrics' in model and 'test' in model['metrics'] and 'c_index' in model['metrics']['test']:
            test_c_indices.append(model['metrics']['test']['c_index'])
        else:
            test_c_indices.append(np.nan)
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.25
    
    # Create bars
    ax.bar(x - width, train_c_indices, width, label='Train', alpha=0.7)
    ax.bar(x, val_c_indices, width, label='Validation', alpha=0.7)
    ax.bar(x + width, test_c_indices, width, label='Test', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('C-index')
    ax.set_title('C-index Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "c_index_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create IBS comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract model names and IBS values
    train_ibs = []
    val_ibs = []
    test_ibs = []
    
    for model in comparison['models']:
        if 'metrics' in model and 'train' in model['metrics'] and 'integrated_brier_score' in model['metrics']['train']:
            train_ibs.append(model['metrics']['train']['integrated_brier_score'])
        else:
            train_ibs.append(np.nan)
            
        if 'metrics' in model and 'val' in model['metrics'] and 'integrated_brier_score' in model['metrics']['val']:
            val_ibs.append(model['metrics']['val']['integrated_brier_score'])
        else:
            val_ibs.append(np.nan)
            
        if 'metrics' in model and 'test' in model['metrics'] and 'integrated_brier_score' in model['metrics']['test']:
            test_ibs.append(model['metrics']['test']['integrated_brier_score'])
        else:
            test_ibs.append(np.nan)
    
    # Create bars
    ax.bar(x - width, train_ibs, width, label='Train', alpha=0.7)
    ax.bar(x, val_ibs, width, label='Validation', alpha=0.7)
    ax.bar(x + width, test_ibs, width, label='Test', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Integrated Brier Score')
    ax.set_title('Integrated Brier Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ibs_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create time-dependent AUC comparison plot for test set
    time_points = comparison['time_points']
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract time-dependent AUC values for each model
    for model in comparison['models']:
        if 'metrics' in model and 'test' in model['metrics'] and 'time_auc' in model['metrics']['test']:
            time_auc = model['metrics']['test']['time_auc']
            auc_values = [time_auc.get(str(t), np.nan) for t in time_points]
            ax.plot(time_points, auc_values, marker='o', label=model['name'])
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('AUC')
    ax.set_title('Time-dependent AUC Comparison (Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "time_auc_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_comparison_html_report(comparison: Dict[str, Any], output_path: str) -> str:
    """
    Create an HTML report comparing multiple models.
    
    Args:
        comparison: Dictionary containing comparison data
        output_path: Path where plot images are saved
        
    Returns:
        HTML report as a string
    """
    import os
    
    # Create HTML header
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Survival Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-section {{ margin-bottom: 30px; }}
            .plot-section {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
            .plot-container {{ max-width: 100%; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Survival Model Comparison Report</h1>
        <p><strong>Timestamp:</strong> {comparison['timestamp']}</p>
        <p><strong>Time Points:</strong> {', '.join(map(str, comparison['time_points']))}</p>
        <p><strong>Models:</strong> {', '.join(model['name'] for model in comparison['models'])}</p>
    """
    
    # Add comparison plots
    html += """
    <div class="plot-section">
        <div class="plot-container">
            <h2>C-index Comparison</h2>
            <img src="c_index_comparison.png" alt="C-index Comparison">
        </div>
        <div class="plot-container">
            <h2>Integrated Brier Score Comparison</h2>
            <img src="ibs_comparison.png" alt="Integrated Brier Score Comparison">
        </div>
        <div class="plot-container">
            <h2>Time-dependent AUC Comparison</h2>
            <img src="time_auc_comparison.png" alt="Time-dependent AUC Comparison">
        </div>
    </div>
    """
    
    # Add detailed metrics table
    html += """
    <div class="metric-section">
        <h2>Detailed Metrics</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>C-index</th>
                <th>Integrated Brier Score</th>
    """
    
    # Add time-dependent AUC columns
    for t in comparison['time_points']:
        html += f"""
                <th>AUC at t={t}</th>
        """
    
    html += """
            </tr>
    """
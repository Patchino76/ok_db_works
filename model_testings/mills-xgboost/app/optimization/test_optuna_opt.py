import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import logging

# Add project root to sys.path for relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the model class
from app.models.xgboost_model import MillsXGBoostModel

# Configure logging
def setup_logger(log_file=None):
    """Set up logger with file and console handlers"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger (will be configured with file in main)
logger = setup_logger()

class BlackBoxFunction:
    """
    A black box function that loads an XGBoost model and predicts output based on input features.
    This function will be optimized using Optuna.
    """
    
    def __init__(self, model_id: str, maximize: bool = True):
        """
        Initialize the black box function with a specific model.
        
        Args:
            model_id: The ID of the model to load (e.g., "xgboost_PSI80_model")
            maximize: Whether to maximize (True) or minimize (False) the objective function
        """
        self.model_id = model_id
        self.maximize = maximize
        self.model = None
        self.scaler = None
        self.metadata = None
        self.features = None
        self.target_col = None
        self.parameter_bounds = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the XGBoost model, scaler, and metadata from the models folder"""
        try:
            # Determine file paths based on model_id
            models_dir = os.path.join(project_root, 'models')
            self.target_col = "PSI80"  # Default
            
            
            # Get the base name without extension for metadata and scaler
            model_base = self.model_id.split(".")[0]
            model_path = os.path.join(models_dir, f"{model_base}_model.json")
            metadata_path = os.path.join(models_dir, f"{model_base}_metadata.json")
            scaler_path = os.path.join(models_dir, f"{model_base}_scaler.pkl")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}. Using default features.")
                self.features = [
                    'Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 
                    'DensityHC', 'MotorAmp', 'Shisti', 'Daiki'
                ]
            else:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.features = self.metadata.get('features', [
                    'Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 
                    'DensityHC', 'MotorAmp', 'Shisti', 'Daiki'
                ])
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            # Create and load model using MillsXGBoostModel
            self.xgb_model = MillsXGBoostModel()
            self.xgb_model.load_model(model_path, scaler_path, metadata_path if os.path.exists(metadata_path) else None)
            
            logger.info(f"Successfully loaded model {self.model_id}")
            logger.info(f"Features: {self.features}")
            logger.info(f"Target column: {self.target_col}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def set_parameter_bounds(self, parameter_bounds: Dict[str, List[float]]):
        """
        Set bounds for the parameters to optimize.
        
        Args:
            parameter_bounds: Dictionary mapping feature names to [min, max] bounds
        """
        self.parameter_bounds = parameter_bounds
        
        # Validate that bounds are provided for features in the model
        for feature in parameter_bounds:
            if feature not in self.features:
                logger.warning(f"Parameter bound provided for feature '{feature}' which is not in the model features.")
        
        missing_bounds = [f for f in self.features if f not in parameter_bounds]
        if missing_bounds:
            logger.warning(f"No bounds provided for features: {missing_bounds}")
    
    def __call__(self, **features) -> float:
        """
        Predict the target value based on the provided features.
        
        Args:
            features: Feature values as keyword arguments
            
        Returns:
            float: The predicted value
        """
        if not self.xgb_model:
            raise ValueError("Model not loaded")
        
        try:
            # Create a dictionary with all features
            input_data = {feature: features.get(feature, 0.0) for feature in self.features}
            
            # Make prediction
            prediction = self.xgb_model.predict(input_data)[0]
            
            # Return prediction (negated if minimizing)
            return prediction if self.maximize else -prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            # Return a very bad value in case of error
            return -1e9 if self.maximize else 1e9


def optimize_with_optuna(
    black_box_func: BlackBoxFunction, 
    n_trials: int = 100,
    timeout: int = None
) -> Tuple[Dict[str, float], float, optuna.study.Study]:
    """
    Optimize the black box function using Optuna.
    
    Args:
        black_box_func: The black box function to optimize
        n_trials: Number of optimization trials
        timeout: Timeout in seconds (optional)
        
    Returns:
        Tuple of (best_params, best_value, study)
    """
    if not black_box_func.parameter_bounds:
        raise ValueError("Parameter bounds must be set before optimization")
    
    # Define the objective function for Optuna
    def objective(trial):
        # Suggest values for each parameter within bounds
        params = {}
        for feature, bounds in black_box_func.parameter_bounds.items():
            params[feature] = trial.suggest_float(feature, bounds[0], bounds[1])
        
        # Call the black box function
        return black_box_func(**params)
    
    # Create and run the study
    direction = "maximize" if black_box_func.maximize else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters and value
    best_params = study.best_params
    best_value = study.best_value
    
    return best_params, best_value, study


def export_study_to_csv(study: optuna.study.Study, save_path: str) -> pd.DataFrame:
    """
    Export the Optuna study trials to a DataFrame and save as CSV.
    
    Args:
        study: The completed Optuna study
        save_path: Path to save the CSV file
        
    Returns:
        DataFrame containing the study trials data
    """
    # Create a list to hold all trial data
    trials_data = []
    
    # Extract trial information
    for i, trial in enumerate(study.trials):
        # Get basic trial info
        trial_dict = {
            "trial_number": i,
            "value": trial.value,  # The objective value
            "state": trial.state,  # COMPLETE, PRUNED, etc.
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
            "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
        }
        
        # Add parameters used in this trial
        for param_name, param_value in trial.params.items():
            trial_dict[f"param_{param_name}"] = param_value
            
        # Add intermediate values if any (for trials that use pruning)
        for step, intermediate_value in trial.intermediate_values.items():
            trial_dict[f"intermediate_{step}"] = intermediate_value
            
        trials_data.append(trial_dict)
    
    # Convert to DataFrame
    trials_df = pd.DataFrame(trials_data)
    
    # Save to CSV
    trials_df.to_csv(save_path, index=False)
    
    return trials_df


def plot_optimization_results(study: optuna.study.Study, black_box_func: BlackBoxFunction):
    """
    Plot the optimization results.
    
    Args:
        study: The completed Optuna study
        black_box_func: The black box function that was optimized
    """
    # Get path to results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "optimization_results")
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(f"Optimization History for {black_box_func.model_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "optimization_history.png"))
    
    # Plot parameter importances if there are enough trials
    if len(study.trials) > 10:
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f"Parameter Importances for {black_box_func.model_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "parameter_importances.png"))
    
    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title(f"Parallel Coordinate Plot for {black_box_func.model_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "parallel_coordinate.png"))


def main(log_file=None):
    """Main function to demonstrate usage"""
    # Configure logger with file if specified
    global logger
    if log_file:
        logger = setup_logger(log_file)
        logger.info(f"Logging to file: {log_file}")
    
    # Example model ID and parameter bounds
    model_id = "xgboost_PSI80_mill8"
    parameter_bounds = {
        "Ore": [160.0, 200.0],
        "WaterMill": [5.0, 20.0],
        "WaterZumpf": [160.0, 250.0],
        "PressureHC": [0.3, 0.5],
        "DensityHC": [1600, 1900],
        "MotorAmp": [180, 220],
        "Shisti": [0.05, 0.3],
        "Daiki": [0.1, 0.4]
    }
    
    # Create the black box function
    logger.info(f"Creating black box function with model {model_id}")
    black_box = BlackBoxFunction(model_id=model_id, maximize=True)
    
    # Set parameter bounds
    black_box.set_parameter_bounds(parameter_bounds)
    
    # Run optimization
    logger.info("Starting optimization...")
    best_params, best_value, study = optimize_with_optuna(
        black_box_func=black_box,
        n_trials=150  # Reduced for testing, increase for better results
    )
    
    # Log results
    logger.info(f"Best value: {best_value}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create results directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "optimization_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot results
    logger.info("Generating plots...")
    plot_optimization_results(study, black_box)
    
    # Return results as dictionary
    result = {
        "model_id": model_id,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials)
    }
    
    # Get path to results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "optimization_results")
    results_file = os.path.join(results_dir, "optimization_results.json")
    
    # Save results to file
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Export study trials to CSV
    csv_file = os.path.join(results_dir, "optuna_trials.csv")
    trials_df = export_study_to_csv(study, csv_file)
    logger.info(f"Exported {len(trials_df)} trials to {csv_file}")
    
    # Log summary statistics of trials
    logger.info(f"Trial statistics:")
    logger.info(f"  Mean value: {trials_df['value'].mean():.4f}")
    logger.info(f"  Std deviation: {trials_df['value'].std():.4f}")
    logger.info(f"  Min value: {trials_df['value'].min():.4f}")
    logger.info(f"  Max value: {trials_df['value'].max():.4f}")
    
    return result


if __name__ == "__main__":
    import os
    
    # Define directories for logs and results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "logs")
    results_dir = os.path.join(current_dir, "optimization_results")
    
    # Create directories if they don't exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Use constant log file name based on model ID
    model_id = "xgboost_PSI80_mill8"
    log_file = os.path.join(logs_dir, f"optuna_optimization_{model_id}.log")
    
    main(log_file=log_file)
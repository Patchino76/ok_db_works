from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os
import json
import uuid
import logging
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from ..database.db_connector import MillsDataConnector
from ..models.xgboost_model import MillsXGBoostModel
from ..models.data_processor import DataProcessor
# Optimization imports will be added later
from .schemas import (
    TrainingRequest, TrainingResponse, 
    PredictionRequest, PredictionResponse,
    OptimizationRequest, OptimizationResponse,
    ParameterRecommendation
)

# Configure logging
logger = logging.getLogger(__name__)


class BlackBoxFunction:
    """
    A black box function that loads an XGBoost model and predicts output based on input features.
    This function will be optimized using Optuna.
    """
    
    def __init__(self, model_id: str, xgb_model=None, maximize: bool = True):
        """
        Initialize the black box function with a specific model.
        
        Args:
            model_id: The ID of the model to load (e.g., "xgboost_PSI80_model")
            xgb_model: Optional, pre-loaded XGBoost model instance
            maximize: Whether to maximize (True) or minimize (False) the objective function
        """
        self.model_id = model_id
        self.maximize = maximize
        self.xgb_model = xgb_model
        self.scaler = None
        self.metadata = None
        self.features = None
        self.target_col = None
        self.parameter_bounds = None
        
        # If model is not provided, load it
        if self.xgb_model is None:
            self._load_model()
        else:
            # Get features from the provided model
            self.features = xgb_model.features
            self.target_col = xgb_model.target_col
    
    def _load_model(self):
        """Load the XGBoost model, scaler, and metadata from the models folder"""
        try:
            # Determine file paths based on model_id
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            models_dir = os.path.join(project_root, 'models')
            
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
                self.target_col = self.metadata.get('target_col', 'PSI80')
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            # Create and load model using MillsXGBoostModel
            from ..models.xgboost_model import MillsXGBoostModel
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


# Create router
router = APIRouter()

# Storage for models and related objects
# In a production environment, this should be replaced with a proper database or storage system
models_store = {}

@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train a new XGBoost model with the specified parameters"""
    try:
        # Generate a unique ID for this model
        model_id = str(uuid.uuid4())
        
        # Create database connector
        db_connector = MillsDataConnector(
            host=request.db_config.host,
            port=request.db_config.port,
            dbname=request.db_config.dbname,
            user=request.db_config.user,
            password=request.db_config.password
        )
        
        # Get combined data
        logger.info(f"Fetching data for mill {request.mill_number} from {request.start_date} to {request.end_date}")
        df = db_connector.get_combined_data(
            mill_number=request.mill_number,
            start_date=request.start_date,
            end_date=request.end_date,
            resample_freq='1min'  # For 1-minute intervals as mentioned in the memory
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data found for the specified parameters")
        
        # Set features if not provided
        features = request.features or [
            'Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 
            'DensityHC', 'MotorAmp', 'Shisti', 'Daiki'
        ]
        
        # Process data
        data_processor = DataProcessor()
        X_scaled, y, scaler = data_processor.preprocess(df, features, request.target_col)
        
        # Split data - use time-ordered split for time series (no shuffling)
        # Calculate the split point for time series data
        split_idx = int(len(X_scaled) * (1 - request.test_size))
        
        # Time-ordered split (training data is earlier, test data is later)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Time-ordered train-test split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        logger.info(f"Training data time range: earliest {split_idx} records")
        logger.info(f"Test data time range: latest {len(X_scaled) - split_idx} records")
        
        # Create and train model
        xgb_model = MillsXGBoostModel(features=features, target_col=request.target_col)
        
        # Use provided parameters if available
        params = None
        if request.params:
            params = request.params.dict()
        
        # Train the model
        training_results = xgb_model.train(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            params=params
        )
        
        # Save the model with mill number in filename
        os.makedirs("models", exist_ok=True)
        save_results = xgb_model.save_model(directory="models", mill_number=request.mill_number)
        
        # Store model in memory
        models_store[model_id] = {
            "model": xgb_model,
            "file_paths": save_results,
            "created_at": datetime.now().isoformat(),
            "features": features,
            "target_col": request.target_col
        }
        
        # Prepare response
        response = {
            "model_id": model_id,
            **training_results
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict", response_model=PredictionResponse) 
async def predict(request: PredictionRequest):
    """Make predictions using a trained model (no caching - always loads fresh)"""
    try:
        # Always load model fresh from disk (no caching)
        logger.info(f"Loading model {request.model_id} fresh from disk")
        
        # Import the model class
        from ..models.xgboost_model import MillsXGBoostModel
        import os
        import glob
        
        # Create an instance and load the model
        model = MillsXGBoostModel()
        
        # Use the full model name as provided
        model_id = request.model_id
        
        # Use absolute path to mills-xgboost/models directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        base_dir = os.path.join(project_root, 'models')
        
        # Check if the directory exists
        if not os.path.exists(base_dir):
            logger.error(f"Models directory does not exist: {base_dir}")
            raise FileNotFoundError(f"Models directory not found: {base_dir}")
        
        # Log the directory we're using
        logger.info(f"Using models directory: {base_dir}")
        
        # Clean the model_id by removing any extension or existing suffix
        base_model_id = model_id.replace('.json', '').replace('_model', '').replace('_metadata', '').replace('_scaler', '')
        logger.info(f"Using base model ID: {base_model_id}")
        
        # Construct paths with standardized suffixes
        model_path = os.path.join(base_dir, f"{base_model_id}_model.json")
        scaler_path = os.path.join(base_dir, f"{base_model_id}_scaler.pkl")
        metadata_path = os.path.join(base_dir, f"{base_model_id}_metadata.json")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            
            # Try to find matching model files in case there's a different naming pattern
            all_model_files = glob.glob(os.path.join(base_dir, "*.json"))
            matching_files = [f for f in all_model_files if base_model_id in os.path.basename(f)]
            
            if matching_files:
                model_path = matching_files[0]
                logger.info(f"Found alternate model file: {model_path}")
                # Update other paths based on the found file
                file_base = os.path.basename(model_path).replace('.json', '')
                scaler_path = os.path.join(base_dir, f"{file_base.replace('_model', '')}_scaler.pkl")
                metadata_path = os.path.join(base_dir, f"{file_base.replace('_model', '')}_metadata.json")
            else:
                raise FileNotFoundError(f"No model file found for {base_model_id}")
        
        # Log the files we're using
        logger.info(f"Using model files:")
        logger.info(f"- Model:    {os.path.basename(model_path)}")
        logger.info(f"- Scaler:   {os.path.basename(scaler_path)}")
        logger.info(f"- Metadata: {os.path.basename(metadata_path)}")
        
        # Load model with paths
        model.load_model(model_path, scaler_path, metadata_path)
        
        # Debug logging to verify features
        logger.info(f"Model loaded with features: {model.features}")
        
        # Make prediction
        prediction = model.predict(request.data)[0]
        
        return {
            "prediction": float(prediction),
            "model_id": request.model_id,
            "target_col": model.target_col,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_parameters(request: OptimizationRequest):
    """Optimize XGBoost hyperparameters using Bayesian Optimization"""
    try:
        # Log optimization request
        logger.info(f"Optimization request received for model {request.model_id}")
        
        # Check if model exists in memory
        if request.model_id not in models_store:
            # Try to load from disk
            try:
                logger.info(f"Model {request.model_id} not found in memory, attempting to load from disk")
                # Create an instance and load the model
                black_box = BlackBoxFunction(model_id=request.model_id, maximize=request.maximize)
                
                # Store in memory for future use
                models_store[request.model_id] = {
                    "model": black_box.xgb_model,
                    "target_col": black_box.target_col,
                    "features": black_box.features
                }
                logger.info(f"Successfully loaded model {request.model_id} from disk")
            except Exception as e:
                logger.error(f"Failed to load model from disk: {str(e)}")
                raise HTTPException(status_code=404, detail="Model not found")
        
        # Get model
        model_info = models_store[request.model_id]
        model = model_info["model"]
        target_col = model_info.get("target_col", "PSI80")
        
        # Create black box function with the loaded model
        black_box = BlackBoxFunction(
            model_id=request.model_id,
            xgb_model=model,
            maximize=request.maximize
        )
        
        # Set parameter bounds
        if not request.parameter_bounds:
            # Use default bounds if not provided
            logger.warning(f"No parameter bounds provided, using default bounds")
            parameter_bounds = {
                "Ore": [150.0, 200.0],
                "WaterMill": [10.0, 20.0],
                "WaterZumpf": [180.0, 250.0],
                "PressureHC": [70.0, 90.0],
                "DensityHC": [1.5, 1.9],
                "MotorAmp": [30.0, 50.0],
                "Shisti": [0.05, 0.2],
                "Daiki": [0.2, 0.5]
            }
        else:
            parameter_bounds = request.parameter_bounds
        
        black_box.set_parameter_bounds(parameter_bounds)
        
        # Configure optimization parameters
        n_trials = request.n_iter if request.n_iter else 25
        init_points = request.init_points if request.init_points else 5
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Run optimization
        best_params, best_value, study = optimize_with_optuna(
            black_box_func=black_box,
            n_trials=n_trials
        )
        
        # If we're minimizing, we need to negate the value back since BlackBoxFunction negates it internally
        if not request.maximize:
            best_value = -best_value
            
        # Generate recommendations from top trials
        recommendations = []
        for trial in sorted(study.trials, key=lambda t: t.value if request.maximize else -t.value, reverse=request.maximize)[:5]:
            value = trial.value if request.maximize else -trial.value
            recommendations.append({
                "params": trial.params,
                "predicted_value": float(value)
            })
        
        # Create results directory to save optimization artifacts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, '..', 'optimization', 'optimization_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Export study trials to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(results_dir, f"optuna_trials_{request.model_id}_{timestamp}.csv")
        
        # Export trial data
        trials_data = []
        for i, trial in enumerate(study.trials):
            trial_dict = {
                "trial_number": i,
                "value": trial.value if request.maximize else -trial.value,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete
            }
            for param_name, param_value in trial.params.items():
                trial_dict[f"param_{param_name}"] = param_value
            trials_data.append(trial_dict)
            
        # Convert to DataFrame and save
        trials_df = pd.DataFrame(trials_data)
        trials_df.to_csv(csv_file, index=False)
            
        # Return optimized parameters
        return OptimizationResponse(
            best_params=best_params,
            best_target=float(best_value),
            target_col=target_col,
            maximize=request.maximize,
            recommendations=recommendations,
            model_id=request.model_id
        )
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List all available models by scanning the models directory"""
    try:
        import os
        import glob
        import json
        from datetime import datetime
        
        # Use absolute path to mills-xgboost/models directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_dir = os.path.join(project_root, 'models')
        
        # Check if directory exists
        if not os.path.exists(models_dir):
            logger.error(f"Models directory does not exist: {models_dir}")
            return {"error": "Models directory not found"}
        
        # Find all metadata files
        metadata_files = glob.glob(os.path.join(models_dir, '*_metadata.json'))
        logger.info(f"Found {len(metadata_files)} model metadata files")
        
        result = {}
        for metadata_file in metadata_files:
            try:
                # Extract model ID from filename
                filename = os.path.basename(metadata_file)
                model_id = filename.replace('_metadata.json', '')
                
                # Read metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if corresponding model and scaler files exist
                model_file = os.path.join(models_dir, f"{model_id}_model.json")
                scaler_file = os.path.join(models_dir, f"{model_id}_scaler.pkl")
                
                files_exist = os.path.exists(model_file) and os.path.exists(scaler_file)
                
                # Get file modification time as a proxy for creation date
                try:
                    mod_time = os.path.getmtime(metadata_file)
                    mod_time_str = datetime.fromtimestamp(mod_time).isoformat()
                except:
                    mod_time_str = "Unknown"
                
                # Get required information
                result[model_id] = {
                    "name": model_id,
                    "features": metadata.get("features", []),
                    "target_col": metadata.get("target_col", ""),
                    "last_trained": metadata.get("last_trained", mod_time_str),
                    "files_complete": files_exist
                }
            except Exception as e:
                logger.error(f"Error processing metadata file {metadata_file}: {str(e)}")
                # Continue with next file
        
        return result
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

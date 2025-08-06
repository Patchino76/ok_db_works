import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import logging
import os
import json
from datetime import datetime

# Get logger without reconfiguring - use the main app's logger configuration
logger = logging.getLogger(__name__)

class MillsXGBoostModel:
    """
    Production-ready XGBoost model for mill data regression
    """
    
    def __init__(self, features=None, target_col='PSI80'):
        """
        Initialize the XGBoost model
        
        Args:
            features: List of feature column names (default: None, will be loaded from metadata)
            target_col: Target column name (default: 'PSI80')
        """
        # Only set features if explicitly provided, otherwise leave None to be loaded from metadata
        self.features = features
        self.target_col = target_col
        self.model = None
        self.scaler = None
        self.training_history = {}
        self.feature_importance = None
        self.model_params = {}
        
        # Model initialized - no need to log this basic initialization
    
    def train(self, X_train, X_test, y_train, y_test, scaler, params=None):
        """
        Train the XGBoost model with early stopping
        
        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled)
            y_train: Training target values
            y_test: Test target values
            scaler: The scaler used for feature scaling
            params: Optional dictionary of XGBoost parameters
            
        Returns:
            Dictionary with training results
        """
        start_time = datetime.now()
        logger.info(f"Starting XGBoost model training at {start_time}")
        
        # Store the scaler
        self.scaler = scaler
        
        # Set default parameters or use provided ones
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 30
        }
        
        self.model_params = params or default_params
        
        # Create and train the model
        # Add eval_metric to model_params instead of fit method
        params_with_eval = self.model_params.copy()
        params_with_eval['eval_metric'] = 'mae'
        self.model = xgb.XGBRegressor(**params_with_eval)
        
        # Fit the model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False  # Suppress verbose output in production
        )
        
        # Store training results
        self.training_history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, train_pred)
        test_metrics = self._calculate_metrics(y_test, test_pred)
        
        # Log metrics
        logger.info(f"Training metrics: MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}")
        logger.info(f"Test metrics: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Log top features
        top_features = self.feature_importance.head(3)['Feature'].tolist()
        logger.info(f"Top 3 important features: {', '.join(top_features)}")
        
        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model training completed in {duration:.2f} seconds")
        
        # Return results
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.feature_importance.to_dict(),
            'training_duration': duration,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
    
    def predict(self, data):
        """
        Make predictions using the trained model
        
        Args:
            data: DataFrame or dictionary with features
            
        Returns:
            Numpy array with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.features is None:
            raise ValueError("Model features not initialized. Load model from metadata first.")
        
        try:
            # Convert dictionary to DataFrame if necessary
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Ensure all required features are present
            missing_features = [f for f in self.features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            
            # Extract features and scale
            X = data[self.features]
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            predictions = self.model.predict(X_scaled)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, directory='models', mill_number=None):
        """
        Save the model, scaler, and metadata to disk
        
        Args:
            directory: Directory to save model files
            mill_number: Mill number to include in the filename (optional)
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            # Use absolute path for models directory
            if not os.path.isabs(directory):
                # Create absolute path relative to the project root
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                directory = os.path.join(project_root, directory)
                logger.info(f"Using absolute path for models directory: {directory}")
            
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Use fixed filenames that overwrite on each training
            # Include mill number in filename if provided
            mill_suffix = f'_mill{mill_number}' if mill_number is not None else ''
            
            model_path = os.path.join(directory, f'xgboost_{self.target_col}{mill_suffix}_model.json')
            scaler_path = os.path.join(directory, f'xgboost_{self.target_col}{mill_suffix}_scaler.pkl')
            metadata_path = os.path.join(directory, f'xgboost_{self.target_col}{mill_suffix}_metadata.json')
            self.model.save_model(model_path)
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'features': self.features,
                'target_col': self.target_col,
                'model_params': self.model_params,
                'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
                'training_history': self.training_history,
                'last_trained': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path, scaler_path, metadata_path=None):
        """
        Load model, scaler, and metadata from disk
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            metadata_path: Path to metadata file
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load model
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            
            # Load metadata if provided
            if metadata_path:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Always use features from metadata, with fallback to default if not found
                self.features = metadata.get('features', [
                    'Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 
                    'DensityHC', 'MotorAmp', 'Shisti', 'Daiki'
                ])
                self.target_col = metadata.get('target_col', self.target_col)
                self.model_params = metadata.get('model_params', {})
                self.training_history = metadata.get('training_history', {})
                
                # Recreate feature importance DataFrame if available
                if 'feature_importance' in metadata and metadata['feature_importance']:
                    importance_data = metadata['feature_importance']
                    self.feature_importance = pd.DataFrame({
                        'Feature': importance_data.get('Feature', []),
                        'Importance': importance_data.get('Importance', [])
                    })
                
                logger.info(f"Metadata loaded from {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression performance metrics
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        if self.model is None:
            logger.warning("Cannot calculate feature importance - model not trained")
            return
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for better organization
        self.feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        return self.feature_importance
    
    def get_model_summary(self):
        """
        Get a summary of the model for logging purposes
        
        Returns:
            Dictionary with model summary information
        """
        if self.model is None:
            return {"status": "Not trained"}
        
        return {
            "target": self.target_col,
            "features": self.features,
            "params": self.model_params,
            "feature_importance": self.feature_importance.to_dict() if self.feature_importance is not None else None,
            "training_history": self.training_history
        }

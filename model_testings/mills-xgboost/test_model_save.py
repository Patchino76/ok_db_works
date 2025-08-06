import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

# Set up logging to both console and file
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import our model class
sys.path.append(os.path.abspath('.'))  # Add current directory to path
from app.models.xgboost_model import MillsXGBoostModel

def test_model_save():
    """Test function to check if model saving works correctly"""
    logger.info("Starting model save test")
    
    # Create a simple test dataset
    np.random.seed(42)
    n_samples = 100
    
    # Create features and target
    features = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'MotorAmp', 'Shisti', 'Daiki']
    X = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in features
    })
    y = X['Ore'] * 0.5 + X['WaterMill'] * 0.3 + X['MotorAmp'] * 0.2 + np.random.normal(0, 0.1, n_samples)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    logger.info("Initializing XGBoost model")
    model = MillsXGBoostModel(features=features, target_col='PSI80')
    
    # Set some basic parameters
    params = {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 3
    }
    
    # Train the model
    logger.info("Training model")
    try:
        # Make sure the params include early_stopping_rounds 
        params['early_stopping_rounds'] = 10
        params['eval_metric'] = 'mae'
        model.train(X_train, X_test, y_train, y_test, scaler, params)
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Try without early stopping as a fallback
        logger.info("Retrying without early stopping")
        del params['early_stopping_rounds']
        model.model = xgb.XGBRegressor(**params)
        model.model.fit(X_train, y_train)
        model.scaler = scaler
    
    # Save the model
    logger.info("Saving model")
    save_results = model.save_model(directory='models_test')
    
    # Print save paths
    logger.info(f"Model saved to: {save_results['model_path']}")
    logger.info(f"Scaler saved to: {save_results['scaler_path']}")
    logger.info(f"Metadata saved to: {save_results['metadata_path']}")
    
    # Check if files exist
    for path_key, file_path in save_results.items():
        if os.path.exists(file_path):
            logger.info(f"File exists: {file_path}")
        else:
            logger.error(f"File does not exist: {file_path}")

if __name__ == "__main__":
    logger.info(f"Current working directory: {os.getcwd()}")
    test_model_save()

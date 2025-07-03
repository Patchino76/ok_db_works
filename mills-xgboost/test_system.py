import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from app.database.db_connector import MillsDataConnector
from app.models.data_processor import DataProcessor
from app.models.xgboost_model import MillsXGBoostModel
from app.optimization.bayesian_opt import MillBayesianOptimizer

def test_database_connection(db_config):
    """Test database connection and data retrieval"""
    logger.info("Testing database connection...")
    
    connector = MillsDataConnector(
        host=db_config["host"],
        port=db_config["port"],
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"]
    )
    
    # Get data from the past week for mill 1
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    # Get mill data
    mill_data = connector.get_mill_data(
        mill_number=1,
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Retrieved {len(mill_data)} mill data records")
    
    # Get ore quality data
    ore_data = connector.get_ore_quality_data(
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Retrieved {len(ore_data)} ore quality records")
    
    # Get combined data
    combined_data = connector.get_combined_data(
        mill_number=1,
        start_date=start_date,
        end_date=end_date,
        resample_freq='1min'
    )
    logger.info(f"Combined data has {len(combined_data)} records and columns: {combined_data.columns.tolist()}")
    
    return combined_data

def test_model_training(data):
    """Test model training with data"""
    logger.info("Testing model training...")
    
    # Define features and target
    features = [
        'Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 
        'DensityHC', 'MotorAmp', 'Shisti', 'Daiki'
    ]
    target_col = 'PSI80'
    
    # Preprocess data
    data_processor = DataProcessor()
    X_scaled, y, scaler = data_processor.preprocess(data, features, target_col)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    xgb_model = MillsXGBoostModel(features=features, target_col=target_col)
    
    training_results = xgb_model.train(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler
    )
    
    logger.info(f"Training completed with test RMSE: {training_results['test_metrics']['rmse']:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    save_path = xgb_model.save_model(directory="models")
    logger.info(f"Model saved to {save_path}")
    
    return xgb_model

def test_prediction(model, test_data):
    """Test prediction with model"""
    logger.info("Testing model prediction...")
    
    # Make predictions for a few samples
    if isinstance(test_data, pd.DataFrame) and len(test_data) > 0:
        sample_data = test_data.iloc[:5].copy()
        predictions = model.predict(sample_data)
        
        for i, pred in enumerate(predictions):
            logger.info(f"Sample {i+1} prediction: {pred:.4f}")
    
    # Test single prediction with dict input
    sample_dict = {
        'Ore': 180.0,
        'WaterMill': 150.0,
        'WaterZumpf': 70.0,
        'PressureHC': 35.0,
        'DensityHC': 1.8,
        'MotorAmp': 250.0,
        'Shisti': 0.5,
        'Daiki': 0.3
    }
    
    prediction = model.predict(sample_dict)[0]
    logger.info(f"Single prediction with dict input: {prediction:.4f}")

def test_bayesian_optimization(model):
    """Test Bayesian optimization with model"""
    logger.info("Testing Bayesian optimization...")
    
    # Create optimizer
    optimizer = MillBayesianOptimizer(
        xgboost_model=model,
        target_col=model.target_col,
        maximize=False  # Set to minimize PSI80
    )
    
    # Set parameter bounds
    bounds = {
        'Ore': (160.0, 200.0),
        'WaterMill': (120.0, 180.0),
        'WaterZumpf': (40.0, 100.0),
        'PressureHC': (20.0, 50.0),
        'DensityHC': (1.5, 2.2),
        'MotorAmp': (200.0, 300.0)
    }
    optimizer.set_parameter_bounds(bounds)
    
    # Add a simple constraint (e.g., WaterMill should be at least twice WaterZumpf)
    constraints = {
        'water_ratio': lambda params: params.get('WaterMill', 0) >= 1.5 * params.get('WaterZumpf', 0)
    }
    optimizer.set_constraints(constraints)
    
    # Run a quick optimization (fewer iterations for testing)
    os.makedirs("optimization_results", exist_ok=True)
    results = optimizer.optimize(
        init_points=2,
        n_iter=5,
        save_dir="optimization_results"
    )
    
    logger.info(f"Best parameters found: {json.dumps(results['best_params'], indent=2)}")
    logger.info(f"Best target value: {results['best_target']:.4f}")
    
    # Get recommendations
    recommendations = optimizer.recommend_parameters(n_recommendations=2)
    logger.info(f"Top recommendations: {json.dumps(recommendations, indent=2)}")

def main():
    """Main test function"""
    logger.info("Starting system test...")
    
    # Database configuration (replace with your actual credentials)
    db_config = {
        "host": "em-m-db4.ellatzite-med.com",
        "port": 5432,
        "dbname": "em_pulse_data",
        "user": "s.lyubenov",
        "password": "YOUR_PASSWORD_HERE"  # Replace with actual password
    }
    
    try:
        # Test database connection
        logger.info("=" * 50)
        logger.info("TESTING DATABASE CONNECTION")
        logger.info("=" * 50)
        data = None
        
        # Check if we have test data saved already to avoid database connection if not needed
        test_data_path = "test_data.pkl"
        if os.path.exists(test_data_path):
            logger.info(f"Loading test data from {test_data_path}")
            data = pd.read_pickle(test_data_path)
        else:
            # Skip DB connection if no password provided
            if db_config["password"] == "YOUR_PASSWORD_HERE":
                logger.warning("No database password provided. Skipping database connection test.")
                logger.warning("Please edit the script with your actual database credentials.")
            else:
                data = test_database_connection(db_config)
                # Save data for future tests
                if data is not None:
                    data.to_pickle(test_data_path)
                    logger.info(f"Saved test data to {test_data_path}")
        
        # If we have data, continue with tests
        if data is not None:
            # Test model training
            logger.info("=" * 50)
            logger.info("TESTING MODEL TRAINING")
            logger.info("=" * 50)
            model = test_model_training(data)
            
            # Test prediction
            logger.info("=" * 50)
            logger.info("TESTING PREDICTION")
            logger.info("=" * 50)
            test_prediction(model, data)
            
            # Test Bayesian optimization
            logger.info("=" * 50)
            logger.info("TESTING BAYESIAN OPTIMIZATION")
            logger.info("=" * 50)
            test_bayesian_optimization(model)
        else:
            logger.warning("No data available. Please provide database credentials or test data.")
        
        logger.info("=" * 50)
        logger.info("TEST COMPLETED")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for preprocessing data for XGBoost models, focusing on mill data.
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def preprocess(self, df, features, target_col, filter_data=True):
        """
        Preprocess data for XGBoost model training or inference
        
        Args:
            df: DataFrame with raw data
            features: List of feature column names
            target_col: Target column name
            filter_data: Whether to apply filters for Ore and target column
            
        Returns:
            Tuple of (X_scaled, y, scaler)
        """
        # Store feature names
        self.feature_names = features
        
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Select only the required features and target
        try:
            data = data[features + [target_col]].copy()
        except KeyError as e:
            missing_cols = [col for col in features + [target_col] if col not in df.columns]
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
        # Drop rows with NaN values
        data_clean = data.dropna()
        dropped_rows = len(data) - len(data_clean)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with missing values ({dropped_rows/len(data):.2%} of data)")
        
        # Apply filters based on feature bounds (similar to the notebook)
        if filter_data:
            logger.info("Applying filters on features")
            # These filters match those in the notebook
            if 'Ore' in data_clean.columns and target_col in data_clean.columns:
                data_filtered = data_clean[
                    (data_clean['Ore'] > 160) & 
                    (data_clean['Ore'] < 200)
                ]
                
                # Apply filter based on target column (PSI80 or FR200)
                if target_col == 'PSI80':
                    data_filtered = data_filtered[data_filtered[target_col] < 55]
                elif target_col == 'FR200':
                    data_filtered = data_filtered[data_filtered[target_col] < 30]
                
                filtered_rows = len(data_clean) - len(data_filtered)
                logger.info(f"Filtered out {filtered_rows} rows ({filtered_rows/len(data_clean):.2%} of data)")
                data_clean = data_filtered
        
        # Split features and target
        X = data_clean[features]
        y = data_clean[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, self.scaler
    
    def transform_new_data(self, df, features):
        """
        Transform new data using the fitted scaler
        
        Args:
            df: DataFrame with new data
            features: List of feature column names to use
            
        Returns:
            Scaled features as numpy array
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fit yet. Run preprocess() first.")
        
        # Check for missing features
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing features in input data: {missing_cols}")
        
        # Extract features and scale
        X = df[features]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

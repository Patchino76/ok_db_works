import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import seaborn as sns

# Data preparation
def prepare_data(df, features, target_col, test_size=0.2, random_state=42):
    """
    Prepare data for XGBoost model by splitting into train/test sets and standardizing features.
    
    Args:
        df: DataFrame containing the data
        features: List of feature column names
        target_col: Target column name
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing (X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, feature_names)
    """
    # Select only the required features and target
    data = df[features + [target_col]].copy()
    
    # Drop rows with NaN values
    data_clean = data.dropna()
    dropped_rows = len(data) - len(data_clean)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values ({dropped_rows/len(data):.2%} of data)")
    
    # Check for remaining missing values
    if data_clean.isnull().sum().sum() > 0:
        print("Warning: Still have missing values after dropna()")
        print(data_clean.isnull().sum())
    
    # Split features and target
    X = data_clean[features]
    y = data_clean[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, X.columns)

# Model training with early stopping
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost regression model with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Trained XGBoost model
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,  # Reduced from 1000 for faster execution
        learning_rate=0.1,  # Increased from 0.05 for faster convergence
        max_depth=5,       # Slightly reduced complexity
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,  # Reduced from 50
        eval_metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10  # Show progress every 10 iterations instead of 100
    )
    
    return model

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

# Prediction visualization
def plot_predictions(y_true, y_pred, title):
    """
    Create scatter plot of actual vs predicted values.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png")
    plt.show()

# Training history plot
def plot_training_history(model):
    """
    Plot the training history showing MAE over boosting rounds.
    
    Args:
        model: Trained XGBoost model
    """
    results = model.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mae'], label='Training MAE')
    plt.plot(x_axis, results['validation_1']['mae'], label='Validation MAE')
    plt.legend()
    plt.ylabel('MAE')
    plt.xlabel('Boosting Rounds')
    plt.title('XGBoost Training History')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("XGBoost_Training_History.png")
    plt.show()

# Feature importance plot
def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig("XGBoost_Feature_Importance.png")
    plt.show()
    
    return feat_imp

# Time series plot of actual vs predicted values
def plot_time_series_predictions(df, target_col, predictions, test_indices, title):
    """
    Plot time series of actual vs predicted values.
    
    Args:
        df: Original DataFrame with timestamps as index
        target_col: Target column name
        predictions: Predicted values
        test_indices: Indices of test data
        title: Plot title
    """
    # Create a copy of the DataFrame with predictions
    df_plot = df.loc[test_indices].copy()
    df_plot['Predicted'] = predictions
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot[target_col], label='Actual', alpha=0.7)
    plt.plot(df_plot.index, df_plot['Predicted'], label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png")
    plt.show()

# Main workflow
def main():
    # Load data
    print("Loading data...")
    csv_path = r'c:\Projects\ok_db_works\mill_ore_quality_06.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"Loaded data with shape: {df.shape}")
    
    # Display basic information
    print("\nDataFrame info:")
    print(df.info())
    
    print("\nDataFrame head:")
    print(df.head())
    
    print("\nDataFrame statistics:")
    print(df.describe())
    
    # Define features and target
    features = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'MotorAmp', 'Shisti']
    target_col = 'PSI80'
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler_X, feature_names = prepare_data(
        df, features, target_col, test_size=0.2
    )
    
    # Train model
    print("\nTraining XGBoost model...")
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Generate predictions
    print("\nGenerating predictions...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    
    print(f"\nTraining Metrics: MAE={train_metrics['MAE']:.4f}, RMSE={train_metrics['RMSE']:.4f}, R²={train_metrics['R2']:.4f}")
    print(f"Test Metrics: MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}, R²={test_metrics['R2']:.4f}")
    
    # Visualize results
    print("\nPlotting training history...")
    plot_training_history(model)
    
    print("Plotting feature importance...")
    importance_df = plot_feature_importance(model, feature_names)
    print("\nFeature Importance:")
    print(importance_df)
    
    print("\nPlotting predictions...")
    plot_predictions(y_train, train_pred, 'Training: Actual vs Predicted')
    plot_predictions(y_test, test_pred, 'Testing: Actual vs Predicted')
    
    # Get the test indices from the original dataframe
    # This requires keeping track of indices during train_test_split, which we didn't do
    # Instead, we'll create a time series plot with a sample of the data
    print("\nPlotting time series predictions (sample)...")
    # We'll use a different approach to demonstrate time series plotting
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_pred
    })
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Actual'], label='Actual', alpha=0.7)
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted', alpha=0.7)
    plt.title('Time Series: Actual vs Predicted (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Time_Series_Predictions.png")
    plt.show()

if __name__ == '__main__':
    main()
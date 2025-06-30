import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# Load sample dataset (replace with your data)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Data preparation
def prepare_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler_X)

# Model training with early stopping
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# Prediction visualization
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Training history plot
def plot_training_history(model):
    results = model.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mae'], label='Validation MAE')
    plt.legend()
    plt.ylabel('MAE')
    plt.xlabel('Boosting Rounds')
    plt.title('Training History')
    plt.show()

# Main workflow
def main():
    # Prepare data
    X_train, X_test, y_train, y_test, scaler_X = prepare_data(df, 'Target')
    
    # Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Generate predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    
    print(f"Training Metrics: MAE={train_metrics['MAE']:.4f}, R²={train_metrics['R2']:.4f}")
    print(f"Test Metrics: MAE={test_metrics['MAE']:.4f}, R²={test_metrics['R2']:.4f}")
    
    # Visualize results
    plot_training_history(model)
    plot_predictions(y_train, train_pred, 'Training: Actual vs Predicted')
    plot_predictions(y_test, test_pred, 'Testing: Actual vs Predicted')

if __name__ == '__main__':
    main()
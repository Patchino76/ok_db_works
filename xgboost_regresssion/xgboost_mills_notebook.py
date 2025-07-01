#%% [markdown]
# XGBoost Regression for Mill Data
# 
# This notebook analyzes mill data using XGBoost regression.
# Features: Ore, WaterMill, WaterZumpf, PressureHC, DensityHC, MotorAmp, Shisti
# Target: PSI80

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

#%% Load and explore data
# Load the mill data
print("Loading data...")
csv_path = r'c:\Projects\ok_db_works\mill_ore_quality_08.csv'
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
print(f"Loaded data with shape: {df.shape}")

# Display basic information
print("\nDataFrame info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

#%% Define smoothing functions
# Three different approaches to smooth time series data

def smooth_rolling_mean(dataframe, window_size=5, min_periods=1):
    """
    Apply rolling mean smoothing to all numeric columns in the dataframe.
    
    Args:
        dataframe: Input DataFrame with time series data
        window_size: Size of the rolling window (default: 5)
        min_periods: Minimum number of observations required (default: 1)
        
    Returns:
        DataFrame with smoothed values
    """
    df_smooth = dataframe.copy()
    numeric_cols = df_smooth.select_dtypes(include=['number']).columns
    
    # Apply rolling mean with centered window
    for col in numeric_cols:
        df_smooth[col] = df_smooth[col].rolling(
            window=window_size, 
            min_periods=min_periods, 
            center=True
        ).mean()
    
    # Fill any NaN values at edges
    df_smooth = df_smooth.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Applied rolling mean smoothing with window size {window_size}")
    return df_smooth

def smooth_ewm(dataframe, span=5, min_periods=1):
    """
    Apply exponentially weighted moving average smoothing to all numeric columns.
    This gives more weight to recent observations.
    
    Args:
        dataframe: Input DataFrame with time series data
        span: Specify decay in terms of span (default: 5)
        min_periods: Minimum number of observations required (default: 1)
        
    Returns:
        DataFrame with smoothed values
    """
    df_smooth = dataframe.copy()
    numeric_cols = df_smooth.select_dtypes(include=['number']).columns
    
    # Apply EWM smoothing
    for col in numeric_cols:
        df_smooth[col] = df_smooth[col].ewm(
            span=span, 
            min_periods=min_periods, 
            adjust=True
        ).mean()
    
    print(f"Applied exponentially weighted moving average smoothing with span {span}")
    return df_smooth

def smooth_savgol(dataframe, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay filter for smoothing.
    This preserves features of the distribution such as peaks better than rolling averages.
    
    Args:
        dataframe: Input DataFrame with time series data
        window_length: Length of the filter window (must be odd number, default: 11)
        polyorder: Order of the polynomial (default: 2)
        
    Returns:
        DataFrame with smoothed values
    """
    from scipy.signal import savgol_filter
    
    df_smooth = dataframe.copy()
    numeric_cols = df_smooth.select_dtypes(include=['number']).columns
    
    # Make sure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply Savitzky-Golay filter
    for col in numeric_cols:
        # Skip columns with NaN values
        if df_smooth[col].isnull().sum() > 0:
            continue
            
        df_smooth[col] = savgol_filter(
            df_smooth[col].values, 
            window_length=window_length, 
            polyorder=polyorder
        )
    
    print(f"Applied Savitzky-Golay filter with window length {window_length} and polynomial order {polyorder}")
    return df_smooth

# Choose one smoothing method to apply (uncomment the one you want to use)
print("\nApplying data smoothing...")
# df = smooth_rolling_mean(df, window_size=5)  # Option 1: Rolling mean
# df = smooth_ewm(df, span=5)                 # Option 2: Exponential weighted moving average
df = smooth_savgol(df, window_length=11)    # Option 3: Savitzky-Golay filter
# print("No smoothing applied - uncomment one of the smoothing functions above to enable")

#%% Define features and target
# Specify the features and target variable
features = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'MotorAmp', 'Shisti']
target_col = 'PSI80'

# Check if all required columns exist
missing_cols = [col for col in features + [target_col] if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns in dataset: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
else:
    print("All required columns are present in the dataset")

#%% Prepare data for modeling
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
else:
    print("No missing values in the cleaned dataset")
    
# Filter data based on feature bounds
print("\nFiltering data based on feature bounds:")
print(f"Original data shape: {data_clean.shape}")

# Apply filters for Ore and PSI80
filtered_data = data_clean.copy()
filtered_data = filtered_data[(filtered_data['Ore'] > 160) & 
                             (filtered_data['Ore'] < 200) & 
                             (filtered_data['PSI80'] < 55)]

# Report on filtering results
filtered_rows = len(data_clean) - len(filtered_data)
print(f"Filtered out {filtered_rows} rows ({filtered_rows/len(data_clean):.2%} of data)")
print(f"Remaining data shape: {filtered_data.shape}")
print(f"Ore range: {filtered_data['Ore'].min():.2f} to {filtered_data['Ore'].max():.2f}")
print(f"PSI80 range: {filtered_data['PSI80'].min():.2f} to {filtered_data['PSI80'].max():.2f}")

# Use the filtered data for further processing
data_clean = filtered_data

# Split features and target
X = data_clean[features]
y = data_clean[target_col]

# Time series split - chronological without shuffling
# For time series data, we should split by time rather than randomly
# Use the last 20% of the data as the test set

# Sort by index (timestamp) to ensure chronological order
data_clean = data_clean.sort_index()
X = data_clean[features]
y = data_clean[target_col]

# Calculate the split point at 80% of the data
split_idx = int(len(X) * 0.8)

# Split the data chronologically
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Training set: {X_train.shape[0]} samples (from {X_train.index.min()} to {X_train.index.max()})")
print(f"Test set: {X_test.shape[0]} samples (from {X_test.index.min()} to {X_test.index.max()})")
print(f"Using chronological split for time series data (no shuffling)")


# Feature scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

#%% Correlation analysis
# Create correlation matrix
corr_matrix = data_clean.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("Correlation_Matrix.png")
plt.show()

# Correlation with target
target_corr = corr_matrix[target_col].sort_values(ascending=False)
print("\nCorrelation with target variable (PSI80):")
print(target_corr)

#%% Train XGBoost model
# Configure and train the model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='mae'
)

# Train the model with early stopping
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=10
)

#%% Generate predictions
# Make predictions on train and test sets
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

#%% Evaluate model performance
# Calculate metrics
train_mae = mean_absolute_error(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

# Print metrics
print("\nModel Performance Metrics:")
print(f"Training: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
print(f"Test: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

#%% Visualize training history
# Plot training history
results = model.evals_result()
epochs = len(results['validation_0']['mae'])
x_axis = range(epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['mae'], label='Training MAE')
plt.plot(x_axis, results['validation_1']['mae'], label='Validation MAE')
plt.legend()
plt.ylabel('Mean Absolute Error')
plt.xlabel('Boosting Rounds')
plt.title('XGBoost Training History')
plt.grid(True)
plt.tight_layout()
plt.savefig("XGBoost_Training_History.png")
plt.show()

#%% Analyze feature importance
# Get and plot feature importance
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig("XGBoost_Feature_Importance.png")
plt.show()

print("\nFeature Importance:")
print(feat_imp)

#%% Visualize predictions
# Plot actual vs predicted values for training set (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training: Actual vs Predicted (Scatter)')
plt.grid(True)
plt.tight_layout()
plt.savefig("Training_Predictions_Scatter.png")
plt.show()

# Plot actual vs predicted values for test set (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Testing: Actual vs Predicted (Scatter)')
plt.grid(True)
plt.tight_layout()
plt.savefig("Testing_Predictions_Scatter.png")
plt.show()

#%% Trend line visualization
# Create trend line plots showing actual vs predicted values over sample indices

# For training data - show a sample of 500 points for better visibility
sample_size = min(500, len(y_train))
sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
sample_indices.sort()  # Sort indices to maintain time order

plt.figure(figsize=(12, 6))
plt.plot(y_train.iloc[sample_indices].values, 'b-', label='Actual Values', linewidth=2)
plt.plot(train_pred[sample_indices], 'r-', label='Predicted Values', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel(target_col)
plt.title('Training: Actual vs Predicted Values (Trend)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Training_Predictions_Trend.png")
plt.show()

# For test data - show a sample of 500 points for better visibility
sample_size = min(500, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
sample_indices.sort()  # Sort indices to maintain time order

plt.figure(figsize=(12, 6))
plt.plot(y_test.iloc[sample_indices].values, 'b-', label='Actual Values', linewidth=2)
plt.plot(test_pred[sample_indices], 'r-', label='Predicted Values', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel(target_col)
plt.title('Testing: Actual vs Predicted Values (Trend)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Testing_Predictions_Trend.png")
plt.show()

#%% Residual analysis
# Calculate residuals
train_residuals = y_train - train_pred
test_residuals = y_test - test_pred

# Plot residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(train_pred, train_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Training Residuals')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(test_pred, test_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Test Residuals')
plt.grid(True)

plt.tight_layout()
plt.savefig("Residual_Analysis.png")
plt.show()

# Distribution of residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(train_residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Training Residuals Distribution')

plt.subplot(1, 2, 2)
sns.histplot(test_residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Test Residuals Distribution')

plt.tight_layout()
plt.savefig("Residual_Distribution.png")
plt.show()

#%% Save model for future use
# Save the trained model
model.save_model("xgboost_mill_psi80_model.json")
print("Model saved to 'xgboost_mill_psi80_model.json'")

# Save the scaler for future preprocessing
import pickle
with open('scaler_mill_psi80.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
print("Scaler saved to 'scaler_mill_psi80.pkl'")

#%% Summary
print("\nModel Summary:")
print(f"- Target variable: {target_col}")
print(f"- Features used: {', '.join(features)}")
print(f"- Top 3 important features: {', '.join(feat_imp['Feature'].head(3).tolist())}")
print(f"- Model performance (Test): MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
print("- Model and scaler saved for future use")

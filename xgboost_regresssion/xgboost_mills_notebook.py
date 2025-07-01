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
csv_path = r'c:\Projects\ok_db_works\mill_ore_quality_06.csv'
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
print(f"Loaded data with shape: {df.shape}")

# Display basic information
print("\nDataFrame info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

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

# Split features and target
X = data_clean[features]
y = data_clean[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

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

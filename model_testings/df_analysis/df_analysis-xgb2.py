#%% Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set display options for better readability
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

#%% Define parameters
parameters = [
    'Ore',
    'WaterMill',
    'WaterZumpf',
    'Power',
    'ZumpfLevel',
    'PressureHC',
    'DensityHC',
    'PulpHC',
    'PumpRPM',
    'MotorAmp',
    'PSI80',
    'PSI200',
    'Class_15',
    'Class_12',
    'Grano',
    'Daiki',
    'Shisti'
]

#%% Load and prepare the data
# Load the combined mill data
base_dir = Path(__file__).resolve().parents[1]  # .../model_testings
file_path = base_dir / 'data' / 'combined_data_mill7.csv'
df = pd.read_csv(file_path, parse_dates=['TimeStamp'], index_col='TimeStamp')
# First filter the DataFrame with the original parameters
df = df[parameters].copy()

# Filter by date range
start_date = pd.Timestamp('2025-06-15 06:00')
end_date = pd.Timestamp('2025-08-12 22:00')
df = df.loc[start_date:end_date].copy()

# Apply data quality constraints
initial_count = len(df)

# Filter based on physical constraints
df = df[
    (df['Ore'].between(150, 200)) &  
    (df['PSI200'].between(15, 35)) &   
    (df['WaterMill'].between(5, 20)) & 
    (df['WaterZumpf'].between(140, 250)) &
    (df['DensityHC'].between(1500, 1800)) &
    (df['MotorAmp'].between(170, 220))
].copy()



# Display basic info about the dataframe
print("DataFrame Info:")
df.info()

# Display first and last few rows
print("\nFirst 2 rows of the dataframe:")
print(df.head(2))
print("\nLast 2 rows of the dataframe:")
print(df.tail(2))
#%%
def estimate_lag_ccf(
    df: pd.DataFrame,
    x: str,
    y: str,
    max_lag: int = 120,
    detrend_window: int | None = 60,
    standardize: bool = True,
    return_series: bool = True,
    plot: bool = False,
    ax=None,
):
   

    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Columns not found: x='{x}', y='{y}'. Available: {list(df.columns)}")

    # Align and drop rows where either is missing
    s = df[[x, y]].copy()
    s = s.dropna(how='any')
    if s.empty:
        raise ValueError("After dropping NaNs there is no data to compute CCF.")

    xs = s[x].astype(float)
    ys = s[y].astype(float)

    # Detrend with rolling mean to remove slow drift (uses past and current values only)
    if detrend_window and detrend_window > 1:
        xs = xs - xs.rolling(detrend_window, min_periods=1).mean()
        ys = ys - ys.rolling(detrend_window, min_periods=1).mean()

    if standardize:
        # Z-score; guard against zero variance
        x_std = xs.std(ddof=0)
        y_std = ys.std(ddof=0)
        xs = (xs - xs.mean()) / (x_std if x_std != 0 else 1.0)
        ys = (ys - ys.mean()) / (y_std if y_std != 0 else 1.0)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    # Positive lag -> correlate y_t with x_{t-lag} by shifting x forward
    for l in lags:
        corrs.append(ys.corr(xs.shift(l)))

    corrs = pd.Series(corrs, index=lags)
    # Choose lag by absolute correlation (strength), keep sign
    if corrs.abs().isna().all():
        raise ValueError("All correlation values are NaN; check data and parameters.")
    best_lag = int(corrs.abs().idxmax())
    best_corr = float(corrs.loc[best_lag])

    # Infer sampling period if DateTimeIndex
    step_td = None
    lag_td = None
    if isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 1:
        diffs = s.index.to_series().diff().dropna()
        if not diffs.empty:
            step_td = diffs.median()
            try:
                lag_td = step_td * best_lag
            except Exception:
                lag_td = None

    if plot:
        _ax = ax
        if _ax is None:
            fig, _ax = plt.subplots(figsize=(7.5, 3.5))
        _ax.stem(corrs.index, corrs.values, linefmt='0.7', markerfmt='o', basefmt=' ')
        _ax.axvline(0, color='k', lw=0.8)
        _ax.axvline(best_lag, color='C1', lw=1.2, ls='--', label=f"best lag = {best_lag}")
        _ax.set_xlabel('Lag (steps) — positive: x leads y (y after x)')
        _ax.set_ylabel('Correlation')
        title_td = f" (~{lag_td})" if lag_td is not None else ""
        _ax.set_title(f"Cross-correlation {x} → {y}: best lag {best_lag}{title_td}, r = {best_corr:.3f}")
        _ax.legend(loc='best')
        plt.tight_layout()

    return {
        'best_lag': best_lag,
        'best_corr': best_corr,
        'corrs': corrs if return_series else None,
        'lag_timedelta': lag_td,
        'step_timedelta': step_td,
        'x': x,
        'y': y,
    }
#%%
def shift_feature_trim(df: pd.DataFrame, feature: str, shift: int) -> pd.DataFrame:
    """
    Shift an existing feature by `shift` steps and trim only the edge rows that
    become NaN due to the shift. The column name is preserved (no new column).

    Convention: positive `shift` means the value at time t becomes the past
    value feature[t-shift] (i.e., align past values to current time).
    """

    if feature not in df.columns:
        raise KeyError(f"'{feature}' not found in DataFrame columns: {list(df.columns)}")

    out = df.copy()
    out[feature] = out[feature].shift(shift)

    # Edge trimming to remove NaNs created by shifting
    if shift > 0:
        out = out.iloc[shift:].copy()
    elif shift < 0:
        out = out.iloc[:shift].copy()  # trims last -shift rows

    # Drop any residual NaNs in the shifted feature (e.g., internal gaps)
    out = out.dropna(subset=[feature])

    return out
#%% Smoothing function and demonstration
# --------------------------------------- SMOOTH DF -------------------------------
def smooth_time_series(data, method='savgol', window_size=None, polyorder=2, alpha=0.3, span=5, loess_frac=0.3, **kwargs):

    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: smooth_time_series(x, method, window_size, polyorder, alpha, span, loess_frac, **kwargs))
    
    if method == 'savgol':
        from scipy.signal import savgol_filter
        window = window_size or min(11, len(data) - 1) if len(data) > 1 else 1
        window = window if window % 2 == 1 else window - 1  # Ensure odd window size
        poly = min(polyorder, window - 1)
        return pd.Series(
            savgol_filter(data.values, window_length=window, polyorder=poly, **kwargs),
            index=data.index,
            name=data.name
        )
    
    elif method == 'ema':
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    elif method == 'rolling':
        return data.rolling(window=window_size or span, min_periods=1, center=True).mean()
    
    elif method == 'median':
        return data.rolling(window=window_size or span, min_periods=1, center=True).median()
    
    elif method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(data.values, 
                       range(len(data)), 
                       frac=loess_frac,
                       **kwargs)
        return pd.Series(smooth[:, 1], index=data.index, name=data.name)
    
    elif method == 'hp':
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(data, **kwargs)
        return pd.Series(trend, index=data.index, name=data.name)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

# Example usage with different smoothing methods
print("\nApplying different smoothing methods to the data...")

# Create a copy of the dataframe for demonstration
plot_df = df[['Ore', 'PSI200']].copy()

# Apply different smoothing methods
methods = {
    'Original': plot_df,
    'EMA (α=0.1)': smooth_time_series(plot_df, 'ema', alpha=0.1),
    'Savgol (w=21, p=2)': smooth_time_series(plot_df, 'savgol', window_size=21, polyorder=2),
    'Rolling Mean (w=15)': smooth_time_series(plot_df, 'rolling', window_size=15),
    'Rolling Median (w=15)': smooth_time_series(plot_df, 'median', window_size=15),
    'LOWESS (frac=0.1)': smooth_time_series(plot_df, 'lowess', loess_frac=0.1)
}

# df = smooth_time_series(df, 'rolling', window_size=15)
# df_plot = smooth_time_series(plot_df, 'rolling', window_size=15)
# df_plot.plot()
#-----------------------------------------------------------------------
#%% Create scatter plot matrix of parameters
print("\nGenerating scatter plot matrix of parameters...")

# Select a subset of features for visualization to avoid overcrowding
# Using all features can make the plot too dense
plot_features = ['Ore', 'WaterMill', 'WaterZumpf', 'DensityHC', 'MotorAmp', 'PSI200']
plot_df = df[plot_features].copy()

# Create figure with subplots
n = len(plot_features)
fig, axes = plt.subplots(n, n, figsize=(20, 18))
fig.suptitle('Scatter Plot Matrix of Parameters', y=1.02, fontsize=16)

# Customize the appearance
plt.rcParams.update({'font.size': 10})
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10

# Create scatter plots
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        
        # Diagonal: Show KDE plot
        if i == j:
            sns.kdeplot(data=plot_df.iloc[:, i], ax=ax, fill=True, color='skyblue')
            ax.set_ylabel('Density')
        # Off-diagonal: Show scatter plot
        else:
            ax.scatter(
                plot_df.iloc[:, j], 
                plot_df.iloc[:, i],
                alpha=0.5,
                s=10,
                color='royalblue',
                edgecolor='none'
            )
        
        # Only show x labels on bottom row
        if i == n - 1:
            ax.set_xlabel(plot_df.columns[j])
        else:
            ax.set_xticklabels([])
            
        # Only show y labels on first column
        if j == 0:
            ax.set_ylabel(plot_df.columns[i])
        else:
            ax.set_yticklabels([])
            
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.show()

# Add correlation heatmap for reference
plt.figure(figsize=(12, 10))
corr = plot_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, 
    mask=mask, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1,
    fmt='.2f',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Heatmap of Parameters', pad=20)
plt.tight_layout()
plt.show()

#%% Prepare data for modeling
# Define target and features
target = 'PSI200'
features = [
    'Ore',
    'WaterMill',
    'WaterZumpf',
    # 'Power',
    # 'ZumpfLevel',
    'PressureHC',
    'DensityHC',
    'PulpHC',
    # 'PumpRPM',
    'MotorAmp',
    # 'Class_15',
    # 'Class_12',
    # 'Grano',
    # 'Daiki',
    # 'Shisti'
]

# Estimate best lag for each feature vs target and shift dataframe accordingly
print("\nEstimating best lag for each feature relative to target 'PSI200'...")
lag_results: dict[str, dict] = {}
for feat in features:
    try:
        res = estimate_lag_ccf(
            df=df,
            x=feat,
            y=target,
            max_lag=90,
            detrend_window=60,
            standardize=True,
            return_series=False,
            plot=True,
        )
        lag_results[feat] = res
        lag_td = res.get('lag_timedelta')
        print(f"- {feat}: best_lag = {res['best_lag']:+d} steps" + (f" (~{lag_td})" if lag_td is not None else "") + f", corr = {res['best_corr']:.3f}")
    except Exception as e:
        print(f"! Failed lag estimation for {feat}: {e}")

print("\nShifting features by their estimated lags...\n(Note: positive lag means the feature leads the target; we shift it forward)")
for feat, res in lag_results.items():
    lag = int(res['best_lag'])
    if lag == 0:
        print(f"  - {feat}: lag=0, no shift applied")
        continue
    before = len(df)
    df = shift_feature_trim(df, feature=feat, shift=lag)
    after = len(df)
    print(f"  - {feat}: shifted by {lag:+d} steps | rows {before} -> {after}")

# Final alignment to ensure no NaNs remain in modeling columns
df = df.dropna(subset=[target] + features)
print(f"Post-shift DataFrame shape: {df.shape}")

# Split into features (X) and target (y)
X = df[features].copy()
y = df[target].copy()

# Split into train and test sets (80% train, 20% test) without shuffling
test_size = 0.2
split_idx = int(len(X) * (1 - test_size))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Initialize and fit the scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# Display information about the prepared data
print("\nData Preparation Summary:")
print(f"- Original dataset shape: {df.shape}")
print(f"- Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"- Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
print("\nFirst few rows of scaled training features:")
print(X_train_scaled.head(2))

# Save the prepared data for future use
prepared_data = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'features': features,
    'target': target,
    'scaler': scaler
}

print("\nData preparation complete. Prepared data is available in the 'prepared_data' dictionary.")

#%% Train and evaluate Linear Regression model
print("\n" + "="*80)
print("TRAINING MULTIVARIATE LINEAR REGRESSION MODEL")
print("="*80)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Initialize and train the model
print("\nTraining the XGBoost model...")
# model = XGBRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=5,
#     min_child_weight=1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )
model = XGBRegressor(
    n_estimators=2000,          # Reduce from 100
    learning_rate=0.03,       # Reduce from 0.1
    max_depth=8,              # Reduce from 5
    min_child_weight=7,       # Increase from 1
    subsample=0.8,            # Reduce from 0.8
    colsample_bytree=0.8,     # Reduce from 0.8
    reg_alpha=0.2,            # Add L1 regularization
    reg_lambda=5.0,           # Add L2 regularization
    random_state=42,
    objective='reg:absoluteerror',
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- MAE: {mae:.4f}")
    print(f"- R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Get metrics for both train and test sets
train_metrics = calculate_metrics(y_train, y_train_pred, "Training Set")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test Set")

# Calculate and print correlation coefficient
correlation = np.corrcoef(y_test, y_test_pred)[0, 1]
print(f"\nCorrelation test set: {correlation:.4f}")


# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Plot actual vs predicted values with regression line
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Data Points')

# Calculate and plot the perfect prediction line (y=x)
perfect_line = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(perfect_line, perfect_line, 'r--', label='Perfect Prediction', linewidth=1.5)

# Calculate the actual regression line between actual and predicted values
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(y_test.values.reshape(-1, 1), y_test_pred)
slope = reg.coef_[0]
intercept = reg.intercept_

# Generate points for the regression line
regression_x = np.linspace(y_test.min(), y_test.max(), 100)
regression_y = intercept + slope * regression_x

# Plot the actual regression line
plt.plot(regression_x, regression_y, 'g-', 
         label=f'Regression Line (slope={slope:.2f})', 
         linewidth=2)

# Add statistics to the plot
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

stats_text = (
    f'$R^2$ = {r2:.3f}\n'
    f'MAE = {mae:.3f}\n'
    f'RMSE = {rmse:.3f}'
)

plt.gca().text(
    0.02, 0.98, stats_text,
    transform=plt.gca().transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
)

plt.title('Actual vs Predicted Values (Test Set)')
plt.xlabel('Actual PSI200')
plt.ylabel('Predicted PSI200')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Add model to prepared_data
prepared_data['model'] = model
prepared_data['feature_importance'] = feature_importance

#%% Plot actual vs predicted trends over time
print("\nGenerating trend plot of actual vs predicted values...")

# Create a figure with two subplots (one above the other)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot actual vs predicted values
ax1.plot(y_test.index, y_test, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
ax1.plot(y_test.index, y_test_pred, 'r--', label='Predicted', alpha=0.9, linewidth=1.2)
ax1.set_ylabel('PSI200 Value')
ax1.set_title('Actual vs Predicted Values Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot the error (residuals)
error = y_test - y_test_pred
ax2.plot(y_test.index, error, 'g-', alpha=0.7, linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time')
ax2.set_ylabel('Prediction Error')
ax2.set_title('Prediction Error Over Time')
ax2.grid(True, alpha=0.3)

# Add overall statistics
stats_text = (
    f'Model Performance Statistics\n'
    f'Mean Absolute Error: {mae:.2f}\n'
    f'Root Mean Squared Error: {rmse:.2f}\n'
    f'R² Score: {r2:.3f}'
)

plt.figtext(
    0.99, 0.5, stats_text,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round'),
    verticalalignment='center',
    horizontalalignment='right',
    fontsize=10
)

plt.tight_layout()
plt.show()

#%% Residual Analysis
print("\nPerforming residual analysis...")
residuals = y_test - y_test_pred

plt.figure(figsize=(15, 4))

# Plot 1: Residuals vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals vs Actual  
plt.subplot(1, 3, 2)
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual')
plt.grid(True, alpha=0.3)

# Plot 3: Distribution of residuals
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%
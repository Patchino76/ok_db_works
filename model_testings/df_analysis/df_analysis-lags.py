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
file_path = base_dir / 'data' / 'combined_data_mill8.csv'
df = pd.read_csv(file_path, parse_dates=['TimeStamp'], index_col='TimeStamp')
# First filter the DataFrame with the original parameters
df = df[parameters].copy()

# Filter by date range
start_date = pd.Timestamp('2025-06-15 06:00')
end_date = pd.Timestamp('2025-08-04 22:00')
df = df.loc[start_date:end_date].copy()

# Apply data quality constraints
initial_count = len(df)

# Ensure a continuous 1-min time grid (do not drop rows)
df = df.sort_index()
full_index = pd.date_range(df.index.min(), df.index.max(), freq='1min')
df = df.reindex(full_index)

# Build a quality mask instead of dropping rows
mask_ok = (
    df['Ore'].between(160, 200) &
    df['PSI200'].between(18, 35) &
    df['WaterMill'].between(10, 20) &
    df['WaterZumpf'].between(150, 250) &
    df['DensityHC'].between(1600, 1800) &
    df['MotorAmp'].between(190, 220)
)

# Prepare a copy for lag estimation; mask only the series we analyze
df_ccf = df.copy()
ccf_cols = ['Ore', 'DensityHC', 'PSI200']
existing_cols = [c for c in ccf_cols if c in df_ccf.columns]
df_ccf.loc[~mask_ok, existing_cols] = np.nan



# Display basic info about the dataframe
print("DataFrame Info:")
df.info()

# Display first and last few rows
print("\nFirst 2 rows of the dataframe:")
print(df.head(2))
print("\nLast 2 rows of the dataframe:")
print(df.tail(2))

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

# Example usage with different smoothing methods (disabled for lag analysis to avoid non-causal effects)
print("\nSmoothing demo skipped to keep lag analysis causal and avoid NaN issues.")



# Important: DO NOT replace df with non-causal smoothing for lag estimation
# If needed for visualization, use causal EMA on a copy, e.g.:
# df_ema_vis = smooth_time_series(df_ccf[['Ore','PSI200','DensityHC']], 'ema', alpha=0.1)
# df_ema_vis.plot()

#%% Create scatter plot matrix of parameters


#%% Prepare data for modeling
# Define target and features
target = 'PSI200'
features = [
    'Ore',
    'WaterMill',
    'WaterZumpf',
    # 'Power',
    # 'ZumpfLevel',
    # 'PressureHC',
    'DensityHC',
    # 'PulpHC',
    # 'PumpRPM',
    'MotorAmp',
    # 'Class_15',
    # 'Class_12',
    # 'Grano',
    # 'Daiki',
    # 'Shisti'
]

# Build a modeling DataFrame without breaking the time grid used for CCF
df_model = df.loc[mask_ok].dropna(subset=features + [target]).copy()

# Split into features (X) and target (y) from df_model
X = df_model[features].copy()
y = df_model[target].copy()

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
print(f"- Modeling dataset shape (after mask/dropna): {df_model.shape}")
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
    """
    Estimate delay (lag) between two time series columns using cross-correlation.

    Positive lag means: x leads y by `lag` steps (i.e., y reacts AFTER x by `lag`).

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame containing both columns.
    x, y : str
        Column names in `df`.
    max_lag : int, default 120
        Maximum number of steps (rows) to search in both directions.
        Assumes a roughly constant sampling interval.
    detrend_window : int | None, default 60
        If set, subtract a rolling mean of this window from each series to
        reduce spurious correlation due to slow drift. Set to None to skip.
    standardize : bool, default True
        If True, z-score each series after detrending.
    return_series : bool, default True
        If True, include the full correlation-by-lag Series in the return dict.
    plot : bool, default False
        If True, draw a stem plot of correlation vs lag and mark the best lag.
    ax : matplotlib axis, optional
        Axis to plot on when plot=True. If None, a new figure/axis is created.

    Returns
    -------
    dict
        {
          'best_lag': int,                      # steps; positive => y after x
          'best_corr': float,                    # correlation at best_lag
          'corrs': pd.Series | None,             # correlation vs lag
          'lag_timedelta': pd.Timedelta | None,  # best_lag in time units (if DateTimeIndex)
          'step_timedelta': pd.Timedelta | None, # median sampling step (if DateTimeIndex)
          'x': str,
          'y': str,
        }
    """

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
# Run CCF on masked, unsmoothed data
estimate_lag_ccf(df_ccf, 'Ore', 'PSI200', max_lag=120, detrend_window=30, plot=True)
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

df = shift_feature_trim(df, 'Ore', 12)
df.tail()

# %%
df_ccf_shifted = shift_feature_trim(df_ccf, 'Ore', 12)
df_ccf_shifted.tail()
estimate_lag_ccf(df_ccf_shifted, 'Ore', 'PSI200', max_lag=120, detrend_window=30, plot=True)
# %%

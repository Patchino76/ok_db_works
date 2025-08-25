#%% Imports and configuration
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)

#%% Parameters to keep (mirrors df_analysis-xgb2)
parameters = [
     'Ore',
     'WaterMill',
     'WaterZumpf',
     'Power',
     'ZumpfLevel',
     'PressureHC',
     'DensityHC',
     'FE',
     'PulpHC',
     'PumpRPM',
     'MotorAmp',
     'PSI200',
     'Class_15',
     'Class_12',
     'Grano',
     'Daiki',
     'Shisti',
 ]

#%% Data loading (similar to df_analysis-xgb2)
def load_data(resample: str | None = None):
    """
    Load combined mill data, subset to key parameters, apply date filter and
    basic physical constraints, and return a cleaned DataFrame indexed by TimeStamp.

    Args:
        resample: Optional pandas offset alias (e.g., '5T'). If provided, data is
                  resampled by median after filtering columns and date range.
    """
    # .../model_testings
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / 'data' / 'combined_data_mill6.csv'

    # Read CSV with TimeStamp as DateTimeIndex
    df = pd.read_csv(file_path, parse_dates=['TimeStamp'], index_col='TimeStamp')

    # Keep only the specified parameters that exist in the file
    keep_cols = [c for c in parameters if c in df.columns]
    df = df[keep_cols].copy()

    # Date range filter (mirror df_analysis-xgb2)
    start_date = pd.Timestamp('2025-06-15 06:00')
    end_date = pd.Timestamp('2025-08-24 22:00')
    df = df.loc[start_date:end_date].copy()

    # Optional resampling to smooth noise and speed up algorithms
    if resample:
        # Use median to be robust to outliers; keep timestamp index
        df = df.resample(resample).median().dropna(how='all')

    # Apply basic physical constraints (same as df_analysis-xgb2)
    df = df[
        (df['Ore'].between(150, 200)) &
        (df['PSI200'].between(15, 35)) &
        (df['WaterMill'].between(5, 20)) &
        (df['WaterZumpf'].between(140, 250)) &
        (df['DensityHC'].between(1500, 1800)) &
        (df['MotorAmp'].between(170, 220))
    ].copy()

    return df

#%%
df = load_data()
print(df.head())

# %%

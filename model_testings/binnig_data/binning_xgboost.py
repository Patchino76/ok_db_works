#%% Imports and configuration
import pandas as pd
import numpy as np
from pathlib import Path
from raw_idea import smart_binning, analyze_binning_quality

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
     'PSI80',
     'PSI200',
     'Class_15',
     'Class_12',
     'Grano',
     'Daiki',
     'Shisti',
 ]

#%% Data loading (similar to df_analysis-xgb2)
def load_data():
    """
    Load combined mill data, subset to key parameters, apply date filter and
    basic physical constraints, and return a cleaned DataFrame indexed by TimeStamp.
    """
    # .../model_testings
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / 'data' / 'combined_data_mill8.csv'

    # Read CSV with TimeStamp as DateTimeIndex
    df = pd.read_csv(file_path, parse_dates=['TimeStamp'], index_col='TimeStamp')

    # Keep only the specified parameters that exist in the file
    keep_cols = [c for c in parameters if c in df.columns]
    df = df[keep_cols].copy()

    # Date range filter (mirror df_analysis-xgb2)
    start_date = pd.Timestamp('2025-06-15 06:00')
    end_date = pd.Timestamp('2025-08-18 22:00')
    df = df.loc[start_date:end_date].copy()

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

# if __name__ == "__main__":
#     df = load_data()

#     print("DataFrame Info:")
#     df.info()

#     print("\nFirst 2 rows of the dataframe:")
#     print(df.head(2))

#     print("\nLast 2 rows of the dataframe:")
#     print(df.tail(2))

# %%

df = load_data()
binned_df, bin_info = smart_binning(df, n_bins=40, method='kmeans', target_col='PSI80')

# Analyze binning quality
analyze_binning_quality(df, binned_df, bin_info)

# Save binning information
bin_info_df = pd.DataFrame.from_dict(bin_info, orient='index')
bin_info_df.to_csv('bin_info.csv')

# Save binned data
binned_df.to_csv('binned_data.csv')

# Save original data
df.to_csv('original_data.csv')

print("\nBinning completed successfully.")

# %%

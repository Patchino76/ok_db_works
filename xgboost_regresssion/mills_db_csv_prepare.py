import pandas as pd
from sqlalchemy import create_engine
from matplotlib import pyplot as plt

pg_params = {
    'host': 'em-m-db4.ellatzite-med.com',
    'port': 5432,
    'dbname': 'em_pulse_data',
    'user': 's.lyubenov',
    'password': 'tP9uB7sH7mK6zA7t'
}

engine = create_engine(f"postgresql://{pg_params['user']}:{pg_params['password']}@{pg_params['host']}:{pg_params['port']}/{pg_params['dbname']}")

def load_tables_to_dataframes():
    df_mill_06 = pd.read_sql_table('MILL_06', con=engine, schema='mills', index_col='TimeStamp')
    print(f"Loaded mill_06 table with {len(df_mill_06)} rows")
    
    df_mill_07 = pd.read_sql_table('MILL_07', con=engine, schema='mills', index_col='TimeStamp')
    print(f"Loaded mill_07 table with {len(df_mill_07)} rows")
    
    df_mill_08 = pd.read_sql_table('MILL_08', con=engine, schema='mills', index_col='TimeStamp')
    print(f"Loaded mill_08 table with {len(df_mill_08)} rows")
    
    df_ore_quality = pd.read_sql_table('ore_quality', con=engine, schema='mills')
    
    print(f"Loaded ore_quality table with {len(df_ore_quality)} rows from mills.ore_quality")
    
    return df_mill_06, df_mill_07, df_mill_08, df_ore_quality


def process_dataframe(df, start_date, end_date, resample_freq='1min'):
    df_processed = df.copy()
    
    if not isinstance(df_processed.index, pd.DatetimeIndex):
        if 'TimeStamp' in df_processed.columns:
            df_processed.set_index('TimeStamp', inplace=True)
    
    print(f"DataFrame date range: {df_processed.index.min()} to {df_processed.index.max()}")
    
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    
    if df_processed.index.tz is not None:
        df_processed.index = df_processed.index.tz_localize(None)
    
    df_processed = df_processed[(df_processed.index >= start_date) & (df_processed.index <= end_date)]
    print(f"Filtered DataFrame from {len(df)} to {len(df_processed)} rows")
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        # Resample and interpolate in one step for smoother transitions
        df_processed = df_processed[numeric_cols].resample(resample_freq).mean().interpolate(method='linear')
        print(f"Resampled DataFrame to {resample_freq} frequency with interpolation, resulting in {len(df_processed)} rows")
    
    df_processed = df_processed.interpolate().ffill().bfill()
    
    # Apply smoothing using rolling window
    # Window size of 5 is a good starting point, can be adjusted as needed
    window_size = 15
    df_processed = df_processed.rolling(window=window_size, min_periods=1, center=True).mean()
    print(f"Applied smoothing with rolling window of size {window_size}")
    
    return df_processed


def join_dataframes_on_timestamp(df1, df2):

    # Verify both dataframes have the same index
    if not df1.index.equals(df2.index):
        raise ValueError("Dataframes must have identical timestamp indices")
    
    # Join the dataframes on their indices
    joined_df = df1.join(df2)
    
    return joined_df

def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.show()

def main():
    # Load tables
    df_mill_06, df_mill_07, df_mill_08, df_ore_quality = load_tables_to_dataframes()
    
    # Check if all tables were loaded successfully
    if df_mill_06 is not None and df_mill_07 is not None and df_mill_08 is not None and df_ore_quality is not None:
        # Print shape of loaded DataFrames
        print(f"\nDataFrame shapes before processing:")
        print(f"mill_06: {df_mill_06.shape}")
        print(f"mill_07: {df_mill_07.shape}")
        print(f"mill_08: {df_mill_08.shape}")
        print(f"ore_quality: {df_ore_quality.shape}")
        
        # Get the current time for end date
        start_time = pd.Timestamp('2025-05-31 06:00:00').tz_localize(None)
        current_time = pd.Timestamp('2025-06-29 06:00:00').tz_localize(None)
        
        print(f"Using date range: {start_time} to {current_time}")
        
        # Process DataFrames
        print("\nProcessing mill_06...")
        processed_mill_06 = process_dataframe(
            df_mill_06, 
            start_date=start_time, 
            end_date=current_time, 
            resample_freq='1min'
        )
        
        print("\nProcessing mill_07...")
        processed_mill_07 = process_dataframe(
            df_mill_07,
            start_date=start_time,
            end_date=current_time,
            resample_freq='1min'
        )
        
        print("\nProcessing mill_08...")
        processed_mill_08 = process_dataframe(
            df_mill_08,
            start_date=start_time,
            end_date=current_time,
            resample_freq='1min'
        )
        
        print("\nProcessing ore_quality...")
        processed_ore_quality = process_dataframe(
            df_ore_quality,
            start_date=start_time,
            end_date=current_time,
            resample_freq='1min'  # Using a different frequency for ore_quality as it might have a different time granularity
        )
                
        # Print shape of processed DataFrames before alignment
        print(f"\nDataFrame shapes after initial processing:")
        if processed_mill_06 is not None:
            print(f"processed_mill_06: {processed_mill_06.shape}")
        if processed_mill_07 is not None:
            print(f"processed_mill_07: {processed_mill_07.shape}")
        if processed_mill_08 is not None:
            print(f"processed_mill_08: {processed_mill_08.shape}")
        if processed_ore_quality is not None:
            print(f"processed_ore_quality: {processed_ore_quality.shape}")
        
        # Ensure all DataFrames have the same length and aligned indices
        print("\nAligning DataFrames to ensure consistent indices...")
        # aligned_dfs = ensure_consistent_dataframes(
        #     [processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality],
        #     start_time,
        #     current_time,
        #     '1min'
        # )
        
        # processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality = aligned_dfs
        
        # Print shape of final aligned DataFrames
        print(f"\nDataFrame shapes after alignment:")
        if processed_mill_06 is not None:
            print(f"aligned_mill_06: {processed_mill_06.shape}")
            processed_mill_06.to_csv('processed_mill_06.csv', index=True)
        if processed_mill_07 is not None:
            print(f"aligned_mill_07: {processed_mill_07.shape}")
        if processed_mill_08 is not None:
            print(f"aligned_mill_08: {processed_mill_08.shape}")
        if processed_ore_quality is not None:
            # print(processed_ore_quality.info())
            processed_ore_quality = processed_ore_quality[['Class_12', 'Grano', 'Daiki', 'Shisti']]
            # print(processed_ore_quality.head())
            print(f"aligned_ore_quality: {processed_ore_quality.shape}")
            processed_ore_quality.to_csv('processed_ore_quality.csv', index=True)
            plot_data(processed_ore_quality['Shisti'], processed_ore_quality['Grano'])
        
        mills_and_ore_quality = join_dataframes_on_timestamp(processed_mill_06, processed_ore_quality)
        mills_and_ore_quality.to_csv('mill_ore_quality_06.csv', index=True)

        mills_and_ore_quality = join_dataframes_on_timestamp(processed_mill_07, processed_ore_quality)
        mills_and_ore_quality.to_csv('mill_ore_quality_07.csv', index=True)

        mills_and_ore_quality = join_dataframes_on_timestamp(processed_mill_08, processed_ore_quality)
        mills_and_ore_quality.to_csv('mill_ore_quality_08.csv', index=True)

if __name__ == "__main__":
    main()
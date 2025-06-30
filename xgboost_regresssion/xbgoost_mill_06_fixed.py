import pandas as pd
from sqlalchemy import create_engine

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
    
    try:
        df_ore_quality = pd.read_sql_table('ore_quality', con=engine, schema='mills')
    except:
        df_ore_quality = pd.read_sql_query("SELECT * FROM mills.ore_quality", engine)
    
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
        df_processed = df_processed[numeric_cols].resample(resample_freq).mean()
        print(f"Resampled DataFrame to {resample_freq} frequency, resulting in {len(df_processed)} rows")
    
    df_processed = df_processed.interpolate().ffill().bfill()
    
    return df_processed


def ensure_consistent_dataframes(dfs, start_date, end_date, freq='1min'):
    common_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    print(f"Created common date range with {len(common_index)} points")
    
    aligned_dfs = []
    for df in dfs:
        if df is not None and not df.empty:
            aligned_dfs.append(df.reindex(common_index))
        else:
            aligned_dfs.append(None)
    
    return aligned_dfs


def main():
    df_mill_06, df_mill_07, df_mill_08, df_ore_quality = load_tables_to_dataframes()
    
    print(f"\nDataFrame shapes before processing:")
    print(f"mill_06: {df_mill_06.shape}")
    print(f"mill_07: {df_mill_07.shape}")
    print(f"mill_08: {df_mill_08.shape}")
    print(f"ore_quality: {df_ore_quality.shape}")
    
    current_time = pd.Timestamp('2025-06-30T09:42:10+03:00').tz_localize(None)
    start_time = pd.Timestamp('2025-05-25 02:01:00').tz_localize(None)
    
    print(f"Using date range: {start_time} to {current_time}")
    
    print("\nProcessing mill_06...")
    processed_mill_06 = process_dataframe(df_mill_06, start_time, current_time)
    
    print("\nProcessing mill_07...")
    processed_mill_07 = process_dataframe(df_mill_07, start_time, current_time)
    
    print("\nProcessing mill_08...")
    processed_mill_08 = process_dataframe(df_mill_08, start_time, current_time)
    
    print("\nProcessing ore_quality...")
    if 'date' in df_ore_quality.columns:
        df_ore_quality['date'] = pd.to_datetime(df_ore_quality['date'])
        df_ore_quality.set_index('date', inplace=True)
    
    processed_ore_quality = process_dataframe(df_ore_quality, start_time, current_time)
    
    print("\nAligning DataFrames...")
    aligned_dfs = ensure_consistent_dataframes(
        [processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality],
        start_time, current_time
    )
    
    processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality = aligned_dfs
    
    print(f"\nDataFrame shapes after alignment:")
    print(f"aligned_mill_06: {processed_mill_06.shape}")
    print(f"aligned_mill_07: {processed_mill_07.shape}")
    print(f"aligned_mill_08: {processed_mill_08.shape}")
    print(f"aligned_ore_quality: {processed_ore_quality.shape}")


if __name__ == "__main__":
    main()

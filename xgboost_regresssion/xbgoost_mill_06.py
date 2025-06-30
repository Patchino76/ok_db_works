import pandas as pd
from sqlalchemy import create_engine
import os
import sys

# PostgreSQL connection parameters
pg_params = {
    'host': 'em-m-db4.ellatzite-med.com',
    'port': 5432,
    'dbname': 'em_pulse_data',
    'user': 's.lyubenov',
    'password': 'tP9uB7sH7mK6zA7t'
}

def get_db_connection():
    """Create and return a SQLAlchemy engine for PostgreSQL connection"""
    pg_host = pg_params['host']
    pg_port = pg_params['port']
    pg_dbname = pg_params['dbname']
    pg_user = pg_params['user']
    pg_password = pg_params['password']
    
    connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_dbname}"
    engine = create_engine(connection_string)
    
    return engine

def load_tables_to_dataframes():
    """Load mill_06, mill_07, mill_08 and ore_quality tables into separate DataFrames"""
    # Get database connection
    engine = get_db_connection()
    
    # Initialize DataFrames
    df_mill_06 = None
    df_mill_07 = None
    df_mill_08 = None
    df_ore_quality = None
    
    # Define schema for mill tables
    mill_schema = 'mills'
    
    # Try loading mill_06
    try:
        df_mill_06 = pd.read_sql_table('MILL_06', con=engine, schema=mill_schema, index_col='TimeStamp')
        print(f"Loaded mill_06 table with {len(df_mill_06)} rows")
    except Exception as e:
        print(f"Error loading mill_06 table: {e}")
    
    # Try loading mill_07
    try:
        df_mill_07 = pd.read_sql_table('MILL_07', con=engine, schema=mill_schema, index_col='TimeStamp')
        print(f"Loaded mill_07 table with {len(df_mill_07)} rows")
    except Exception as e:
        print(f"Error loading mill_07 table: {e}")
    
    # Try loading mill_08
    try:
        df_mill_08 = pd.read_sql_table('MILL_08', con=engine, schema=mill_schema, index_col='TimeStamp')
        print(f"Loaded mill_08 table with {len(df_mill_08)} rows")
    except Exception as e:
        print(f"Error loading mill_08 table: {e}")
    
    # Try different possible schemas for ore_quality table
    schemas_to_try = ['public', 'ore', 'mills']
    table_names_to_try = ['ore_quality', 'ORE_QUALITY']
    
    for schema in schemas_to_try:
        for table_name in table_names_to_try:
            try:
                df_ore_quality = pd.read_sql_table(table_name, con=engine, schema=schema)
                print(f"Loaded ore_quality table with {len(df_ore_quality)} rows from {schema}.{table_name}")
                break  # Break out of the inner loop if successful
            except Exception:
                continue  # Try the next combination
        if df_ore_quality is not None:
            break  # Break out of the outer loop if successful
    
    if df_ore_quality is None:
        print("Could not find ore_quality table in any of the checked schemas")
        
        # As a last resort, try using a custom SQL query to list available tables
        try:
            # Query to list all tables from all schemas
            tables_query = """
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema') 
            ORDER BY table_schema, table_name
            """
            available_tables = pd.read_sql(tables_query, con=engine)
            print("\nAvailable tables in the database:")
            print(available_tables)
            
            # Look for tables that might contain ore quality data
            ore_tables = available_tables[available_tables['table_name'].str.contains('ore', case=False)]
            if not ore_tables.empty:
                print("\nPossible ore quality tables found:")
                print(ore_tables)
                
        except Exception as e:
            print(f"Error listing available tables: {e}")
    
    return df_mill_06, df_mill_07, df_mill_08, df_ore_quality


    

def process_dataframe(df, start_date=None, end_date=None, resample_freq='1min', interpolation_method='linear'):
    """
    Process a DataFrame by filtering based on date range, cleaning NaN values through interpolation,
    and resampling to specified frequency.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process
    start_date (str or datetime, optional): Start date for filtering, e.g., '2024-01-01'
    end_date (str or datetime, optional): End date for filtering, e.g., '2024-06-30'
    resample_freq (str, optional): Frequency for resampling, default is '1min'
    interpolation_method (str, optional): Method for interpolating missing values, default is 'linear'
    
    Returns:
    pd.DataFrame: The processed DataFrame
    """
    if df is None:
        print("Warning: DataFrame is None, cannot process")
        return None
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check if the DataFrame has a DatetimeIndex
    if not isinstance(df_processed.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not a DatetimeIndex, attempting to convert")
        try:
            if 'TimeStamp' in df_processed.columns:
                df_processed.set_index('TimeStamp', inplace=True)
            else:
                print("Warning: No TimeStamp column found, cannot process this DataFrame")
                return df_processed
        except Exception as e:
            print(f"Error converting index to DatetimeIndex: {e}")
            return df_processed
            
    # Print the actual date range in the DataFrame for debugging
    if not df_processed.empty:
        print(f"DataFrame date range: {df_processed.index.min()} to {df_processed.index.max()}")
    
    # Convert start_date and end_date to pandas Timestamp objects without timezone info
    if start_date is not None:
        start_date = pd.to_datetime(start_date).tz_localize(None)
        print(f"Using start_date: {start_date}")
    if end_date is not None:
        end_date = pd.to_datetime(end_date).tz_localize(None)
        print(f"Using end_date: {end_date}")
    
    # Filter based on date range if provided
    original_row_count = len(df_processed)
    
    # Convert DataFrame index to naive datetime (no timezone) if it has timezone info
    if df_processed.index.tz is not None:
        df_processed.index = df_processed.index.tz_localize(None)
    
    # Apply the date filters
    if start_date is not None:
        df_processed = df_processed[df_processed.index >= start_date]
    if end_date is not None:
        df_processed = df_processed[df_processed.index <= end_date]
    
    if len(df_processed) < original_row_count:
        print(f"Filtered DataFrame from {original_row_count} to {len(df_processed)} rows")
    
    # Check if we still have data after filtering
    if df_processed.empty:
        print("Warning: DataFrame is empty after date filtering")
        return df_processed
    
    # Handle different column types for ore_quality table
    # First, convert object columns to numeric when possible
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            try:
                # Try to convert to numeric, coerce errors to NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                print(f"Converted column '{col}' to numeric type")
            except Exception as e:
                print(f"Could not convert column '{col}' to numeric: {e}")
    
    # Identify numeric columns that can be resampled
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = [col for col in df_processed.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        print(f"Non-numeric columns that will be excluded from resampling: {non_numeric_cols}")
    
    # Resample to the specified frequency for numeric columns only
    try:
        if numeric_cols:
            # If we have numeric columns, resample those
            df_numeric = df_processed[numeric_cols].resample(resample_freq).mean()
            print(f"Resampled DataFrame to {resample_freq} frequency, resulting in {len(df_numeric)} rows")
            
            # For non-numeric columns, we can either drop them or fill with mode/most frequent value
            if non_numeric_cols:
                # Option 1: Drop non-numeric columns
                print("Non-numeric columns are excluded from the result")
            
            df_processed = df_numeric
        else:
            print("No numeric columns found for resampling. Skipping resample step.")
    except Exception as e:
        print(f"Error resampling DataFrame: {e}")
    
    # Infer objects before interpolation (to address the FutureWarning)
    try:
        df_processed = df_processed.infer_objects(copy=False)
    except Exception as e:
        print(f"Error inferring object types: {e}")
    
    # Interpolate NaN values
    try:
        df_processed = df_processed.interpolate(method=interpolation_method)
        # Get count of remaining NaNs after interpolation
        nan_count = df_processed.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values remain after interpolation")
            # Forward fill and backward fill to handle edge NaN values that interpolation might miss
            df_processed = df_processed.ffill().bfill()
            nan_count_after = df_processed.isna().sum().sum()
            if nan_count_after > 0:
                print(f"Warning: {nan_count_after} NaN values still remain after ffill/bfill")
        else:
            print("Successfully interpolated all NaN values")
    except Exception as e:
        print(f"Error interpolating DataFrame: {e}")
    
    return df_processed

def ensure_consistent_dataframes(dfs, start_date, end_date, freq='1min'):
    """
    Ensure all DataFrames have the same index by creating a common date range and reindexing.
    
    Parameters:
    dfs (list): List of DataFrames to align
    start_date (datetime): Start date for the common date range
    end_date (datetime): End date for the common date range
    freq (str): Frequency for the date range
    
    Returns:
    list: List of aligned DataFrames with the same index
    """
    # Create a common date range
    common_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    print(f"Created common date range with {len(common_index)} points from {common_index.min()} to {common_index.max()}")
    
    aligned_dfs = []
    for df in dfs:
        if df is not None and not df.empty:
            # Reindex the DataFrame to the common index
            aligned_df = df.reindex(common_index)
            aligned_dfs.append(aligned_df)
        else:
            aligned_dfs.append(None)
    
    return aligned_dfs

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
        current_time = pd.Timestamp('2025-06-30T09:42:10+03:00').tz_localize(None)
        start_time = pd.Timestamp('2025-05-25 02:01:00').tz_localize(None)
        
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
        
        # For ore_quality, we need to check if it has a datetime index
        print("\nProcessing ore_quality...")
        if 'date' in df_ore_quality.columns:
            # If ore_quality has a 'date' column, set it as the index first
            df_ore_quality['date'] = pd.to_datetime(df_ore_quality['date'])
            df_ore_quality.set_index('date', inplace=True)
        
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
        aligned_dfs = ensure_consistent_dataframes(
            [processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality],
            start_time,
            current_time,
            '1min'
        )
        
        processed_mill_06, processed_mill_07, processed_mill_08, processed_ore_quality = aligned_dfs
        
        # Print shape of final aligned DataFrames
        print(f"\nDataFrame shapes after alignment:")
        if processed_mill_06 is not None:
            print(f"aligned_mill_06: {processed_mill_06.shape}")
        if processed_mill_07 is not None:
            print(f"aligned_mill_07: {processed_mill_07.shape}")
        if processed_mill_08 is not None:
            print(f"aligned_mill_08: {processed_mill_08.shape}")
        if processed_ore_quality is not None:
            print(f"aligned_ore_quality: {processed_ore_quality.shape}")
        
        # Now you have clean, interpolated, resampled, and aligned DataFrames ready for further analysis

if __name__ == "__main__":
    main()
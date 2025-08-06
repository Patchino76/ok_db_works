import pandas as pd
from sqlalchemy import create_engine
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

class MillsDataConnector:
    """
    Class to handle database connections and data retrieval from PostgreSQL server
    for mill and ore quality data.
    """
    
    def __init__(self, host, port, dbname, user, password):
        """
        Initialize database connection parameters
        
        Args:
            host: PostgreSQL host address
            port: PostgreSQL port number
            dbname: Database name
            user: PostgreSQL username
            password: PostgreSQL password
        """
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = None
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    def get_mill_data(self, mill_number, start_date=None, end_date=None):
        """
        Retrieve mill data from PostgreSQL for a specific mill number and date range
        
        Args:
            mill_number: Mill number (6, 7, or 8)
            start_date: Start date for data retrieval (default: None)
            end_date: End date for data retrieval (default: None)
            
        Returns:
            DataFrame with mill data
        """
        try:
            mill_table = f"MILL_{mill_number:02d}"
            
            # Build query
            query = f"SELECT * FROM mills.{mill_table}"
            
            # Add date filters if provided
            conditions = []
            if start_date:
                conditions.append(f"\"TimeStamp\" >= '{start_date}'")
            if end_date:
                conditions.append(f"\"TimeStamp\" <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Execute query
            df = pd.read_sql_query(query, self.engine, index_col='TimeStamp')
            logger.info(f"Retrieved {len(df)} rows for Mill {mill_number}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving mill data: {e}")
            raise

    def get_ore_quality(self, start_date=None, end_date=None):
        """
        Retrieve ore quality data from PostgreSQL for a specific date range
        
        Args:
            start_date: Start date for data retrieval (default: None)
            end_date: End date for data retrieval (default: None)
            
        Returns:
            DataFrame with ore quality data
        """
        try:
            # Build query
            query = "SELECT * FROM mills.ore_quality"
            
            # Add date filters if provided
            conditions = []
            if start_date:
                conditions.append(f"\"TimeStamp\" >= '{start_date}'")
            if end_date:
                conditions.append(f"\"TimeStamp\" <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Execute query
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Retrieved {len(df)} rows of ore quality data")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving ore quality data: {e}")
            raise
            
    def process_dataframe(self, df, start_date=None, end_date=None, resample_freq='1min'):
        """
        Process a dataframe for use in modeling - handles resampling, smoothing, etc.
        
        Args:
            df: Input DataFrame
            start_date: Optional start date filter
            end_date: Optional end date filter
            resample_freq: Frequency for resampling time series
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            if 'TimeStamp' in df_processed.columns:
                df_processed.set_index('TimeStamp', inplace=True)
        
        # Apply date range filtering if provided
        if start_date or end_date:
            start = pd.to_datetime(start_date).tz_localize(None) if start_date else None
            end = pd.to_datetime(end_date).tz_localize(None) if end_date else None
            
            # Remove timezone info if present
            if df_processed.index.tz is not None:
                df_processed.index = df_processed.index.tz_localize(None)
            
            # Apply filters
            if start:
                df_processed = df_processed[df_processed.index >= start]
            if end:
                df_processed = df_processed[df_processed.index <= end]
            
        # Convert object columns to numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Get only numeric columns
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Resample and interpolate
            df_processed = df_processed[numeric_cols].resample(resample_freq).mean().interpolate(method='linear')
            
            # Fill remaining NAs
            df_processed = df_processed.interpolate().ffill().bfill()
            
            # Apply smoothing using rolling window
            window_size = 15  # Same as in the notebook
            df_processed = df_processed.rolling(window=window_size, min_periods=1, center=True).mean()
            logger.info(f"Applied smoothing with rolling window of size {window_size}")
            
        return df_processed
    
    def join_dataframes_on_timestamp(self, df1, df2):
        """
        Join two dataframes on their timestamp indices
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Joined DataFrame
        """
        # Make sure both dataframes have datetime indices
        for df in [df1, df2]:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'TimeStamp' in df.columns:
                    df.set_index('TimeStamp', inplace=True)
        
        # Align indices by time
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_index]
        df2_aligned = df2.loc[common_index]
        
        # Join the dataframes
        joined_df = df1_aligned.join(df2_aligned)
        logger.info(f"Joined dataframes with {len(joined_df)} rows")
        
        return joined_df
    
    def get_combined_data(self, mill_number, start_date=None, end_date=None, resample_freq='1min'):
        """
        Get combined mill and ore quality data, processed and joined
        
        Args:
            mill_number: Mill number (6, 7, or 8)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            resample_freq: Frequency for resampling time series
            
        Returns:
            Combined DataFrame with mill and ore quality data
        """
        try:
            # Get raw data
            mill_data = self.get_mill_data(mill_number, start_date, end_date)
            ore_data = self.get_ore_quality(start_date, end_date)
            
            # Process each dataframe
            processed_mill = self.process_dataframe(mill_data, start_date, end_date, resample_freq)
            processed_ore = self.process_dataframe(ore_data, start_date, end_date, resample_freq)
            
            # Join the dataframes
            combined_data = self.join_dataframes_on_timestamp(processed_mill, processed_ore)
            logger.info(f"Combined data has {len(combined_data)} rows and {len(combined_data.columns)} columns")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            raise

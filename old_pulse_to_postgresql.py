import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pyodbc
import sys
import os
import logging
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pulse_db_transform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PulseDBTransformer:
    def __init__(self, pg_host='localhost', pg_port=5432, pg_dbname='em_pulse_data', pg_user='postgres', pg_password='postgres'):
        # SQL Server connection parameters
        self.server = '10.20.2.10'
        self.database = 'pulse'
        self.username = 'Pulse_RO'
        self.password = 'PD@T@r3@der'
        self.connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        self.engine = create_engine("mssql+pyodbc:///?odbc_connect=" + self.connection_string)
        
        # Target PostgreSQL connection parameters
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_dbname = pg_dbname
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_engine = create_engine(f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_dbname}")
        
        # Mill names and sensor tags
        # self.mills = ['Mill01', 'Mill02', 'Mill03', 'Mill04', 'Mill05', 'Mill06',
        #              'Mill07', 'Mill08', 'Mill09', 'Mill10', 'Mill11', 'Mill12']
        self.mills = ['Mill06', 'Mill07', 'Mill08']
        
        # SQL tags dictionary from SQL_Data_Pulse_9.py
        self.sql_tags = {
            'Ore': {"485" : "Mill01", "488" : "Mill02", "491" : "Mill03", "494" : "Mill04", "497" : "Mill05", "500" : "Mill06",
                    "455" : "Mill07", "467" : "Mill08", "476" : "Mill09", "479" : "Mill10", "482" : "Mill11", "3786" : "Mill12"},
            
            'WaterMill': {"560" : "Mill01", "580" : "Mill02", "530" : "Mill03", "1227" : "Mill04", "1237" : "Mill05", "1219" : "Mill06",
                          "461" : "Mill07", "474" : "Mill08", "509" : "Mill09", "517" : "Mill10", "522" : "Mill11", "3790" : "Mill12"},
            
            'WaterZumpf': {"561" : "Mill01", "581" : "Mill02", "531" : "Mill03", "1228" : "Mill04", "1238" : "Mill05", "1220" : "Mill06",
                            "462" : "Mill07", "475" : "Mill08", "510" : "Mill09", "518" : "Mill10", "523" : "Mill11", "3792" : "Mill12"},
            
            'Power': {"487" : "Mill01", "490" : "Mill02", "493" : "Mill03", "496" : "Mill04", "499" : "Mill05", "502" : "Mill06",
                            "460" : "Mill07", "471" : "Mill08", "478" : "Mill09", "481" : "Mill10", "483" : "Mill11", "3773" : "Mill12"},
            
            'ZumpfLevel': {"486" : "Mill01", "489" : "Mill02", "492" : "Mill03", "495" : "Mill04", "498" : "Mill05", "501" : "Mill06",
                            "458" : "Mill07", "470" : "Mill08", "477" : "Mill09", "480" : "Mill10", "484" : "Mill11", "3747" : "Mill12"},
            
            'PressureHC': {"558" : "Mill01", "578" : "Mill02", "528" : "Mill03", "1225" : "Mill04", "1235" : "Mill05", "1217" : "Mill06",
                            "459" : "Mill07", "472" : "Mill08", "507" : "Mill09", "515" : "Mill10", "2687" : "Mill11", "3774" : "Mill12"},
            
            'DensityHC': {"557" : "Mill01", "577" : "Mill02", "527" : "Mill03", "1224" : "Mill04", "1234" : "Mill05", "1216" : "Mill06",
                            "457" : "Mill07", "469" : "Mill08", "506" : "Mill09", "514" : "Mill10", "2658" : "Mill11", "3742" : "Mill12"},
            
            # 'PulpHC_old': {"559" : "Mill01", "579" : "Mill02", "529" : "Mill03", "1226" : "Mill04", "1236" : "Mill05", "1218" : "Mill06",
            #                 "3640" : "Mill07", "1000" : "Mill08", "508" : "Mill09", "516" : "Mill10", "2691" : "Mill11", "3788" : "Mill12"},
            'PulpHC': {"2386" : "Mill01", "2160" : "Mill02", "2609" : "Mill03", "2726" : "Mill04", "2217" : "Mill05", "2777" : "Mill06",
                            "2837" : "Mill07", "1833" : "Mill08", "2485" : "Mill09", "2434" : "Mill10", "2670" : "Mill11", "3755" : "Mill12"},

            'PumpRPM': {"2405" : "Mill01", "2198" : "Mill02", "2629" : "Mill03", "2745" : "Mill04", "1652" : "Mill05", "2796" : "Mill06",
                            "2856" : "Mill07", "1800" : "Mill08", "2471" : "Mill09", "2452" : "Mill10", "2690" : "Mill11", "3780" : "Mill12"},

            'MotorAmp': {"2379" : "Mill01", "2153" : "Mill02", "2602" : "Mill03", "2719" : "Mill04", "2210" : "Mill05", "2770" : "Mill06",
                            "2830" : "Mill07", "1805" : "Mill08", "2478" : "Mill09", "2427" : "Mill10", "2663" : "Mill11", "3748" : "Mill12"},

            'PSI80': {"2379" : "Mill01", "2153" : "Mill02", "2602" : "Mill03", "2719" : "Mill04", "2210" : "Mill05", "5329" : "Mill06",
                            "5331" : "Mill07", "5332" : "Mill08", "2478" : "Mill09", "2427" : "Mill10", "2663" : "Mill11", "3748" : "Mill12"},
            
            'PSI200': {"2379" : "Mill01", "2153" : "Mill02", "2602" : "Mill03", "2719" : "Mill04", "2210" : "Mill05", "5328" : "Mill06",
                            "5330" : "Mill07", "5333" : "Mill08", "2478" : "Mill09", "2427" : "Mill10", "2663" : "Mill11", "3748" : "Mill12"},

            # 'CIDRA200': {"2875" : "Mill01", "2876" : "Mill02", "2877" : "Mill03", "2878" : "Mill04", "2879" : "Mill05", "2880" : "Mill06",
            #                 "2881" : "Mill07", "2882" : "Mill08", "2883" : "Mill09", "2884" : "Mill10", "2885" : "Mill11", "3775" : "Mill12"}
        }
        
        # Table names from SQL Server
        self.table_names = ['LoggerValues', 'LoggerValues_Archive_Jun2025']

        # self.table_names = ['LoggerValues', 
        #         'LoggerValues_Archive_Jan2025', 'LoggerValues_Archive_Dec2024', 
        #         'LoggerValues_Archive_Nov2024', 'LoggerValues_Archive_Oct2024',
        #         'LoggerValues_Archive_Sep2024', 'LoggerValues_Archive_Aug2024',
        #         'LoggerValues_Archive_Jul2024', 'LoggerValues_Archive_Jun2024',
        #         'LoggerValues_Archive_May2024', 'LoggerValues_Archive_Apr2024',  
        #         'LoggerValues_Archive_Mar2024', 'LoggerValues_Archive_Feb2024',
        #         'LoggerValues_Archive_Jan2024',
        #         ]
        
        # Initialize timestamp filter to None (get all data)
        self.filter_timestamp = None
    
    def read_sql_table(self, table_name, feature):
        """Read data from SQL Server for a specific feature"""
        tags = "LoggerTagID = " + " OR LoggerTagID = ".join(self.sql_tags[feature].keys())
        
        # Add timestamp filter if we're in append mode
        timestamp_filter = ""
        if hasattr(self, 'filter_timestamp') and self.filter_timestamp is not None:
            since_str = self.filter_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            timestamp_filter = f" AND IndexTime > '{since_str}'"
            logger.info(f"  - Filtering {table_name}.{feature} data after: {since_str}")
        
        query_str = f'SELECT IndexTime, LoggerTagID, Value FROM {table_name} WHERE {tags}{timestamp_filter} ORDER BY IndexTime DESC'
        logger.debug(f"  - SQL Query: {query_str}")
        
        try:
            query = pd.read_sql_query(query_str, self.engine)
        except Exception as e:
            logger.error(f"  - Error executing SQL query for {table_name}.{feature}: {e}")
            return pd.DataFrame()
        
        # If no data found, return empty dataframe
        if query.empty:
            logger.info(f"  - No data found in {table_name} for {feature} with the current filter")
            return pd.DataFrame()
        else:
            logger.info(f"  - Found {len(query)} rows in {table_name} for {feature}")
        
        try:
            # Process the data
            query = query.drop_duplicates(subset='IndexTime', keep='last')
            df = query.pivot(index="IndexTime", columns="LoggerTagID", values="Value")
            df = df.ffill().bfill()  # Using newer pandas methods
            df = df.resample("1min").mean()
            
            # Rename columns to mill names
            df.columns = [self.sql_tags[feature][str(k)] for k in df.columns if str(k) in self.sql_tags[feature]]
            df.index.names = ['TimeStamp']
            df.sort_index(axis=1, inplace=True)
            
            logger.debug(f"  - Processed {len(df)} rows for {table_name}.{feature} after resampling")
            return df
        except Exception as e:
            logger.error(f"  - Error processing data for {table_name}.{feature}: {e}")
            return pd.DataFrame()

    def compose_feature(self, feature):
        """Combine data from all tables for a specific feature"""
        frames = []
        for tbl in self.table_names:
            logger.info(f"Processing {tbl} for {feature}")
            frame = self.read_sql_table(tbl, feature)
            if not frame.empty:
                frames.append(frame)
        
        if not frames:
            logger.warning(f"No data found for feature {feature} across all tables")
            return pd.DataFrame()
        
        try:
            df = pd.concat(frames)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicate timestamps
            df = df.shift(2, freq='h')  # Apply 2-hour shift
            df = df.ffill().bfill()  # Using newer pandas methods
            logger.info(f"Composed feature {feature}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"Error composing feature {feature}: {e}")
            return pd.DataFrame()

    def create_mill_dataframe(self, mill):
        """Create a dataframe for a specific mill with all features"""
        logger.info(f"Creating dataframe for {mill}")
        all_data = []
        common_index = None
        
        for feature in self.sql_tags.keys():
            logger.info(f"Processing {feature} for {mill}")
            feature_df = self.compose_feature(feature)
            if not feature_df.empty and mill in feature_df.columns:
                feature_series = feature_df[mill]
                if common_index is None:
                    common_index = feature_series.index
                    logger.info(f"Initial index for {mill}: {len(common_index)} timestamps from {common_index.min()} to {common_index.max()}")
                else:
                    old_len = len(common_index)
                    common_index = common_index.intersection(feature_series.index)
                    logger.info(f"Index intersection for {mill}.{feature}: {old_len} -> {len(common_index)} timestamps")
                all_data.append((feature, feature_series))
            else:
                logger.warning(f"No data found for {feature} in {mill}")
        
        if not all_data:
            logger.warning(f"No features found for {mill}")
            return pd.DataFrame()
        
        # Create dataframe with aligned index
        mill_df = pd.DataFrame(index=common_index)
        for feature, series in all_data:
            mill_df[feature] = series.reindex(common_index)
        
        logger.info(f"Created dataframe for {mill}: {len(mill_df)} rows, {len(mill_df.columns)} features")
        if not mill_df.empty:
            logger.info(f"  - Time range: {mill_df.index.min()} to {mill_df.index.max()}")
        
        return mill_df
    
    def save_to_postgresql(self, schema='mills'):
        """Save all mill data to PostgreSQL database, replacing existing data"""
        logger.info("=== STARTING FULL REPLACE OPERATION TO POSTGRESQL ===")
        
        # Create schema if it doesn't exist
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
                conn.commit()
            logger.info(f"Schema '{schema}' verified/created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating schema: {e}")
            return
        
        replace_summary = {}
        
        for mill in self.mills:
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING {mill} FOR FULL REPLACE")
            logger.info(f"{'='*50}")
            
            try:
                mill_df = self.create_mill_dataframe(mill)
                
                # Convert mill name to table name (e.g., Mill01 -> MILL_01)
                table_name = f"MILL_{mill[4:].zfill(2)}"
                
                # Save to PostgreSQL
                if not mill_df.empty:
                    logger.info(f"ЁЯФД REPLACE MODE: Replacing all data in {schema}.{table_name}")
                    
                    mill_df.to_sql(name=table_name, schema=schema, con=self.pg_engine, 
                                   if_exists='replace', index=True, index_label='TimeStamp')
                    
                    logger.info(f"тЬЕ REPLACED {schema}.{table_name} with {len(mill_df)} rows")
                    logger.info(f"  - Time range: {mill_df.index.min()} to {mill_df.index.max()}")
                    replace_summary[mill] = {'action': 'replace', 'rows': len(mill_df)}
                else:
                    logger.warning(f"тЪая╕П  No data to save for {mill}")
                    replace_summary[mill] = {'action': 'no_data', 'rows': 0}
                    
            except Exception as e:
                logger.error(f"тЭМ Error saving {mill} to PostgreSQL: {e}")
                replace_summary[mill] = {'action': 'error', 'error': str(e)}
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("FULL REPLACE OPERATION SUMMARY")
        logger.info(f"{'='*60}")
        
        for mill, summary in replace_summary.items():
            action = summary['action']
            if action == 'replace':
                logger.info(f"тЬЕ {mill}: REPLACED with {summary['rows']} rows")
            elif action == 'no_data':
                logger.warning(f"тЪая╕П  {mill}: No data available")
            elif action == 'error':
                logger.error(f"тЭМ {mill}: Error - {summary['error']}")
        
        logger.info("=== FULL REPLACE OPERATION COMPLETED ===")

    def append_to_postgresql(self, schema='mills'):
        """Append new data to existing PostgreSQL database tables, only adding records newer than the latest timestamp"""
        logger.info("=== STARTING APPEND OPERATION TO POSTGRESQL ===")
        
        # Create schema if it doesn't exist
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
                conn.commit()
            logger.info(f"Schema '{schema}' verified/created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating schema: {e}")
            return
        
        # Store original table_names to restore later
        original_table_names = self.table_names.copy()
        append_summary = {}
        
        for mill in self.mills:
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING {mill} FOR APPEND OPERATION")
            logger.info(f"{'='*50}")
            
            # Convert mill name to table name (e.g., Mill01 -> MILL_01)
            table_name = f"MILL_{mill[4:].zfill(2)}"
            
            try:
                # Check if the table exists
                with self.pg_engine.connect() as conn:
                    exists_query = text(f"""
                        SELECT EXISTS (
                            SELECT FROM pg_tables
                            WHERE schemaname = '{schema}'
                            AND tablename = '{table_name}'
                        );
                    """)
                    result = conn.execute(exists_query).fetchone()
                    table_exists = result[0] if result else False
                
                if table_exists:
                    logger.info(f"тЬУ Table {schema}.{table_name} exists - checking for latest timestamp")
                    
                    # Get the latest timestamp and row count from the existing table
                    with self.pg_engine.connect() as conn:
                        query = text(f"""
                            SELECT MAX("TimeStamp") as max_ts, COUNT(*) as row_count 
                            FROM {schema}."{table_name}";
                        """)
                        result = conn.execute(query).fetchone()
                        latest_timestamp = result[0] if result and result[0] else None
                        existing_row_count = result[1] if result else 0
                    
                    logger.info(f"  - Existing rows in table: {existing_row_count}")
                    
                    if latest_timestamp:
                        logger.info(f"  - Latest timestamp in table: {latest_timestamp}")
                        
                        # Parse the timestamp string to datetime object if needed
                        if not isinstance(latest_timestamp, datetime):
                            latest_dt = pd.to_datetime(latest_timestamp)
                        else:
                            latest_dt = latest_timestamp
                        
                        # Set filter timestamp with buffer for the 2-hour shift
                        # The buffer ensures we capture data that might be shifted into our time range
                        buffer_hours = 2.5  # Slightly larger buffer to be safe
                        self.filter_timestamp = latest_dt - pd.Timedelta(hours=buffer_hours)
                        logger.info(f"  - Set filter_timestamp to: {self.filter_timestamp} (buffer: {buffer_hours}h)")
                        
                        logger.info(f"ЁЯФД APPEND MODE: Fetching data newer than {latest_dt}")
                        
                        # Get mill dataframe with filtering
                        mill_df = self.create_mill_dataframe(mill)
                        
                        if not mill_df.empty:
                            logger.info(f"  - Retrieved {len(mill_df)} rows from source (after filtering)")
                            logger.info(f"  - Source time range: {mill_df.index.min()} to {mill_df.index.max()}")
                            
                            # Apply final filter to only include timestamps after the latest timestamp
                            new_data = mill_df[mill_df.index > latest_dt]
                            logger.info(f"  - After final timestamp filter: {len(new_data)} new rows")
                            
                            if not new_data.empty:
                                logger.info(f"  - New data time range: {new_data.index.min()} to {new_data.index.max()}")
                                
                                # Append to PostgreSQL table
                                new_data.to_sql(name=table_name, schema=schema, con=self.pg_engine, 
                                               if_exists='append', index=True, index_label='TimeStamp')
                                
                                logger.info(f"тЬЕ SUCCESSFULLY APPENDED {len(new_data)} new rows to {schema}.{table_name}")
                                append_summary[mill] = {'action': 'append', 'rows': len(new_data), 'new_total': existing_row_count + len(new_data)}
                            else:
                                logger.info(f"тД╣я╕П  No new data to append for {mill} - all data is older than {latest_dt}")
                                append_summary[mill] = {'action': 'no_new_data', 'rows': 0, 'existing_rows': existing_row_count}
                        else:
                            logger.warning(f"тЪая╕П  No data retrieved from source for {mill}")
                            append_summary[mill] = {'action': 'no_source_data', 'rows': 0, 'existing_rows': existing_row_count}
                    else:
                        logger.warning(f"тЪая╕П  Table {schema}.{table_name} exists but has no data - recreating")
                        self.filter_timestamp = None
                        
                        logger.info(f"ЁЯФД RECREATE MODE: Creating table from scratch")
                        mill_df = self.create_mill_dataframe(mill)
                        
                        if not mill_df.empty:
                            mill_df.to_sql(name=table_name, schema=schema, con=self.pg_engine, 
                                          if_exists='replace', index=True, index_label='TimeStamp')
                            logger.info(f"тЬЕ RECREATED table {schema}.{table_name} with {len(mill_df)} rows")
                            append_summary[mill] = {'action': 'recreate_empty', 'rows': len(mill_df), 'new_total': len(mill_df)}
                        else:
                            logger.error(f"тЭМ No data found for {mill} - table remains empty")
                            append_summary[mill] = {'action': 'no_data', 'rows': 0, 'new_total': 0}
                else:
                    logger.info(f"тД╣я╕П  Table {schema}.{table_name} does not exist - creating new table")
                    self.filter_timestamp = None
                    
                    logger.info(f"ЁЯФД CREATE MODE: Creating new table from scratch")
                    mill_df = self.create_mill_dataframe(mill)
                    
                    if not mill_df.empty:
                        mill_df.to_sql(name=table_name, schema=schema, con=self.pg_engine, 
                                      if_exists='replace', index=True, index_label='TimeStamp')
                        logger.info(f"тЬЕ CREATED new table {schema}.{table_name} with {len(mill_df)} rows")
                        logger.info(f"  - Time range: {mill_df.index.min()} to {mill_df.index.max()}")
                        append_summary[mill] = {'action': 'create_new', 'rows': len(mill_df), 'new_total': len(mill_df)}
                    else:
                        logger.error(f"тЭМ No data found for {mill} - cannot create table")
                        append_summary[mill] = {'action': 'no_data', 'rows': 0, 'new_total': 0}
                        
            except Exception as e:
                logger.error(f"тЭМ Error processing {mill}: {e}")
                append_summary[mill] = {'action': 'error', 'error': str(e)}
            
            # Reset filter for next mill
            self.filter_timestamp = None
        
        # Restore original state
        self.table_names = original_table_names
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("APPEND OPERATION SUMMARY")
        logger.info(f"{'='*60}")
        
        for mill, summary in append_summary.items():
            action = summary['action']
            if action == 'append':
                logger.info(f"тЬЕ {mill}: APPENDED {summary['rows']} rows (total: {summary['new_total']})")
            elif action == 'create_new':
                logger.info(f"ЁЯЖХ {mill}: CREATED with {summary['rows']} rows")
            elif action == 'recreate_empty':
                logger.info(f"ЁЯФД {mill}: RECREATED with {summary['rows']} rows")
            elif action == 'no_new_data':
                logger.info(f"тД╣я╕П  {mill}: No new data (existing: {summary['existing_rows']} rows)")
            elif action == 'no_source_data':
                logger.warning(f"тЪая╕П  {mill}: No source data available")
            elif action == 'no_data':
                logger.error(f"тЭМ {mill}: No data found")
            elif action == 'error':
                logger.error(f"тЭМ {mill}: Error - {summary['error']}")
        
        logger.info("=== APPEND OPERATION COMPLETED ===")

def main():
    # PostgreSQL connection parameters
    pg_host = 'em-m-db4.ellatzite-med.com'  # PostgreSQL server host
    pg_port = 5432                           # PostgreSQL server port
    pg_dbname = 'em_pulse_data'              # PostgreSQL database name
    pg_user = 's.lyubenov'                   # PostgreSQL username
    pg_password = 'tP9uB7sH7mK6zA7t'         # PostgreSQL password
    
    # Initialize the transformer with PostgreSQL connection parameters
    transformer = PulseDBTransformer(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password
    )
    
    # Save or append based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == 'append':
        transformer.append_to_postgresql()
    else:
        transformer.save_to_postgresql()

if __name__ == "__main__":
    main()

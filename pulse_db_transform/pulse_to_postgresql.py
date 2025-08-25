import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import pyodbc
# import  
import sys
import os
from sqlalchemy.exc import SQLAlchemyError

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
        # Now mapping: feature -> { MillName: TagID }
        self.sql_tags = {
            'Ore': {"Mill01": "485", "Mill02": "488", "Mill03": "491", "Mill04": "494", "Mill05": "497", "Mill06": "500",
                    "Mill07": "455", "Mill08": "467", "Mill09": "476", "Mill10": "479", "Mill11": "482", "Mill12": "3786"},
            
            'WaterMill': {"Mill01": "560", "Mill02": "580", "Mill03": "530", "Mill04": "1227", "Mill05": "1237", "Mill06": "1219",
                          "Mill07": "461", "Mill08": "474", "Mill09": "509", "Mill10": "517", "Mill11": "522", "Mill12": "3790"},
            
            'WaterZumpf': {"Mill01": "561", "Mill02": "581", "Mill03": "531", "Mill04": "1228", "Mill05": "1238", "Mill06": "1220",
                            "Mill07": "462", "Mill08": "475", "Mill09": "510", "Mill10": "518", "Mill11": "523", "Mill12": "3792"},
            
            'Power': {"Mill01": "487", "Mill02": "490", "Mill03": "493", "Mill04": "496", "Mill05": "499", "Mill06": "502",
                            "Mill07": "460", "Mill08": "471", "Mill09": "478", "Mill10": "481", "Mill11": "483", "Mill12": "3773"},
            
            'ZumpfLevel': {"Mill01": "486", "Mill02": "489", "Mill03": "492", "Mill04": "495", "Mill05": "498", "Mill06": "501",
                            "Mill07": "458", "Mill08": "470", "Mill09": "477", "Mill10": "480", "Mill11": "484", "Mill12": "3747"},
            
            'PressureHC': {"Mill01": "558", "Mill02": "578", "Mill03": "528", "Mill04": "1225", "Mill05": "1235", "Mill06": "1217",
                            "Mill07": "459", "Mill08": "472", "Mill09": "507", "Mill10": "515", "Mill11": "2687", "Mill12": "3774"},
            
            'DensityHC': {"Mill01": "557", "Mill02": "577", "Mill03": "527", "Mill04": "1224", "Mill05": "1234", "Mill06": "1216",
                            "Mill07": "457", "Mill08": "469", "Mill09": "506", "Mill10": "514", "Mill11": "2658", "Mill12": "3742"},
            
            'FE': {"Mill01": "1762", "Mill02": "1762", "Mill03": "1762", "Mill04": "1762", "Mill05": "1762", "Mill06": "1762",
                            "Mill07": "1762", "Mill08": "1762", "Mill09": "1762", "Mill10": "1762", "Mill11": "1762", "Mill12": "1762"},

            'PulpHC': {"Mill01": "2386", "Mill02": "2160", "Mill03": "2609", "Mill04": "2726", "Mill05": "2217", "Mill06": "2777",
                            "Mill07": "2837", "Mill08": "1833", "Mill09": "2485", "Mill10": "2434", "Mill11": "2670", "Mill12": "3755"},

            'PumpRPM': {"Mill01": "2405", "Mill02": "2198", "Mill03": "2629", "Mill04": "2745", "Mill05": "1652", "Mill06": "2796",
                            "Mill07": "2856", "Mill08": "1800", "Mill09": "2471", "Mill10": "2452", "Mill11": "2690", "Mill12": "3780"},

            'MotorAmp': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "2770",
                            "Mill07": "2830", "Mill08": "1805", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},

            'PSI80': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "5329",
                            "Mill07": "5331", "Mill08": "5332", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},
            
            'PSI200': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "5328",
                            "Mill07": "5330", "Mill08": "5333", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},

            # 'CIDRA200': {"Mill01": "2875", "Mill02": "2876", "Mill03": "2877", "Mill04": "2878", "Mill05": "2879", "Mill06": "2880",
            #                 "Mill07": "2881", "Mill08": "2882", "Mill09": "2883", "Mill10": "2884", "Mill11": "2885", "Mill12": "3775"}
        }
        
        # Table names from SQL Server - only tables needed for June 2025 onwards
        self.table_names = ['LoggerValues', 'LoggerValues_Archive_Jul2025', 'LoggerValues_Archive_Jun2025']

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
        # Flag to indicate if we're creating a new table (removes time limitations)
        self.creating_new_table = False
    
    def read_sql_table(self, table_name, feature):
        """Read data from SQL Server for a specific feature"""
        # Get current time in UTC
        current_time = datetime.utcnow()
        
        # Get the tag IDs we're interested in (now values())
        tag_ids = list(self.sql_tags[feature].values())
        tag_conditions = " OR ".join([f"LoggerTagID = {tag_id}" for tag_id in tag_ids])
        
        # Build the WHERE clause
        where_clauses = [f"({tag_conditions})"]
        
        # Only apply time filters if we're not creating a new table
        if not self.creating_new_table:
            # Add timestamp filter if we're in append mode
            if hasattr(self, 'filter_timestamp') and self.filter_timestamp is not None:
                since_str = self.filter_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                where_clauses.append(f"IndexTime > '{since_str}'")
                print(f"  - Filtering data after: {since_str}")
            else:
                # If not in append mode and not creating new table, limit to last 7 days instead of 24 hours
                # This helps capture more data during recovery operations
                default_start_time = (current_time - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                where_clauses.append(f"IndexTime >= '{default_start_time}'")
                print(f"  - Limiting to last 7 days (since {default_start_time})")
        else:
            print(f"  - Creating new table: fetching ALL available data from {table_name}")
        
        # Build the final query - ORDER BY ASC to get chronological data
        # Remove TOP limit when creating new tables to ensure all data is captured
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        if self.creating_new_table:
            # No limit for new tables to ensure complete data capture
            query_str = f"""
            SELECT IndexTime, LoggerTagID, Value 
            FROM {table_name} 
            WHERE {where_clause}
            ORDER BY IndexTime ASC
            """
        else:
            # For updates, still use a reasonable limit but order chronologically
            query_str = f"""
            SELECT TOP 500000 IndexTime, LoggerTagID, Value 
            FROM {table_name} 
            WHERE {where_clause}
            ORDER BY IndexTime ASC
            """
        
        # Log the query for debugging (without the actual values for security)
        print(f"  - Executing query on {table_name} for {feature}")
        # print(f"  - SQL Query: {query_str}")
        
        query = pd.read_sql_query(query_str, self.engine)
        
        # If no data found, return empty dataframe
        if query.empty:
            print(f"  - No data found in {table_name} for {feature} with the current filter")
            return pd.DataFrame()
        else:
            print(f"  - Found {len(query)} rows in {table_name} for {feature}")
        
        # Process the data
        query = query.drop_duplicates(subset='IndexTime', keep='last')
        df = query.pivot(index="IndexTime", columns="LoggerTagID", values="Value")
        df = df.ffill().bfill()  # Using newer pandas methods
        df = df.resample("1min").mean()
        
        # Rename columns to mill names using a reverse map: tag_id -> mill
        reverse_map = {str(tag): mill for mill, tag in self.sql_tags[feature].items()}
        df.columns = [reverse_map[str(k)] for k in df.columns if str(k) in reverse_map]
        df.index.names = ['TimeStamp']
        df.sort_index(axis=1, inplace=True)
        
        return df

    def compose_feature(self, feature):
        """Combine data from all tables for a specific feature"""
        frames = []
        for tbl in self.table_names:
            print(f"Processing {tbl} for {feature}")
            frames.append(self.read_sql_table(tbl, feature))
        
        df = pd.concat(frames)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicate timestamps
        df = df.shift(2, freq='h')
        df = df.ffill().bfill()  # Using newer pandas methods
        return df

    def create_mill_dataframe(self, mill):
        """Create a dataframe for a specific mill with all features"""
        all_data = []
        common_index = None
        
        for feature in self.sql_tags.keys():
            print(f"Processing {feature} for {mill}")
            feature_df = self.compose_feature(feature)
            # FE is a global feature shared across all mills (single tag). After pivot it
            # will appear as a single column (renamed unpredictably due to duplicate tag mapping).
            # Handle FE specially: take the single series and use it for every mill.
            if feature == 'FE' and not feature_df.empty:
                feature_series = feature_df.iloc[:, 0]
                if common_index is None:
                    common_index = feature_series.index
                else:
                    common_index = common_index.intersection(feature_series.index)
                all_data.append((feature, feature_series))
                continue

            if mill in feature_df.columns:
                feature_series = feature_df[mill]
                if common_index is None:
                    common_index = feature_series.index
                else:
                    common_index = common_index.intersection(feature_series.index)
                all_data.append((feature, feature_series))
        
        # Create dataframe with aligned index
        mill_df = pd.DataFrame(index=common_index)
        for feature, series in all_data:
            mill_df[feature] = series.reindex(common_index)
        
        return mill_df
    
    def save_to_postgresql(self, schema='mills'):
        """Save all mill data to PostgreSQL database, replacing existing data"""
        print("\nStarting to save all mills to PostgreSQL...")
        
        # Set flag to indicate we're creating new tables
        self.creating_new_table = True
        
        # Create schema if it doesn't exist
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
                conn.commit()
            print(f"Schema '{schema}' created or already exists.")
        except SQLAlchemyError as e:
            print(f"Error creating schema: {e}")
            return
        
        for mill in self.mills:
            print(f"\nProcessing {mill}...")
            mill_df = self.create_mill_dataframe(mill)
            
            # Convert mill name to table name (e.g., Mill01 -> MILL_01)
            table_name = f"MILL_{mill[4:].zfill(2)}"
            
            # Save to PostgreSQL
            if not mill_df.empty:
                try:
                    mill_df.to_sql(name=table_name, schema=schema, con=self.pg_engine, 
                                    if_exists='replace', index=True, index_label='TimeStamp')
                    print(f"Saved {mill} to {schema}.{table_name} in PostgreSQL with {len(mill_df)} rows")
                except Exception as e:
                    print(f"Error saving {mill} to PostgreSQL: {e}")
            else:
                print(f"No data to save for {mill}")
        
        # Reset flag
        self.creating_new_table = False
        print("\nAll mills have been processed and saved to the PostgreSQL database")

    def append_to_postgresql(self, schema='mills'):
        """Append new data to existing PostgreSQL database tables, only adding records newer than the latest timestamp"""
        print("\n🚀 Starting data synchronization to PostgreSQL...")
        start_time = datetime.now()
        
        # Create schema if it doesn't exist
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
                conn.commit()
            print(f"✅ Schema '{schema}' created or already exists.")
        except SQLAlchemyError as e:
            print(f"❌ Error creating schema: {e}")
            return
        
        # Process each mill
        for mill in self.mills:
            mill_start = datetime.now()
            table_name = f"MILL_{mill[4:].zfill(2)}"
            print(f"\n🔍 Processing {mill} (table: {schema}.{table_name})...")
            
            try:
                # Check if table exists
                with self.pg_engine.connect() as conn:
                    exists = conn.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM pg_tables
                            WHERE schemaname = :schema 
                            AND tablename = :table
                        );
                    """), {'schema': schema, 'table': table_name}).scalar()
                
                if exists:
                    # Get latest timestamp from the table
                    with self.pg_engine.connect() as conn:
                        latest_ts = conn.execute(text(f'SELECT MAX("TimeStamp") FROM {schema}."{table_name}"')).scalar()
                    
                    if latest_ts:
                        # Convert to datetime if it's a string
                        latest_dt = pd.to_datetime(latest_ts) if not isinstance(latest_ts, datetime) else latest_ts
                        
                        # Set filter to get only new data (with buffer for timezone shifts)
                        # Increase buffer to 6 hours to handle potential gaps
                        self.filter_timestamp = latest_dt - pd.Timedelta('6 hours')
                        self.creating_new_table = False  # We're updating existing table
                        print(f"  ⏱️  Last record in DB: {latest_dt}")
                        print(f"  🔍 Fetching data after: {self.filter_timestamp} (6-hour buffer)")
                        
                        # Get new data
                        mill_df = self.create_mill_dataframe(mill)
                        
                        if not mill_df.empty:
                            # Filter to only include new records (after the latest in DB)
                            new_data = mill_df[mill_df.index > latest_dt]
                            
                            if not new_data.empty:
                                # Insert new data
                                new_data.to_sql(
                                    name=table_name, 
                                    schema=schema, 
                                    con=self.pg_engine,
                                    if_exists='append',
                                    index=True,
                                    index_label='TimeStamp',
                                    method='multi',
                                    chunksize=1000
                                )
                                duration = (datetime.now() - mill_start).total_seconds()
                                print(f"  ✅ Appended {len(new_data)} new rows in {duration:.1f} seconds")
                                print(f"  📊 New data range: {new_data.index.min()} to {new_data.index.max()}")
                            else:
                                print("  ℹ️  No new data to append")
                        else:
                            print("  ℹ️  No data returned from source")
                    else:
                        print(f"  ⚠️  Table exists but is empty - will repopulate")
                        self._repopulate_table(mill, schema, table_name)
                else:
                    print(f"  🆕 Table doesn't exist - creating with initial data")
                    self._repopulate_table(mill, schema, table_name)
                    
            except Exception as e:
                print(f"  ❌ Error processing {mill}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            mill_duration = (datetime.now() - mill_start).total_seconds()
            print(f"  ⏱️  Completed in {mill_duration:.1f} seconds")
        
        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Synchronization completed in {total_duration:.1f} seconds")
    
    def _repopulate_table(self, mill, schema, table_name):
        """Helper method to repopulate a table with fresh data"""
        print(f"  🔄 Repopulating {schema}.{table_name} with ALL available data...")
        self.filter_timestamp = None  # Get all data
        self.creating_new_table = True  # Remove time limitations
        mill_df = self.create_mill_dataframe(mill)
        
        if not mill_df.empty:
            mill_df.to_sql(
                name=table_name,
                schema=schema,
                con=self.pg_engine,
                if_exists='replace',
                index=True,
                index_label='TimeStamp',
                method='multi',
                chunksize=1000  # Process in smaller chunks for large datasets
            )
            print(f"  ✅ Created {schema}.{table_name} with {len(mill_df)} rows")
            print(f"  📅 Data range: {mill_df.index.min()} to {mill_df.index.max()}")
        else:
            print("  ⚠️  No data available to populate table")
        
        # Reset flag
        self.creating_new_table = False

def main():
    # PostgreSQL connection parameters
    pg_host = 'em-m-db4.ellatzite-med.com'  # PostgreSQL server host
    pg_port = 5432                           # PostgreSQL server port
    pg_dbname = 'em_pulse_data'              # PostgreSQL database name
    pg_user = 's.lyubenov'                   # PostgreSQL username
    pg_password = 'tP9uB7sH7mK6zA7t'         # PostgreSQL password
    
    print("🚀 Starting Pulse to PostgreSQL data synchronization...")
    print(f"Connecting to {pg_host}:{pg_port}/{pg_dbname} as {pg_user}")
    
    try:
        # Initialize the transformer with PostgreSQL connection parameters
        transformer = PulseDBTransformer(
            pg_host=pg_host,
            pg_port=pg_port,
            pg_dbname=pg_dbname,
            pg_user=pg_user,
            pg_password=pg_password
        )
        
        # Always use append mode - it will create tables if they don't exist
        transformer.append_to_postgresql()
        
    except Exception as e:
        print(f"❌ Error during synchronization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("✅ Data synchronization completed successfully!")
    return 0

if __name__ == "__main__":
    main()
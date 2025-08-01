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
        mill_count = len(self.sql_tags[feature])
        
        # Add timestamp filter if we're in append mode
        timestamp_filter = ""
        if hasattr(self, 'filter_timestamp') and self.filter_timestamp is not None:
            since_str = self.filter_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            timestamp_filter = f" AND IndexTime > '{since_str}'"
            print(f"  - [🕒] Filtering for data after: {since_str}")
        
        query_str = f'SELECT IndexTime, LoggerTagID, Value FROM {table_name} WHERE {tags}{timestamp_filter} ORDER BY IndexTime DESC'
        
        print(f"  - [🔍] Querying {table_name} for {feature} (expecting {mill_count} mills)...")
        start_time = datetime.now()
        
        try:
            query = pd.read_sql_query(query_str, self.engine)
            query_time = datetime.now() - start_time
            
            # If no data found, return empty dataframe
            if query.empty:
                print(f"  - [❌] No data found in {table_name} for {feature} with the current filter")
                return pd.DataFrame()
                
            # Log basic stats
            time_range = f"from {query['IndexTime'].min()} to {query['IndexTime'].max()}" if not query.empty else "no data"
            print(f"  - [✅] Found {len(query):,} rows in {query_time.total_seconds():.1f}s | {time_range}")
            
            # Process the data
            print("  - [⚙️] Processing data...")
            start_process = datetime.now()
            
            # Drop duplicates and pivot
            unique_count = len(query)
            query = query.drop_duplicates(subset='IndexTime', keep='last')
            dup_count = unique_count - len(query)
            if dup_count > 0:
                print(f"    - Removed {dup_count:,} duplicate timestamps")
            
            df = query.pivot(index="IndexTime", columns="LoggerTagID", values="Value")
            
            # Handle missing mills
            found_mills = [str(k) for k in df.columns if str(k) in self.sql_tags[feature]]
            missing_mills = [k for k in self.sql_tags[feature].keys() if k not in found_mills]
            
            if missing_mills:
                print(f"    - ⚠️ Missing data for mills: {', '.join(missing_mills)}")
            
            # Forward fill, backfill, and resample
            df = df.ffill().bfill()
            df = df.resample("1min").mean()
            
            # Rename columns to mill names
            df.columns = [self.sql_tags[feature][str(k)] for k in df.columns if str(k) in self.sql_tags[feature]]
            df.index.names = ['TimeStamp']
            df.sort_index(axis=1, inplace=True)
            
            process_time = datetime.now() - start_process
            print(f"  - [✨] Processed {len(df):,} rows in {process_time.total_seconds():.1f}s")
            
            return df
            
        except Exception as e:
            print(f"  - [❌] Error processing {table_name} for {feature}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def compose_feature(self, feature):
        """Combine data from all tables for a specific feature"""
        print(f"\n📊 Composing feature: {feature}")
        print("-" * 50)
        
        frames = []
        total_rows = 0
        
        for tbl in self.table_names:
            print(f"\n📂 Processing table: {tbl}")
            table_data = self.read_sql_table(tbl, feature)
            
            if not table_data.empty:
                frames.append(table_data)
                total_rows += len(table_data)
                print(f"  - Added {len(table_data):,} rows from {tbl}")
            else:
                print(f"  - No data found in {tbl} for {feature}")
        
        if not frames:
            print("⚠️ No data found for any tables")
            return pd.DataFrame()
        
        print(f"\n🔗 Combining {len(frames)} data frames...")
        start_combine = datetime.now()
        
        # Combine all frames
        df = pd.concat(frames)
        
        # Sort and remove duplicates
        df = df.sort_index()
        initial_count = len(df)
        df = df[~df.index.duplicated(keep='first')]
        dup_count = initial_count - len(df)
        
        if dup_count > 0:
            print(f"  - Removed {dup_count:,} duplicate timestamps")
        
        # Apply time shift (2 hours forward)
        print("  - Applying 2-hour time shift...")
        df = df.shift(2, freq='h')
        
        # Fill any remaining gaps
        print("  - Filling missing values...")
        df = df.ffill().bfill()
        
        combine_time = datetime.now() - start_combine
        print(f"\n✅ Combined {len(df):,} rows in {combine_time.total_seconds():.1f}s")
        print("-" * 50)
        
        return df

    def create_mill_dataframe(self, mill):
        """Create a dataframe for a specific mill with all features"""
        all_data = []
        common_index = None
        
        for feature in self.sql_tags.keys():
            print(f"Processing {feature} for {mill}")
            feature_df = self.compose_feature(feature)
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
        
        print("\nAll mills have been processed and saved to the PostgreSQL database")

    def table_exists(self, schema, table_name):
        """Check if a table exists in the database"""
        try:
            with self.pg_engine.connect() as conn:
                exists_query = text(f"""
                    SELECT EXISTS (
                        SELECT FROM pg_tables
                        WHERE schemaname = :schema
                        AND tablename = :table_name
                    );
                """)
                result = conn.execute(exists_query, {'schema': schema, 'table_name': table_name}).fetchone()
                return result[0] if result else False
        except Exception as e:
            print(f"Error checking if table {schema}.{table_name} exists: {e}")
            return False

    def get_latest_timestamp(self, schema, table_name):
        """Get the latest timestamp from a table"""
        try:
            with self.pg_engine.connect() as conn:
                query = text(f'SELECT MAX("TimeStamp") FROM {schema}."{table_name}"')
                result = conn.execute(query).fetchone()
                return result[0] if result and result[0] else None
        except Exception as e:
            print(f"Error getting latest timestamp from {schema}.{table_name}: {e}")
            return None

    def create_or_append_to_table(self, schema, table_name, mill):
        """Create a new table or append to an existing one"""
        try:
            # Check if table exists
            if not self.table_exists(schema, table_name):
                print(f"🆕 Table {schema}.{table_name} doesn't exist, creating new table")
                self.filter_timestamp = None  # Get all data for new tables
                
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
                        chunksize=1000
                    )
                    print(f"✅ Created new table {schema}.{table_name} with {len(mill_df)} rows")
                else:
                    print(f"⚠️ No data found for {mill}, table not created")
                return

            # Table exists, get latest timestamp
            latest_timestamp = self.get_latest_timestamp(schema, table_name)
            
            if latest_timestamp:
                if not isinstance(latest_timestamp, datetime):
                    latest_timestamp = pd.to_datetime(latest_timestamp)
                
                # Add a buffer to account for the 2-hour shift in compose_feature
                buffer_time = pd.Timedelta(hours=2.1)
                self.filter_timestamp = latest_timestamp - buffer_time
                
                print(f"🔍 Found existing data in {schema}.{table_name} up to {latest_timestamp}")
                print(f"   Filtering for data after {self.filter_timestamp} (with {buffer_time} buffer)")
                
                # Get new data
                mill_df = self.create_mill_dataframe(mill)
                
                if not mill_df.empty:
                    # Filter for only new records
                    new_records = mill_df[mill_df.index > latest_timestamp]
                    
                    if not new_records.empty:
                        print(f"📥 Found {len(new_records)} new records to append")
                        
                        # Append new data
                        new_records.to_sql(
                            name=table_name,
                            schema=schema,
                            con=self.pg_engine,
                            if_exists='append',
                            index=True,
                            index_label='TimeStamp',
                            method='multi',
                            chunksize=1000
                        )
                        print(f"✅ Appended {len(new_records)} new rows to {schema}.{table_name}")
                    else:
                        print(f"ℹ️ No new data to append for {mill} (all data already exists)")
                else:
                    print(f"ℹ️ No new data found for {mill}")
            else:
                # Table exists but is empty, treat as new table
                print(f"ℹ️ Table {schema}.{table_name} exists but is empty, populating with all data")
                self.filter_timestamp = None
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
                        chunksize=1000
                    )
                    print(f"✅ Populated empty table {schema}.{table_name} with {len(mill_df)} rows")
                
        except Exception as e:
            print(f"❌ Error processing {mill}: {str(e)}")
            import traceback
            traceback.print_exc()

    def append_to_postgresql(self, schema='mills'):
        """Append new data to existing PostgreSQL database tables, only adding records newer than the latest timestamp"""
        print("\n🚀 Starting to synchronize data with PostgreSQL...")
        start_time = datetime.now()
        
        # Create schema if it doesn't exist
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                conn.commit()
            print(f"✅ Schema '{schema}' created or already exists")
        except SQLAlchemyError as e:
            print(f"❌ Error creating schema: {e}")
            return
        
        # Store original table_names to restore later
        original_table_names = self.table_names.copy()
        
        try:
            # Process each mill
            for mill in self.mills:
                print(f"\n🔧 Processing {mill}...")
                table_name = f"MILL_{mill[4:].zfill(2)}"
                self.create_or_append_to_table(schema, table_name, mill)
                
        finally:
            # Always restore original state
            self.filter_timestamp = None
            self.table_names = original_table_names
        
        duration = datetime.now() - start_time
        print(f"\n✨ All mills processed in {duration}")
        print("✅ Data synchronization completed successfully!")

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
    
    # Automatically handle table creation and data appending
    print("🔄 Starting data synchronization...")
    transformer.append_to_postgresql()

if __name__ == "__main__":
    main()

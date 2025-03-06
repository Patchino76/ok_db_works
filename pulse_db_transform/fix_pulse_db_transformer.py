import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import datetime

class FixedPulseDBTransformer:
    def __init__(self):
        # SQL Server connection parameters
        self.server = '10.20.2.10'
        self.database = 'pulse'
        self.username = 'Pulse_RO'
        self.password = 'PD@T@r3@der'
        self.connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        self.engine = create_engine("mssql+pyodbc:///?odbc_connect=" + self.connection_string)
        
        # Mill names and sensor tags
        self.mills = ['Mill01']
        
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
            
            'PulpHC': {"559" : "Mill01", "579" : "Mill02", "529" : "Mill03", "1226" : "Mill04", "1236" : "Mill05", "1218" : "Mill06",
                            "3640" : "Mill07", "1000" : "Mill08", "508" : "Mill09", "516" : "Mill10", "2691" : "Mill11", "3788" : "Mill12"},

            'PumpRPM': {"2405" : "Mill01", "2198" : "Mill02", "2629" : "Mill03", "2745" : "Mill04", "1652" : "Mill05", "2796" : "Mill06",
                            "2856" : "Mill07", "1800" : "Mill08", "2471" : "Mill09", "2452" : "Mill10", "2690" : "Mill11", "3780" : "Mill12"},

            'MotorAmp': {"2379" : "Mill01", "2153" : "Mill02", "2602" : "Mill03", "2719" : "Mill04", "2210" : "Mill05", "2770" : "Mill06",
                            "2830" : "Mill07", "1805" : "Mill08", "2478" : "Mill09", "2427" : "Mill10", "2663" : "Mill11", "3748" : "Mill12"},

            'CIDRA200': {"2875" : "Mill01", "2876" : "Mill02", "2877" : "Mill03", "2878" : "Mill04", "2879" : "Mill05", "2880" : "Mill06",
                            "2881" : "Mill07", "2882" : "Mill08", "2883" : "Mill09", "2884" : "Mill10", "2885" : "Mill11", "3775" : "Mill12"}
        }
        
        # Table names from SQL Server
        self.table_names = [
            'LoggerValues', 
            'LoggerValues_Archive_Jan2025', 'LoggerValues_Archive_Dec2024', 
            'LoggerValues_Archive_Nov2024', 'LoggerValues_Archive_Oct2024',
            'LoggerValues_Archive_Sep2024', 'LoggerValues_Archive_Aug2024',
            'LoggerValues_Archive_Jul2024', 'LoggerValues_Archive_Jun2024',
            'LoggerValues_Archive_May2024', 'LoggerValues_Archive_Apr2024',  
            'LoggerValues_Archive_Mar2024', 'LoggerValues_Archive_Feb2024',
            'LoggerValues_Archive_Jan2024',
        ]
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
            print(f"  - Filtering data after: {since_str}")
        
        query_str = f'SELECT IndexTime, LoggerTagID, Value FROM {table_name} WHERE {tags}{timestamp_filter} ORDER BY IndexTime DESC'
        
        try:
            query = pd.read_sql_query(query_str, self.engine)
            
            # If no data found, return empty dataframe
            if query.empty:
                print(f"  - No data found in {table_name} for {feature} with the current filter")
                return pd.DataFrame()
            else:
                print(f"  - Found {len(query)} rows in {table_name} for {feature}")
            
            # Process the data
            query = query.drop_duplicates(subset=['IndexTime', 'LoggerTagID'], keep='last')
            df = query.pivot(index="IndexTime", columns="LoggerTagID", values="Value")
            df = df.ffill().bfill()  # Using newer pandas methods
            df = df.resample("1min").mean()
            
            # Rename columns to mill names
            df.columns = [self.sql_tags[feature][str(k)] for k in df.columns if str(k) in self.sql_tags[feature]]
            df.index.names = ['TimeStamp']
            df.sort_index(axis=1, inplace=True)
            
            return df
        except Exception as e:
            print(f"  - Error reading {table_name} for {feature}: {str(e)}")
            return pd.DataFrame()

    def compose_feature(self, feature):
        """Combine data from all tables for a specific feature"""
        frames = []
        for tbl in self.table_names:
            print(f"Processing {tbl} for {feature}")
            frames.append(self.read_sql_table(tbl, feature))
        
        # Filter out empty frames
        frames = [df for df in frames if not df.empty]
        
        if not frames:
            print(f"No data found for {feature} in any table")
            # Return an empty DataFrame with the correct structure
            return pd.DataFrame(columns=self.mills)
        
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
        
        # First, collect all feature series
        feature_series_data = {}
        for feature in self.sql_tags.keys():
            print(f"Processing {feature} for {mill}")
            feature_df = self.compose_feature(feature)
            
            if mill in feature_df.columns:
                feature_series = feature_df[mill]
                feature_series_data[feature] = feature_series
                
                # Print date range for debugging
                if not feature_series.empty:
                    print(f"  - Date range for {feature}: {feature_series.index.min()} to {feature_series.index.max()}")
                    print(f"  - Number of records: {len(feature_series)}")
                else:
                    print(f"  - No data for {feature}")
        
        # Find features with data
        features_with_data = [f for f, s in feature_series_data.items() if not s.empty]
        print(f"Features with data: {features_with_data}")
        
        if not features_with_data:
            print(f"No data found for {mill}")
            return pd.DataFrame()
        
        # Calculate the common index using only features that have data
        for feature in features_with_data:
            series = feature_series_data[feature]
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)
        
        # Print common index range for debugging
        if common_index is not None and len(common_index) > 0:
            print(f"Common index range: {common_index.min()} to {common_index.max()}")
            print(f"Number of records in common index: {len(common_index)}")
        else:
            print("No common index found")
            return pd.DataFrame()
        
        # Create dataframe with aligned index
        mill_df = pd.DataFrame(index=common_index)
        for feature in features_with_data:
            mill_df[feature] = feature_series_data[feature].reindex(common_index)
        
        return mill_df

    def save_to_sqlite(self, output_db_path='fixed_mills.sqlite'):
        """Save all mill data to SQLite database"""
        conn = sqlite3.connect(output_db_path)
        
        for mill in self.mills:
            print(f"\nProcessing {mill}...")
            mill_df = self.create_mill_dataframe(mill)
            
            if mill_df.empty:
                print(f"No data to save for {mill}")
                continue
            
            # Convert mill name to table name (e.g., Mill01 -> MILL_01)
            table_name = f"MILL_{mill[4:]}".upper()
            
            # Save to SQLite
            mill_df.to_sql(table_name, conn, if_exists='replace', index=True)
            print(f"Saved {table_name} to SQLite database")
            
            # Print date range of saved data
            cursor = conn.cursor()
            cursor.execute(f"SELECT MIN(TimeStamp), MAX(TimeStamp), COUNT(*) FROM {table_name}")
            min_time, max_time, count = cursor.fetchone()
            print(f"Data range in {table_name}:")
            print(f"  First timestamp: {min_time}")
            print(f"  Last timestamp: {max_time}")
            print(f"  Total records: {count}")
        
        conn.close()
        print("\nAll mills have been processed and saved to the SQLite database")

def main():
    transformer = FixedPulseDBTransformer()
    transformer.save_to_sqlite('fixed_mills.sqlite')

if __name__ == "__main__":
    main()

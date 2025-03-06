import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import datetime

class PulseDBTransformer:
    def __init__(self):
        # SQL Server connection parameters
        self.server = '10.20.2.10'
        self.database = 'pulse'
        self.username = 'Pulse_RO'
        self.password = 'PD@T@r3@der'
        self.connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        self.engine = create_engine("mssql+pyodbc:///?odbc_connect=" + self.connection_string)
        
        # Mill names and sensor tags
        self.mills = ['Mill01', 'Mill02', 'Mill03', 'Mill04', 'Mill05', 'Mill06',
                     'Mill07', 'Mill08', 'Mill09', 'Mill10', 'Mill11', 'Mill12']
        
        # SQL tags dictionary from SQL_Data_Pulse_9.py
        self.sql_tags = {
            'Ore': {"485": "Mill01", "488": "Mill02", "491": "Mill03", "494": "Mill04", "497": "Mill05", "500": "Mill06",
                   "455": "Mill07", "467": "Mill08", "476": "Mill09", "479": "Mill10", "482": "Mill11", "3786": "Mill12"},
            'WaterMill': {"560": "Mill01", "580": "Mill02", "530": "Mill03", "1227": "Mill04", "1237": "Mill05", "1219": "Mill06",
                         "461": "Mill07", "474": "Mill08", "509": "Mill09", "517": "Mill10", "522": "Mill11", "3790": "Mill12"},
            'Power': {"487": "Mill01", "490": "Mill02", "493": "Mill03", "496": "Mill04", "499": "Mill05", "502": "Mill06",
                     "460": "Mill07", "471": "Mill08", "478": "Mill09", "481": "Mill10", "483": "Mill11", "3773": "Mill12"},
            'DensityHC': {"557": "Mill01", "577": "Mill02", "527": "Mill03", "1224": "Mill04", "1234": "Mill05", "1216": "Mill06",
                         "457": "Mill07", "469": "Mill08", "506": "Mill09", "514": "Mill10", "2658": "Mill11", "3742": "Mill12"}
        }
        
        # Table names from SQL Server
        self.table_names = ['LoggerValues',
                          'LoggerValues_Archive_Jan2025', 'LoggerValues_Archive_Dec2024',
                          'LoggerValues_Archive_Nov2024', 'LoggerValues_Archive_Oct2024']

    def read_sql_table(self, table_name, feature):
        """Read data from SQL Server for a specific feature"""
        tags = "LoggerTagID = " + " OR LoggerTagID = ".join(self.sql_tags[feature].keys())
        
        query = pd.read_sql_query(f'SELECT IndexTime, LoggerTagID, Value FROM {table_name} WHERE {tags} ORDER BY IndexTime DESC', 
                                self.engine)
        
        # Process the data
        query = query.drop_duplicates(subset='IndexTime', keep='last')
        df = query.pivot(index="IndexTime", columns="LoggerTagID", values="Value")
        df = df.ffill().bfill()  # Using newer pandas methods
        df = df.resample("1min").mean()
        
        # Rename columns to mill names
        df.columns = [self.sql_tags[feature][str(k)] for k in df.columns if str(k) in self.sql_tags[feature]]
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

    def save_to_sqlite(self, output_db_path='mills.sqlite'):
        """Save all mill data to SQLite database"""
        conn = sqlite3.connect(output_db_path)
        
        for mill in self.mills:
            print(f"\nProcessing {mill}...")
            mill_df = self.create_mill_dataframe(mill)
            
            # Convert mill name to table name (e.g., Mill01 -> MILL_01)
            table_name = f"MILL_{mill[4:]}".upper()
            
            # Save to SQLite
            mill_df.to_sql(table_name, conn, if_exists='replace', index=True)
            print(f"Saved {table_name} to SQLite database")
        
        conn.close()
        print("\nAll mills have been processed and saved to the SQLite database")

def main():
    transformer = PulseDBTransformer()
    transformer.save_to_sqlite('mills.sqlite')

if __name__ == "__main__":
    main()

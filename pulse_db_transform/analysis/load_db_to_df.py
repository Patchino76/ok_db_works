import sqlite3
import pandas as pd


class MillDataLoader:
    
    def __init__(self, db_file="D:\\DataSets\\MFC\\SQLITE3_NEW\\mills.sqlite"):
        self.db_file = db_file
    
    def get_mill_dataframe(self, mill_table="MILL_01"):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{mill_table}'")
        if cursor.fetchone() is None:
            conn.close()
            return None
        
        df = pd.read_sql_query(f"SELECT * FROM {mill_table}", conn)
        conn.close()
        

        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df = df.set_index("TimeStamp")
        
        return df

if __name__ == "__main__":
    loader = MillDataLoader()
    df_mill01 = loader.get_mill_dataframe("MILL_01")
    
    if df_mill01 is not None:
        print(df_mill01.head())
        print(df_mill01.info())

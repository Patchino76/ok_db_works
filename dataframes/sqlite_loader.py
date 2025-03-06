import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import os

class SQLiteDBLoader:
    def __init__(self, db_path="D:\\DataSets\\MFC\\SQLITE3_NEW\\mills.sqlite"):
        """
        Initialize the SQLite database loader.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.engine = None
        
        # Verify the database file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database file not found at: {db_path}")
        
        # Create SQLAlchemy engine for pandas operations
        self.engine = create_engine(f"sqlite:///{db_path}")
        
    def connect(self):
        """Establish a connection to the SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"Connected to SQLite database: {self.db_path}")
            return self.connection
        except sqlite3.Error as e:
            print(f"Error connecting to SQLite database: {e}")
            raise
    
    def close(self):
        """Close the database connection if open"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Connection to SQLite database closed")
    
    def get_table_names(self):
        """Get a list of all tables in the database"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            return tables
        except sqlite3.Error as e:
            print(f"Error getting table names: {e}")
            raise
        
    def get_table_schema(self, table_name):
        """Get the schema for a specific table"""
        try:
            if not self.connection:
                self.connect()
                
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()
            return schema
        except sqlite3.Error as e:
            print(f"Error getting schema for table {table_name}: {e}")
            raise
    
    def query_to_dataframe(self, query):
        """
        Execute a SQL query and return the results as a pandas DataFrame
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame: Results of the query
        """
        try:
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            raise
    
    def table_to_dataframe(self, table_name, columns=None, where_clause=None, limit=None):
        """
        Load a table or subset of a table into a pandas DataFrame
        
        Args:
            table_name (str): Name of the table to load
            columns (list, optional): List of columns to select. Defaults to all columns.
            where_clause (str, optional): WHERE clause for filtering. Defaults to None.
            limit (int, optional): Limit the number of rows returned. Defaults to None.
            
        Returns:
            pandas.DataFrame: Table data as a DataFrame
        """
        # Build the query
        cols_str = "*" if not columns else ", ".join(columns)
        query = f"SELECT {cols_str} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
            
        if limit:
            query += f" LIMIT {limit}"
            
        return self.query_to_dataframe(query)
    
    def execute_query(self, query, params=None):
        """
        Execute a SQL query that doesn't return data (INSERT, UPDATE, DELETE)
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query. Defaults to None.
            
        Returns:
            int: Number of rows affected
        """
        try:
            if not self.connection:
                self.connect()
                
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            self.connection.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            if params:
                print(f"Params: {params}")
            raise
    
    def __enter__(self):
        """Support for context manager protocol"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager protocol"""
        self.close()


# Example usage
if __name__ == "__main__":
    # Create an instance of the SQLiteDBLoader
    db_loader = SQLiteDBLoader()
    
    try:
        # Connect to the database
        db_loader.connect()
        
        # Get table names
        tables = db_loader.get_table_names()
        print(f"Tables in database: {tables}")
        
        # Get schema for the first table
        if tables:
            schema = db_loader.get_table_schema(tables[0])
            print(f"Schema for table {tables[0]}:")
            for col in schema:
                print(f"  {col}")
        
        # Load data from a table
        if tables:
            df = db_loader.table_to_dataframe(tables[0], limit=5)
            print(f"\nSample data from {tables[0]}:")
            print(df.head())
            
    finally:
        # Close the connection
        db_loader.close()

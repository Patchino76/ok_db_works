#%%
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

# Add parent directory to path so we can import the columns_dictionary module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# %%
# Load the CSV data
file_path = "combined_dispatcher_data_en.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime if it's not already
if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df.tail()

# %%
# Save DataFrame to SQLite database
sqlite_path = "D:/DataSets/MFC/SQLITE3_NEW/mills.sqlite"

# Connect to SQLite database
print(f"Connecting to SQLite database: {sqlite_path}")
conn = sqlite3.connect(sqlite_path)

# Save DataFrame to SQLite table
print(f"Saving DataFrame to 'Dispatchers' table...")
df.to_sql('Dispatchers', conn, if_exists='replace', index=False)

# Verify the table was created
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Dispatchers'")
if cursor.fetchone():
    print("Table 'Dispatchers' created successfully")
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM Dispatchers")
    row_count = cursor.fetchone()[0]
    print(f"Inserted {row_count} rows into the Dispatchers table")
    
    # Get column names
    cursor.execute("PRAGMA table_info(Dispatchers)")
    columns = cursor.fetchall()
    print(f"Table has {len(columns)} columns:")
    for col in columns[:5]:  # Show first 5 columns
        print(f"  - {col[1]} ({col[2]})")
    if len(columns) > 5:
        print(f"  - ... and {len(columns) - 5} more columns")
else:
    print("Error: Table 'Dispatchers' was not created")

# Close the connection
conn.close()
print("SQLite connection closed")

# %%

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('mills.sqlite')
cursor = conn.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"- {table[0]}")

# For each table, get the first row to understand structure and last timestamp
for table_name in [t[0] for t in tables]:
    print(f"\nTable: {table_name}")
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"Row count: {row_count}")
    
    # Get max timestamp (assuming 'TimeStamp' column exists)
    try:
        cursor.execute(f"SELECT MAX(TimeStamp) FROM {table_name}")
        max_ts = cursor.fetchone()[0]
        print(f"Latest timestamp: {max_ts}")
    except sqlite3.OperationalError:
        print("No TimeStamp column found")

conn.close()

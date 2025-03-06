import sqlite3

def clear_all_tables():
    conn = sqlite3.connect('mills.sqlite')
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Enable foreign key support
    cursor.execute("PRAGMA foreign_keys = OFF;")
    
    try:
        # Begin transaction
        cursor.execute("BEGIN TRANSACTION;")
        
        # Delete all rows from each table
        for table in tables:
            table_name = table[0]
            print(f"Clearing table: {table_name}")
            cursor.execute(f"DELETE FROM {table_name};")
        
        # Commit the transaction
        conn.commit()
        print("All tables have been cleared successfully!")
        
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        print(f"An error occurred: {e}")
        
    finally:
        # Re-enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.close()
        conn.close()

if __name__ == "__main__":
    clear_all_tables()

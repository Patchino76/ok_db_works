import sys
from pulse_to_postgresql import PulseDBTransformer

def main():
    # PostgreSQL connection parameters
    pg_host = 'em-m-db4.ellatzite-med.com'
    pg_port = 5432
    pg_dbname = 'em_pulse_data'
    pg_user = 's.lyubenov'
    pg_password = 'tP9uB7sH7mK6zA7t'
    
    print("Initializing PulseDBTransformer...")
    transformer = PulseDBTransformer(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password
    )
    
    print("Running append_to_postgresql...")
    # Limit the number of tables processed for testing
    # Use a specific mill for testing to reduce processing time
    transformer.mills = ['Mill01']  # Just process Mill01 for testing
    transformer.table_names = ['LoggerValues']  # Just process current table
    
    transformer.append_to_postgresql()
    print("Append operation completed.")

if __name__ == "__main__":
    main()

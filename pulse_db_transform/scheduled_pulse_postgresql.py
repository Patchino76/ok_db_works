import threading
import time
import sys
from datetime import datetime
from pulse_to_postgresql import PulseDBTransformer

def run_hourly():
    """Run pulse DB transformation every hour in a separate thread"""
    while True:
        # Wait until the next hour
        now = datetime.now()
        print(f"\nRunning scheduled transformation at {now}")
        
        # Transform and append data
        try:
            # Create transformer with PostgreSQL connection parameters
            transformer = PulseDBTransformer(
                pg_host=pg_host,
                pg_port=pg_port,
                pg_dbname=pg_dbname,
                pg_user=pg_user,
                pg_password=pg_password
            )
            transformer.append_to_postgresql()
            print(f"Successfully appended data at {datetime.now()}")
        except Exception as e:
            print(f"Error appending data: {e}")
        
        # Wait until the next hour
        now = datetime.now()
        seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
        time.sleep(seconds_to_next_hour)

# PostgreSQL connection parameters
pg_host = 'em-m-db4.ellatzite-med.com'  # PostgreSQL server host
pg_port = 5432                          # PostgreSQL server port
pg_dbname = 'em_pulse_data'             # PostgreSQL database name
pg_user = 's.lyubenov'                 # PostgreSQL username
pg_password = 'tP9uB7sH7mK6zA7t'       # PostgreSQL password

# Run initial transformation and start hourly thread
if __name__ == "__main__":
    # Create transformer with PostgreSQL connection parameters
    transformer = PulseDBTransformer(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password
    )
    
    # Run immediately
    transformer.append_to_postgresql()
    
    # Start hourly thread
    thread = threading.Thread(target=run_hourly)
    thread.daemon = True
    thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped by user")
        sys.exit(0)

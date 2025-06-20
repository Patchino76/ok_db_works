import threading
import time
import sys
from datetime import datetime, timedelta
from pulse_to_postgresql import PulseDBTransformer

def run_hourly():
    """Run pulse DB transformation every hour (3600 seconds) in a separate thread"""
    while True:
        start_time = datetime.now()
        print(f"\nRunning scheduled transformation at {start_time}")
        
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
        
        # Calculate time to wait until exactly one hour has passed since start_time
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        seconds_to_wait = max(0, 3600 - elapsed_seconds)  # 3600 seconds = 1 hour
        print(f"Next execution scheduled in {seconds_to_wait:.1f} seconds (at {(start_time + timedelta(seconds=3600)).strftime('%H:%M:%S')})")
        time.sleep(seconds_to_wait)

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

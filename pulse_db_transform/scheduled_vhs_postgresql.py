"""Периодична (почасова) синхронизация на VHS тагове към PostgreSQL.

Аналогично на scheduled_pulse_postgresql.py, но за schema vhs / table VHS_TAGS.
"""
import threading
import time
import sys
from datetime import datetime, timedelta

from vhs_to_postgresql import VHSTransformer

# PostgreSQL connection parameters
pg_host = 'em-m-db4.ellatzite-med.com'
pg_port = 5432
pg_dbname = 'em_pulse_data'
pg_user = 's.lyubenov'
pg_password = 'tP9uB7sH7mK6zA7t'


def make_transformer():
    return VHSTransformer(
        pg_host=pg_host, pg_port=pg_port, pg_dbname=pg_dbname,
        pg_user=pg_user, pg_password=pg_password
    )


def run_hourly():
    """Run VHS synchronization every hour (3600 seconds) in a separate thread."""
    while True:
        start_time = datetime.now()
        print(f"\nRunning scheduled VHS transformation at {start_time}")
        try:
            make_transformer().append_to_postgresql()
            print(f"Successfully appended VHS data at {datetime.now()}")
        except Exception as e:
            print(f"Error appending VHS data: {e}")

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        seconds_to_wait = max(0, 3600 - elapsed_seconds)
        print(f"Next execution scheduled in {seconds_to_wait:.1f} seconds "
              f"(at {(start_time + timedelta(seconds=3600)).strftime('%H:%M:%S')})")
        time.sleep(seconds_to_wait)


if __name__ == "__main__":
    # Run immediately
    make_transformer().append_to_postgresql()

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

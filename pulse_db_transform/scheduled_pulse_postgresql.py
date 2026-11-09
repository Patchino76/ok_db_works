"""Обединен scheduler: Pulse SQL Server -> PostgreSQL за мелници и VHS тагове.

Създава ЕДНА връзка към Pulse SQL (и една към PostgreSQL) и я споделя между
двата трансформатора, които се изпълняват последователно всеки час.
"""
import threading
import time
import sys
from datetime import datetime, timedelta

from pulse_config import create_pulse_engine, create_pg_engine
from pulse_to_postgresql import PulseDBTransformer
from vhs_to_postgresql import VHSTransformer

# Една обща връзка към Pulse SQL и една към PostgreSQL за двата job-а
pulse_engine = create_pulse_engine()
pg_engine = create_pg_engine()


def run_sync():
    """Изпълнява синхронизацията на мелниците и VHS таговете (споделени връзки)."""
    print(f"\n=== Synchronization started at {datetime.now()} ===")

    try:
        PulseDBTransformer(engine=pulse_engine, pg_engine=pg_engine).append_to_postgresql()
        print(f"Mills appended at {datetime.now()}")
    except Exception as e:
        print(f"Error appending mills: {e}")

    try:
        VHSTransformer(engine=pulse_engine, pg_engine=pg_engine).append_to_postgresql()
        print(f"VHS appended at {datetime.now()}")
    except Exception as e:
        print(f"Error appending VHS: {e}")


def run_hourly():
    """Изпълнява обединената синхронизация на всеки час (3600 секунди)."""
    while True:
        start_time = datetime.now()
        run_sync()

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        seconds_to_wait = max(0, 600 - elapsed_seconds)
        print(f"Next execution scheduled in {seconds_to_wait:.1f} seconds "
              f"(at {(start_time + timedelta(seconds=300)).strftime('%H:%M:%S')})")
        time.sleep(seconds_to_wait)


if __name__ == "__main__":
    # Първоначално изпълнение
    run_sync()

    # Стартиране на почасовия thread
    thread = threading.Thread(target=run_hourly)
    thread.daemon = True
    thread.start()

    # Държим main thread жива
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped by user")
        sys.exit(0)

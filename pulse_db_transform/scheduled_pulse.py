import threading
import time
from datetime import datetime
from pulse_db_transformer import PulseDBTransformer

def run_hourly():
    """Run pulse DB transformation every hour in a separate thread"""
    while True:
        try:
            print(f"{datetime.now().strftime('%H:%M:%S')}: Running transformation...")
            transformer = PulseDBTransformer()
            transformer.append_to_sqlite('mills.sqlite')
            print(f"{datetime.now().strftime('%H:%M:%S')}: Complete")
        except Exception as e:
            print(f"Error: {e}")
            
        # Wait for next hour
        now = datetime.now()
        seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
        time.sleep(seconds_to_next_hour)

# Run initial transformation and start hourly thread
if __name__ == "__main__":
    # Run immediately
    PulseDBTransformer().append_to_sqlite('mills.sqlite')
    
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

#!/usr/bin/env python3
"""
Create a Windows Service for the Pulse Scheduler
This creates a proper Windows service that runs in the background
"""

import os
import sys
import subprocess
import win32service
import win32serviceutil
import win32event
import servicemanager
import socket
import time
from scheduled_pulse_postgresql import run_hourly

class PulseSchedulerService(win32serviceutil.ServiceFramework):
    """Windows Service for Pulse Scheduler"""
    
    _svc_name_ = "PulseScheduler"
    _svc_display_name_ = "Pulse Data Scheduler Service"
    _svc_description_ = "Automatically processes pulse data every hour"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.is_alive = True
        
    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, '')
        )
        
    def SvcDoRun(self):
        """Main service loop"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        # Start the hourly scheduler in a separate thread
        import threading
        scheduler_thread = threading.Thread(target=run_hourly)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Keep the service running
        while self.is_alive:
            # Wait for stop event or timeout
            win32event.WaitForSingleObject(self.hWaitStop, 5000)  # 5 second timeout

def install_service():
    """Install the Windows service"""
    try:
        # Install the service
        win32serviceutil.InstallService(
            PulseSchedulerService._svc_reg_class_,
            PulseSchedulerService._svc_name_,
            PulseSchedulerService._svc_display_name_,
            description=PulseSchedulerService._svc_description_
        )
        
        print(f"✅ Service '{PulseSchedulerService._svc_name_}' installed successfully")
        print("   You can now start it from Windows Services or run:")
        print(f"   python {__file__} start")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install service: {e}")
        return False

def start_service():
    """Start the Windows service"""
    try:
        win32serviceutil.StartService(PulseSchedulerService._svc_name_)
        print(f"✅ Service '{PulseSchedulerService._svc_name_}' started successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return False

def stop_service():
    """Stop the Windows service"""
    try:
        win32serviceutil.StopService(PulseSchedulerService._svc_name_)
        print(f"✅ Service '{PulseSchedulerService._svc_name_}' stopped successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to stop service: {e}")
        return False

def remove_service():
    """Remove the Windows service"""
    try:
        win32serviceutil.RemoveService(PulseSchedulerService._svc_name_)
        print(f"✅ Service '{PulseSchedulerService._svc_name_}' removed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to remove service: {e}")
        return False

def main():
    """Main function for service management"""
    if len(sys.argv) == 1:
        print("Pulse Scheduler Windows Service Manager")
        print("=" * 40)
        print("Usage:")
        print(f"  python {__file__} install   - Install the service")
        print(f"  python {__file__} start     - Start the service")
        print(f"  python {__file__} stop      - Stop the service")
        print(f"  python {__file__} remove    - Remove the service")
        print(f"  python {__file__} debug     - Run in debug mode")
        return
    
    command = sys.argv[1].lower()
    
    if command == "install":
        install_service()
    elif command == "start":
        start_service()
    elif command == "stop":
        stop_service()
    elif command == "remove":
        remove_service()
    elif command == "debug":
        print("Running in debug mode (Ctrl+C to stop)...")
        try:
            run_hourly()
        except KeyboardInterrupt:
            print("Stopped by user")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: install, start, stop, remove, debug")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "debug":
        main()
    else:
        # Run as service
        win32serviceutil.HandleCommandLine(PulseSchedulerService)

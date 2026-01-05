@echo off
title Pulse Data Scheduler
echo Starting Pulse Data Scheduler...
echo This window can be minimized to the taskbar
echo Press Ctrl+C to stop the scheduler
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python script directly
C:\venv\crewai312\Scripts\python scheduled_pulse_postgresql.py

REM If the script exits, wait for user input before closing
echo.
echo Scheduler has stopped. Press any key to exit...
pause

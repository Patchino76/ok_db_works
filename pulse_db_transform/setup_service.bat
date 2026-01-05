@echo off
title Pulse Scheduler Service Setup
echo Pulse Scheduler Windows Service Setup
echo =====================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running as administrator
) else (
    echo ❌ This script must be run as administrator
    echo Right-click the script and select "Run as administrator"
    pause
    exit /b 1
)

REM Install required packages
echo 📦 Installing required packages...
pip install pywin32
if %errorLevel% neq 0 (
    echo ❌ Failed to install pywin32
    pause
    exit /b 1
)

REM Install the service
echo 🔧 Installing Windows service...
python create_windows_service.py install
if %errorLevel% neq 0 (
    echo ❌ Failed to install service
    pause
    exit /b 1
)

echo.
echo ✅ Service installed successfully!
echo.
echo Next steps:
echo 1. Start the service: python create_windows_service.py start
echo 2. Or start it from Windows Services (services.msc)
echo 3. Check service status in Windows Services
echo.
echo To stop the service: python create_windows_service.py stop
echo To remove the service: python create_windows_service.py remove
echo.
pause

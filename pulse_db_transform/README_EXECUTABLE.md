# Creating a Windows Executable for Pulse Scheduler

This guide shows how to create a Windows executable that runs the pulse data scheduler as a resident application that can be minimized to the taskbar.

## Option 1: Simple Batch Script (Recommended)

This is the easiest approach - just run the Python script through a batch file.

### Quick Start

1. **Run the scheduler:**
   - Double-click `run_pulse_scheduler.bat`
   - A console window will open titled "Pulse Data Scheduler"
   - The window can be minimized to the taskbar

### Usage

1. **Start the scheduler:**

   - Run `run_pulse_scheduler.bat`
   - A console window will open and can be minimized to taskbar

2. **Monitor activity:**

   - The console shows hourly execution logs
   - Each run displays start time, duration, and next execution time

3. **Stop the scheduler:**
   - Restore the console window from taskbar
   - Press `Ctrl+C` to stop
   - Or close the console window

## Option 2: Windows Service (Professional)

For a more professional deployment that runs completely in the background.

### Installation

1. **Run as Administrator:**

   - Right-click `setup_service.bat` and select "Run as administrator"
   - This will install the Windows service

2. **Start the service:**
   ```cmd
   python create_windows_service.py start
   ```
   - Or use Windows Services (services.msc)

### Service Management

```cmd
# Install service
python create_windows_service.py install

# Start service
python create_windows_service.py start

# Stop service
python create_windows_service.py stop

# Remove service
python create_windows_service.py remove

# Debug mode (run in console)
python create_windows_service.py debug
```

### Service Features

- **Background Operation:** Runs without any user interface
- **Automatic Startup:** Can be configured to start with Windows
- **Persistent:** Continues running after user logout
- **Professional:** Managed through Windows Services panel

## Option 3: Standalone Executable (Advanced)

Create a single .exe file that doesn't require Python installation.

### Build Requirements

Due to pip hash validation issues, use this manual approach:

1. **Install PyInstaller manually:**

   ```cmd
   pip install --no-cache-dir pyinstaller
   ```

2. **Build the executable:**

   ```cmd
   python build_executable.py
   ```

3. **Run the executable:**
   - Use `run_pulse_scheduler.bat` to launch with console
   - Or run `dist/PulseScheduler.exe` directly

## Comparison of Options

| Feature                | Batch Script      | Windows Service    | Standalone EXE    |
| ---------------------- | ----------------- | ------------------ | ----------------- |
| **Ease of Use**        | ✅ Easiest        | ⚠️ Complex         | ⚠️ Complex        |
| **Background**         | ❌ Visible window | ✅ True background | ❌ Visible window |
| **Auto-start**         | ❌ Manual         | ✅ With Windows    | ❌ Manual         |
| **No Python Required** | ❌ Needs Python   | ❌ Needs Python    | ✅ Self-contained |
| **Professional**       | ⚠️ Basic          | ✅ Professional    | ✅ Professional   |

## Recommendation

**Start with Option 1 (Batch Script)** - it's simple and works immediately. If you need true background operation later, migrate to Option 2 (Windows Service).

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Errors:**

   - Verify connection parameters in `scheduled_pulse_postgresql.py`
   - Check network connectivity to database server
   - Ensure database user has required permissions

2. **Permission Issues:**

   - Run as administrator for Windows Service installation
   - Check antivirus isn't blocking the executable

3. **Python Path Issues:**
   - Ensure Python is in system PATH
   - Use full Python path if needed: `C:\Python311\python.exe`

### Modifying Settings

Edit connection parameters in `scheduled_pulse_postgresql.py`:

```python
pg_host = 'your_host'
pg_port = 5432
pg_dbname = 'your_database'
pg_user = 'your_username'
pg_password = 'your_password'
```

#!/usr/bin/env python3
"""
Script to build a Windows executable from scheduled_pulse_postgresql.py
Run this script to create an executable that can run residently in Windows
"""

import os
import sys
import subprocess
import shutil

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller is already installed")
        return True
    except ImportError:
        print("📦 Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✅ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyInstaller: {e}")
            return False

def create_spec_file():
    """Create a PyInstaller spec file with proper configuration"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['scheduled_pulse_postgresql.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['threading', 'time', 'datetime', 'psycopg2', 'pandas', 'sqlalchemy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PulseScheduler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console window visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file path here if desired
)
'''
    
    with open('pulse_scheduler.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("✅ Created pulse_scheduler.spec file")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")
    
    # Clean previous builds
    for dir_name in ['build', 'dist', '__pycache__']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🧹 Cleaned {dir_name} directory")
    
    # Build the executable
    try:
        subprocess.check_call([
            sys.executable, 
            '-m', 
            'PyInstaller', 
            '--clean', 
            '--noconfirm', 
            'pulse_scheduler.spec'
        ])
        print("✅ Executable built successfully!")
        print(f"📁 Executable location: {os.path.abspath('dist/PulseScheduler.exe')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build executable: {e}")
        return False

def create_startup_script():
    """Create a batch script to run the executable with console window"""
    batch_content = '''@echo off
title Pulse Data Scheduler
echo Starting Pulse Data Scheduler...
echo This window can be minimized to the taskbar
echo Press Ctrl+C to stop the scheduler
echo.

REM Change to the directory where the executable is located
cd /d "%~dp0dist"

REM Run the executable
PulseScheduler.exe

REM If the executable exits, wait for user input before closing
pause
'''
    
    with open('run_pulse_scheduler.bat', 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print("✅ Created run_pulse_scheduler.bat")

def create_startup_shortcut():
    """Create a shortcut for easy access"""
    import winshell
    from win32com.client import Dispatch
    
    try:
        desktop = winshell.desktop()
        path = os.path.join(desktop, "Pulse Scheduler.lnk")
        target = os.path.abspath("run_pulse_scheduler.bat")
        wDir = os.path.abspath(".")
        icon = target
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        print(f"✅ Created desktop shortcut: {path}")
    except ImportError:
        print("⚠️ Could not create desktop shortcut (winshell not installed)")
        print("   Install with: pip install winshell pywin32")

def main():
    """Main build process"""
    print("🚀 Building Pulse Scheduler executable...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('scheduled_pulse_postgresql.py'):
        print("❌ scheduled_pulse_postgresql.py not found in current directory")
        print("   Please run this script from the pulse_db_transform directory")
        return False
    
    # Install PyInstaller
    if not install_pyinstaller():
        return False
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    if not build_executable():
        return False
    
    # Create startup script
    create_startup_script()
    
    # Create desktop shortcut
    try:
        create_startup_shortcut()
    except Exception as e:
        print(f"⚠️ Could not create desktop shortcut: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Build completed successfully!")
    print("\n📋 Usage Instructions:")
    print("1. Run 'run_pulse_scheduler.bat' to start the scheduler")
    print("2. The console window will open and can be minimized to taskbar")
    print("3. Press Ctrl+C in the console to stop the scheduler")
    print("4. The executable will run the data transformation every hour")
    print(f"\n📁 Files created:")
    print(f"   - Executable: {os.path.abspath('dist/PulseScheduler.exe')}")
    print(f"   - Batch file: {os.path.abspath('run_pulse_scheduler.bat')}")
    
    return True

if __name__ == "__main__":
    main()

@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv-win\Scripts\python.exe" (
  echo Run setup_windows.bat first. Do not copy a macOS virtual environment to Windows.
  pause
  exit /b 1
)
"%~dp0.venv-win\Scripts\python.exe" -u -m camix_director.web %*
if errorlevel 1 pause

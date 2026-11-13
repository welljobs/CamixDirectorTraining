@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv-win\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  exit /b 1
)
"%~dp0.venv-win\Scripts\python.exe" -m pytest -q
exit /b %ERRORLEVEL%

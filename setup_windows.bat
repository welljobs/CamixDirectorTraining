@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=cpu"
set "INDEX="
if /I "%MODE%"=="cpu" set "INDEX=https://download.pytorch.org/whl/cpu"
if /I "%MODE%"=="cu118" set "INDEX=https://download.pytorch.org/whl/cu118"
if /I "%MODE%"=="cu126" set "INDEX=https://download.pytorch.org/whl/cu126"
if /I "%MODE%"=="cu128" set "INDEX=https://download.pytorch.org/whl/cu128"
if not defined INDEX goto bad_mode
py -3.11 -c "import struct,platform; assert struct.calcsize('P')==8 and platform.machine().lower() in ('amd64','x86_64'), 'Use 64-bit x86 Python 3.11'" >nul 2>nul
if errorlevel 1 goto need_python
if not exist ".venv-win\Scripts\python.exe" (
  py -3.11 -m venv .venv-win
  if errorlevel 1 goto failed
)
set "PY=%~dp0.venv-win\Scripts\python.exe"
"%PY%" -m pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 --index-url "%INDEX%"
if errorlevel 1 goto failed
"%PY%" -m pip install -r requirements-train.txt
if errorlevel 1 goto failed
"%PY%" -m pip check
if errorlevel 1 goto failed
"%PY%" -c "import torch,av,numpy; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('PyAV:',av.__version__)"
if errorlevel 1 goto failed
echo.
echo Setup complete. Run start_web_windows.bat to open the local console.
echo CUDA builds still require a compatible NVIDIA GPU and driver. CPU is always an option.
exit /b 0
:need_python
echo Install Python 3.11 x64 with the Windows py launcher, then run this script again.
echo This setup does not support 32-bit Python or claim Windows ARM compatibility.
echo See WINDOWS.md.
pause
exit /b 1
:bad_mode
echo Usage: setup_windows.bat [cpu^|cu118^|cu126^|cu128]
exit /b 2
:failed
echo Setup failed. Keep this output for diagnosis. No system Python settings were changed.
pause
exit /b 1

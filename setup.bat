@echo off
setlocal
rem KittenTTS 2 Web UI - one-time setup.
rem Creates .\venv next to this script and installs the requirements.
rem Everything (venv, pip cache, model cache) stays inside this folder.
cd /d "%~dp0"

set "PIP_CACHE_DIR=%~dp0cache\pip"
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 -m venv "%~dp0venv" || py -3 -m venv "%~dp0venv" || goto :fail
) else (
  python -m venv "%~dp0venv" || goto :fail
)

"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail

rem PyTorch with CUDA: a plain "pip install torch" on Windows gives the CPU-only build.
rem Install the CUDA build first from the PyTorch index. cu128 matches recent NVIDIA
rem drivers; pick the right index-url for your setup at https://pytorch.org/get-started/locally/
"%~dp0venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || goto :fail

rem Check that the CUDA build of torch can see the GPU.
"%~dp0venv\Scripts\python.exe" -c "import torch,sys; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); sys.exit(0 if ok else 3)"
if errorlevel 3 (
  echo.
  echo WARNING: PyTorch cannot see an NVIDIA GPU. The app will run on the CPU and be very slow.
  echo Update your NVIDIA driver ^(570 or newer for CUDA 12.8; check with nvidia-smi^) and run setup.bat again.
  echo.
)

"%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" || goto :fail

if not exist "%~dp0.env" if exist "%~dp0.env.example" copy "%~dp0.env.example" "%~dp0.env" >nul

echo.
echo Setup finished. Run start.bat to launch the web UI.
echo The model (about 4.7 GB of VRAM when loaded) downloads from Hugging Face on first start.
endlocal
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
endlocal
exit /b 1

@echo off
setlocal
rem KittenTTS 2 Web UI - start the local server.
rem Settings come from .env (see .env.example). Paths are relative to this folder.
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\activate.bat" (
  echo No venv found. Run setup.bat first.
  pause
  exit /b 1
)
call "%~dp0venv\Scripts\activate.bat"

rem HF_HOME, output and cache folders are resolved by webui\app.py from .env (defaults: .\cache\hf, .\output).
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
rem Read the port from .env for the browser link (the server reads .env itself).
if not defined KITTEN_WEBUI_PORT if exist "%~dp0.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do if /i "%%A"=="KITTEN_WEBUI_PORT" set "KITTEN_WEBUI_PORT=%%B"
)
if not defined KITTEN_WEBUI_PORT set KITTEN_WEBUI_PORT=7860

echo Starting KittenTTS 2 Web UI on http://127.0.0.1:%KITTEN_WEBUI_PORT%/
echo First load takes a while and uses about 4.7 GB of VRAM. Leave this window open.
rem Open the browser only once the server answers (up to 2 minutes), so the first page load does not fail.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$u='http://127.0.0.1:%KITTEN_WEBUI_PORT%/'; for($i=0; $i -lt 120; $i++){ try { Invoke-WebRequest ($u + 'api/status') -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 } }; Start-Process $u"
python "%~dp0webui\app.py"
endlocal

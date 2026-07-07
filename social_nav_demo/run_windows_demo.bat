@echo off
setlocal

set "ROOT=%~dp0"
set "DEMO_DIR=%ROOT%static_demo"
set "INDEX_FILE=%DEMO_DIR%\index.html"
set "PORT=8000"

if not exist "%INDEX_FILE%" (
  echo Could not find "%INDEX_FILE%".
  echo Run this file from the social_nav_demo folder after static_demo has been generated.
  pause
  exit /b 1
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  echo Starting demo at http://localhost:%PORT%/
  start "" "http://localhost:%PORT%/"
  echo.
  echo Keep this window open while viewing the demo.
  echo Press Ctrl+C to stop the server.
  py -3 -m http.server %PORT% --directory "%DEMO_DIR%"
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  echo Starting demo at http://localhost:%PORT%/
  start "" "http://localhost:%PORT%/"
  echo.
  echo Keep this window open while viewing the demo.
  echo Press Ctrl+C to stop the server.
  python -m http.server %PORT% --directory "%DEMO_DIR%"
  exit /b %ERRORLEVEL%
)

echo Python was not found, so opening the static HTML directly.
echo The saved demo trials will still display in the browser.
start "" "%INDEX_FILE%"
pause

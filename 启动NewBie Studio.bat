@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\daily_start.ps1"
set "START_EXIT=%ERRORLEVEL%"
if not "%START_EXIT%"=="0" (
  echo.
  echo Startup failed. Keep this window open and review the error above.
  pause
)
exit /b %START_EXIT%

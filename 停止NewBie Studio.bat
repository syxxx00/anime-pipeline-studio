@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1"
set "STOP_EXIT=%ERRORLEVEL%"
if not "%STOP_EXIT%"=="0" (
  echo.
  echo Stop failed. Keep this window open and review the error above.
  pause
  exit /b %STOP_EXIT%
)
echo.
pause
exit /b %STOP_EXIT%

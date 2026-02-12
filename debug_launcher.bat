@echo off
set "APP_DIR=%LOCALAPPDATA%\Privox"
set "LIB_DIR=%APP_DIR%\_internal_libs"
set "PYTHONPATH=%LIB_DIR%;%CD%\src"
set "PRIVOX_DEBUG=1"

echo ========================================================
echo Debug Launcher for Privox (Bypassing Bootstrap)
echo ========================================================
echo APP_DIR: %APP_DIR%
echo LIB_DIR: %LIB_DIR%
echo PYTHONPATH: %PYTHONPATH%
echo.

echo Launching python...
C:\Python313\python.exe src\voice_input.py > debug_output.txt 2>&1

echo.
echo Launch finished. Check debug_output.txt for errors.
pause

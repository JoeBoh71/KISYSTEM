@echo off
echo.
echo ====================================================================
echo KISYSTEM - Python 3.12
echo ====================================================================
echo.
if not exist "C:\Python312\python.exe" (
    echo ERROR: Python 3.12 not found
    exit /b 1
)
"C:\Python312\python.exe" --version
"C:\Python312\python.exe" run_autonomous.py %*

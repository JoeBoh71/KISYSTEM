# KISYSTEM Launcher
Write-Host `"
Write-Host `"===================================================================`"
Write-Host `"KISYSTEM - Python 3.12`"
Write-Host `"===================================================================`"
Write-Host `"
if (-not (Test-Path `"C:\Python312\python.exe`")) {
    Write-Host `"ERROR: Python 3.12 not found`"
    exit 1
}
& `"C:\Python312\python.exe`" --version
& `"C:\Python312\python.exe`" run_autonomous.py `$args

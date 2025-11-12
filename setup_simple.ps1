# KISYSTEM Python 3.12 Setup - Ultra Simple Version
# No fancy characters, no complex here-strings
# Run: powershell -ExecutionPolicy Bypass -File setup_simple.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================================================="
Write-Host "KISYSTEM - Python 3.12 + CuPy Installer"
Write-Host "====================================================================="
Write-Host ""

# Config
$PY_VERSION = "3.12.7"
$PY_DIR = "C:\Python312"
$PY_URL = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
$PY_INSTALLER = "$env:TEMP\python-3.12.7-amd64.exe"
$KISYSTEM = "C:\KISYSTEM"

# Step 1: Check Python
Write-Host ""
Write-Host "Step 1: Checking Python 3.12..."
Write-Host ""

if (Test-Path "$PY_DIR\python.exe") {
    Write-Host "[OK] Python 3.12 found at $PY_DIR"
    $ver = & "$PY_DIR\python.exe" --version 2>&1
    Write-Host "     Version: $ver"
    $SKIP_INSTALL = $true
} else {
    Write-Host "[!] Python 3.12 not found"
    Write-Host "    Will download and install..."
    $SKIP_INSTALL = $false
}

# Step 2: Download
if (-not $SKIP_INSTALL) {
    Write-Host ""
    Write-Host "Step 2: Downloading Python 3.12..."
    Write-Host ""
    Write-Host "URL: $PY_URL"
    Write-Host "Size: 25 MB (1-2 min)"
    Write-Host ""
    
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $PY_URL -OutFile $PY_INSTALLER
        $ProgressPreference = 'Continue'
        Write-Host "[OK] Download complete"
    }
    catch {
        Write-Host "[ERROR] Download failed: $_"
        Write-Host ""
        Write-Host "Manual download:"
        Write-Host "1. Go to: https://www.python.org/downloads/"
        Write-Host "2. Download Python 3.12.7 (64-bit)"
        Write-Host "3. Install to: $PY_DIR"
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Step 3: Install
    Write-Host ""
    Write-Host "Step 3: Installing Python 3.12..."
    Write-Host ""
    Write-Host "Target: $PY_DIR"
    Write-Host "This takes 2-3 minutes..."
    Write-Host ""
    
    $args = @(
        "/quiet"
        "InstallAllUsers=0"
        "PrependPath=0"
        "Include_test=0"
        "Include_pip=1"
        "Include_launcher=0"
        "TargetDir=$PY_DIR"
    )
    
    Start-Process -FilePath $PY_INSTALLER -ArgumentList $args -Wait -NoNewWindow
    
    if (Test-Path "$PY_DIR\python.exe") {
        Write-Host "[OK] Python 3.12 installed"
        $ver = & "$PY_DIR\python.exe" --version 2>&1
        Write-Host "     Version: $ver"
    } else {
        Write-Host "[ERROR] Installation failed"
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Remove-Item $PY_INSTALLER -ErrorAction SilentlyContinue
}

# Step 4: Upgrade pip
Write-Host ""
Write-Host "Step 4: Upgrading pip..."
Write-Host ""

try {
    & "$PY_DIR\python.exe" -m pip install --upgrade pip --quiet
    Write-Host "[OK] pip upgraded"
}
catch {
    Write-Host "[WARNING] pip upgrade failed (non-critical)"
}

# Step 5: Install CuPy
Write-Host ""
Write-Host "Step 5: Installing CuPy + NumPy..."
Write-Host ""
Write-Host "This takes 2-5 minutes (1 GB download)"
Write-Host ""

$CUPY_OK = $false

Write-Host "Trying cupy-cuda12x..."
try {
    & "$PY_DIR\python.exe" -m pip install cupy-cuda12x numpy --quiet
    Write-Host "[OK] CuPy (CUDA 12.x) installed"
    $CUPY_VERSION = "cuda12x"
    $CUPY_OK = $true
}
catch {
    Write-Host "[!] CUDA 12.x failed, trying 11.x..."
    try {
        & "$PY_DIR\python.exe" -m pip install cupy-cuda11x numpy --quiet
        Write-Host "[OK] CuPy (CUDA 11.x) installed"
        $CUPY_VERSION = "cuda11x"
        $CUPY_OK = $true
    }
    catch {
        Write-Host "[ERROR] CuPy installation failed"
        Write-Host ""
        Write-Host "Try manual install:"
        Write-Host "$PY_DIR\python.exe -m pip install cupy-cuda12x"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Step 6: Verify GPU
if ($CUPY_OK) {
    Write-Host ""
    Write-Host "Step 6: Verifying GPU..."
    Write-Host ""
    
    $testCode = "import cupy as cp; import numpy as np; print('CuPy:', cp.__version__); print('NumPy:', np.__version__); device = cp.cuda.Device(0); cc = device.compute_capability; print(f'GPU: Compute {cc[0]}.{cc[1]}'); mem = device.mem_info[1] // (1024**2); print(f'Memory: {mem} MB'); a = cp.random.rand(1000,1000); b = cp.random.rand(1000,1000); c = cp.matmul(a,b); cp.cuda.Stream.null.synchronize(); print('GPU test: PASSED')"
    
    try {
        & "$PY_DIR\python.exe" -c $testCode
        Write-Host ""
        Write-Host "[OK] GPU verification successful"
    }
    catch {
        Write-Host "[WARNING] GPU verification had errors"
        Write-Host "          (CuPy installed, but GPU may not be accessible)"
    }
}

# Step 7: Create Launchers
Write-Host ""
Write-Host "Step 7: Creating launchers..."
Write-Host ""

# Batch file
$batContent = "@echo off`r`n"
$batContent += "echo.`r`n"
$batContent += "echo ====================================================================`r`n"
$batContent += "echo KISYSTEM - Python 3.12`r`n"
$batContent += "echo ====================================================================`r`n"
$batContent += "echo.`r`n"
$batContent += "if not exist `"$PY_DIR\python.exe`" (`r`n"
$batContent += "    echo ERROR: Python 3.12 not found`r`n"
$batContent += "    exit /b 1`r`n"
$batContent += ")`r`n"
$batContent += "`"$PY_DIR\python.exe`" --version`r`n"
$batContent += "`"$PY_DIR\python.exe`" run_autonomous.py %*`r`n"

$batFile = "$KISYSTEM\run_kisystem.bat"
[System.IO.File]::WriteAllText($batFile, $batContent)
Write-Host "[OK] Created: $batFile"

# PowerShell file
$ps1Content = "# KISYSTEM Launcher`r`n"
$ps1Content += "Write-Host ```"`r`n"
$ps1Content += "Write-Host ```"===================================================================```"`r`n"
$ps1Content += "Write-Host ```"KISYSTEM - Python 3.12```"`r`n"
$ps1Content += "Write-Host ```"===================================================================```"`r`n"
$ps1Content += "Write-Host ```"`r`n"
$ps1Content += "if (-not (Test-Path ```"$PY_DIR\python.exe```")) {`r`n"
$ps1Content += "    Write-Host ```"ERROR: Python 3.12 not found```"`r`n"
$ps1Content += "    exit 1`r`n"
$ps1Content += "}`r`n"
$ps1Content += "& ```"$PY_DIR\python.exe```" --version`r`n"
$ps1Content += "& ```"$PY_DIR\python.exe```" run_autonomous.py ```$args`r`n"

$ps1File = "$KISYSTEM\run_kisystem.ps1"
[System.IO.File]::WriteAllText($ps1File, $ps1Content)
Write-Host "[OK] Created: $ps1File"

# Step 8: Deploy Agent
Write-Host ""
Write-Host "Step 8: PythonTesterAgent..."
Write-Host ""

$agentSrc = "$KISYSTEM\python_tester_agent_cupy.py"
$agentDst = "$KISYSTEM\agents\python_tester_agent.py"

if (Test-Path $agentSrc) {
    Copy-Item $agentSrc $agentDst -Force
    Write-Host "[OK] Deployed python_tester_agent.py"
} else {
    Write-Host "[!] python_tester_agent_cupy.py not found"
    Write-Host "    Download from Claude and copy manually"
}

# Summary
Write-Host ""
Write-Host "====================================================================="
Write-Host "INSTALLATION COMPLETE"
Write-Host "====================================================================="
Write-Host ""
Write-Host "Installed:"
Write-Host "  - Python 3.12 at $PY_DIR"
Write-Host "  - CuPy ($CUPY_VERSION) with GPU support"
Write-Host "  - NumPy for computing"
Write-Host "  - Launchers created"
Write-Host ""
Write-Host "Run KISYSTEM:"
Write-Host "  .\run_kisystem.bat"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Test: $PY_DIR\python.exe test_installation.py"
Write-Host "  2. Run: .\run_kisystem.bat"
Write-Host ""

Read-Host "Press Enter to exit"

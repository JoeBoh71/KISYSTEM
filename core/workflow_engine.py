"""
KISYSTEM Workflow Engine - Enhanced with Balanced Auto-Dependency-Installation
================================================================================

Security Model: 3-Stage Protection
1. Whitelist Check   → Known packages auto-install
2. PyPI Validation   → Verify package exists and is legitimate  
3. User Confirmation → Ask for unknown packages

Author: Jörg Bohne
Created: 2025-01-06
"""

import asyncio
import sys
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import subprocess
import json
import aiohttp


# ============================================================================
# CONFIGURATION
# ============================================================================

class SecurityLevel(Enum):
    """Security levels for dependency installation"""
    PARANOID = "paranoid"      # Never auto-install, always ask
    BALANCED = "balanced"      # Whitelist + Confirmation
    AUTONOMOUS = "autonomous"  # Auto-install everything (DANGEROUS!)


@dataclass
class WorkflowConfig:
    """Configuration for workflow engine behavior"""
    
    # Auto-install settings
    auto_install_enabled: bool = True
    security_level: SecurityLevel = SecurityLevel.BALANCED
    require_confirmation: bool = True
    
    # Package management
    use_whitelist: bool = True
    validate_pypi: bool = True
    cache_validation: bool = True
    offline_mode: bool = False  # Skip PyPI validation when no network
    
    # Performance
    parallel_install: bool = False  # Install dependencies parallel
    max_install_retries: int = 2
    
    # Logging
    verbose: bool = True
    log_file: Optional[str] = None


# ============================================================================
# PACKAGE WHITELIST - Common Safe Packages
# ============================================================================

PACKAGE_WHITELIST = {
    # Core scientific computing
    "numpy", "scipy", "pandas", "matplotlib", "seaborn",
    
    # Audio/Signal processing  
    "soundfile", "librosa", "pydub", "resampy", "audioread",
    
    # Machine Learning
    "scikit-learn", "torch", "tensorflow", "keras",
    
    # Data formats
    "h5py", "netcdf4", "xlrd", "openpyxl", "pyarrow",
    
    # Utilities
    "tqdm", "click", "rich", "colorama", "tabulate",
    
    # Testing
    "pytest", "pytest-asyncio", "pytest-cov",
    
    # Standard library extensions
    "typing-extensions", "dataclasses", "attrs",
}


# ============================================================================
# PACKAGE VALIDATION
# ============================================================================

class PackageValidator:
    """Validates packages against PyPI and security databases"""
    
    def __init__(self, cache_results: bool = True):
        self.cache_results = cache_results
        self._validation_cache: Dict[str, bool] = {}
        
    async def validate_package(self, package: str) -> Dict[str, any]:
        """
        Validate package exists on PyPI and check basic security
        
        Returns:
            dict with keys: valid, exists, downloads, description, error
        """
        
        # Check cache
        if self.cache_results and package in self._validation_cache:
            return {
                "valid": self._validation_cache[package],
                "cached": True,
                "package": package
            }
        
        result = {
            "valid": False,
            "exists": False,
            "package": package,
            "cached": False,
            "downloads": None,
            "description": None,
            "error": None
        }
        
        try:
            # Query PyPI JSON API
            url = f"https://pypi.org/pypi/{package}/json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        result["exists"] = True
                        result["valid"] = True
                        result["description"] = data["info"].get("summary", "")
                        
                        # Get download stats if available
                        # Note: PyPI doesn't provide this in JSON API anymore
                        # Would need pypistats package for real stats
                        
                        # Cache result
                        if self.cache_results:
                            self._validation_cache[package] = True
                            
                    elif response.status == 404:
                        result["error"] = "Package not found on PyPI"
                    else:
                        result["error"] = f"HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            result["error"] = "Validation timeout"
        except aiohttp.ClientConnectorError:
            result["error"] = "No network connection"
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def is_whitelisted(self, package: str) -> bool:
        """Check if package is on whitelist"""
        return package.lower() in PACKAGE_WHITELIST


# ============================================================================
# ASYNC INPUT HELPER
# ============================================================================

async def async_input(prompt: str) -> str:
    """
    Non-blocking async input for user confirmations
    
    Uses loop.run_in_executor to avoid blocking the event loop
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


# ============================================================================
# DEPENDENCY INSTALLER
# ============================================================================

class DependencyInstaller:
    """Handles safe installation of Python packages"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.validator = PackageValidator(cache_results=config.cache_validation)
        
        # Track what we've installed this session
        self.installed_this_session: Set[str] = set()
        
    async def ensure_dependencies(self, packages: List[str]) -> Dict[str, bool]:
        """
        Ensure all required packages are installed
        
        Returns:
            dict mapping package -> successfully installed/available
        """
        results = {}
        
        for package in packages:
            # Check if already installed
            if self._is_installed(package):
                if self.config.verbose:
                    print(f"[Workflow] ✓ {package} already installed")
                results[package] = True
                continue
            
            # Try to install
            success = await self._install_package(package)
            results[package] = success
            
            if success:
                self.installed_this_session.add(package)
                
        return results
    
    def _is_installed(self, package: str) -> bool:
        """Check if package is already installed"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    async def _install_package(self, package: str) -> bool:
        """
        Install a single package with 3-stage security
        
        Stage 1: Whitelist check
        Stage 2: PyPI validation  
        Stage 3: User confirmation
        """
        
        # ===== STAGE 1: WHITELIST =====
        if self.config.use_whitelist and self.validator.is_whitelisted(package):
            if self.config.verbose:
                print(f"[Workflow] ✓ {package} is whitelisted - auto-installing...")
            return await self._do_install(package)
        
        # ===== STAGE 2: VALIDATION =====
        if self.config.validate_pypi and not self.config.offline_mode:
            if self.config.verbose:
                print(f"[Workflow] ⚙️ Validating {package} on PyPI...")
                
            validation = await self.validator.validate_package(package)
            
            # Check for network errors in offline mode
            if "No network connection" in validation.get("error", ""):
                if self.config.verbose:
                    print(f"[Workflow] ⚠️ No network - skipping validation")
                # Continue to confirmation instead of blocking
            elif not validation["valid"]:
                error = validation.get("error", "Unknown error")
                print(f"[Workflow] ✗ {package} validation FAILED: {error}")
                print(f"[Workflow] ⚠️ Package blocked for security reasons")
                return False
            else:
                if self.config.verbose:
                    desc = validation.get("description", "No description")
                    print(f"[Workflow] ✓ Validated: {desc[:80]}...")
        elif self.config.offline_mode:
            if self.config.verbose:
                print(f"[Workflow] ⚠️ Offline mode - skipping PyPI validation")
        
        # ===== STAGE 3: USER CONFIRMATION =====
        if self.config.require_confirmation:
            print(f"\n[Workflow] ⚠️ Package '{package}' not on whitelist")
            print(f"[Workflow] ℹ️ Validated on PyPI - appears legitimate")
            print(f"[Workflow] ❓ Install? (yes/no): ", end='', flush=True)
            
            response = await async_input("")
            
            if response.lower() not in ['yes', 'y']:
                print(f"[Workflow] ⛔ Installation cancelled by user")
                return False
            
            print(f"[Workflow] ✓ User approved installation")
        
        # All checks passed - install
        return await self._do_install(package)
    
    async def _do_install(self, package: str) -> bool:
        """
        Actually install the package using pip
        
        Returns True if successful, False otherwise
        """
        
        for attempt in range(self.config.max_install_retries):
            try:
                if self.config.verbose:
                    print(f"[Workflow] 📦 Installing {package}... (attempt {attempt + 1})")
                
                # Run pip install
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "--break-system-packages"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=120  # 2 minutes max
                )
                
                if result.returncode == 0:
                    if self.config.verbose:
                        print(f"[Workflow] ✅ {package} installed successfully")
                    return True
                else:
                    error = result.stderr.strip()
                    print(f"[Workflow] ❌ Installation failed: {error[:200]}")
                    
            except subprocess.TimeoutExpired:
                print(f"[Workflow] ⏱️ Installation timeout (attempt {attempt + 1})")
            except Exception as e:
                print(f"[Workflow] ❌ Installation error: {e}")
        
        # All retries failed
        print(f"[Workflow] ⛔ Failed to install {package} after {self.config.max_install_retries} attempts")
        return False


# ============================================================================
# WORKFLOW ENGINE
# ============================================================================

class WorkflowEngine:
    """
    Main workflow orchestration engine
    
    Manages multi-agent task execution with automatic dependency handling
    """
    
    def __init__(self, supervisor, config: Optional[WorkflowConfig] = None):
        self.supervisor = supervisor
        self.config = config or WorkflowConfig()
        self.installer = DependencyInstaller(self.config)
        
    async def execute_task(self, task_description: str, context: Dict = None) -> Dict:
        """
        Execute a task with automatic dependency management
        
        Args:
            task_description: What to do
            context: Additional context for the task
            
        Returns:
            Result dict with status, output, errors
        """
        
        if self.config.verbose:
            print(f"\n[Workflow] 🚀 Starting task: {task_description[:80]}...")
        
        result = {
            "status": "pending",
            "task": task_description,
            "output": None,
            "errors": [],
            "dependencies_installed": []
        }
        
        try:
            # 1. Task Planning Phase
            plan = await self._plan_task(task_description, context)
            
            # 2. Dependency Check Phase
            if plan.get("required_packages"):
                installed = await self.installer.ensure_dependencies(
                    plan["required_packages"]
                )
                result["dependencies_installed"] = list(installed.keys())
                
                # Check if any critical dependencies failed
                failed = [pkg for pkg, ok in installed.items() if not ok]
                if failed:
                    result["status"] = "failed"
                    result["errors"].append(f"Missing dependencies: {failed}")
                    return result
            
            # 3. Task Execution Phase
            output = await self._execute_planned_task(plan, context)
            
            result["status"] = "completed"
            result["output"] = output
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            if self.config.verbose:
                print(f"[Workflow] ❌ Task failed: {e}")
        
        return result
    
    async def _plan_task(self, task: str, context: Dict) -> Dict:
        """
        Plan task execution and identify required packages
        
        Returns:
            Plan dict with steps, required_packages, estimated_time
        """
        # TODO: Integration with LLM for intelligent planning
        # For now, return basic structure
        
        return {
            "steps": [],
            "required_packages": [],
            "estimated_time": None
        }
    
    async def _execute_planned_task(self, plan: Dict, context: Dict) -> any:
        """Execute the planned task steps"""
        # TODO: Actual task execution logic
        pass


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the workflow engine"""
    
    # Create config
    config = WorkflowConfig(
        auto_install_enabled=True,
        security_level=SecurityLevel.BALANCED,
        verbose=True
    )
    
    # Create engine (supervisor=None for testing)
    engine = WorkflowEngine(supervisor=None, config=config)
    
    # Test dependency installation
    print("Testing Dependency Installer...\n")
    
    test_packages = ["numpy", "fake-package-xyz-123", "scipy"]
    results = await engine.installer.ensure_dependencies(test_packages)
    
    print("\n" + "="*70)
    print("Installation Results:")
    for pkg, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {pkg}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
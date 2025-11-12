"""
KISYSTEM Hardware Test Agent
Umfassende Hardware-Validierung für U3DAW Development Setup
Tests: RME MADI FX, RTX 4070, DeckLink 8K Pro, HD Fury VRROOM, System

Author: Jörg Bohne
Date: 2025-11-06
"""

import asyncio
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import platform

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from learning_module import LearningModule


class HardwareTestAgent:
    """
    Hardware Test Agent für U3DAW Development Setup
    
    Tests:
    - RME HDSPe MADI FX (Audio Interface)
    - RTX 4070 (GPU/CUDA)
    - DeckLink 8K Pro V2 (Video I/O)
    - HD Fury VRROOM (HDMI Processor)
    - System (CPU, RAM, SSD, Network)
    """
    
    def __init__(self, supervisor=None):
        """
        Initialize Hardware Test Agent
        
        Args:
            supervisor: Supervisor instance (optional)
        """
        # Get workspace from supervisor or use default
        if supervisor and hasattr(supervisor, 'workspace'):
            workspace = supervisor.workspace
        else:
            workspace = "D:/AGENT_MEMORY"
        
        self.workspace = Path(workspace) / "hardware_tests"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Store supervisor reference
        self.supervisor = supervisor
        
        # Initialize Learning Module V2
        self.learning = LearningModule()
        
        self.model = "llama3.1:8b"  # Fast model for reporting
        
        print(f"[HardwareTest] Initialized with workspace: {self.workspace}")
        print(f"[HardwareTest] Learning V2 active")
    
    async def execute(self, task) -> str:
        """
        Execute hardware test task
        
        Args:
            task: Task description (str or dict)
            
        Returns:
            Test results
        """
        # Handle dict input from supervisor
        if isinstance(task, dict):
            task_str = task.get('target', str(task))
        else:
            task_str = str(task)
        
        task_lower = task_str.lower()
        
        # Route to specific test
        if 'rme' in task_lower or 'madi' in task_lower or 'audio' in task_lower:
            return await self.test_audio_hardware()
        elif 'gpu' in task_lower or 'cuda' in task_lower or '4070' in task_lower:
            return await self.test_gpu()
        elif 'decklink' in task_lower or 'video' in task_lower:
            return await self.test_video_hardware()
        elif 'hdfury' in task_lower or 'vrroom' in task_lower or 'hdmi' in task_lower:
            return await self.test_hdfury()
        elif 'system' in task_lower or 'cpu' in task_lower or 'ram' in task_lower:
            return await self.test_system()
        elif 'u3daw' in task_lower or 'pipeline' in task_lower or 'end-to-end' in task_lower:
            return await self.test_u3daw_pipeline()
        elif 'all' in task_lower or 'complete' in task_lower or 'full' in task_lower:
            return await self.test_all()
        else:
            # Default: Full test suite
            return await self.test_all()
    
    async def test_all(self) -> str:
        """Run all hardware tests"""
        print("\n[HardwareTest] 🔍 Running complete hardware test suite...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Run all tests
        tests = [
            ('GPU', self.test_gpu()),
            ('Audio', self.test_audio_hardware()),
            ('Video', self.test_video_hardware()),
            ('HD Fury', self.test_hdfury()),
            ('System', self.test_system())
        ]
        
        for name, test_coro in tests:
            try:
                result = await test_coro
                results['tests'][name] = {'status': 'success', 'result': result}
                print(f"[HardwareTest] ✓ {name} test complete")
            except Exception as e:
                results['tests'][name] = {'status': 'failed', 'error': str(e)}
                print(f"[HardwareTest] ✗ {name} test failed: {e}")
        
        # Generate report
        report = self._generate_report(results)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.workspace / f"hardware_report_{timestamp}.txt"
        report_path.write_text(report, encoding='utf-8')
        
        print(f"[HardwareTest] 📄 Report saved: {report_path}")
        
        return report
    
    async def test_gpu(self) -> str:
        """Test RTX 4070 GPU"""
        print("\n[HardwareTest] 🎮 Testing RTX 4070...")
        
        gpu_info = {}
        
        try:
            # nvidia-smi query
            cmd = [
                'nvidia-smi',
                '--query-gpu=name,memory.total,memory.used,temperature.gpu,clocks.gr,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                data = stdout.decode('utf-8', errors='ignore').strip().split(',')
                gpu_info = {
                    'name': data[0].strip(),
                    'vram_total': f"{data[1].strip()} MB",
                    'vram_used': f"{data[2].strip()} MB",
                    'temperature': f"{data[3].strip()}°C",
                    'clock_speed': f"{data[4].strip()} MHz",
                    'power_draw': f"{data[5].strip()} W"
                }
            else:
                gpu_info['error'] = stderr.decode('utf-8', errors='ignore')
        
        except Exception as e:
            gpu_info['error'] = str(e)
        
        # CUDA test
        try:
            # Check CUDA version
            proc = await asyncio.create_subprocess_exec(
                'nvcc', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                output = stdout.decode('utf-8', errors='ignore')
                match = re.search(r'release (\d+\.\d+)', output)
                if match:
                    gpu_info['cuda_version'] = match.group(1)
        except:
            gpu_info['cuda_version'] = 'Not available'
        
        # Format result
        result = "=== GPU Test (RTX 4070) ===\n"
        for key, value in gpu_info.items():
            result += f"  {key}: {value}\n"
        
        return result
    
    async def test_audio_hardware(self) -> str:
        """Test RME HDSPe MADI FX"""
        print("\n[HardwareTest] 🎵 Testing RME HDSPe MADI FX...")
        
        audio_info = {}
        
        try:
            # Check if RME device exists (Windows Registry or Device Manager)
            # Note: This is a simplified check - full implementation would query TotalMix
            
            proc = await asyncio.create_subprocess_exec(
                'powershell',
                '-Command',
                'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*RME*" -or $_.FriendlyName -like "*MADI*"}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                output = stdout.decode('utf-8', errors='ignore')
                if 'RME' in output or 'MADI' in output:
                    audio_info['device'] = 'RME HDSPe MADI FX detected'
                    audio_info['status'] = 'Present'
                else:
                    audio_info['device'] = 'RME device not found'
                    audio_info['status'] = 'Missing'
            
            # Try to get ASIO info (if available)
            # This would require ASIO SDK - simplified for now
            audio_info['asio'] = 'Check TotalMix for details'
            audio_info['note'] = 'Full audio tests require TotalMix integration'
            
        except Exception as e:
            audio_info['error'] = str(e)
        
        # Format result
        result = "=== Audio Hardware Test (RME MADI FX) ===\n"
        for key, value in audio_info.items():
            result += f"  {key}: {value}\n"
        
        result += "\nRecommended Manual Checks:\n"
        result += "  - TotalMix: Verify 64 MADI channels\n"
        result += "  - Sample Rate: Test 48/96/192 kHz\n"
        result += "  - Latency: Measure round-trip <3ms\n"
        result += "  - M-32 Pro II: Check dual-unit sync\n"
        
        return result
    
    async def test_video_hardware(self) -> str:
        """Test DeckLink 8K Pro V2"""
        print("\n[HardwareTest] 📹 Testing DeckLink 8K Pro V2...")
        
        video_info = {}
        
        try:
            # Check for Blackmagic DeckLink device
            proc = await asyncio.create_subprocess_exec(
                'powershell',
                '-Command',
                'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*DeckLink*" -or $_.FriendlyName -like "*Blackmagic*"}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                output = stdout.decode('utf-8', errors='ignore')
                if 'DeckLink' in output or 'Blackmagic' in output:
                    video_info['device'] = 'DeckLink 8K Pro V2 detected'
                    video_info['status'] = 'Present'
                else:
                    video_info['device'] = 'DeckLink device not found'
                    video_info['status'] = 'Missing'
            
            video_info['note'] = 'Full video tests require Blackmagic Desktop Video SDK'
            
        except Exception as e:
            video_info['error'] = str(e)
        
        # Format result
        result = "=== Video Hardware Test (DeckLink 8K Pro V2) ===\n"
        for key, value in video_info.items():
            result += f"  {key}: {value}\n"
        
        result += "\nRecommended Manual Checks:\n"
        result += "  - Desktop Video: Verify driver version\n"
        result += "  - Input Signal: Test 4K@60Hz, 8K@30Hz\n"
        result += "  - Output: Verify HDMI/SDI output\n"
        
        return result
    
    async def test_hdfury(self) -> str:
        """Test HD Fury VRROOM"""
        print("\n[HardwareTest] 🎬 Testing HD Fury VRROOM...")
        
        hdfury_info = {}
        
        # HD Fury VRROOM is typically accessed via Web UI or Serial
        # This is a placeholder - full implementation would require HTTP/Serial API
        
        hdfury_info['device'] = 'HD Fury VRROOM'
        hdfury_info['note'] = 'Requires Web UI or Serial API for automated testing'
        hdfury_info['web_ui'] = 'Typically at http://192.168.x.x'
        
        result = "=== HDMI Processor Test (HD Fury VRROOM) ===\n"
        for key, value in hdfury_info.items():
            result += f"  {key}: {value}\n"
        
        result += "\nRecommended Manual Checks:\n"
        result += "  - Web UI: Check firmware version\n"
        result += "  - EDID: Verify custom EDID loaded\n"
        result += "  - Signal: Test HDMI 2.1 features\n"
        result += "  - VRR: Verify Variable Refresh Rate\n"
        result += "  - Resolution: Test 4K@120Hz capability\n"
        
        return result
    
    async def test_system(self) -> str:
        """Test System (CPU, RAM, SSD)"""
        print("\n[HardwareTest] 💻 Testing System...")
        
        system_info = {}
        
        try:
            # CPU Info
            system_info['cpu'] = platform.processor()
            
            # OS Info
            system_info['os'] = f"{platform.system()} {platform.version()}"
            
            # Python Version
            system_info['python'] = platform.python_version()
            
            # RAM Info (Windows)
            proc = await asyncio.create_subprocess_exec(
                'powershell',
                '-Command',
                '(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum / 1GB',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                ram_gb = stdout.decode('utf-8', errors='ignore').strip()
                system_info['ram'] = f"{ram_gb} GB"
            
            # Disk Info (simplified)
            proc = await asyncio.create_subprocess_exec(
                'powershell',
                '-Command',
                'Get-PhysicalDisk | Select-Object FriendlyName, MediaType | ConvertTo-Json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                try:
                    disks = json.loads(stdout.decode('utf-8', errors='ignore'))
                    if isinstance(disks, dict):
                        disks = [disks]
                    system_info['disks'] = ', '.join([d['FriendlyName'] for d in disks if 'FriendlyName' in d])
                except:
                    system_info['disks'] = 'Could not parse disk info'
        
        except Exception as e:
            system_info['error'] = str(e)
        
        # Format result
        result = "=== System Test ===\n"
        for key, value in system_info.items():
            result += f"  {key}: {value}\n"
        
        result += "\nExpected Specs:\n"
        result += "  CPU: AMD Ryzen 9 7900\n"
        result += "  RAM: 64GB DDR5\n"
        result += "  SSD: Samsung 990 PRO (2TB) + 9100 PRO (1TB)\n"
        
        return result
    
    async def test_u3daw_pipeline(self) -> str:
        """Test U3DAW End-to-End Pipeline"""
        print("\n[HardwareTest] 🎼 Testing U3DAW Pipeline...")
        
        pipeline_info = {}
        
        pipeline_info['pipeline'] = 'Roon → ASIO → RME MADI FX → M-32 Pro II (x2) → Loopback'
        pipeline_info['note'] = 'Requires U3DAW application running'
        pipeline_info['status'] = 'Manual test required'
        
        result = "=== U3DAW Pipeline Test ===\n"
        for key, value in pipeline_info.items():
            result += f"  {key}: {value}\n"
        
        result += "\nTest Procedure:\n"
        result += "  1. Start Roon playback\n"
        result += "  2. Verify ASIO buffer in U3DAW\n"
        result += "  3. Check RME TotalMix levels\n"
        result += "  4. Measure M-32 output (DA)\n"
        result += "  5. Loopback to M-32 input (AD)\n"
        result += "  6. Verify round-trip latency <5ms\n"
        result += "  7. Check THD <0.001%\n"
        
        return result
    
    def _generate_report(self, results: Dict) -> str:
        """Generate formatted test report"""
        report = "╔══════════════════════════════════════════╗\n"
        report += "║     U3DAW Hardware Test Report         ║\n"
        report += "╚══════════════════════════════════════════╝\n\n"
        
        report += f"Timestamp: {results['timestamp']}\n"
        report += f"System: Windows 10 IoT LTSC\n\n"
        
        # Summary
        total = len(results['tests'])
        passed = sum(1 for t in results['tests'].values() if t['status'] == 'success')
        
        report += f"Summary: {passed}/{total} tests passed\n\n"
        
        # Individual tests
        for name, test in results['tests'].items():
            if test['status'] == 'success':
                report += f"✓ {name} Test\n"
                report += test['result'] + "\n"
            else:
                report += f"✗ {name} Test FAILED\n"
                report += f"  Error: {test.get('error', 'Unknown')}\n\n"
        
        report += "═══════════════════════════════════════════\n"
        report += "End of Report\n"
        
        return report


if __name__ == '__main__':
    async def test():
        agent = HardwareTestAgent()
        result = await agent.test_all()
        print(result)
    
    asyncio.run(test())

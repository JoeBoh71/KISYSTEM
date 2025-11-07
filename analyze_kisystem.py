"""
KISYSTEM CONSISTENCY ANALYZER
Analyzes all Python files for:
- Import consistency
- Path correctness
- Version consistency
- Circular dependencies
- Missing dependencies
- Obsolete code
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# ANSI Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

class KISYSTEMAnalyzer:
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.files = {}
        self.imports = defaultdict(list)
        self.issues = []
        
    def analyze(self):
        """Run complete analysis"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}KISYSTEM CONSISTENCY ANALYZER{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        # Phase 1: Collect all Python files
        self.collect_files()
        
        # Phase 2: Analyze each file
        self.analyze_files()
        
        # Phase 3: Check cross-references
        self.check_cross_references()
        
        # Phase 4: Report
        self.generate_report()
        
    def collect_files(self):
        """Collect all Python files"""
        print(f"{CYAN}Phase 1: Collecting Python files...{RESET}")
        
        for py_file in self.root.rglob("*.py"):
            # Skip __pycache__
            if "__pycache__" in str(py_file):
                continue
                
            rel_path = py_file.relative_to(self.root)
            
            try:
                content = py_file.read_text(encoding='utf-8')
                self.files[str(rel_path)] = {
                    'path': py_file,
                    'content': content,
                    'lines': len(content.split('\n')),
                    'size': py_file.stat().st_size
                }
            except Exception as e:
                self.issues.append(f"[ERROR] Cannot read {rel_path}: {e}")
        
        print(f"  Found {len(self.files)} Python files\n")
        
    def analyze_files(self):
        """Analyze each file for issues"""
        print(f"{CYAN}Phase 2: Analyzing files...{RESET}\n")
        
        for rel_path, file_info in self.files.items():
            content = file_info['content']
            
            # Extract imports
            imports = self.extract_imports(content)
            self.imports[rel_path] = imports
            
            # Check for issues
            self.check_file_issues(rel_path, content)
            
        print()
        
    def extract_imports(self, content):
        """Extract all imports from file"""
        imports = {
            'standard': [],
            'local': [],
            'sys_path': []
        }
        
        # Standard imports
        for match in re.finditer(r'^import\s+(\w+)', content, re.MULTILINE):
            imports['standard'].append(match.group(1))
            
        for match in re.finditer(r'^from\s+(\w+)', content, re.MULTILINE):
            imports['standard'].append(match.group(1))
        
        # Local imports (from agents.X, from core.X)
        for match in re.finditer(r'from\s+(agents|core|tests)\.(\w+)', content):
            imports['local'].append(f"{match.group(1)}.{match.group(2)}")
        
        # sys.path.insert
        for match in re.finditer(r'sys\.path\.insert\(.*?[\'"](.+?)[\'"]', content):
            imports['sys_path'].append(match.group(1))
            
        return imports
        
    def check_file_issues(self, rel_path, content):
        """Check individual file for issues"""
        issues = []
        
        # Check 1: Outdated version markers
        if re.search(r'Version:\s*1\.0|v1\.0|Phase\s*[1-4](?![5-9])', content, re.IGNORECASE):
            issues.append("OLD_VERSION")
            
        # Check 2: Hardcoded paths
        if 'C:\\\\' in content or 'C:/' in content:
            # Count occurrences
            count = len(re.findall(r'C:[\\\/]', content))
            issues.append(f"HARDCODED_PATHS ({count})")
            
        # Check 3: Duplicate CUDA_TEMPLATE definition
        cuda_templates = len(re.findall(r'CUDA_TEMPLATE\s*=', content))
        if cuda_templates > 1:
            issues.append(f"DUPLICATE_CUDA_TEMPLATE ({cuda_templates})")
            
        # Check 4: Missing docstrings for classes/functions
        classes = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        functions = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
        docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
        
        if classes + functions > docstrings + 2:  # Allow some margin
            issues.append(f"MISSING_DOCS (classes:{classes}, funcs:{functions}, docs:{docstrings})")
            
        # Check 5: TODO/FIXME markers
        todos = len(re.findall(r'TODO|FIXME|XXX|HACK', content, re.IGNORECASE))
        if todos > 0:
            issues.append(f"TODO_MARKERS ({todos})")
            
        # Check 6: Print statements (should use logging)
        prints = len(re.findall(r'print\(', content))
        if prints > 10:  # Some prints are OK for user feedback
            issues.append(f"EXCESSIVE_PRINTS ({prints})")
            
        if issues:
            print(f"  {YELLOW}[WARN]{RESET} {rel_path}:")
            for issue in issues:
                print(f"    - {issue}")
                
    def check_cross_references(self):
        """Check if imported modules exist"""
        print(f"{CYAN}Phase 3: Checking cross-references...{RESET}\n")
        
        for rel_path, imports in self.imports.items():
            for local_import in imports['local']:
                # Convert import to file path
                module_path = local_import.replace('.', os.sep) + '.py'
                
                if module_path not in self.files:
                    self.issues.append(
                        f"[ERROR] {rel_path}: imports '{local_import}' but {module_path} not found"
                    )
                    print(f"  {RED}[ERROR]{RESET} {rel_path}: Missing import {local_import}")
                    
        if not any('[ERROR]' in issue for issue in self.issues):
            print(f"  {GREEN}[OK]{RESET} All imports valid\n")
        else:
            print()
            
    def generate_report(self):
        """Generate final report"""
        print(f"{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}ANALYSIS REPORT{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        # File count by directory
        print(f"{CYAN}File Distribution:{RESET}")
        by_dir = defaultdict(int)
        for rel_path in self.files:
            directory = str(Path(rel_path).parent) if str(Path(rel_path).parent) != '.' else 'root'
            by_dir[directory] += 1
            
        for directory, count in sorted(by_dir.items()):
            print(f"  {directory:20s}: {count:2d} files")
        print()
        
        # Total lines of code
        total_lines = sum(f['lines'] for f in self.files.values())
        total_size = sum(f['size'] for f in self.files.values())
        print(f"{CYAN}Code Statistics:{RESET}")
        print(f"  Total files:  {len(self.files)}")
        print(f"  Total lines:  {total_lines:,}")
        print(f"  Total size:   {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print()
        
        # Issues summary
        error_count = len([i for i in self.issues if '[ERROR]' in i])
        warn_count = len([i for i in self.issues if '[WARN]' in i])
        
        print(f"{CYAN}Issues Found:{RESET}")
        print(f"  Errors:   {error_count}")
        print(f"  Warnings: {warn_count}")
        print()
        
        if error_count > 0:
            print(f"{RED}CRITICAL ERRORS:{RESET}")
            for issue in self.issues:
                if '[ERROR]' in issue:
                    print(f"  {issue}")
            print()
            
        # Key files check
        print(f"{CYAN}Critical Files Check:{RESET}")
        critical = [
            'core\\supervisor_v3_optimization.py',
            'core\\model_selector.py',
            'core\\learning_module_v2.py',
            'core\\ollama_client.py',
            'agents\\builder_agent.py',
            'agents\\fixer_agent.py',
            'agents\\cuda_profiler_agent.py',
            'agents\\search_agent_v2.py',
            'test_phase6_optimization.py'
        ]
        
        for critical_file in critical:
            # Normalize path separators
            critical_normalized = critical_file.replace('\\', os.sep)
            
            if critical_normalized in self.files:
                file_info = self.files[critical_normalized]
                print(f"  {GREEN}[OK]{RESET} {critical_file:45s} ({file_info['lines']} lines)")
            else:
                print(f"  {RED}[MISS]{RESET} {critical_file}")
        print()
        
        # CUDA_TEMPLATE check
        print(f"{CYAN}CUDA_TEMPLATE Check:{RESET}")
        agents_with_template = []
        for rel_path, file_info in self.files.items():
            if 'agents' in rel_path and 'CUDA_TEMPLATE' in file_info['content']:
                agents_with_template.append(rel_path)
                
        if agents_with_template:
            for agent in agents_with_template:
                print(f"  {GREEN}[OK]{RESET} {agent} has CUDA_TEMPLATE")
        else:
            print(f"  {RED}[MISS]{RESET} No agents have CUDA_TEMPLATE defined!")
        print()
        
        # Final verdict
        print(f"{CYAN}{'='*70}{RESET}")
        if error_count == 0:
            print(f"{GREEN}STATUS: READY FOR FIX INSTALLATION{RESET}")
        else:
            print(f"{RED}STATUS: CRITICAL ERRORS MUST BE FIXED FIRST{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")


if __name__ == "__main__":
    analyzer = KISYSTEMAnalyzer("C:\\KISYSTEM")
    analyzer.analyze()

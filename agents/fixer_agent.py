"""
FixerAgent v2.8.0 - ERROR-PATTERN AUTO-FIX

MAJOR UPGRADE v2.8.0 (2025-11-12 - Phase 1):
- ✅ Error-Pattern-Database Integration
- ✅ Automatic fixes for known CUDA errors (80% instant fix)
- ✅ Regex-based transformations (# → //, __declspec → __global__)
- ✅ VLA detection and conversion guidance
- ✅ Add missing includes automatically
- ✅ LLM fallback for unknown errors

INHERITED FROM v2.7.4:
- ✅ Ultra-conservative MINIMAL fixes
- ✅ Kernel deletion detection and rejection
- ✅ CUDA syntax validator
- ✅ Diff marker stripping
- ✅ Dynamic temperature adjustment
- ✅ Model escalation support
- ✅ Learning module integration

PERFORMANCE IMPACT:
- Known errors: 0.001s (regex) vs 5-10s (LLM)
- Expected: 80% of fixes instant, 20% LLM fallback
- Learning accumulation: Pattern database grows over time

Author: Jörg Bohne / Claude (Anthropic)
Created: 2025-11-12
Version: 2.8.0
Status: PHASE 1 - PATTERN AUTO-FIX
"""

import logging
import re
import json
import sys
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FixerAgent:
    """
    Intelligent Code Fixing Agent with Error Pattern Auto-Fix
    
    Features:
    - Pattern-based auto-fixes (instant, no LLM)
    - Regex transformations for common errors
    - LLM fallback for unknown errors
    - Learning from fixes
    
    API compliant with supervisor_v3.py v3.11+.
    """
    
    def __init__(
        self,
        learning_module=None,
        search_agent=None,
        error_patterns_path: str = "C:/KISYSTEM/config/cuda_error_patterns.json"
    ):
        """
        Initialize FixerAgent with error pattern database.
        
        Args:
            learning_module: Optional learning module for caching
            search_agent: Optional search agent for future enhancements
            error_patterns_path: Path to CUDA error patterns JSON
            
        This signature is backward-compatible with supervisor_v3.py.
        """
        # Create internal OllamaClient (pattern from BuilderAgent)
        sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
        from ollama_client import OllamaClient
        
        self.ollama_client = OllamaClient()
        self.learning_module = learning_module
        self.search_agent = search_agent
        self.name = "FixerAgent"
        self.version = "2.8.0"
        
        # Load error patterns
        self.error_patterns = self._load_error_patterns(error_patterns_path)
        self.patterns_loaded = len(self.error_patterns) if self.error_patterns else 0
        
        # Statistics
        self.stats = {
            'total_fixes': 0,
            'pattern_fixes': 0,
            'llm_fixes': 0,
            'failed_fixes': 0
        }
        
        logger.info(f"{self.name} v{self.version} initialized with {self.patterns_loaded} patterns")
        print(f"[{self.name}] ✓ Initialized v{self.version}")
        if self.patterns_loaded > 0:
            print(f"[{self.name}] ✓ Loaded {self.patterns_loaded} error patterns")
        else:
            print(f"[{self.name}] ⚠ No error patterns loaded (LLM-only mode)")
    
    def _load_error_patterns(self, path: str) -> Optional[List[Dict]]:
        """Load error patterns from JSON file"""
        try:
            pattern_file = Path(path)
            if not pattern_file.exists():
                logger.warning(f"Pattern file not found: {path}")
                return None
            
            with open(pattern_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            patterns = data.get('patterns', [])
            # Sort by priority (lower number = higher priority)
            patterns.sort(key=lambda p: p.get('priority', 999))
            
            logger.info(f"Loaded {len(patterns)} error patterns from {path}")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return None
    
    def _match_error_pattern(self, error: str) -> Optional[Dict]:
        """
        Match error message against known patterns.
        Returns pattern dict if match found, None otherwise.
        """
        if not self.error_patterns:
            return None
        
        for pattern in self.error_patterns:
            # Check regex match
            error_regex = pattern.get('error_regex')
            if error_regex and re.search(error_regex, error, re.IGNORECASE | re.DOTALL):
                logger.info(f"Matched pattern: {pattern['name']} (ID: {pattern['id']})")
                return pattern
            
            # Check nvcc_error_patterns (exact substring match)
            nvcc_patterns = pattern.get('nvcc_error_patterns', [])
            for nvcc_pattern in nvcc_patterns:
                if nvcc_pattern.lower() in error.lower():
                    logger.info(f"Matched pattern: {pattern['name']} (ID: {pattern['id']})")
                    return pattern
        
        return None
    
    def _apply_pattern_fix(self, code: str, pattern: Dict, error: str) -> Tuple[str, bool, str]:
        """
        Apply automatic fix based on pattern.
        
        Returns:
            (fixed_code, success, fix_description)
        """
        fix_type = pattern.get('fix_type')
        pattern_name = pattern.get('name', 'Unknown')
        
        try:
            if fix_type == 'regex_replace':
                # Simple regex replacement
                search = pattern.get('search_pattern')
                replace = pattern.get('replace_pattern', '')
                
                if not search:
                    return code, False, "No search pattern defined"
                
                fixed_code = re.sub(search, replace, code, flags=re.MULTILINE)
                
                # Check if anything changed
                if fixed_code != code:
                    lines_changed = len([i for i, (a, b) in enumerate(zip(code.split('\n'), fixed_code.split('\n'))) if a != b])
                    desc = f"Pattern fix: {pattern_name} ({lines_changed} lines)"
                    logger.info(desc)
                    print(f"[{self.name}] ✓ {desc}")
                    return fixed_code, True, desc
                else:
                    return code, False, "No changes made (pattern didn't match)"
            
            elif fix_type == 'remove_lines':
                # Remove lines matching pattern
                search = pattern.get('search_pattern')
                if not search:
                    return code, False, "No search pattern defined"
                
                lines = code.split('\n')
                filtered_lines = [
                    line for line in lines 
                    if not re.search(search, line)
                ]
                
                if len(filtered_lines) < len(lines):
                    fixed_code = '\n'.join(filtered_lines)
                    removed = len(lines) - len(filtered_lines)
                    desc = f"Pattern fix: {pattern_name} (removed {removed} lines)"
                    logger.info(desc)
                    print(f"[{self.name}] ✓ {desc}")
                    return fixed_code, True, desc
                else:
                    return code, False, "No lines matched removal pattern"
            
            elif fix_type == 'add_include':
                # Add missing include at top
                include = pattern.get('include')
                if not include:
                    return code, False, "No include specified"
                
                # Check if already present
                if include in code:
                    return code, False, f"{include} already present"
                
                # Add after last #include or at top
                lines = code.split('\n')
                last_include_idx = -1
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include_idx = i
                
                if last_include_idx >= 0:
                    # Insert after last include
                    lines.insert(last_include_idx + 1, include)
                else:
                    # Insert at beginning
                    lines.insert(0, include)
                    lines.insert(1, '')  # Add blank line
                
                fixed_code = '\n'.join(lines)
                desc = f"Pattern fix: {pattern_name} (added {include})"
                logger.info(desc)
                print(f"[{self.name}] ✓ {desc}")
                return fixed_code, True, desc
            
            elif fix_type == 'add_defines':
                # Add math defines
                defines = pattern.get('defines', [])
                if not defines:
                    return code, False, "No defines specified"
                
                # Check if already present
                if all(d in code for d in defines):
                    return code, False, "Defines already present"
                
                # Find last #include
                lines = code.split('\n')
                last_include_idx = -1
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include_idx = i
                
                # Insert defines after last include
                insert_idx = last_include_idx + 1 if last_include_idx >= 0 else 0
                
                for define in reversed(defines):
                    if define not in code:
                        lines.insert(insert_idx, define)
                
                lines.insert(insert_idx, '')  # Add blank line
                
                fixed_code = '\n'.join(lines)
                desc = f"Pattern fix: {pattern_name} (added {len(defines)} defines)"
                logger.info(desc)
                print(f"[{self.name}] ✓ {desc}")
                return fixed_code, True, desc
            
            elif fix_type == 'vla_to_malloc':
                # VLA requires code analysis - provide guidance to LLM
                desc = f"Pattern detected: {pattern_name} (needs LLM for conversion)"
                logger.info(f"{desc} - falling back to LLM with guidance")
                print(f"[{self.name}] ⚠ {desc}")
                # Return False to trigger LLM, but with VLA-specific prompt
                return code, False, "VLA detected, requires malloc conversion (LLM needed)"
            
            else:
                logger.warning(f"Unknown fix_type: {fix_type}")
                return code, False, f"Unknown fix_type: {fix_type}"
                
        except Exception as e:
            logger.error(f"Pattern fix failed: {e}")
            return code, False, f"Pattern fix error: {e}"
    
    def _count_cuda_kernels(self, code: str) -> int:
        """Count __global__ kernels in CUDA code."""
        return len(re.findall(r'__global__\s+void\s+\w+\s*\(', code))
    
    def _validate_cuda_syntax(self, code: str, language: str) -> str:
        """
        Validate and fix CUDA syntax issues.
        
        NOTE: This is now redundant with pattern-based fixes,
        but kept for backward compatibility.
        """
        if language.lower() not in ['cuda', 'cu', 'cpp', 'c++', 'c']:
            return code
        
        lines = code.split('\n')
        fixed_lines = []
        
        # Valid preprocessor directives
        valid_directives = (
            '#include', '#define', '#pragma', '#ifndef', 
            '#ifdef', '#endif', '#if', '#else', '#elif', '#undef'
        )
        
        for line in lines:
            stripped = line.lstrip()
            
            # Preserve valid preprocessor directives
            if stripped.startswith(valid_directives):
                fixed_lines.append(line)
                continue
            
            # Convert invalid # comments to // comments
            if stripped.startswith('#'):
                fixed_line = line.replace('#', '//', 1)
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _strip_diff_markers(self, code: str) -> str:
        """Strip diff markers that LLMs sometimes generate."""
        # Remove diff-style markers
        code = re.sub(r'(OLD|NEW|CHANGED)\s*\(.*?\):\s*\n?', '', code, flags=re.MULTILINE)
        
        # Remove markdown code fences
        code = re.sub(r'```(?:cuda|cpp|c\+\+)?\s*\n?', '', code)
        code = re.sub(r'```\s*\n?', '', code)
        
        # Remove explanatory preambles
        code = re.sub(r'^.*?(here\s+is|fixed|corrected).*?code.*?:\s*\n', '', code, flags=re.IGNORECASE | re.MULTILINE)
        
        return code.strip()
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response, handling various formats."""
        # Strip diff markers first
        code = self._strip_diff_markers(response)
        
        # If response looks like pure code (starts with # or /), return as is
        if code.startswith(('#', '/', '__', 'extern', 'namespace', 'using', 'class', 'struct')):
            return code
        
        # Try to find code block
        code_match = re.search(r'```(?:cuda|cpp|c\+\+)?\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        return code
    
    def _create_ultra_conservative_prompt(
        self,
        code: str,
        error: str,
        failure_count: int,
        matched_pattern: Optional[Dict] = None
    ) -> str:
        """
        Ultra-conservative fix prompt.
        
        If pattern was matched but couldn't be auto-fixed (e.g. VLA),
        include pattern-specific guidance.
        """
        
        # Truncate error if too long
        error_summary = error[:500] + '...' if len(error) > 500 else error
        
        # Add pattern-specific guidance if available
        pattern_guidance = ""
        if matched_pattern:
            pattern_guidance = f"""
PATTERN DETECTED: {matched_pattern['name']}
DESCRIPTION: {matched_pattern['description']}
CORRECT EXAMPLE: {matched_pattern['examples']['correct']}
WRONG EXAMPLE: {matched_pattern['examples']['wrong']}
"""
        
        prompt = f"""You are fixing a compilation error. Be EXTREMELY conservative.

{pattern_guidance}

CRITICAL RULES (FAILURE = REJECTION):
1. Change EXACTLY ONE LINE (or add ONE line if something is missing)
2. NEVER delete working code
3. NEVER rewrite entire functions
4. NEVER change code that isn't broken
5. Output ONLY the complete fixed code (no explanations, no diffs, no markdown)

CUDA SYNTAX RULES:
- Use // for comments, NEVER use # for comments
- WRONG: # Define constants
- CORRECT: // Define constants
- Only use # for preprocessor: #include, #define, #pragma
- WRONG: __declspec(__global__) void kernel()
- CORRECT: __global__ void kernel()

COMPILATION ERROR:
{error_summary}

ORIGINAL CODE (DO NOT DELETE ANYTHING):
{code}

EXAMPLES OF GOOD FIXES:

Example 1 - Missing constant:
ERROR: 'M_PI' was not declared
FIX: Add ONE line at top: #define M_PI 3.14159265358979323846

Example 2 - Wrong variable scope:
ERROR: identifier "coeffs" is undefined in device code
FIX: Add __constant__ before the variable declaration

Example 3 - Invalid # comment:
ERROR: Ungültiger Präprozessorbefehl "Define"
FIX: Change # Define constants → // Define constants

Example 4 - __declspec error:
ERROR: cannot overload functions distinguished by return type alone
FIX: Change __declspec(__global__) → __global__

YOUR TASK:
1. Find the EXACT line causing the error
2. Make the SMALLEST possible change to fix it
3. Output the COMPLETE corrected code (preserve everything else exactly)
4. DO NOT write "OLD:" or "NEW:" or any diff markers
5. DO NOT explain - just output the fixed code
6. Use // for comments, NOT #

FIXED CODE:"""
        
        return prompt
    
    def _create_deep_debugging_prompt(
        self,
        code: str,
        error: str,
        failure_count: int,
        matched_pattern: Optional[Dict] = None
    ) -> str:
        """
        Deep debugging prompt for persistent errors.
        """
        
        pattern_guidance = ""
        if matched_pattern:
            pattern_guidance = f"""
PATTERN: {matched_pattern['name']}
{matched_pattern['description']}
Example Fix: {matched_pattern['examples']['correct']}
"""
        
        prompt = f"""You are debugging a persistent CUDA compilation error (attempt {failure_count}/5).

{pattern_guidance}

ANALYSIS REQUIRED:
1. What is the ROOT CAUSE of this error?
2. What is the SMALLEST possible fix?
3. Are there multiple related errors that need ONE fix?

CRITICAL RULES:
- Output ONLY complete working code (no explanations, no diffs)
- Change as LITTLE as possible
- NEVER delete working __global__ kernels
- NEVER rewrite large code sections
- Use // for comments in CUDA code, NOT # (except #include, #define, #pragma)
- Use __global__ NOT __declspec(__global__)

ERROR (attempt {failure_count}):
{error}

CURRENT CODE:
{code}

THINK STEP BY STEP:
1. Which specific line/symbol is causing the error?
2. What is the MINIMAL change needed?
3. Will this change break anything else?
4. Are there any # comments that should be //?
5. Are there any __declspec that should be __global__?

OUTPUT: Complete fixed code only (no markdown, no explanations)

FIXED CODE:"""
        
        return prompt
    
    async def fix(
        self,
        code: str,
        error: str,
        language: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Fix code with pattern auto-fix + LLM fallback.
        
        NEW in v2.8: Checks error against pattern database first,
        applies automatic fix if pattern matches. Only calls LLM
        if no pattern match or pattern fix fails.
        
        Args:
            code: Broken code to fix
            error: Compilation error message
            language: Programming language (cuda, python, cpp)
            context: Optional dict with:
                - failure_count: Number of previous attempts (0-indexed)
                - model_override: Specific model to use
                
        Returns:
            Dict with:
                - status: 'success' or 'error'
                - code: Fixed code (or original if failed)
                - model_used: Model that generated the fix (or 'pattern' for auto-fix)
                - explanation: What was fixed
                - metadata: Additional info
        """
        
        self.stats['total_fixes'] += 1
        
        print(f"[{self.name}] 🔧 Fixing {language} code...")
        print(f"[{self.name}] Error (first 200 chars): {error[:200]}...")
        
        # Extract context
        context = context or {}
        failure_count = context.get('failure_count', 0) + 1  # Convert to 1-indexed
        model_override = context.get('model_override')
        
        # Count kernels in original
        original_kernel_count = self._count_cuda_kernels(code)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: PATTERN MATCHING & AUTO-FIX
        # ═══════════════════════════════════════════════════════════════
        
        matched_pattern = self._match_error_pattern(error)
        
        if matched_pattern:
            print(f"[{self.name}] 🎯 Pattern matched: {matched_pattern['name']}")
            
            # Try automatic fix
            fixed_code, fix_success, fix_desc = self._apply_pattern_fix(code, matched_pattern, error)
            
            if fix_success:
                # PATTERN FIX SUCCESSFUL - Skip LLM entirely!
                self.stats['pattern_fixes'] += 1
                
                # Validate CUDA syntax (redundant but safe)
                fixed_code = self._validate_cuda_syntax(fixed_code, language)
                
                # Check for kernel deletion
                fixed_kernel_count = self._count_cuda_kernels(fixed_code)
                
                if fixed_kernel_count < original_kernel_count:
                    logger.error(f"Pattern fix deleted kernels ({original_kernel_count} → {fixed_kernel_count})")
                    print(f"[{self.name}] ⚠️ Pattern fix rejected: Kernels deleted")
                    # Fall through to LLM
                else:
                    # SUCCESS - Return immediately
                    print(f"[{self.name}] ✅ PATTERN FIX SUCCESSFUL (No LLM needed)")
                    
                    # Log to learning module
                    if self.learning_module:
                        try:
                            self.learning_module.log_result(
                                task_type='fix_pattern',
                                model_used='pattern_' + matched_pattern['id'],
                                success=True,
                                domain=language,
                                error_type=matched_pattern['id']
                            )
                        except Exception as e:
                            logger.warning(f"Failed to log to learning module: {e}")
                    
                    return {
                        'status': 'success',
                        'code': fixed_code,
                        'model_used': f"pattern_{matched_pattern['id']}",
                        'explanation': fix_desc,
                        'metadata': {
                            'original_size': len(code),
                            'fixed_size': len(fixed_code),
                            'pattern_id': matched_pattern['id'],
                            'pattern_name': matched_pattern['name'],
                            'fix_type': matched_pattern['fix_type'],
                            'llm_used': False,
                            'instant_fix': True
                        }
                    }
            else:
                # Pattern matched but couldn't auto-fix (e.g. VLA)
                print(f"[{self.name}] ⚠️ Pattern fix not applicable: {fix_desc}")
                print(f"[{self.name}] → Falling back to LLM with pattern guidance")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: LLM FALLBACK (No pattern match or pattern fix failed)
        # ═══════════════════════════════════════════════════════════════
        
        self.stats['llm_fixes'] += 1
        
        # Determine model
        model = model_override if model_override else "deepseek-coder-v2:16b"
        
        # Select prompt based on failure count (with pattern guidance if available)
        if failure_count <= 2:
            prompt = self._create_ultra_conservative_prompt(code, error, failure_count, matched_pattern)
        else:
            prompt = self._create_deep_debugging_prompt(code, error, failure_count, matched_pattern)
        
        print(f"[{self.name}] Step 4: Generating fix with {model}...")
        
        # Dynamic temperature
        temperature = min(0.3 + (failure_count * 0.1), 0.8)
        
        try:
            # Generate fix
            response = await self.ollama_client.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=8000
            )
            
            # Extract code
            fixed_code = self._extract_code(response)
            
            if not fixed_code or len(fixed_code) < 50:
                logger.error(f"Generated code too short: {len(fixed_code)} chars")
                self.stats['failed_fixes'] += 1
                return {
                    'status': 'error',
                    'code': code,
                    'model_used': model,
                    'error': f'Generated code too short ({len(fixed_code)} chars)'
                }
            
            # Validate CUDA syntax
            fixed_code = self._validate_cuda_syntax(fixed_code, language)
            
            # Check for kernel deletion
            fixed_kernel_count = self._count_cuda_kernels(fixed_code)
            
            if fixed_kernel_count < original_kernel_count:
                logger.error(
                    f"FIX REJECTED: Kernels deleted "
                    f"({original_kernel_count} → {fixed_kernel_count})"
                )
                print(f"[{self.name}] ⚠️ Fix rejected: Kernels deleted")
                self.stats['failed_fixes'] += 1
                
                return {
                    'status': 'error',
                    'code': code,
                    'model_used': model,
                    'error': f'Fix deleted {original_kernel_count - fixed_kernel_count} kernels (forbidden)',
                    'rejection_reason': 'kernel_deletion'
                }
            
            # Check code size change
            original_size = len(code)
            fixed_size = len(fixed_code)
            size_change_pct = abs(fixed_size - original_size) / original_size * 100
            
            if size_change_pct > 20:
                logger.warning(f"Large code change: {size_change_pct:.1f}%")
            
            print(f"[{self.name}] ✓ Fix generated")
            
            # Log to learning module
            if self.learning_module:
                try:
                    self.learning_module.log_result(
                        task_type='fix_llm',
                        model_used=model,
                        success=True,
                        domain=language,
                        error_type='compilation_error'
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to learning module: {e}")
            
            # Return success
            return {
                'status': 'success',
                'code': fixed_code,
                'model_used': model,
                'explanation': f'Fixed compilation error (attempt {failure_count})',
                'metadata': {
                    'original_size': original_size,
                    'fixed_size': fixed_size,
                    'size_change_pct': size_change_pct,
                    'temperature': temperature,
                    'failure_count': failure_count,
                    'cuda_syntax_validated': True,
                    'kernels_preserved': fixed_kernel_count == original_kernel_count,
                    'llm_used': True,
                    'pattern_guidance': matched_pattern['id'] if matched_pattern else None
                }
            }
            
        except Exception as e:
            logger.error(f"Fix generation error: {e}")
            self.stats['failed_fixes'] += 1
            return {
                'status': 'error',
                'code': code,
                'model_used': model,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get FixerAgent statistics"""
        if self.stats['total_fixes'] > 0:
            pattern_rate = self.stats['pattern_fixes'] / self.stats['total_fixes'] * 100
            llm_rate = self.stats['llm_fixes'] / self.stats['total_fixes'] * 100
            fail_rate = self.stats['failed_fixes'] / self.stats['total_fixes'] * 100
        else:
            pattern_rate = llm_rate = fail_rate = 0.0
        
        return {
            **self.stats,
            'pattern_fix_rate': pattern_rate,
            'llm_fix_rate': llm_rate,
            'fail_rate': fail_rate,
            'patterns_loaded': self.patterns_loaded
        }


def main():
    """Test FixerAgent v2.8.0 - Pattern Auto-Fix"""
    import asyncio
    
    async def test():
        print("\n" + "="*70)
        print("FIXERAGENT v2.8.0 - PATTERN AUTO-FIX TEST")
        print("="*70)
        
        # Test 1: Initialization
        print("\nTest 1: Initialization with Patterns")
        agent = FixerAgent(
            learning_module=None,
            search_agent=None,
            error_patterns_path="C:/KISYSTEM/config/cuda_error_patterns.json"
        )
        print(f"✓ Patterns loaded: {agent.patterns_loaded}")
        
        # Test 2: Pattern matching
        print("\nTest 2: Pattern Matching")
        test_error = 'error: invalid preprocessing directive #Define'
        pattern = agent._match_error_pattern(test_error)
        if pattern:
            print(f"✓ Matched pattern: {pattern['name']}")
        else:
            print("✗ No pattern matched")
        
        # Test 3: Pattern auto-fix
        print("\nTest 3: Pattern Auto-Fix")
        test_code = "# Define constants\n__global__ void test() {}"
        test_error = 'error: invalid preprocessing directive'
        
        result = await agent.fix(
            code=test_code,
            error=test_error,
            language='cuda',
            context={'failure_count': 0}
        )
        
        print(f"Status: {result['status']}")
        print(f"Model: {result['model_used']}")
        print(f"Instant Fix: {result['metadata'].get('instant_fix', False)}")
        
        # Test 4: Stats
        print("\nTest 4: Statistics")
        stats = agent.get_stats()
        print(f"Total fixes: {stats['total_fixes']}")
        print(f"Pattern fixes: {stats['pattern_fixes']} ({stats['pattern_fix_rate']:.1f}%)")
        print(f"LLM fixes: {stats['llm_fixes']} ({stats['llm_fix_rate']:.1f}%)")
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
    
    asyncio.run(test())


if __name__ == "__main__":
    main()

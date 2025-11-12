"""
FixerAgent V2.6.5 - CRITICAL: Fixed Explanation Detection Bug

CHANGES FROM V2.6.4:
- FIX: Explanation detection now checks FIRST 300 chars only (not full response)
- FIX: Removed false-positive protection that broke detection
- FIX: Added more explanation phrases ('to fix', 'the fix', 'we need to')

BUG in v2.6.4: 
  Line 329 had `and phrase not in original_code.lower()` 
  → If original code contained phrase, detection failed
  → LLM wrote "Based on..." directly into .cu file
  → Result: "identifier 'Based' is undefined"

Version History:
- v2.5: Improved prompts with "MINIMAL SURGICAL FIX"
- v2.6: Diff-mode + validation guards (RUN 37.2+)
- v2.6.1: Added .fix() wrapper for Supervisor v3.7 compatibility
- v2.6.2: Fixed CodeExtractor import error
- v2.6.3: Fixed __init__ parameter mismatch
- v2.6.4: Strict diff-mode enforcement (RUN 37.2 Fix)
- v2.6.5: Fixed explanation detection (checks first 300 chars only)
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
import re

from core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class FixerAgent:
    """
    Fixes compilation and runtime errors in generated code.
    V2.6.5: Fixed explanation detection bug (checks first 300 chars only).
    """
    
    def __init__(
        self, 
        workspace: str = None,
        learning_module = None,
        search_agent = None
    ):
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.ollama = OllamaClient()
        self.learning = learning_module  # For future use
        self.search = search_agent       # For future use
        
        # Escalation chain with stop-loss
        self.escalation_chain = [
            'deepseek-coder-v2:16b',  # Level 1: Fast, good for simple fixes
            'qwen2.5:32b',             # Level 2: Better reasoning
            'qwen2.5-coder:32b'        # Level 3: Complex CUDA fixes
        ]
        
        # Stop-loss: Max 2 escalations
        self.max_escalation_level = 2
        
        logger.info("[FixerAgent] ✓ Initialized with Smart Escalation + Auto-Dependencies")
    
    def _analyze_fix_scope(self, code: str, error: str, language: str) -> Dict:
        """
        Analyze error and determine minimal fix scope.
        Returns fix strategy to guide LLM.
        """
        error_lower = error.lower()
        
        # Pattern 1: Trivial typo/syntax
        if any(pattern in error_lower for pattern in ['#define', 'undefined', 'undeclared']):
            if 'Define' in code and '#define' in error_lower:
                return {
                    'scope': 'SINGLE_LINE',
                    'strategy': 'Fix typo: #Define → #define',
                    'max_changes': 1,
                    'forbidden': ['delete functions', 'rewrite logic', 'change algorithms']
                }
            if 'M_PI' in error or 'PI' in error:
                return {
                    'scope': 'SINGLE_LINE',
                    'strategy': 'Add #define M_PI 3.14159265358979323846',
                    'max_changes': 1,
                    'forbidden': ['delete code', 'rewrite functions']
                }
        
        # Pattern 2: VLA (Variable Length Array)
        if 'constant value' in error_lower or 'vla' in error_lower:
            return {
                'scope': 'ALLOCATION_FIX',
                'strategy': 'Replace stack array with dynamic allocation or template',
                'max_changes': 3,
                'forbidden': ['delete main()', 'remove other functions', 'change algorithm']
            }
        
        # Pattern 3: Missing entry point
        if 'entry point' in error_lower or 'einstiegspunkt' in error_lower:
            return {
                'scope': 'ADD_MAIN',
                'strategy': 'Add missing main() function',
                'max_changes': 5,
                'forbidden': ['delete existing code', 'rewrite kernels']
            }
        
        # Pattern 4: CUDA-specific
        if language == 'cuda':
            if '__host__' in error or '__device__' in error:
                return {
                    'scope': 'CUDA_SPECIFIER',
                    'strategy': 'Add correct __constant__, __device__, or __host__ specifier',
                    'max_changes': 2,
                    'forbidden': ['rewrite kernel', 'change logic']
                }
        
        # Default: Conservative
        return {
            'scope': 'MINIMAL',
            'strategy': 'Fix only the specific error, preserve all other code',
            'max_changes': 5,
            'forbidden': ['delete functions', 'rewrite algorithms', 'change working code']
        }
    
    def _create_diff_prompt(
        self,
        code: str,
        error: str,
        fix_scope: Dict,
        language: str,
        search_results: Optional[str] = None,
        failure_count: int = 0
    ) -> str:
        """
        Create a diff-based fix prompt that enforces minimal changes.
        """
        # Temperature based on failure count
        temperature = 0.3 + (failure_count * 0.1)
        
        # Extract relevant error lines
        error_context = self._extract_error_context(error, code)
        
        prompt = f"""You are a SURGICAL CODE FIXER. Output ONLY a diff, NOTHING else.

FIX SCOPE ANALYSIS:
- Scope: {fix_scope['scope']}
- Strategy: {fix_scope['strategy']}
- Maximum changes allowed: {fix_scope['max_changes']} lines
- FORBIDDEN: {', '.join(fix_scope['forbidden'])}

ERROR TO FIX:
{error_context}

CURRENT CODE:
```{language}
{code}
```

CRITICAL OUTPUT RULES - READ CAREFULLY:
==================================================
DO NOT OUTPUT ANY TEXT EXCEPT THE DIFF FORMAT BELOW.
NO "Based on..."
NO "Here's what..."
NO "The issue is..."
NO EXPLANATIONS.
NO MARKDOWN BLOCKS (```diff).
NO COMMENTARY.
JUST THE OLD/NEW PAIRS.
==================================================

OUTPUT FORMAT - EXACTLY THIS, NOTHING MORE:

OLD (Line X):
<exact line from code above>

NEW (Line X):
<fixed line>

OLD (Line Y):
<exact line from code above>

NEW (Line Y):
<fixed line>

REPEAT FOR EACH LINE THAT NEEDS FIXING.
MAXIMUM {fix_scope['max_changes']} LINES.
DO NOT OUTPUT ANYTHING ELSE.

"""
        
        if search_results:
            prompt += f"""
SEARCH RESULTS (if helpful):
{search_results[:500]}

"""
        
        prompt += """
==================================================
EXAMPLE OF CORRECT OUTPUT (NO OTHER TEXT):

OLD (Line 7):
#Define M_PI 3.14159

NEW (Line 7):
#define M_PI 3.14159

==================================================
NOW OUTPUT YOUR DIFF. NO EXPLANATIONS. JUST OLD/NEW PAIRS:
"""
        
        return prompt
    
    def _extract_error_context(self, error: str, code: str) -> str:
        """
        Extract relevant error information and code context.
        """
        lines = error.split('\n')
        
        # Find line numbers in error
        line_numbers = []
        for line in lines:
            match = re.search(r'line (\d+)', line, re.IGNORECASE)
            if match:
                line_numbers.append(int(match.group(1)))
            match = re.search(r'\((\d+)\)', line)
            if match:
                line_numbers.append(int(match.group(1)))
        
        if not line_numbers:
            return error[:500]  # Just return truncated error
        
        # Extract code context around error lines
        code_lines = code.split('\n')
        context = []
        for line_num in set(line_numbers):
            if 1 <= line_num <= len(code_lines):
                start = max(0, line_num - 3)
                end = min(len(code_lines), line_num + 2)
                context.append(f"Lines {start+1}-{end}:")
                context.extend(f"  {i+1}: {code_lines[i]}" for i in range(start, end))
        
        return error[:300] + "\n\nCODE CONTEXT:\n" + "\n".join(context)
    
    def _validate_fix(
        self,
        old_code: str,
        new_code: str,
        error: str,
        language: str,
        fix_scope: Dict
    ) -> Dict:
        """
        Validate that fix is acceptable (not too aggressive).
        Returns {'valid': bool, 'reason': str}
        """
        old_size = len(old_code)
        new_size = len(new_code)
        size_change = abs(new_size - old_size) / old_size if old_size > 0 else 0
        
        # Rule 1: Size change < 20% (unless adding main)
        if size_change > 0.20 and fix_scope['scope'] != 'ADD_MAIN':
            return {
                'valid': False,
                'reason': f"Code size changed {size_change*100:.1f}% (>{20}% threshold). Too aggressive!"
            }
        
        # Rule 2: Shrinkage > 50% is CRITICAL (almost always wrong)
        # v2.6.4: Changed from 15% to 50% threshold
        if new_size < old_size * 0.50 and fix_scope['scope'] != 'ADD_MAIN':
            return {
                'valid': False,
                'reason': f"CRITICAL: Code shrank from {old_size} to {new_size} chars (-{(1-new_size/old_size)*100:.1f}%). Deleted >50% of code!"
            }
        
        # Rule 3: main() must not disappear (unless it's the fix)
        if language in ['cuda', 'cpp']:
            had_main = 'main(' in old_code
            has_main = 'main(' in new_code
            if had_main and not has_main and fix_scope['scope'] != 'ADD_MAIN':
                return {
                    'valid': False,
                    'reason': "main() function deleted! This is forbidden."
                }
        
        # Rule 4: CUDA kernels must not disappear
        if language == 'cuda':
            old_kernels = old_code.count('__global__')
            new_kernels = new_code.count('__global__')
            if old_kernels > new_kernels:
                return {
                    'valid': False,
                    'reason': f"CUDA kernels deleted ({old_kernels} → {new_kernels}). This is forbidden."
                }
        
        # Rule 5: Check if error line still exists unchanged (bad fix)
        error_lines = re.findall(r'line (\d+)', error.lower())
        if error_lines:
            line_num = int(error_lines[0])
            old_lines = old_code.split('\n')
            new_lines = new_code.split('\n')
            if 0 < line_num <= min(len(old_lines), len(new_lines)):
                if old_lines[line_num-1].strip() == new_lines[line_num-1].strip():
                    return {
                        'valid': False,
                        'reason': f"Error line {line_num} unchanged. Fix didn't address the problem."
                    }
        
        return {'valid': True, 'reason': 'Validation passed'}
    
    def _apply_diff_to_code(self, original_code: str, llm_output: str) -> Optional[str]:
        """
        If LLM returned a diff, apply it to original code.
        If LLM returned complete code, extract it.
        
        V2.6.5: Now checks ONLY first 300 chars for explanations (fixed bug from v2.6.4).
        """
        # V2.6.5 FIX: Check ONLY first 300 chars for explanatory text
        # Bug in v2.6.4: Checked full response AND required phrase NOT in original_code
        # → Failed to reject when LLM started with "Based on..." 
        explanation_phrases = [
            'based on',
            'here is',
            'here are',
            "here's",
            'the issue',
            'the problem',
            'the error',
            'to fix this',
            'the fix',
            'we need to',
            'you need to',
            'i suggest',
            'i recommend',
            'the solution',
            'it appears',
            'we should'
        ]
        
        # Only check BEGINNING of response (first 300 chars)
        # If LLM starts with explanation → it's not following diff format
        llm_start = llm_output[:300].lower()
        
        for phrase in explanation_phrases:
            if phrase in llm_start:
                # LLM gave explanation instead of diff
                logger.error(f"[FixerAgent] ❌ LLM gave explanatory text ('{phrase}') instead of diff - REJECTING")
                logger.error(f"[FixerAgent] LLM output (first 500 chars): {llm_output[:500]}...")
                return None
        
        # Check if output contains diff markers
        if 'OLD (' in llm_output and 'NEW (' in llm_output:
            # Parse diff format
            changes = []
            lines = llm_output.split('\n')
            i = 0
            while i < len(lines):
                if lines[i].startswith('OLD (Line'):
                    # Extract line number
                    match = re.search(r'Line (\d+)', lines[i])
                    if match:
                        line_num = int(match.group(1))
                        old_content = lines[i+1] if i+1 < len(lines) else ""
                        # Find corresponding NEW
                        if i+2 < len(lines) and lines[i+2].startswith('NEW (Line'):
                            new_content = lines[i+3] if i+3 < len(lines) else ""
                            changes.append((line_num, old_content.strip(), new_content.strip()))
                            i += 4
                            continue
                i += 1
            
            if changes:
                # Apply changes
                code_lines = original_code.split('\n')
                for line_num, old, new in changes:
                    if 0 < line_num <= len(code_lines):
                        code_lines[line_num - 1] = new
                return '\n'.join(code_lines)
        
        # Fallback: Extract complete code from LLM output
        # Simple inline extraction - look for code blocks
        code_block_pattern = r'```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, llm_output, re.DOTALL)
        
        if matches:
            # Return first code block found
            return matches[0].strip()
        
        # Last resort: if output looks like code (has #include, __global__, etc)
        if any(marker in llm_output for marker in ['#include', '__global__', '__device__', 'int main']):
            # Remove common markdown artifacts
            cleaned = llm_output.strip()
            cleaned = re.sub(r'^```.*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
            return cleaned.strip()
        
        return None
    
    async def fix_code(
        self,
        code: str,
        error: str,
        language: str = 'cuda',
        domain: str = 'cuda',
        search_results: Optional[str] = None,
        failure_count: int = 0,
        model_override: Optional[str] = None
    ) -> Dict:
        """
        Fix code with minimal surgical changes.
        
        Args:
            code: The code to fix
            error: The error message
            language: Programming language
            domain: Problem domain (for learning)
            search_results: Optional search results for context
            failure_count: Number of previous failures (for escalation)
            model_override: Force specific model (for Phase 7)
        
        Returns:
            Dict with 'status', 'code', 'model_used', 'escalation_level'
        """
        print("="*70)
        print("[FixerAgent] 🔧 Fixing {} code...".format(language))
        print(f"[FixerAgent] Error (first 200 chars): {error[:200]}...")
        print(f"[FixerAgent] Full error ({len(error)} chars):")
        print(error)
        print("="*70)
        
        # Step 1: Analyze fix scope
        fix_scope = self._analyze_fix_scope(code, error, language)
        logger.info(f"[FixerAgent] Fix scope: {fix_scope['scope']} - {fix_scope['strategy']}")
        
        # Step 2: Determine escalation level
        escalation_level = min(failure_count, self.max_escalation_level)
        
        # Step 3: Select model
        if model_override:
            # Phase 7: Use model from Meta-Supervisor
            if hasattr(model_override, 'model'):
                model = model_override.model
            else:
                model = str(model_override)
            logger.info(f"[FixerAgent] Using Phase 7 model: {model}")
        else:
            model = self.escalation_chain[escalation_level]
            logger.info(f"[FixerAgent] Escalation level {escalation_level}: {model}")
        
        # Step 4: Trigger search if high failure count
        if failure_count >= 3 and not search_results:
            logger.warning(f"[FixerAgent] ⚠️ {failure_count} failures - web search recommended")
        
        # Step 5: Create diff-based prompt
        prompt = self._create_diff_prompt(
            code=code,
            error=error,
            fix_scope=fix_scope,
            language=language,
            search_results=search_results,
            failure_count=failure_count
        )
        
        # Step 6: Generate fix
        print(f"[FixerAgent] Step 4: Generating fix with {model}...")
        
        temperature = 0.3 + (failure_count * 0.1)
        response = await self.ollama.generate(
            model=model,
            prompt=prompt,
            temperature=min(temperature, 0.8)
        )
        
        # Step 7: Apply diff or extract code
        fixed_code = self._apply_diff_to_code(code, response)
        
        if not fixed_code:
            logger.error("[FixerAgent] Failed to extract fixed code")
            return {
                'status': 'error',
                'error': 'Could not extract fixed code from LLM response',
                'model_used': model,
                'escalation_level': escalation_level
            }
        
        # Step 8: Validate fix
        validation = self._validate_fix(
            old_code=code,
            new_code=fixed_code,
            error=error,
            language=language,
            fix_scope=fix_scope
        )
        
        if not validation['valid']:
            logger.warning(f"[FixerAgent] ⚠️ Fix validation failed: {validation['reason']}")
            # Still return the code, but flag it
            return {
                'status': 'warning',
                'code': fixed_code,
                'model_used': model,
                'escalation_level': escalation_level,
                'validation_warning': validation['reason'],
                'old_size': len(code),
                'new_size': len(fixed_code)
            }
        
        print("[FixerAgent] ✓ Fix generated")
        print("="*70)
        print("[FixerAgent] Status: COMPLETED")
        print(f"[FixerAgent] Escalation Level: {escalation_level}")
        print(f"[FixerAgent] Code size: {len(code)} → {len(fixed_code)} chars ({((len(fixed_code)/len(code)-1)*100):+.1f}%)")
        print("="*70)
        
        return {
            'status': 'success',
            'code': fixed_code,
            'model_used': model,
            'escalation_level': escalation_level,
            'old_size': len(code),
            'new_size': len(fixed_code)
        }


    async def fix(
        self,
        code: str,
        error: str,
        language: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        COMPATIBILITY WRAPPER for old Supervisor API.
        Translates old .fix() calls to new .fix_code() API.
        
        Args:
            code: Code to fix
            error: Error message
            language: Programming language
            context: Optional context dict with failure_count, search_results
        
        Returns:
            Dict with 'status', 'fixed_code', 'model_used'
        """
        context = context or {}
        failure_count = context.get('failure_count', 0)
        search_results = context.get('search_results')
        domain = context.get('domain', 'cuda' if language == 'cuda' else 'generic')
        
        result = await self.fix_code(
            code=code,
            error=error,
            language=language,
            domain=domain,
            search_results=search_results,
            failure_count=failure_count
        )
        
        # Translate new format to old format
        if result['status'] in ['success', 'warning']:
            return {
                'status': 'completed',
                'fixed_code': result['code'],
                'model_used': result['model_used'],
                'escalation_level': result.get('escalation_level', 0)
            }
        else:
            return {
                'status': 'failed',
                'error': result.get('error', 'Unknown error'),
                'model_used': result.get('model_used', 'unknown')
            }


if __name__ == '__main__':
    # Test the fixer
    import sys
    
    test_code = """
#Define M_PI 3.14159

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= M_PI;
    }
}

int main() {
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    kernel<<<1, 256>>>(d_data, 1024);
    cudaFree(d_data);
    return 0;
}
"""
    
    test_error = """
code.cu(7): fatal error C1021: Ungültiger Präprozessorbefehl "Define".
"""
    
    async def test():
        fixer = FixerAgent()
        result = await fixer.fix_code(
            code=test_code,
            error=test_error,
            language='cuda',
            failure_count=0
        )
        print("\n" + "="*70)
        print("TEST RESULT:")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Fixed code:\n{result['code']}")
        print("="*70)
    
    asyncio.run(test())
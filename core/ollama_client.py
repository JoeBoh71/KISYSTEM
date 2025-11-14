"""
KISYSTEM Ollama Client
Real integration with Ollama models

Author: Jörg Bohne
Date: 2025-11-12
Version: 1.1 - CUDA C++ explicit prompting added
"""

import aiohttp
import json
from typing import Dict, List, Optional, AsyncIterator
import asyncio

# Import CUDA prompt template (Phase 2 Enhancement)
try:
    from cuda_prompt_template import CUDAPromptTemplate
    CUDA_TEMPLATE_AVAILABLE = True
except ImportError:
    CUDA_TEMPLATE_AVAILABLE = False
    print("[OllamaClient] Warning: cuda_prompt_template not found, using basic CUDA prompts")



class OllamaClient:
    """
    Client for communicating with Ollama API
    
    Features:
    - Async streaming responses
    - Token counting
    - Error handling
    - Timeout protection
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: int = 300
    ) -> str:
        """
        Generate response from Ollama model
        
        Args:
            model: Model name (e.g., "llama3.1:8b")
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max tokens to generate
            stream: Stream response (not implemented yet)
            timeout: Request timeout in seconds
            
        Returns:
            Generated text response
        """
        
        # CRITICAL: Check if model exists before attempting generation
        # Prevents waiting full timeout (up to 30min) for non-existent model
        model_available = await self.check_model(model)
        if not model_available:
            raise ValueError(
                f"Model '{model}' not available in Ollama. "
                f"Run 'ollama list' to see available models."
            )
        
        # Build request
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # We'll use non-streaming for simplicity
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            request_data["system"] = system
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/generate",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error ({response.status}): {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timed out after {timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 300
    ) -> str:
        """
        Chat completion (multi-turn conversation)
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout
            
        Returns:
            Assistant's response
        """
        
        # Check if model exists
        model_available = await self.check_model(model)
        if not model_available:
            raise ValueError(
                f"Model '{model}' not available in Ollama. "
                f"Run 'ollama list' to see available models."
            )
        
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error ({response.status}): {error_text}")
                    
                    result = await response.json()
                    return result.get("message", {}).get("content", "")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timed out after {timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    async def check_model(self, model: str) -> bool:
        """
        Check if model is available
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is available
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return model in models
            return False
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """
        List all available models
        
        Returns:
            List of model names
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplates:
    """Pre-built prompt templates for common tasks"""
    
    @staticmethod
    def code_generation(task: str, language: str, context: Optional[str] = None) -> str:
        """
        Template for code generation with explicit CUDA C++ handling
        
        CRITICAL: CUDA/cu language gets explicit prompt to prevent Numba/CuPy confusion.
        LLMs often interpret "cuda code" as Python Numba (@cuda.jit) or CuPy,
        which causes nvcc compilation errors.
        
        Args:
            task: Description of what to generate
            language: Programming language (cuda/cu gets special treatment)
            context: Optional additional context
            
        Returns:
            Prompt string optimized for the target language
        """
        
        # CUDA gets explicit treatment to prevent Numba/CuPy confusion
        # CUDA gets explicit treatment using enhanced template
        if language.lower() in ['cuda', 'cu']:
            if CUDA_TEMPLATE_AVAILABLE:
                # Use enhanced CUDA template with comprehensive rules
                return CUDAPromptTemplate.get_code_generation_prompt(task, context)
            else:
                # Fallback to basic prompt if template not available
                prompt = f"""Generate NATIVE CUDA C++ code (NOT Python, NOT Numba, NOT CuPy).

Task: {task}

CRITICAL: 
- Use __global__ for kernels
- Call kernels with <<<blocks, threads>>> syntax  
- Use // for comments (NOT #)
- Include <cuda_runtime.h>

Provide ONLY the code:
"""
                if context:
                    prompt += f"\nContext: {context}\n"
                return prompt
        else:
            # Standard prompt for other languages (Python, C++, etc.)
            prompt = f"""Generate clean, production-ready {language} code for the following task:

Task: {task}

Requirements:
- Write complete, working code
- Include error handling
- Add docstrings/comments
- Follow best practices for {language}
- Make it efficient and maintainable
"""
        
        if context:
            prompt += f"\nAdditional Context:\n{context}\n"
        
        prompt += f"\nProvide ONLY the code, no explanations:\n\n"
        
        return prompt
    
    @staticmethod
    def test_generation(code: str, framework: str, language: str) -> str:
        """Template for test generation"""
        
        return f"""Generate comprehensive {framework} tests for the following {language} code:

Code to test:
```{language}
{code}
```

Requirements:
- Use {framework} framework
- Cover all major functions/methods
- Include edge cases and error handling
- Add test documentation
- Make tests maintainable

Provide ONLY the test code, no explanations:

"""
    
    @staticmethod
    def bug_fix(code: str, error: str, language: str, escalation: int = 0) -> str:
        """Template for bug fixing with CUDA-specific support"""
        
        # Use enhanced CUDA template for CUDA code
        if language.lower() in ['cuda', 'cu'] and CUDA_TEMPLATE_AVAILABLE:
            return CUDAPromptTemplate.get_bug_fix_prompt(code, error, escalation)
        
        # Standard fix for other languages
        if escalation >= 3:
            # Deep debugging mode
            return f"""DEEP DEBUGGING MODE - Root Cause Analysis

The following {language} code has failed {escalation} times.
Previous fixes did not work. Perform thorough root cause analysis.

Broken Code:
```{language}
{code}
```

Error:
{error}

Analyze:
1. What is the ROOT CAUSE of this error?
2. Why would previous standard fixes have failed?
3. What non-obvious issues might exist?
4. What is the CORRECT solution?

Provide the fixed code with brief explanation of the root cause:

"""
        else:
            # Standard fix
            return f"""Fix the following {language} code that produces an error:

Broken Code:
```{language}
{code}
```

Error:
{error}

Requirements:
- Fix the error completely
- Preserve original functionality
- Add comments explaining the fix
- Ensure code is production-ready

Provide ONLY the fixed code:

"""


# ============================================================================
# TESTING
# ============================================================================

async def test_ollama_client():
    """Test Ollama client with real model"""
    
    print("="*70)
    print("OLLAMA CLIENT TEST")
    print("="*70)
    
    client = OllamaClient()
    
    # Check connection
    print("\n[1] Checking Ollama connection...")
    models = await client.list_models()
    
    if not models:
        print("✗ Ollama not running or no models available")
        print("  Start Ollama: ollama serve")
        return False
    
    print(f"✓ Found {len(models)} models:")
    for model in models[:5]:  # Show first 5
        print(f"  - {model}")
    
    # Test simple generation
    print("\n[2] Testing code generation...")
    
    test_model = "llama3.1:8b"
    
    if test_model not in models:
        print(f"✗ {test_model} not available")
        test_model = models[0]
        print(f"  Using {test_model} instead")
    
    prompt = PromptTemplates.code_generation(
        task="Write a Python function that adds two numbers",
        language="python"
    )
    
    print(f"Generating with {test_model}...")
    
    try:
        response = await client.generate(
            model=test_model,
            prompt=prompt,
            temperature=0.3,
            timeout=60
        )
        
        print(f"\n✓ Generated {len(response)} characters")
        print(f"\nResponse preview:")
        print("-" * 70)
        print(response[:300])
        if len(response) > 300:
            print("...")
        print("-" * 70)
        
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_ollama_client())
    exit(0 if success else 1)

"""
KISYSTEM Model Selector - Phase 5 Complete
Smart model routing based on task complexity

Author: Jörg Bohne
Date: 2025-11-06
Version: 2.0 (with Workflow Engine integration)
"""

from typing import Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    size_gb: float
    estimated_time: str
    use_cases: list[str]
    vram_required: int  # in GB


# ============================================================================
# MODEL STRATEGY - Based on available Ollama models
# ============================================================================

MODEL_STRATEGY = {
    "simple": ModelConfig(
        name="llama3.1:8b",
        size_gb=4.9,
        estimated_time="~30s",
        use_cases=["functions", "simple_scripts", "quick_fixes", "utility_code"],
        vram_required=5
    ),
    
    "medium": ModelConfig(
        name="deepseek-coder-v2:16b",
        size_gb=8.9,
        estimated_time="~1min",
        use_cases=["classes", "algorithms", "data_structures", "standard_debug"],
        vram_required=9
    ),
    
    "complex": ModelConfig(
        name="qwen2.5-coder:32b",
        size_gb=19.0,
        estimated_time="~2min",
        use_cases=["cuda", "gpu_code", "frameworks", "architecture", "optimization"],
        vram_required=12  # With quantization
    ),
    
    "deep_debug": ModelConfig(
        name="deepseek-r1:32b",
        size_gb=19.0,
        estimated_time="~2min",
        use_cases=["root_cause_analysis", "complex_bugs", "reasoning", "investigation"],
        vram_required=12
    ),
    
    "search": ModelConfig(
        name="phi4:latest",
        size_gb=9.1,
        estimated_time="~20s",
        use_cases=["web_search", "documentation", "research", "quick_answers"],
        vram_required=9
    )
}


# ============================================================================
# COMPLEXITY DETECTION
# ============================================================================

class ComplexityDetector:
    """Detects task complexity from description and context"""
    
    # Keyword sets for different complexity levels
    COMPLEX_KEYWORDS = {
        'cuda', 'gpu', 'kernel', '__global__', '__device__',
        'framework', 'architecture', 'system_design',
        'multi-threaded', 'async', 'parallel', 'concurrent',
        'optimization', 'performance-critical', 'real-time',
        'asio', 'madi', 'audio_processing', 'dsp'
    }
    
    MEDIUM_KEYWORDS = {
        'class', 'algorithm', 'data structure',
        'sort', 'sorting', 'search', 'searching', 'tree', 'graph',
        'implement', 'build', 'create',
        'refactor', 'optimize', 'improve',
        'interface', 'api', 'module',
        'test suite', 'integration', 'component'
    }
    
    DEBUG_KEYWORDS = {
        'root cause', 'why', 'investigate',
        'mysterious', 'intermittent', 'random',
        'sometimes', 'occasionally', 'strange',
        'undefined behavior', 'memory leak', 'segfault'
    }
    
    SIMPLE_INDICATORS = {
        'function', 'helper', 'utility',
        'print', 'log', 'format',
        'simple', 'quick', 'small'
    }
    
    @staticmethod
    def detect(task: str, context: Optional[Dict] = None) -> str:
        """
        Detect complexity level from task description
        
        Args:
            task: Task description string
            context: Optional context dict with additional info
            
        Returns:
            Complexity level: "simple", "medium", "complex", or "deep_debug"
        """
        
        task_lower = task.lower()
        
        # Check context for override
        if context and "complexity" in context:
            return context["complexity"]
        
        # Deep debug detection (highest priority)
        if any(kw in task_lower for kw in ComplexityDetector.DEBUG_KEYWORDS):
            return "deep_debug"
        
        # Complex task detection
        if any(kw in task_lower for kw in ComplexityDetector.COMPLEX_KEYWORDS):
            return "complex"
        
        # Medium task detection
        if any(kw in task_lower for kw in ComplexityDetector.MEDIUM_KEYWORDS):
            return "medium"
        
        # Explicit simple indicators
        if any(kw in task_lower for kw in ComplexityDetector.SIMPLE_INDICATORS):
            return "simple"
        
        # Code size heuristic (if context provided)
        if context:
            lines = context.get("lines_of_code", 0)
            if lines > 200:
                return "complex"
            elif lines > 50:
                return "medium"
        
        # Default: simple for safety (fast turnaround)
        return "simple"


# ============================================================================
# MODEL SELECTOR
# ============================================================================

class ModelSelector:
    """
    Intelligent model selection based on task complexity
    
    Features:
    - Automatic complexity detection
    - Configurable model strategy
    - Fallback handling
    - Performance tracking
    """
    
    def __init__(self, strategy: Optional[Dict] = None):
        """
        Initialize selector with model strategy
        
        Args:
            strategy: Optional custom model strategy dict
        """
        self.strategy = strategy or MODEL_STRATEGY
        self.detector = ComplexityDetector()
        
        # Performance tracking
        self.stats = {
            "simple": {"count": 0, "avg_time": 0},
            "medium": {"count": 0, "avg_time": 0},
            "complex": {"count": 0, "avg_time": 0},
            "deep_debug": {"count": 0, "avg_time": 0}
        }
    
    def select_model(
        self, 
        task: str, 
        agent_type: str = "builder",
        context: Optional[Dict] = None,
        force_complexity: Optional[str] = None
    ) -> ModelConfig:
        """
        Select appropriate model for task
        
        Args:
            task: Task description
            agent_type: Type of agent (builder/tester/fixer)
            context: Optional additional context
            force_complexity: Override complexity detection
            
        Returns:
            ModelConfig for selected model
        """
        
        # Determine complexity
        if force_complexity:
            complexity = force_complexity
        else:
            complexity = self.detector.detect(task, context)
        
        # Special handling for fixer with repeated failures
        if agent_type == "fixer":
            failure_count = context.get("failure_count", 0) if context else 0
            
            if failure_count >= 3:
                # Escalate to deep debug after 3 failures
                complexity = "deep_debug"
                print(f"[ModelSelector] ⚠️ {failure_count} failures detected - escalating to deep_debug")
        
        # Get model config
        model_config = self.strategy[complexity]
        
        # Update stats
        self.stats[complexity]["count"] += 1
        
        print(f"[ModelSelector] 🎯 Task complexity: {complexity.upper()}")
        print(f"[ModelSelector] 🤖 Selected model: {model_config.name}")
        print(f"[ModelSelector] ⏱️ Estimated time: {model_config.estimated_time}")
        
        return model_config
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.stats
    
    def suggest_upgrade(self, current: str, reason: str = "task_failed") -> str:
        """
        Suggest model upgrade on failure
        
        Args:
            current: Current complexity level
            reason: Reason for upgrade
            
        Returns:
            Suggested upgraded complexity level
        """
        
        upgrade_path = {
            "simple": "medium",
            "medium": "complex",
            "complex": "deep_debug",
            "deep_debug": "deep_debug"  # Already at max
        }
        
        upgraded = upgrade_path.get(current, "medium")
        
        print(f"[ModelSelector] 📈 Upgrading: {current} → {upgraded} (reason: {reason})")
        
        return upgraded


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_model_for_task(task: str, agent_type: str = "builder") -> str:
    """
    Quick helper to get model name for a task
    
    Args:
        task: Task description
        agent_type: Agent type
        
    Returns:
        Model name string
    """
    selector = ModelSelector()
    config = selector.select_model(task, agent_type)
    return config.name


def get_all_models() -> Dict[str, ModelConfig]:
    """Get all available models in strategy"""
    return MODEL_STRATEGY


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL SELECTOR - Phase 5 Complete")
    print("="*70)
    
    selector = ModelSelector()
    
    # Test cases
    test_tasks = [
        ("Write a hello world function", "builder"),
        ("Implement bubble sort algorithm", "builder"),
        ("Create CUDA kernel for matrix multiplication", "builder"),
        ("Debug mysterious segfault in audio processing", "fixer"),
        ("Why does this code work sometimes but not always?", "fixer"),
    ]
    
    print("\nTest Cases:\n")
    
    for task, agent in test_tasks:
        print(f"\nTask: '{task}'")
        print(f"Agent: {agent}")
        config = selector.select_model(task, agent)
        print(f"→ Model: {config.name} ({config.size_gb}GB, {config.estimated_time})")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("Stats:", selector.get_stats())
    print("="*70)

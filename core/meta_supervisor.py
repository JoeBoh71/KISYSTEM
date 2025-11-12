"""
MetaSupervisor - Data-driven task prioritization based on historical performance

Part of Phase 7 optimization - provides Meta-Supervisor decision layer
that uses accumulated learning data to intelligently prioritize tasks
and recommend models based on historical success patterns.

Key Features:
- Historical learning data analysis
- Domain-model performance tracking
- Priority scoring based on multiple factors
- Confidence estimation for recommendations

Author: Jörg Bohne
Date: 2025-11-11
Version: 1.2 (Fixed JSON loading + added _get_recommended_model for HybridDecision)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ModelBias:
    """Represents model performance bias for a domain."""
    domain: str
    model: str
    success_rate: float
    avg_time: float
    total_tasks: int
    recent_failures: int
    confidence: float


@dataclass
class TaskPriority:
    """Represents prioritized task recommendation."""
    domain: str
    model: str
    priority: float
    reasoning: str
    confidence: float
    success_rate: float
    avg_time_hours: float


class MetaSupervisor:
    """
    Meta-Supervisor for data-driven task prioritization.
    
    Analyzes historical learning data to:
    - Identify best-performing models per domain
    - Calculate priority scores based on success patterns
    - Provide confidence estimates for recommendations
    - Track failure patterns and suggest escalations
    
    Priority Formula:
        P = 0.5 * success_rate + 
            0.2 * (1 / (1 + time_since_last)) + 
            0.2 * (cumulative_failures / 20) + 
            0.1 * risk_factor
    """
    
    def __init__(self, learning_log_path: Path):
        """Initialize Meta-Supervisor with learning data."""
        self.learning_log_path = Path(learning_log_path)
        self.learning_data: List[Dict] = []
        self.model_biases: Dict[str, List[ModelBias]] = {}  # domain -> list of model biases
        
        self._load_learning_data()
        self._calculate_model_biases()
    
    def _load_learning_data(self) -> None:
        """Load historical learning data from JSON log."""
        if not self.learning_log_path.exists():
            print(f"[MetaSupervisor] ⚠ Learning log not found: {self.learning_log_path}")
            self.learning_data = []
            return
        
        try:
            with open(self.learning_log_path, 'r', encoding='utf-8') as f:
                log = json.load(f)
            
            # CRITICAL FIX: Extract 'entries' list from the JSON structure
            # JSON format: {"project": "U3DAW", "entries": [...], "version": "2.0"}
            if isinstance(log, dict) and 'entries' in log:
                self.learning_data = log['entries']
            elif isinstance(log, list):
                # Fallback: if JSON is already a list
                self.learning_data = log
            else:
                print(f"[MetaSupervisor] ⚠ Unexpected JSON format: {type(log)}")
                self.learning_data = []
            
            print(f"[MetaSupervisor] ✓ Loaded {len(self.learning_data)} learning entries")
        except Exception as e:
            print(f"[MetaSupervisor] ✗ Error loading learning data: {e}")
            import traceback
            traceback.print_exc()
            self.learning_data = []
    
    def _calculate_model_biases(self) -> None:
        """Calculate model performance biases per domain from historical data."""
        if not self.learning_data:
            print("[MetaSupervisor] ⚠ No learning data available for bias calculation")
            return
        
        # Group by domain and model
        domain_model_stats: Dict[Tuple[str, str], List[Dict]] = {}
        
        for entry in self.learning_data:
            domain = entry.get('domain', 'unknown')
            model = entry.get('model', 'unknown')
            key = (domain, model)
            
            if key not in domain_model_stats:
                domain_model_stats[key] = []
            domain_model_stats[key].append(entry)
        
        # Calculate statistics for each domain-model pair
        for (domain, model), entries in domain_model_stats.items():
            total_tasks = len(entries)
            successes = sum(1 for e in entries if e.get('success', False))
            success_rate = successes / total_tasks if total_tasks > 0 else 0.0
            
            # Average time
            times = [e.get('time_hours', 0.0) for e in entries]
            avg_time = sum(times) / len(times) if times else 0.0
            
            # Recent failures (last 5 tasks)
            recent = sorted(entries, key=lambda x: x.get('date', ''), reverse=True)[:5]
            recent_failures = sum(1 for e in recent if not e.get('success', False))
            
            # Confidence based on sample size
            confidence = min(1.0, total_tasks / 10.0)  # Full confidence at 10+ samples
            
            bias = ModelBias(
                domain=domain,
                model=model,
                success_rate=success_rate,
                avg_time=avg_time,
                total_tasks=total_tasks,
                recent_failures=recent_failures,
                confidence=confidence
            )
            
            if domain not in self.model_biases:
                self.model_biases[domain] = []
            self.model_biases[domain].append(bias)
        
        # Sort by success rate within each domain
        for domain in self.model_biases:
            self.model_biases[domain].sort(key=lambda b: b.success_rate, reverse=True)
        
        print(f"[MetaSupervisor] ✓ Calculated biases for {len(self.model_biases)} domains")
    
    def get_model_bias(self, domain: str, model: str) -> Optional[ModelBias]:
        """Get performance bias for specific domain-model pair."""
        if domain not in self.model_biases:
            return None
        
        for bias in self.model_biases[domain]:
            if bias.model == model:
                return bias
        
        return None
    
    def get_best_model_for_domain(self, domain: str) -> Optional[ModelBias]:
        """Get best performing model for a domain."""
        if domain not in self.model_biases:
            return None
        
        # Return model with highest success rate
        if self.model_biases[domain]:
            return self.model_biases[domain][0]
        
        return None
    
    def _get_recommended_model(self, domain: str) -> Optional[str]:
        """
        Get recommended model name for a domain (for HybridDecision compatibility).
        
        Args:
            domain: Task domain
        
        Returns:
            Model name string, or None if no recommendation
        """
        best = self.get_best_model_for_domain(domain)
        return best.model if best else None
    
    def calculate_priority(
        self, 
        domain: str, 
        model: str,
        time_since_last: float = 1.0,
        cumulative_failures: int = 0,
        risk_factor: float = 0.5
    ) -> float:
        """
        Calculate priority score for a task.
        
        Args:
            domain: Task domain
            model: Model to use
            time_since_last: Days since last similar task
            cumulative_failures: Total failures for this task type
            risk_factor: Risk assessment (0-1)
        
        Returns:
            Priority score (0-1, higher is more urgent)
        """
        bias = self.get_model_bias(domain, model)
        
        if not bias:
            # No historical data - use neutral priority
            return 0.5
        
        # Priority formula components
        success_component = 0.5 * bias.success_rate
        time_component = 0.2 * (1.0 / (1.0 + time_since_last))
        failure_component = 0.2 * min(1.0, cumulative_failures / 20.0)
        risk_component = 0.1 * risk_factor
        
        priority = success_component + time_component + failure_component + risk_component
        
        return priority
    
    def get_top_priorities(self, top_n: int = 5) -> List[TaskPriority]:
        """
        Get top N priority task recommendations.
        
        Returns:
            List of TaskPriority objects sorted by priority
        """
        priorities = []
        
        for domain, biases in self.model_biases.items():
            for bias in biases:
                # Calculate priority assuming neutral time/failure factors
                priority = self.calculate_priority(
                    domain=domain,
                    model=bias.model,
                    time_since_last=7.0,  # Assume weekly
                    cumulative_failures=0,
                    risk_factor=0.5
                )
                
                reasoning = f"Historical {bias.success_rate:.1%} success rate over {bias.total_tasks} tasks"
                if bias.recent_failures > 0:
                    reasoning += f", {bias.recent_failures} recent failures"
                
                priorities.append(TaskPriority(
                    domain=domain,
                    model=bias.model,
                    priority=priority,
                    reasoning=reasoning,
                    confidence=bias.confidence,
                    success_rate=bias.success_rate,
                    avg_time_hours=bias.avg_time
                ))
        
        # Sort by priority
        priorities.sort(key=lambda p: p.priority, reverse=True)
        
        return priorities[:top_n]
    
    def should_escalate_model(self, domain: str, model: str, consecutive_failures: int = 2) -> bool:
        """
        Determine if model should be escalated based on failure pattern.
        
        Args:
            domain: Task domain
            model: Current model
            consecutive_failures: Number of consecutive failures
        
        Returns:
            True if model should be escalated
        """
        bias = self.get_model_bias(domain, model)
        
        if not bias:
            # No data - escalate after 2 failures
            return consecutive_failures >= 2
        
        # Escalate if:
        # 1. Success rate < 50% OR
        # 2. Recent failures >= 3 OR
        # 3. Consecutive failures >= 2
        if bias.success_rate < 0.5:
            return True
        
        if bias.recent_failures >= 3:
            return True
        
        if consecutive_failures >= 2:
            return True
        
        return False
    
    def get_alternative_model(self, domain: str, current_model: str) -> Optional[str]:
        """
        Get alternative model recommendation for a domain.
        
        Args:
            domain: Task domain
            current_model: Current failing model
        
        Returns:
            Alternative model name, or None if no better option
        """
        if domain not in self.model_biases:
            return None
        
        # Find models better than current
        for bias in self.model_biases[domain]:
            if bias.model != current_model and bias.success_rate > 0.7:
                return bias.model
        
        return None
    
    def get_confidence(self, domain: str, model: str) -> float:
        """Get confidence level for domain-model recommendation."""
        bias = self.get_model_bias(domain, model)
        return bias.confidence if bias else 0.0
    
    def export_summary(self) -> Dict:
        """Export summary statistics."""
        return {
            'total_entries': len(self.learning_data),
            'domains': list(self.model_biases.keys()),
            'domain_model_pairs': sum(len(biases) for biases in self.model_biases.values()),
            'best_models_per_domain': {
                domain: self.model_biases[domain][0].model 
                for domain in self.model_biases 
                if self.model_biases[domain]
            }
        }


# Testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = Path(__file__).parent.parent / "projects" / "U3DAW" / "learning_log.json"
    
    print("="*70)
    print("META-SUPERVISOR TEST")
    print("="*70)
    
    meta = MetaSupervisor(log_path)
    
    print(f"\nSummary: {meta.export_summary()}")
    
    print("\n--- TOP 5 PRIORITIES ---")
    for i, priority in enumerate(meta.get_top_priorities(5), 1):
        print(f"{i}. {priority.domain} / {priority.model}")
        print(f"   Priority: {priority.priority:.3f} (confidence: {priority.confidence:.2f})")
        print(f"   {priority.reasoning}")
    
    print("\n--- CUDA DOMAIN ANALYSIS ---")
    if 'cuda' in meta.model_biases:
        for bias in meta.model_biases['cuda']:
            print(f"{bias.model:30s}: {bias.success_rate:.1%} ({bias.total_tasks} tasks)")
    
    print("="*70)
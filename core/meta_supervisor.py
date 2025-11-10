"""
KISYSTEM Meta-Supervisor (Phase 7)
====================================

Data-driven prioritization and model selection optimization.

Priority Formula:
    P = 0.5(1-sr) + 0.2/(1+t) + 0.2c/20 + 0.1R

    Where:
    - sr: success_rate (0-1) - lower rate = higher priority
    - t: time_since_last_attempt (hours) - longer = higher priority
    - c: cumulative_failures (0-20+) - more failures = higher priority
    - R: risk_factor (0-1) - higher risk = higher priority

Author: Jörg Bohne / Bohne Audio
Last Updated: 2025-11-10
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics


@dataclass
class TaskPriority:
    """Task priority calculation result."""
    domain: str
    model: str
    priority: float
    success_rate: float
    time_since_last: float
    cumulative_failures: int
    risk_factor: float
    recommended_model: Optional[str] = None


@dataclass
class ModelBias:
    """Model performance bias for a specific domain."""
    model: str
    domain: str
    success_rate: float
    avg_time: float
    sample_size: int
    confidence: float  # 0-1, based on sample size


class MetaSupervisor:
    """
    Meta-Supervisor for KISYSTEM Phase 7.
    
    Aggregates learning data and provides data-driven recommendations for:
    1. Task prioritization (which tasks need attention)
    2. Model selection bias (which model works best for which domain)
    3. ROI optimization (cost-aware scheduling)
    """
    
    def __init__(self, learning_log_path: Path):
        """
        Initialize Meta-Supervisor.
        
        Args:
            learning_log_path: Path to learning_log.json from learning_module_v2
        """
        self.learning_log_path = learning_log_path
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
                self.learning_data = json.load(f)
            print(f"[MetaSupervisor] ✓ Loaded {len(self.learning_data)} learning entries")
        except Exception as e:
            print(f"[MetaSupervisor] ✗ Error loading learning data: {e}")
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
            success_count = sum(1 for e in entries if e.get('outcome') == 'SUCCESS')
            total_count = len(entries)
            success_rate = success_count / total_count if total_count > 0 else 0.0
            
            # Average time (sum of all phases)
            times = []
            for e in entries:
                timings = e.get('timings', {})
                total_time = timings.get('build', 0) + timings.get('test', 0) + timings.get('profile', 0)
                if total_time > 0:
                    times.append(total_time)
            
            avg_time = statistics.mean(times) if times else 0.0
            
            # Confidence based on sample size (more samples = higher confidence)
            # Sigmoid function: confidence = 1 / (1 + exp(-k(n-n0)))
            # Simplified: confidence = min(1.0, sample_size / 10)
            confidence = min(1.0, total_count / 10.0)
            
            bias = ModelBias(
                model=model,
                domain=domain,
                success_rate=success_rate,
                avg_time=avg_time,
                sample_size=total_count,
                confidence=confidence
            )
            
            if domain not in self.model_biases:
                self.model_biases[domain] = []
            self.model_biases[domain].append(bias)
        
        # Sort biases by success_rate (descending) within each domain
        for domain in self.model_biases:
            self.model_biases[domain].sort(key=lambda b: b.success_rate, reverse=True)
        
        print(f"[MetaSupervisor] ✓ Calculated biases for {len(self.model_biases)} domains")
    
    def calculate_task_priority(
        self,
        domain: str,
        model: str,
        risk_factor: float = 0.5
    ) -> TaskPriority:
        """
        Calculate priority for a task based on historical performance.
        
        Priority Formula:
            P = 0.5(1-sr) + 0.2/(1+t) + 0.2c/20 + 0.1R
        
        Args:
            domain: Task domain (e.g., 'cuda', 'asio', 'audio_dsp')
            model: Model used for task
            risk_factor: Task risk factor (0-1), default 0.5
        
        Returns:
            TaskPriority object with calculated priority and recommendations
        """
        # Filter entries for this domain-model combination
        relevant_entries = [
            e for e in self.learning_data
            if e.get('domain') == domain and e.get('model') == model
        ]
        
        if not relevant_entries:
            # No history: high priority (need to learn)
            return TaskPriority(
                domain=domain,
                model=model,
                priority=0.9,  # High priority for unknown territory
                success_rate=0.0,
                time_since_last=999.0,
                cumulative_failures=0,
                risk_factor=risk_factor,
                recommended_model=self._get_recommended_model(domain)
            )
        
        # Calculate success rate (sr)
        success_count = sum(1 for e in relevant_entries if e.get('outcome') == 'SUCCESS')
        total_count = len(relevant_entries)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Calculate time since last attempt (t) in hours
        latest_entry = max(relevant_entries, key=lambda e: e.get('timestamp', ''))
        latest_timestamp = datetime.fromisoformat(latest_entry.get('timestamp', datetime.now().isoformat()))
        time_since_last = (datetime.now() - latest_timestamp).total_seconds() / 3600.0
        
        # Calculate cumulative failures (c)
        cumulative_failures = sum(1 for e in relevant_entries if e.get('outcome') != 'SUCCESS')
        
        # Apply priority formula
        # P = 0.5(1-sr) + 0.2/(1+t) + 0.2c/20 + 0.1R
        p_success = 0.5 * (1.0 - success_rate)
        p_time = 0.2 / (1.0 + time_since_last)
        p_failures = 0.2 * min(cumulative_failures / 20.0, 1.0)  # Cap at 20 failures
        p_risk = 0.1 * risk_factor
        
        priority = p_success + p_time + p_failures + p_risk
        
        return TaskPriority(
            domain=domain,
            model=model,
            priority=priority,
            success_rate=success_rate,
            time_since_last=time_since_last,
            cumulative_failures=cumulative_failures,
            risk_factor=risk_factor,
            recommended_model=self._get_recommended_model(domain)
        )
    
    def _get_recommended_model(self, domain: str) -> Optional[str]:
        """
        Get recommended model for a domain based on historical performance.
        
        Args:
            domain: Task domain
        
        Returns:
            Recommended model name, or None if no bias data available
        """
        if domain not in self.model_biases or not self.model_biases[domain]:
            return None
        
        # Return model with highest success rate and sufficient confidence
        for bias in self.model_biases[domain]:
            if bias.confidence >= 0.5:  # At least 5 samples
                return bias.model
        
        # If no confident recommendation, return best performer regardless
        return self.model_biases[domain][0].model
    
    def get_model_bias(self, domain: str, model: str) -> Optional[ModelBias]:
        """
        Get performance bias for a specific domain-model combination.
        
        Args:
            domain: Task domain
            model: Model name
        
        Returns:
            ModelBias object, or None if no data available
        """
        if domain not in self.model_biases:
            return None
        
        for bias in self.model_biases[domain]:
            if bias.model == model:
                return bias
        
        return None
    
    def get_top_priorities(self, top_n: int = 10) -> List[TaskPriority]:
        """
        Get top N priority tasks that need attention.
        
        Args:
            top_n: Number of top priority tasks to return
        
        Returns:
            List of TaskPriority objects, sorted by priority (descending)
        """
        # Get unique domain-model combinations from learning data
        domain_model_pairs = set(
            (e.get('domain'), e.get('model'))
            for e in self.learning_data
            if e.get('domain') and e.get('model')
        )
        
        priorities = []
        for domain, model in domain_model_pairs:
            priority = self.calculate_task_priority(domain, model)
            priorities.append(priority)
        
        # Sort by priority (descending)
        priorities.sort(key=lambda p: p.priority, reverse=True)
        
        return priorities[:top_n]
    
    def calculate_roi(
        self,
        priority: float,
        estimated_time: float
    ) -> float:
        """
        Calculate ROI for task scheduling (cost-aware queue).
        
        ROI = Priority / EstimatedTime
        
        Args:
            priority: Task priority (0-1)
            estimated_time: Estimated execution time in seconds
        
        Returns:
            ROI value (higher = better)
        """
        if estimated_time <= 0:
            return 0.0
        
        return priority / estimated_time
    
    def get_domain_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregate statistics per domain.
        
        Returns:
            Dict mapping domain -> statistics dict
        """
        domain_stats: Dict[str, List[Dict]] = {}
        
        for entry in self.learning_data:
            domain = entry.get('domain', 'unknown')
            if domain not in domain_stats:
                domain_stats[domain] = []
            domain_stats[domain].append(entry)
        
        result = {}
        for domain, entries in domain_stats.items():
            success_count = sum(1 for e in entries if e.get('outcome') == 'SUCCESS')
            total_count = len(entries)
            success_rate = success_count / total_count if total_count > 0 else 0.0
            
            times = []
            for e in entries:
                timings = e.get('timings', {})
                total_time = timings.get('build', 0) + timings.get('test', 0) + timings.get('profile', 0)
                if total_time > 0:
                    times.append(total_time)
            
            avg_time = statistics.mean(times) if times else 0.0
            
            result[domain] = {
                'success_rate': success_rate,
                'total_tasks': total_count,
                'avg_time_seconds': avg_time
            }
        
        return result
    
    def print_summary(self) -> None:
        """Print summary of Meta-Supervisor state."""
        print("\n" + "="*60)
        print("META-SUPERVISOR SUMMARY")
        print("="*60)
        
        print(f"\nTotal Learning Entries: {len(self.learning_data)}")
        print(f"Domains Tracked: {len(self.model_biases)}")
        
        print("\n--- DOMAIN STATISTICS ---")
        domain_stats = self.get_domain_statistics()
        for domain, stats in sorted(domain_stats.items()):
            print(f"\n{domain}:")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Total Tasks: {stats['total_tasks']}")
            print(f"  Avg Time: {stats['avg_time_seconds']:.1f}s")
        
        print("\n--- MODEL BIASES (TOP PERFORMERS) ---")
        for domain in sorted(self.model_biases.keys()):
            biases = self.model_biases[domain]
            if biases:
                top_bias = biases[0]
                print(f"\n{domain}:")
                print(f"  Best Model: {top_bias.model}")
                print(f"  Success Rate: {top_bias.success_rate:.1%}")
                print(f"  Avg Time: {top_bias.avg_time:.1f}s")
                print(f"  Confidence: {top_bias.confidence:.1%} ({top_bias.sample_size} samples)")
        
        print("\n--- TOP 5 PRIORITY TASKS ---")
        top_priorities = self.get_top_priorities(top_n=5)
        for i, priority in enumerate(top_priorities, 1):
            print(f"\n{i}. {priority.domain} / {priority.model}")
            print(f"   Priority: {priority.priority:.3f}")
            print(f"   Success Rate: {priority.success_rate:.1%}")
            print(f"   Time Since Last: {priority.time_since_last:.1f}h")
            print(f"   Failures: {priority.cumulative_failures}")
            if priority.recommended_model:
                print(f"   Recommended Model: {priority.recommended_model}")
        
        print("\n" + "="*60 + "\n")


# Example usage / testing
if __name__ == "__main__":
    # Test with learning log
    log_path = Path("D:/AGENT_MEMORY/learning_log.json")
    
    if log_path.exists():
        meta = MetaSupervisor(log_path)
        meta.print_summary()
        
        # Test priority calculation
        print("\n--- TESTING PRIORITY CALCULATION ---")
        priority = meta.calculate_task_priority('cuda', 'deepseek-coder-v2:16b', risk_factor=0.7)
        print(f"CUDA / deepseek-coder-v2:16b -> Priority: {priority.priority:.3f}")
        
        # Test ROI calculation
        print("\n--- TESTING ROI CALCULATION ---")
        roi = meta.calculate_roi(priority.priority, estimated_time=180.0)
        print(f"ROI = {roi:.4f} (Priority {priority.priority:.3f} / 180s)")
    else:
        print(f"[MetaSupervisor] ✗ Learning log not found: {log_path}")
        print("[MetaSupervisor] Create test data first with learning_module_v2.py")
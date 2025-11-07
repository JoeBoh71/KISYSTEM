"""
KISYSTEM Performance Parser
Parses CUDA profiler output (nvprof/nsys) and extracts performance metrics

Author: Jörg Bohne
Date: 2025-11-07
Version: 1.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import re


@dataclass
class PerformanceMetrics:
    """Performance metrics from CUDA profiler"""
    
    gpu_time_ms: float
    cpu_time_ms: float
    occupancy: float  # 0.0-1.0
    memory_efficiency: float  # 0.0-1.0
    compute_efficiency: float  # 0.0-1.0
    bottleneck: str  # 'memory', 'compute', 'occupancy', 'balanced'
    performance_score: float  # 0.0-100.0
    
    raw_metrics: Dict = None  # Original parsed data


class PerformanceParser:
    """
    Parses CUDA profiler output and calculates performance metrics
    """
    
    @staticmethod
    def parse_output(output: str) -> Optional[PerformanceMetrics]:
        """
        Parse nvprof/nsys output and extract metrics
        
        Args:
            output: Raw profiler output (stderr)
            
        Returns:
            PerformanceMetrics or None if parsing failed
        """
        
        if not output or len(output) < 10:
            return None
        
        # Extract GPU time
        gpu_time = PerformanceParser._extract_gpu_time(output)
        
        # Extract occupancy if available
        occupancy = PerformanceParser._extract_occupancy(output)
        
        # Extract memory efficiency if available
        memory_eff = PerformanceParser._extract_memory_efficiency(output)
        
        # Extract compute efficiency (approximate)
        compute_eff = PerformanceParser._extract_compute_efficiency(output)
        
        # Determine bottleneck
        bottleneck = PerformanceParser._determine_bottleneck(
            occupancy, memory_eff, compute_eff
        )
        
        # Calculate overall performance score
        score = PerformanceParser._calculate_score(
            occupancy, memory_eff, compute_eff
        )
        
        return PerformanceMetrics(
            gpu_time_ms=gpu_time,
            cpu_time_ms=0.0,  # Not parsed yet
            occupancy=occupancy,
            memory_efficiency=memory_eff,
            compute_efficiency=compute_eff,
            bottleneck=bottleneck,
            performance_score=score,
            raw_metrics={
                'gpu_time': gpu_time,
                'occupancy': occupancy,
                'memory_eff': memory_eff,
                'compute_eff': compute_eff
            }
        )
    
    @staticmethod
    def _extract_gpu_time(output: str) -> float:
        """Extract GPU time in milliseconds"""
        
        # Pattern: "GPU activities:   99.99%  1.234ms"
        match = re.search(r'GPU activities:.*?(\d+\.?\d*)(us|ms|s)', output)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Convert to ms
            if unit == 'us':
                return value / 1000.0
            elif unit == 's':
                return value * 1000.0
            else:
                return value
        
        # Pattern: "Time(%)" followed by time
        match = re.search(r'Time\(%\).*?(\d+\.?\d*)(us|ms|s)', output, re.MULTILINE)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            if unit == 'us':
                return value / 1000.0
            elif unit == 's':
                return value * 1000.0
            else:
                return value
        
        return 1.0  # Default 1ms
    
    @staticmethod
    def _extract_occupancy(output: str) -> float:
        """Extract achieved occupancy (0.0-1.0)"""
        
        # Pattern: "Achieved Occupancy  0.123456"
        match = re.search(r'[Aa]chieved [Oo]ccupancy.*?(\d+\.?\d*)', output)
        if match:
            return float(match.group(1))
        
        # Pattern: "occupancy: 45%"
        match = re.search(r'occupancy:?\s*(\d+)%', output, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        
        return 0.5  # Default 50% if not found
    
    @staticmethod
    def _extract_memory_efficiency(output: str) -> float:
        """Extract memory efficiency (0.0-1.0)"""
        
        # Pattern: "gld_efficiency  99.99%"
        match = re.search(r'gld_efficiency.*?(\d+\.?\d*)%', output)
        if match:
            return float(match.group(1)) / 100.0
        
        # Pattern: "Global Memory Load Efficiency  99%"
        match = re.search(r'[Gg]lobal [Mm]emory.*?[Ee]fficiency.*?(\d+)%', output)
        if match:
            return float(match.group(1)) / 100.0
        
        return 0.7  # Default 70% if not found
    
    @staticmethod
    def _extract_compute_efficiency(output: str) -> float:
        """Extract compute efficiency (0.0-1.0)"""
        
        # Pattern: "sm_efficiency  99%"
        match = re.search(r'sm_efficiency.*?(\d+\.?\d*)%', output)
        if match:
            return float(match.group(1)) / 100.0
        
        # Approximate from other metrics if not available
        return 0.6  # Default 60% if not found
    
    @staticmethod
    def _determine_bottleneck(
        occupancy: float, 
        memory_eff: float, 
        compute_eff: float
    ) -> str:
        """Determine primary bottleneck"""
        
        # Low occupancy is critical
        if occupancy < 0.3:
            return 'occupancy'
        
        # Check which is lowest
        min_metric = min(memory_eff, compute_eff)
        
        if min_metric > 0.7:
            return 'balanced'
        elif memory_eff < compute_eff:
            return 'memory'
        else:
            return 'compute'
    
    @staticmethod
    def _calculate_score(
        occupancy: float, 
        memory_eff: float, 
        compute_eff: float
    ) -> float:
        """
        Calculate overall performance score (0-100)
        
        Weighted: 40% occupancy, 30% memory, 30% compute
        """
        
        score = (
            occupancy * 40.0 +
            memory_eff * 30.0 +
            compute_eff * 30.0
        )
        
        return round(score, 1)
    
    @staticmethod
    def get_optimization_suggestions(metrics: PerformanceMetrics) -> List[Dict]:
        """
        Generate optimization suggestions based on metrics
        
        Args:
            metrics: PerformanceMetrics instance
            
        Returns:
            List of suggestion dicts with issue/severity/fix/description
        """
        
        suggestions = []
        
        # Low occupancy
        if metrics.occupancy < 0.3:
            suggestions.append({
                'issue': 'low_occupancy',
                'severity': 'high',
                'description': f'Occupancy very low ({metrics.occupancy:.1%})',
                'fix': 'Increase block size or reduce register/shared memory usage'
            })
        elif metrics.occupancy < 0.5:
            suggestions.append({
                'issue': 'medium_occupancy',
                'severity': 'medium',
                'description': f'Occupancy could be improved ({metrics.occupancy:.1%})',
                'fix': 'Consider tuning block size or memory usage'
            })
        
        # Low memory efficiency
        if metrics.memory_efficiency < 0.5:
            suggestions.append({
                'issue': 'uncoalesced_memory',
                'severity': 'high',
                'description': f'Memory efficiency low ({metrics.memory_efficiency:.1%})',
                'fix': 'Improve memory access patterns for coalescing'
            })
        elif metrics.memory_efficiency < 0.7:
            suggestions.append({
                'issue': 'memory_efficiency',
                'severity': 'medium',
                'description': f'Memory efficiency suboptimal ({metrics.memory_efficiency:.1%})',
                'fix': 'Review memory access patterns'
            })
        
        # Low compute efficiency
        if metrics.compute_efficiency < 0.5:
            suggestions.append({
                'issue': 'low_compute',
                'severity': 'medium',
                'description': f'Compute efficiency low ({metrics.compute_efficiency:.1%})',
                'fix': 'Add more compute work or reduce branch divergence'
            })
        
        return suggestions


if __name__ == '__main__':
    # Test with sample nvprof output
    sample_output = """
==12345== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.99%  1.234ms         1  1.234ms  1.234ms  1.234ms  kernel(float*, int)
      
==12345== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce RTX 4070 (0)"
    Kernel: kernel(float*, int)
          1                  achieved_occupancy                        Achieved Occupancy    0.456789    0.456789    0.456789
          1                         gld_efficiency             Global Memory Load Efficiency      85.23%      85.23%      85.23%
"""
    
    parser = PerformanceParser()
    metrics = parser.parse_output(sample_output)
    
    if metrics:
        print("Performance Metrics:")
        print(f"  GPU Time: {metrics.gpu_time_ms:.2f}ms")
        print(f"  Occupancy: {metrics.occupancy:.1%}")
        print(f"  Memory Efficiency: {metrics.memory_efficiency:.1%}")
        print(f"  Compute Efficiency: {metrics.compute_efficiency:.1%}")
        print(f"  Bottleneck: {metrics.bottleneck}")
        print(f"  Performance Score: {metrics.performance_score:.1f}/100")
        
        suggestions = parser.get_optimization_suggestions(metrics)
        if suggestions:
            print("\nOptimization Suggestions:")
            for sugg in suggestions:
                print(f"  - {sugg['issue']}: {sugg['description']}")
                print(f"    Fix: {sugg['fix']}")
    else:
        print("Failed to parse output")

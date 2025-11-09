"""
KISYSTEM Performance Parser V2
Parses CUDA profiler output - NSYS FORMAT (Nsight Systems)

Author: Jörg Bohne
Date: 2025-11-09
Version: 2.0 - Adapted for real nsys output format
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
    
    Supports:
    - nsys (Nsight Systems) - PRIMARY format
    - nvprof (legacy) - FALLBACK
    """
    
    @staticmethod
    def parse_output(output: str) -> Optional[PerformanceMetrics]:
        """
        Parse nsys/nvprof output and extract metrics
        
        Args:
            output: Raw profiler output (stdout + stderr combined)
            
        Returns:
            PerformanceMetrics or None if parsing failed
        """
        
        if not output or len(output) < 10:
            print("[PerformanceParser] ✗ Empty or too short output")
            return None
        
        # Detect format
        is_nsys = 'cuda_gpu_kern_sum' in output or '[6/8]' in output
        
        if is_nsys:
            return PerformanceParser._parse_nsys(output)
        else:
            return PerformanceParser._parse_nvprof(output)
    
    @staticmethod
    def _parse_nsys(output: str) -> Optional[PerformanceMetrics]:
        """
        Parse nsys format output
        
        Format:
        [6/8] Executing 'cuda_gpu_kern_sum' stats report
        Time (%)  Total Time (ns)  Instances  ...  Name
        100,0     1888             1          ...  firFilterKernel(...)
        """
        
        print("[PerformanceParser] Detected nsys format")
        
        # Extract GPU kernel time from cuda_gpu_kern_sum section
        gpu_time_ns = PerformanceParser._extract_nsys_kernel_time(output)
        gpu_time_ms = gpu_time_ns / 1_000_000.0 if gpu_time_ns else 0.0
        
        # Extract memory transfer times (optional)
        mem_time_ns = PerformanceParser._extract_nsys_memory_time(output)
        
        # Extract CUDA API overhead (optional)
        api_overhead_ns = PerformanceParser._extract_nsys_api_overhead(output)
        
        # CRITICAL: nsys without --metrics flag gives NO occupancy/efficiency
        # We use educated defaults based on kernel execution time
        occupancy = PerformanceParser._estimate_occupancy(gpu_time_ns)
        memory_eff = PerformanceParser._estimate_memory_efficiency(mem_time_ns, gpu_time_ns)
        compute_eff = PerformanceParser._estimate_compute_efficiency(gpu_time_ns)
        
        # Determine bottleneck
        bottleneck = PerformanceParser._determine_bottleneck(
            occupancy, memory_eff, compute_eff
        )
        
        # Calculate overall performance score
        score = PerformanceParser._calculate_score(
            occupancy, memory_eff, compute_eff
        )
        
        raw_metrics = {
            'gpu_time_ns': gpu_time_ns,
            'gpu_time_ms': gpu_time_ms,
            'mem_time_ns': mem_time_ns,
            'api_overhead_ns': api_overhead_ns,
            'occupancy': occupancy,
            'memory_eff': memory_eff,
            'compute_eff': compute_eff,
            'metrics_estimated': True  # Flag: no real metrics from nsys
        }
        
        print(f"[PerformanceParser] ✓ Parsed nsys output:")
        print(f"  GPU Time: {gpu_time_ms:.3f} ms")
        print(f"  Occupancy: {occupancy:.1%} (estimated)")
        print(f"  Memory Eff: {memory_eff:.1%} (estimated)")
        
        return PerformanceMetrics(
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=0.0,
            occupancy=occupancy,
            memory_efficiency=memory_eff,
            compute_efficiency=compute_eff,
            bottleneck=bottleneck,
            performance_score=score,
            raw_metrics=raw_metrics
        )
    
    @staticmethod
    def _extract_nsys_kernel_time(output: str) -> float:
        """
        Extract kernel execution time from nsys cuda_gpu_kern_sum section
        
        Pattern:
        [6/8] Executing 'cuda_gpu_kern_sum' stats report
        Time (%)  Total Time (ns)  Instances  ...
        100,0     1888             1          ...  kernelName(...)
        
        Returns:
            Time in nanoseconds
        """
        
        # Find the cuda_gpu_kern_sum section
        section_match = re.search(
            r'\[6/8\].*?cuda_gpu_kern_sum.*?\n.*?\n.*?\n\s*(\d+[,.]?\d*)\s+(\d+)',
            output,
            re.MULTILINE | re.DOTALL
        )
        
        if section_match:
            # group(2) is "Total Time (ns)"
            time_ns_str = section_match.group(2)
            time_ns = float(time_ns_str)
            print(f"[PerformanceParser] Found kernel time: {time_ns} ns")
            return time_ns
        
        # Fallback: search for "Total Time (ns)" column directly
        time_match = re.search(r'Total Time \(ns\)\s+\S+\s+(\d+)', output)
        if time_match:
            time_ns = float(time_match.group(1))
            print(f"[PerformanceParser] Found kernel time (fallback): {time_ns} ns")
            return time_ns
        
        print("[PerformanceParser] ⚠️  Could not extract kernel time")
        return 1000.0  # Default 1µs
    
    @staticmethod
    def _extract_nsys_memory_time(output: str) -> float:
        """
        Extract total memory transfer time from cuda_gpu_mem_time_sum
        
        Returns:
            Total memory transfer time in nanoseconds
        """
        
        # Find cuda_gpu_mem_time_sum section
        section = re.search(
            r'\[7/8\].*?cuda_gpu_mem_time_sum.*?\n(.*?)\n\[8/8\]',
            output,
            re.MULTILINE | re.DOTALL
        )
        
        if not section:
            return 0.0
        
        section_text = section.group(1)
        
        # Extract all "Total Time (ns)" values
        times = re.findall(r'(\d+)\s+\d+\s+\d+[,.]?\d*', section_text)
        
        if times:
            total = sum(float(t) for t in times)
            print(f"[PerformanceParser] Memory transfer time: {total} ns")
            return total
        
        return 0.0
    
    @staticmethod
    def _extract_nsys_api_overhead(output: str) -> float:
        """
        Extract CUDA API call overhead (cudaMalloc, cudaMemcpy, etc.)
        
        Returns:
            Total API overhead in nanoseconds
        """
        
        # Find cuda_api_sum section
        section = re.search(
            r'\[5/8\].*?cuda_api_sum.*?\n(.*?)\n\[6/8\]',
            output,
            re.MULTILINE | re.DOTALL
        )
        
        if not section:
            return 0.0
        
        section_text = section.group(1)
        
        # Extract all "Total Time (ns)" values (2nd column)
        # Pattern: numbers in 2nd column of table
        times = re.findall(r'\d+[,.]?\d*\s+(\d+)\s+\d+', section_text)
        
        if times:
            total = sum(float(t) for t in times)
            print(f"[PerformanceParser] CUDA API overhead: {total/1e6:.2f} ms")
            return total
        
        return 0.0
    
    @staticmethod
    def _estimate_occupancy(gpu_time_ns: float) -> float:
        """
        Estimate occupancy based on kernel execution time
        
        Heuristic:
        - Very fast kernels (< 1µs): likely low occupancy (not enough work)
        - Medium kernels (1-100µs): good occupancy
        - Slow kernels (> 100µs): could be memory-bound
        
        Returns:
            Estimated occupancy (0.0-1.0)
        """
        
        if gpu_time_ns < 1000:  # < 1µs
            return 0.3  # Low occupancy
        elif gpu_time_ns < 100_000:  # < 100µs
            return 0.6  # Good occupancy
        else:
            return 0.5  # Medium (could be memory-bound)
    
    @staticmethod
    def _estimate_memory_efficiency(mem_time_ns: float, gpu_time_ns: float) -> float:
        """
        Estimate memory efficiency from memory/compute ratio
        
        Args:
            mem_time_ns: Total memory transfer time
            gpu_time_ns: Kernel execution time
            
        Returns:
            Estimated memory efficiency (0.0-1.0)
        """
        
        if mem_time_ns == 0 or gpu_time_ns == 0:
            return 0.7  # Neutral default
        
        ratio = mem_time_ns / gpu_time_ns
        
        # If memory time dominates, efficiency is likely poor
        if ratio > 10:
            return 0.4  # Memory-bound
        elif ratio > 1:
            return 0.6  # Somewhat memory-bound
        else:
            return 0.8  # Compute-bound (good)
    
    @staticmethod
    def _estimate_compute_efficiency(gpu_time_ns: float) -> float:
        """
        Estimate compute efficiency based on execution time
        
        Returns:
            Estimated compute efficiency (0.0-1.0)
        """
        
        # Simple heuristic: assume medium efficiency
        # Real metric would need SM utilization from nsys --metrics
        return 0.65
    
    @staticmethod
    def _parse_nvprof(output: str) -> Optional[PerformanceMetrics]:
        """
        Parse legacy nvprof format output (FALLBACK)
        
        Format:
        GPU activities:   99.99%  1.234ms  kernel(...)
        Achieved Occupancy    0.456789
        gld_efficiency        85.23%
        """
        
        print("[PerformanceParser] Detected nvprof format")
        
        # Extract GPU time
        gpu_time = PerformanceParser._extract_nvprof_gpu_time(output)
        
        # Extract occupancy if available
        occupancy = PerformanceParser._extract_nvprof_occupancy(output)
        
        # Extract memory efficiency if available
        memory_eff = PerformanceParser._extract_nvprof_memory_efficiency(output)
        
        # Extract compute efficiency
        compute_eff = PerformanceParser._extract_nvprof_compute_efficiency(output)
        
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
            cpu_time_ms=0.0,
            occupancy=occupancy,
            memory_efficiency=memory_eff,
            compute_efficiency=compute_eff,
            bottleneck=bottleneck,
            performance_score=score,
            raw_metrics={
                'gpu_time': gpu_time,
                'occupancy': occupancy,
                'memory_eff': memory_eff,
                'compute_eff': compute_eff,
                'metrics_estimated': False
            }
        )
    
    @staticmethod
    def _extract_nvprof_gpu_time(output: str) -> float:
        """Extract GPU time from nvprof output (milliseconds)"""
        
        match = re.search(r'GPU activities:.*?(\d+\.?\d*)(us|ms|s)', output)
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
    def _extract_nvprof_occupancy(output: str) -> float:
        """Extract occupancy from nvprof output"""
        
        match = re.search(r'[Aa]chieved [Oo]ccupancy.*?(\d+\.?\d*)', output)
        if match:
            return float(match.group(1))
        
        return 0.5  # Default 50%
    
    @staticmethod
    def _extract_nvprof_memory_efficiency(output: str) -> float:
        """Extract memory efficiency from nvprof output"""
        
        match = re.search(r'gld_efficiency.*?(\d+\.?\d*)%', output)
        if match:
            return float(match.group(1)) / 100.0
        
        return 0.7  # Default 70%
    
    @staticmethod
    def _extract_nvprof_compute_efficiency(output: str) -> float:
        """Extract compute efficiency from nvprof output"""
        
        match = re.search(r'sm_efficiency.*?(\d+\.?\d*)%', output)
        if match:
            return float(match.group(1)) / 100.0
        
        return 0.6  # Default 60%
    
    @staticmethod
    def _determine_bottleneck(
        occupancy: float, 
        memory_eff: float, 
        compute_eff: float
    ) -> str:
        """Determine primary bottleneck"""
        
        if occupancy < 0.3:
            return 'occupancy'
        
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
        
        # Note if metrics are estimated
        if metrics.raw_metrics and metrics.raw_metrics.get('metrics_estimated'):
            suggestions.append({
                'issue': 'estimated_metrics',
                'severity': 'info',
                'description': 'Metrics are estimated (nsys ran without --metrics flag)',
                'fix': 'Run: nsys profile --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed your_program.exe'
            })
        
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
    # Test with real nsys output
    print("="*70)
    print("PERFORMANCE PARSER V2 - NSYS FORMAT TEST")
    print("="*70)
    
    # Load real nsys output from file
    try:
        with open('nsys_raw_output.txt', 'r', encoding='utf-8') as f:
            nsys_output = f.read()
        
        print("\n[Test] Parsing real nsys output...")
        parser = PerformanceParser()
        metrics = parser.parse_output(nsys_output)
        
        if metrics:
            print("\n" + "="*70)
            print("PERFORMANCE METRICS:")
            print("="*70)
            print(f"GPU Time:           {metrics.gpu_time_ms:.3f} ms")
            print(f"Occupancy:          {metrics.occupancy:.1%}")
            print(f"Memory Efficiency:  {metrics.memory_efficiency:.1%}")
            print(f"Compute Efficiency: {metrics.compute_efficiency:.1%}")
            print(f"Bottleneck:         {metrics.bottleneck}")
            print(f"Performance Score:  {metrics.performance_score:.1f}/100")
            
            suggestions = parser.get_optimization_suggestions(metrics)
            if suggestions:
                print("\n" + "="*70)
                print("OPTIMIZATION SUGGESTIONS:")
                print("="*70)
                for i, sugg in enumerate(suggestions, 1):
                    print(f"\n{i}. {sugg['issue'].upper()} [{sugg['severity']}]")
                    print(f"   {sugg['description']}")
                    print(f"   Fix: {sugg['fix']}")
            
            print("\n" + "="*70)
            print("✅ TEST PASSED")
            print("="*70)
        else:
            print("\n❌ TEST FAILED - Could not parse output")
            
    except FileNotFoundError:
        print("\n⚠️  nsys_raw_output.txt not found - using synthetic test")
        
        # Synthetic nsys output for testing
        test_output = """
[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                           Name                         
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ------------------------------------------------------
    100,0             1888          1    1888,0    1888,0      1888      1888          0,0  firFilterKernel(const float *, const float *, float *)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     51,0             3520      2    1760,0    1760,0       704      2816       1493,0  [CUDA memcpy Host-to-Device]
     48,0             3328      1    3328,0    3328,0      3328      3328          0,0  [CUDA memcpy Device-to-Host]
"""
        
        parser = PerformanceParser()
        metrics = parser.parse_output(test_output)
        
        if metrics:
            print("\n✅ Synthetic test passed")
            print(f"   GPU Time: {metrics.gpu_time_ms:.3f} ms")
            print(f"   Score: {metrics.performance_score:.1f}/100")
        else:
            print("\n❌ Synthetic test failed")

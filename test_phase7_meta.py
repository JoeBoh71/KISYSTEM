"""
KISYSTEM Phase 7 Unit Tests
============================

Tests for:
- MetaSupervisor (priority calculation, model bias, ROI)
- HybridDecision (complexity analysis, model selection, failure escalation)

Author: Jörg Bohne / Bohne Audio
Last Updated: 2025-11-10
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from meta_supervisor import MetaSupervisor, TaskPriority, ModelBias
from hybrid_decision import HybridDecision, ComplexityAnalysis, ModelDecision


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_learning_data():
    """Create sample learning data for testing."""
    now = datetime.now()
    data = [
        # CUDA tasks - mixed success
        {
            'run_id': 'run_001',
            'domain': 'cuda',
            'model': 'deepseek-coder-v2:16b',
            'iteration': 1,
            'score_final': 85,
            'outcome': 'SUCCESS',
            'phase': 'TEST',
            'reason': 'All tests passed',
            'timings': {'build': 45.2, 'test': 12.3, 'profile': 0},
            'timestamp': (now - timedelta(hours=2)).isoformat()
        },
        {
            'run_id': 'run_002',
            'domain': 'cuda',
            'model': 'deepseek-coder-v2:16b',
            'iteration': 2,
            'score_final': 90,
            'outcome': 'SUCCESS',
            'phase': 'TEST',
            'reason': 'All tests passed',
            'timings': {'build': 42.1, 'test': 11.8, 'profile': 0},
            'timestamp': (now - timedelta(hours=1)).isoformat()
        },
        {
            'run_id': 'run_003',
            'domain': 'cuda',
            'model': 'qwen2.5-coder:32b',
            'iteration': 1,
            'score_final': 95,
            'outcome': 'SUCCESS',
            'phase': 'PROFILE',
            'reason': 'Optimization complete',
            'timings': {'build': 120.5, 'test': 25.3, 'profile': 450.2},
            'timestamp': (now - timedelta(hours=3)).isoformat()
        },
        {
            'run_id': 'run_004',
            'domain': 'cuda',
            'model': 'llama3.1:8b',
            'iteration': 1,
            'score_final': 0,
            'outcome': 'FAIL',
            'phase': 'BUILD',
            'reason': 'Compilation error',
            'timings': {'build': 30.1, 'test': 0, 'profile': 0},
            'timestamp': (now - timedelta(hours=4)).isoformat()
        },
        # ASIO tasks - high success
        {
            'run_id': 'run_005',
            'domain': 'asio',
            'model': 'deepseek-coder-v2:16b',
            'iteration': 1,
            'score_final': 88,
            'outcome': 'SUCCESS',
            'phase': 'TEST',
            'reason': 'ASIO callbacks working',
            'timings': {'build': 55.0, 'test': 15.0, 'profile': 0},
            'timestamp': (now - timedelta(hours=5)).isoformat()
        },
        {
            'run_id': 'run_006',
            'domain': 'asio',
            'model': 'deepseek-coder-v2:16b',
            'iteration': 2,
            'score_final': 92,
            'outcome': 'SUCCESS',
            'phase': 'TEST',
            'reason': 'Low latency achieved',
            'timings': {'build': 52.0, 'test': 14.5, 'profile': 0},
            'timestamp': (now - timedelta(hours=6)).isoformat()
        },
        # Tests domain - simple tasks
        {
            'run_id': 'run_007',
            'domain': 'tests',
            'model': 'phi4:latest',
            'iteration': 1,
            'score_final': 95,
            'outcome': 'SUCCESS',
            'phase': 'TEST',
            'reason': 'Unit tests complete',
            'timings': {'build': 15.0, 'test': 8.0, 'profile': 0},
            'timestamp': (now - timedelta(hours=7)).isoformat()
        }
    ]
    return data


@pytest.fixture
def temp_learning_log(sample_learning_data):
    """Create temporary learning log file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_learning_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def meta_supervisor(temp_learning_log):
    """Create MetaSupervisor instance with sample data."""
    return MetaSupervisor(temp_learning_log)


@pytest.fixture
def hybrid_decision():
    """Create HybridDecision instance."""
    return HybridDecision()


@pytest.fixture
def hybrid_with_meta(meta_supervisor):
    """Create HybridDecision with MetaSupervisor."""
    return HybridDecision(meta_supervisor=meta_supervisor)


# ============================================================================
# META-SUPERVISOR TESTS
# ============================================================================

class TestMetaSupervisor:
    """Tests for MetaSupervisor class."""
    
    def test_load_learning_data(self, meta_supervisor, sample_learning_data):
        """Test loading learning data from JSON."""
        assert len(meta_supervisor.learning_data) == len(sample_learning_data)
        assert meta_supervisor.learning_data[0]['run_id'] == 'run_001'
    
    def test_calculate_model_biases(self, meta_supervisor):
        """Test model bias calculation."""
        assert 'cuda' in meta_supervisor.model_biases
        assert 'asio' in meta_supervisor.model_biases
        assert 'tests' in meta_supervisor.model_biases
        
        # Check CUDA biases
        cuda_biases = meta_supervisor.model_biases['cuda']
        assert len(cuda_biases) > 0
        
        # Should be sorted by success_rate descending
        for i in range(len(cuda_biases) - 1):
            assert cuda_biases[i].success_rate >= cuda_biases[i + 1].success_rate
    
    def test_priority_calculation_high_success(self, meta_supervisor):
        """Test priority calculation for high success rate (should be low priority)."""
        priority = meta_supervisor.calculate_task_priority('asio', 'deepseek-coder-v2:16b')
        
        assert isinstance(priority, TaskPriority)
        assert priority.domain == 'asio'
        assert priority.model == 'deepseek-coder-v2:16b'
        assert priority.success_rate == 1.0  # 2/2 successes
        assert priority.priority < 0.5  # Low priority due to high success
    
    def test_priority_calculation_low_success(self, meta_supervisor):
        """Test priority calculation for low success rate (should be high priority)."""
        priority = meta_supervisor.calculate_task_priority('cuda', 'llama3.1:8b')
        
        assert priority.domain == 'cuda'
        assert priority.success_rate == 0.0  # 0/1 successes
        assert priority.priority > 0.5  # High priority due to failure
    
    def test_priority_calculation_unknown_domain(self, meta_supervisor):
        """Test priority for unknown domain-model combination."""
        priority = meta_supervisor.calculate_task_priority('unknown', 'unknown_model')
        
        assert priority.priority == 0.9  # High priority for unknown territory
        assert priority.success_rate == 0.0
    
    def test_get_recommended_model(self, meta_supervisor):
        """Test model recommendation based on historical performance."""
        # CUDA: deepseek-coder-v2:16b has best success rate (2/2 vs 1/1)
        cuda_recommendation = meta_supervisor._get_recommended_model('cuda')
        assert cuda_recommendation in ['deepseek-coder-v2:16b', 'qwen2.5-coder:32b']
        
        # ASIO: only one model
        asio_recommendation = meta_supervisor._get_recommended_model('asio')
        assert asio_recommendation == 'deepseek-coder-v2:16b'
    
    def test_get_model_bias(self, meta_supervisor):
        """Test retrieving specific model bias."""
        bias = meta_supervisor.get_model_bias('cuda', 'deepseek-coder-v2:16b')
        
        assert bias is not None
        assert bias.model == 'deepseek-coder-v2:16b'
        assert bias.domain == 'cuda'
        assert bias.success_rate == 1.0  # 2/2 successes
        assert bias.sample_size == 2
    
    def test_calculate_roi(self, meta_supervisor):
        """Test ROI calculation."""
        roi_high = meta_supervisor.calculate_roi(priority=0.8, estimated_time=100.0)
        roi_low = meta_supervisor.calculate_roi(priority=0.4, estimated_time=200.0)
        
        assert roi_high > roi_low  # Higher priority, shorter time = better ROI
        assert roi_high == 0.8 / 100.0
    
    def test_get_top_priorities(self, meta_supervisor):
        """Test getting top priority tasks."""
        top_priorities = meta_supervisor.get_top_priorities(top_n=3)
        
        assert len(top_priorities) <= 3
        assert isinstance(top_priorities[0], TaskPriority)
        
        # Should be sorted by priority descending
        for i in range(len(top_priorities) - 1):
            assert top_priorities[i].priority >= top_priorities[i + 1].priority
    
    def test_get_domain_statistics(self, meta_supervisor):
        """Test domain statistics aggregation."""
        stats = meta_supervisor.get_domain_statistics()
        
        assert 'cuda' in stats
        assert 'asio' in stats
        assert 'tests' in stats
        
        # Check CUDA stats
        cuda_stats = stats['cuda']
        assert 'success_rate' in cuda_stats
        assert 'total_tasks' in cuda_stats
        assert 'avg_time_seconds' in cuda_stats
        assert cuda_stats['total_tasks'] == 4  # 4 CUDA tasks in sample data


# ============================================================================
# HYBRID DECISION TESTS
# ============================================================================

class TestHybridDecision:
    """Tests for HybridDecision class."""
    
    def test_analyze_complexity_cuda_simple(self, hybrid_decision):
        """Test complexity analysis for simple CUDA task."""
        analysis = hybrid_decision.analyze_complexity(
            task_description="Implement vector addition with cudaMalloc",
            code_snippet="__global__ void vectorAdd(float* a, float* b)"
        )
        
        assert analysis.domain == 'cuda'
        assert analysis.complexity_level in ['simple', 'medium']
        assert analysis.confidence > 0.0
    
    def test_analyze_complexity_cuda_complex(self, hybrid_decision):
        """Test complexity analysis for complex CUDA task."""
        analysis = hybrid_decision.analyze_complexity(
            task_description="Implement FFT convolution with cufft and multiple streams",
            code_snippet="cufftPlan1d multi-kernel async cooperative groups"
        )
        
        assert analysis.domain == 'cuda'
        assert analysis.complexity_level in ['complex', 'medium']
        assert len(analysis.keywords_found) > 0
    
    def test_analyze_complexity_asio(self, hybrid_decision):
        """Test complexity analysis for ASIO task."""
        analysis = hybrid_decision.analyze_complexity(
            task_description="Implement ASIO bufferSwitch callback",
            code_snippet="ASIOCallbacks callbacks; bufferSwitch()"
        )
        
        assert analysis.domain == 'asio'
        assert analysis.complexity_level == 'medium'
    
    def test_analyze_complexity_audio_dsp(self, hybrid_decision):
        """Test complexity analysis for Audio DSP task."""
        analysis = hybrid_decision.analyze_complexity(
            task_description="Implement STFT filterbank for audio processing",
            code_snippet="STFT convolution PQMF psychoacoustic"
        )
        
        assert analysis.domain == 'audio_dsp'
        assert analysis.complexity_level in ['simple', 'complex']
    
    def test_analyze_complexity_tests(self, hybrid_decision):
        """Test complexity analysis for test tasks."""
        analysis = hybrid_decision.analyze_complexity(
            task_description="Write unit tests for CUDA kernel",
            code_snippet=""
        )
        
        assert analysis.domain == 'tests'
        assert analysis.complexity_level == 'simple'
    
    def test_calculate_complexity_score(self, hybrid_decision):
        """Test complexity score calculation."""
        # Simple task
        simple_analysis = ComplexityAnalysis(
            complexity_level='simple',
            domain='cuda',
            keywords_found=[],
            confidence=0.8
        )
        score_simple, model_simple = hybrid_decision.calculate_complexity_score(simple_analysis)
        
        # Complex task
        complex_analysis = ComplexityAnalysis(
            complexity_level='complex',
            domain='cuda',
            keywords_found=[],
            confidence=0.9
        )
        score_complex, model_complex = hybrid_decision.calculate_complexity_score(complex_analysis)
        
        assert score_complex > score_simple  # Complex should have higher score
        assert score_simple >= 0.0 and score_simple <= 1.0
        assert score_complex >= 0.0 and score_complex <= 1.0
    
    def test_failure_tracking(self, hybrid_decision):
        """Test failure recording and scoring."""
        # No failures initially
        score_0, escalate_0 = hybrid_decision.calculate_failure_score('cuda')
        assert score_0 == 0.0
        assert escalate_0 is None
        
        # Record first failure
        hybrid_decision.record_failure('cuda', 'deepseek-coder-v2:16b')
        score_1, escalate_1 = hybrid_decision.calculate_failure_score('cuda')
        assert score_1 > 0.0
        assert escalate_1 is not None
        
        # Record second failure
        hybrid_decision.record_failure('cuda', 'qwen2.5-coder:32b')
        score_2, escalate_2 = hybrid_decision.calculate_failure_score('cuda')
        assert score_2 > score_1  # More failures = higher score
    
    def test_clear_failures(self, hybrid_decision):
        """Test clearing failure history."""
        hybrid_decision.record_failure('cuda', 'model1')
        hybrid_decision.record_failure('cuda', 'model2')
        
        score_before, _ = hybrid_decision.calculate_failure_score('cuda')
        assert score_before > 0.0
        
        hybrid_decision.clear_failures('cuda')
        score_after, _ = hybrid_decision.calculate_failure_score('cuda')
        assert score_after == 0.0
    
    def test_decide_model_simple_cuda(self, hybrid_decision):
        """Test model decision for simple CUDA task."""
        decision = hybrid_decision.decide_model(
            task_description="Vector addition with cudaMalloc",
            code_snippet="__global__ void add()"
        )
        
        assert isinstance(decision, ModelDecision)
        assert decision.selected_model in hybrid_decision.MODEL_HIERARCHY
        assert decision.selected_model != 'llama3.1:8b'  # Should not select blacklisted model
        assert len(decision.escalation_path) > 0
        assert decision.confidence >= 0.0 and decision.confidence <= 1.0
    
    def test_decide_model_with_failures(self, hybrid_decision):
        """Test model decision with failure escalation."""
        # Record failures for lower-tier models
        hybrid_decision.record_failure('cuda', 'deepseek-coder-v2:16b')
        hybrid_decision.record_failure('cuda', 'qwen2.5-coder:32b')
        
        decision = hybrid_decision.decide_model(
            task_description="CUDA kernel optimization",
            domain='cuda'
        )
        
        # Should escalate to higher-tier model
        assert decision.failure_score > 0.5
        assert decision.selected_model in ['deepseek-r1:32b', 'qwen2.5:32b']
    
    def test_decide_model_asio(self, hybrid_decision):
        """Test model decision for ASIO task."""
        decision = hybrid_decision.decide_model(
            task_description="ASIO bufferSwitch callback implementation",
            code_snippet="ASIOCallbacks"
        )
        
        assert decision.selected_model in ['deepseek-coder-v2:16b', 'qwen2.5-coder:32b', 'qwen2.5:32b']
    
    def test_decide_model_with_meta(self, hybrid_with_meta):
        """Test model decision with Meta-Supervisor data."""
        decision = hybrid_with_meta.decide_model(
            task_description="CUDA vector operations",
            domain='cuda'
        )
        
        # Should have non-zero meta score due to historical data
        assert decision.meta_score > 0.0
        assert decision.selected_model is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase7Integration:
    """Integration tests for Phase 7 components."""
    
    def test_meta_supervisor_hybrid_decision_integration(self, meta_supervisor):
        """Test integration between MetaSupervisor and HybridDecision."""
        hybrid = HybridDecision(meta_supervisor=meta_supervisor)
        
        # Make decision for CUDA task
        decision = hybrid.decide_model(
            task_description="CUDA kernel optimization",
            domain='cuda'
        )
        
        # Should leverage meta data
        assert decision.meta_score > 0.0
        
        # Get priority for the same task
        priority = meta_supervisor.calculate_task_priority('cuda', decision.selected_model)
        
        assert priority.domain == 'cuda'
        assert priority.recommended_model is not None
    
    def test_escalation_path_validity(self, hybrid_decision):
        """Test that escalation paths are valid."""
        decision = hybrid_decision.decide_model(
            task_description="Complex CUDA FFT implementation",
            domain='cuda'
        )
        
        # All models in escalation path should be valid
        for model in decision.escalation_path:
            assert model in hybrid_decision.MODEL_HIERARCHY
        
        # Selected model should be first in path
        assert decision.escalation_path[0] == decision.selected_model
    
    def test_stoploss_mechanism(self, hybrid_decision):
        """Test Stop-Loss: 2 failures → escalate."""
        # Record 2 failures for same domain
        hybrid_decision.record_failure('cuda', 'deepseek-coder-v2:16b')
        hybrid_decision.record_failure('cuda', 'qwen2.5-coder:32b')
        
        # Should only keep last 2
        assert len(hybrid_decision.failure_history['cuda']) == 2
        
        # Record third failure
        hybrid_decision.record_failure('cuda', 'deepseek-r1:32b')
        
        # Should still only have 2 (oldest removed)
        assert len(hybrid_decision.failure_history['cuda']) == 2
        assert 'deepseek-coder-v2:16b' not in hybrid_decision.failure_history['cuda']


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

"""
KISYSTEM Confidence Scorer V2
Multi-Faktor Confidence-Berechnung für Context-Aware Learning

Author: Jörg Bohne
Date: 2025-11-06
"""

import difflib
from typing import Dict, Tuple
from datetime import datetime, timedelta
import json


class ConfidenceScorer:
    """
    Berechnet Confidence-Score für Solution-Matches
    Multi-Faktor: Text-Similarity, Context-Match, History, Recency
    """
    
    # Gewichtung (Jörg's Wahl: 40/30/20/10)
    WEIGHTS = {
        'text_similarity': 0.40,
        'context_match': 0.30,
        'success_history': 0.20,
        'recency': 0.10
    }
    
    # Context-Match Sub-Gewichtung
    CONTEXT_WEIGHTS = {
        'language': 0.40,       # Python vs C++ = fundamental
        'version': 0.25,        # 3.10 vs 3.11 = wichtig
        'os': 0.20,            # Windows vs Linux = Pfade
        'hardware': 0.15       # CPU vs GPU = relevant
    }
    
    # Thresholds
    THRESHOLDS = {
        'enforce': 0.85,    # Ab hier: Solution erzwingen
        'suggest': 0.70,    # Ab hier: Solution vorschlagen
        'consider': 0.50    # Ab hier: Solution erwähnen
    }
    
    def __init__(self, weights: Dict = None, context_weights: Dict = None):
        """
        Initialize Confidence Scorer
        
        Args:
            weights: Optional custom weights
            context_weights: Optional custom context weights
        """
        if weights:
            self.WEIGHTS.update(weights)
        if context_weights:
            self.CONTEXT_WEIGHTS.update(context_weights)
        
        # Validate weights sum to 1.0
        assert abs(sum(self.WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
        assert abs(sum(self.CONTEXT_WEIGHTS.values()) - 1.0) < 0.01, "Context weights must sum to 1.0"
    
    def calculate_confidence(
        self,
        current_error: str,
        current_context: Dict,
        past_solution: Dict
    ) -> Tuple[float, Dict]:
        """
        Calculate overall confidence score
        
        Args:
            current_error: Current error message
            current_context: Current environment context
            past_solution: Past solution dict from database
            
        Returns:
            (confidence_score, detail_scores)
        """
        # Calculate individual scores
        text_sim = self._calc_text_similarity(current_error, past_solution['error_text'])
        context_match = self._calc_context_match(current_context, past_solution)
        history = self._calc_success_history(past_solution)
        recency = self._calc_recency_score(past_solution)
        
        # Weighted sum
        confidence = (
            text_sim * self.WEIGHTS['text_similarity'] +
            context_match['score'] * self.WEIGHTS['context_match'] +
            history * self.WEIGHTS['success_history'] +
            recency * self.WEIGHTS['recency']
        )
        
        # Detail scores for debugging
        details = {
            'text_similarity': round(text_sim, 3),
            'context_match': round(context_match['score'], 3),
            'success_history': round(history, 3),
            'recency': round(recency, 3),
            'context_breakdown': context_match['breakdown'],
            'action': self._determine_action(confidence),
            'weights_used': self.WEIGHTS.copy()
        }
        
        return round(confidence, 3), details
    
    def _calc_text_similarity(self, error1: str, error2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher
        
        Args:
            error1: First error message
            error2: Second error message
            
        Returns:
            Similarity score [0.0, 1.0]
        """
        # Normalize
        e1 = error1.lower().strip()
        e2 = error2.lower().strip()
        
        # Quick exact match
        if e1 == e2:
            return 1.0
        
        # SequenceMatcher ratio
        similarity = difflib.SequenceMatcher(None, e1, e2).ratio()
        
        return similarity
    
    def _calc_context_match(self, ctx1: Dict, solution: Dict) -> Dict:
        """
        Calculate context match score
        
        Args:
            ctx1: Current context
            solution: Past solution with context
            
        Returns:
            {'score': float, 'breakdown': dict}
        """
        breakdown = {}
        
        # Language match (fundamental)
        lang_match = ctx1.get('language') == solution.get('language')
        breakdown['language_match'] = lang_match
        
        # Version match (wichtig aber fuzzy)
        version_match = self._fuzzy_version_match(
            ctx1.get('language_version'),
            solution.get('language_version')
        )
        breakdown['version_match'] = version_match
        
        # OS match
        os_match = ctx1.get('os') == solution.get('os')
        breakdown['os_match'] = os_match
        
        # Hardware match
        hw_match = ctx1.get('hardware') == solution.get('hardware')
        breakdown['hardware_match'] = hw_match
        
        # Weighted score
        score = (
            (1.0 if lang_match else 0.0) * self.CONTEXT_WEIGHTS['language'] +
            version_match * self.CONTEXT_WEIGHTS['version'] +
            (1.0 if os_match else 0.0) * self.CONTEXT_WEIGHTS['os'] +
            (1.0 if hw_match else 0.0) * self.CONTEXT_WEIGHTS['hardware']
        )
        
        return {'score': score, 'breakdown': breakdown}
    
    def _fuzzy_version_match(self, ver1: str, ver2: str) -> float:
        """
        Fuzzy version matching
        
        Examples:
            3.11.4 vs 3.11.5 -> 0.95 (patch diff)
            3.11 vs 3.12 -> 0.85 (minor diff)
            3.11 vs 4.0 -> 0.5 (major diff)
        
        Args:
            ver1: Version string
            ver2: Version string
            
        Returns:
            Match score [0.0, 1.0]
        """
        if not ver1 or not ver2:
            return 0.5  # Unknown -> neutral
        
        if ver1 == ver2:
            return 1.0
        
        # Parse versions
        def parse_version(v):
            try:
                parts = v.replace('C++', '').strip().split('.')
                return [int(p) for p in parts[:3]]  # Max 3 parts
            except:
                return None
        
        v1 = parse_version(ver1)
        v2 = parse_version(ver2)
        
        if not v1 or not v2:
            return 0.5  # Can't parse -> neutral
        
        # Pad to 3 parts
        while len(v1) < 3: v1.append(0)
        while len(v2) < 3: v2.append(0)
        
        # Major diff -> 0.5
        if v1[0] != v2[0]:
            return 0.5
        
        # Minor diff -> 0.85
        if v1[1] != v2[1]:
            return 0.85
        
        # Patch diff -> 0.95
        if v1[2] != v2[2]:
            return 0.95
        
        return 1.0
    
    def _calc_success_history(self, solution: Dict) -> float:
        """
        Calculate success history score using Bayesian smoothing
        
        Args:
            solution: Past solution dict
            
        Returns:
            History score [0.0, 1.0]
        """
        successes = solution.get('success_count', 0)
        failures = solution.get('failure_count', 0)
        
        # Bayesian smoothing with prior (alpha=1, beta=1)
        # This prevents extreme scores for small samples
        score = (successes + 1) / (successes + failures + 2)
        
        return score
    
    def _calc_recency_score(self, solution: Dict) -> float:
        """
        Calculate recency score with linear decay
        
        Args:
            solution: Past solution dict
            
        Returns:
            Recency score [0.0, 1.0]
        """
        last_success = solution.get('last_success_at')
        if not last_success:
            # No success yet -> use creation date
            last_success = solution.get('created_at')
        
        if not last_success:
            return 0.5  # Unknown -> neutral
        
        # Parse timestamp
        try:
            if isinstance(last_success, str):
                dt = datetime.fromisoformat(last_success)
            else:
                dt = last_success
        except:
            return 0.5
        
        # Calculate age in days
        age_days = (datetime.now() - dt).days
        
        # Linear decay over 180 days (6 months)
        # 0 days -> 1.0
        # 90 days -> 0.5
        # 180+ days -> 0.0
        decay_period = 180
        score = max(0.0, 1.0 - (age_days / decay_period))
        
        return score
    
    def _determine_action(self, confidence: float) -> str:
        """
        Determine action based on confidence score
        
        Args:
            confidence: Overall confidence score
            
        Returns:
            'enforce', 'suggest', 'consider', or 'ignore'
        """
        if confidence >= self.THRESHOLDS['enforce']:
            return 'enforce'
        elif confidence >= self.THRESHOLDS['suggest']:
            return 'suggest'
        elif confidence >= self.THRESHOLDS['consider']:
            return 'consider'
        else:
            return 'ignore'
    
    def batch_score_solutions(
        self,
        current_error: str,
        current_context: Dict,
        solutions: list
    ) -> list:
        """
        Score multiple solutions and sort by confidence
        
        Args:
            current_error: Current error message
            current_context: Current context
            solutions: List of past solutions
            
        Returns:
            Sorted list of (solution, confidence, details)
        """
        scored = []
        
        for solution in solutions:
            confidence, details = self.calculate_confidence(
                current_error,
                current_context,
                solution
            )
            scored.append({
                'solution': solution,
                'confidence': confidence,
                'details': details
            })
        
        # Sort by confidence (descending)
        scored.sort(key=lambda x: x['confidence'], reverse=True)
        
        return scored
    
    def explain_score(self, details: Dict) -> str:
        """
        Human-readable explanation of confidence score
        
        Args:
            details: Detail scores dict
            
        Returns:
            Explanation string
        """
        lines = []
        lines.append(f"Action: {details['action'].upper()}")
        lines.append("\nScore Breakdown:")
        lines.append(f"  Text Similarity:  {details['text_similarity']:.1%} (weight: {self.WEIGHTS['text_similarity']:.0%})")
        lines.append(f"  Context Match:    {details['context_match']:.1%} (weight: {self.WEIGHTS['context_match']:.0%})")
        lines.append(f"  Success History:  {details['success_history']:.1%} (weight: {self.WEIGHTS['success_history']:.0%})")
        lines.append(f"  Recency:          {details['recency']:.1%} (weight: {self.WEIGHTS['recency']:.0%})")
        
        lines.append("\nContext Match Details:")
        breakdown = details['context_breakdown']
        lines.append(f"  Language: {'✓' if breakdown['language_match'] else '✗'}")
        lines.append(f"  Version:  {breakdown['version_match']:.0%} match")
        lines.append(f"  OS:       {'✓' if breakdown['os_match'] else '✗'}")
        lines.append(f"  Hardware: {'✓' if breakdown['hardware_match'] else '✗'}")
        
        return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test
    scorer = ConfidenceScorer()
    
    print("=== Confidence Scorer Test ===\n")
    
    # Mock data
    current_error = "ImportError: No module named 'numpy'"
    current_context = {
        'language': 'python',
        'language_version': '3.11.5',
        'os': 'Windows',
        'hardware': 'GPU'
    }
    
    past_solution = {
        'error_text': "ImportError: No module named 'numpy'",
        'solution': 'pip install numpy',
        'language': 'python',
        'language_version': '3.11.4',
        'os': 'Windows',
        'hardware': 'GPU',
        'success_count': 3,
        'failure_count': 0,
        'last_success_at': '2025-11-01T10:00:00',
        'created_at': '2025-10-15T09:00:00'
    }
    
    confidence, details = scorer.calculate_confidence(
        current_error,
        current_context,
        past_solution
    )
    
    print(f"Overall Confidence: {confidence:.1%}\n")
    print(scorer.explain_score(details))

"""
KISYSTEM Learning Module V2
Context-Aware Learning mit Multi-Faktor Confidence-Scoring

Author: Jörg Bohne
Date: 2025-11-06
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from context_tracker import ContextTracker
from confidence_scorer import ConfidenceScorer


class LearningModuleV2:
    """
    Context-Aware Learning System
    Speichert Fehler/Lösungen mit vollständigem Environment-Context
    """
    
    def __init__(self, db_path: str = "D:/AGENT_MEMORY/memory.db"):
        """
        Initialize Learning Module
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.context_tracker = ContextTracker()
        self.confidence_scorer = ConfidenceScorer()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database with V2 schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create V2 table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Core
                error_text TEXT NOT NULL,
                solution TEXT NOT NULL,
                error_type TEXT,
                
                -- Context (Hard Facts)
                language TEXT,
                language_version TEXT,
                os TEXT,
                os_version TEXT,
                hardware TEXT,
                gpu_model TEXT,
                cuda_version TEXT,
                
                -- Dependencies (JSON)
                dependencies TEXT,
                compiler TEXT,
                
                -- Task Context
                complexity TEXT,
                domain TEXT,
                model_used TEXT,
                
                -- Scoring
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_solve_time REAL,
                
                -- Temporal
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                last_success_at TIMESTAMP,
                last_failure_at TIMESTAMP,
                
                -- Metadata
                notes TEXT,
                confidence_override REAL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON solutions_v2(error_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_env ON solutions_v2(language, os, hardware)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_full ON solutions_v2(language, language_version, os, hardware, gpu_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal ON solutions_v2(last_success_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success_rate ON solutions_v2(success_count, failure_count)")
        
        # Create stats view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS solution_stats AS
            SELECT 
                id,
                error_text,
                solution,
                success_count,
                failure_count,
                CAST(success_count AS REAL) / NULLIF(success_count + failure_count, 0) AS success_rate,
                julianday('now') - julianday(created_at) AS age_days,
                julianday('now') - julianday(last_success_at) AS days_since_success,
                domain,
                complexity
            FROM solutions_v2
        """)
        
        conn.commit()
        conn.close()
    
    def store_solution(
        self,
        error: str,
        solution: str,
        code: str,
        model_used: str,
        success: bool,
        solve_time: float = None
    ) -> int:
        """
        Store new solution with full context
        
        Args:
            error: Error message
            solution: Solution that was applied
            code: Source code that caused error
            model_used: Model name that generated solution
            success: Whether solution worked
            solve_time: Time to solve in seconds
            
        Returns:
            solution_id: ID of stored/updated solution
        """
        # Get full context
        context = self.context_tracker.get_full_context(code, error, model_used)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if similar solution exists
        existing = self._find_exact_match(cursor, error, context)
        
        if existing:
            # Update existing
            solution_id = existing['id']
            if success:
                cursor.execute("""
                    UPDATE solutions_v2 SET
                        success_count = success_count + 1,
                        last_success_at = ?,
                        last_used_at = ?,
                        avg_solve_time = CASE 
                            WHEN avg_solve_time IS NULL THEN ?
                            ELSE (avg_solve_time * success_count + ?) / (success_count + 1)
                        END
                    WHERE id = ?
                """, (datetime.now(), datetime.now(), solve_time, solve_time, solution_id))
            else:
                cursor.execute("""
                    UPDATE solutions_v2 SET
                        failure_count = failure_count + 1,
                        last_failure_at = ?,
                        last_used_at = ?
                    WHERE id = ?
                """, (datetime.now(), datetime.now(), solution_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO solutions_v2 (
                    error_text, solution, error_type,
                    language, language_version, os, os_version,
                    hardware, gpu_model, cuda_version,
                    dependencies, compiler,
                    complexity, domain, model_used,
                    success_count, failure_count, avg_solve_time,
                    last_used_at, last_success_at, last_failure_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                error, solution, context['error_type'],
                context['language'], context['language_version'],
                context['os'], context['os_version'],
                context['hardware'], context['gpu_model'], context['cuda_version'],
                context['dependencies'], context['compiler'],
                context['complexity'], context['domain'], context['model_used'],
                1 if success else 0,
                0 if success else 1,
                solve_time,
                datetime.now(),
                datetime.now() if success else None,
                datetime.now() if not success else None
            ))
            solution_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return solution_id
    
    def _find_exact_match(self, cursor, error: str, context: Dict) -> Optional[Dict]:
        """Find exact match (same error + context)"""
        cursor.execute("""
            SELECT * FROM solutions_v2
            WHERE error_text = ?
                AND language = ?
                AND language_version = ?
                AND os = ?
                AND hardware = ?
            ORDER BY last_used_at DESC
            LIMIT 1
        """, (
            error,
            context['language'],
            context['language_version'],
            context['os'],
            context['hardware']
        ))
        
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def find_similar_solutions(
        self,
        error: str,
        code: str,
        model_used: str,
        min_confidence: float = 0.70,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Find similar solutions with confidence scoring
        
        Args:
            error: Current error message
            code: Current source code
            model_used: Model being used
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results
            
        Returns:
            List of solutions with confidence scores
        """
        # Get current context
        current_context = self.context_tracker.get_full_context(code, error, model_used)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get candidate solutions (broad search)
        # Search by error type and domain first for efficiency
        cursor.execute("""
            SELECT * FROM solutions_v2
            WHERE error_type = ?
                OR domain = ?
                OR language = ?
            ORDER BY success_count DESC, last_success_at DESC
            LIMIT 50
        """, (
            current_context['error_type'],
            current_context['domain'],
            current_context['language']
        ))
        
        columns = [desc[0] for desc in cursor.description]
        candidates = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        if not candidates:
            return []
        
        # Score all candidates
        scored = self.confidence_scorer.batch_score_solutions(
            error,
            current_context,
            candidates
        )
        
        # Filter by min_confidence and limit results
        results = [
            s for s in scored 
            if s['confidence'] >= min_confidence
        ][:max_results]
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get learning statistics
        
        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total solutions
        cursor.execute("SELECT COUNT(*) FROM solutions_v2")
        total_solutions = cursor.fetchone()[0]
        
        # Total successes/failures
        cursor.execute("SELECT SUM(success_count), SUM(failure_count) FROM solutions_v2")
        total_success, total_failure = cursor.fetchone()
        
        # By domain
        cursor.execute("""
            SELECT domain, COUNT(*), SUM(success_count), SUM(failure_count)
            FROM solutions_v2
            GROUP BY domain
        """)
        by_domain = {}
        for row in cursor.fetchall():
            domain, count, succ, fail = row
            by_domain[domain] = {
                'solutions': count,
                'successes': succ or 0,
                'failures': fail or 0,
                'success_rate': succ / (succ + fail) if (succ and fail) else 0
            }
        
        # By complexity
        cursor.execute("""
            SELECT complexity, COUNT(*), SUM(success_count), SUM(failure_count)
            FROM solutions_v2
            GROUP BY complexity
        """)
        by_complexity = {}
        for row in cursor.fetchall():
            comp, count, succ, fail = row
            by_complexity[comp] = {
                'solutions': count,
                'successes': succ or 0,
                'failures': fail or 0
            }
        
        # Top solutions
        cursor.execute("""
            SELECT error_text, solution, success_count, domain, complexity
            FROM solutions_v2
            ORDER BY success_count DESC
            LIMIT 5
        """)
        top_solutions = []
        for row in cursor.fetchall():
            top_solutions.append({
                'error': row[0][:100],  # Truncate
                'solution': row[1][:100],
                'successes': row[2],
                'domain': row[3],
                'complexity': row[4]
            })
        
        conn.close()
        
        return {
            'total_solutions': total_solutions,
            'total_successes': total_success or 0,
            'total_failures': total_failure or 0,
            'success_rate': (total_success / (total_success + total_failure)) if (total_success and total_failure) else 0,
            'by_domain': by_domain,
            'by_complexity': by_complexity,
            'top_solutions': top_solutions
        }
    
    def export_knowledge(self, output_file: str):
        """
        Export knowledge base to JSON
        
        Args:
            output_file: Path to output JSON file
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM solutions_v2 ORDER BY success_count DESC")
        columns = [desc[0] for desc in cursor.description]
        solutions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(solutions, f, indent=2, default=str)
    
    def import_knowledge(self, input_file: str):
        """
        Import knowledge base from JSON
        
        Args:
            input_file: Path to input JSON file
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            solutions = json.load(f)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sol in solutions:
            # Remove id to let SQLite auto-assign
            sol.pop('id', None)
            
            columns = ', '.join(sol.keys())
            placeholders = ', '.join(['?' for _ in sol])
            
            cursor.execute(f"""
                INSERT INTO solutions_v2 ({columns})
                VALUES ({placeholders})
            """, list(sol.values()))
        
        conn.commit()
        conn.close()


if __name__ == '__main__':
    # Quick test
    print("=== Learning Module V2 Test ===\n")
    
    learner = LearningModule()
    
    # Test: Store solution
    test_code = """
import numpy as np
def process(data):
    return np.fft.fft(data)
"""
    test_error = "ImportError: No module named 'numpy'"
    test_solution = "pip install numpy"
    
    solution_id = learner.store_solution(
        error=test_error,
        solution=test_solution,
        code=test_code,
        model_used='test',
        success=True,
        solve_time=5.0
    )
    print(f"Stored solution ID: {solution_id}")
    
    # Test: Find similar
    similar = learner.find_similar_solutions(
        error=test_error,
        code=test_code,
        model_used='test',
        min_confidence=0.5
    )
    print(f"\nFound {len(similar)} similar solutions")
    if similar:
        best = similar[0]
        print(f"Best match: {best['confidence']:.1%} confidence")
        print(f"Action: {best['details']['action']}")
    
    # Test: Statistics
    stats = learner.get_statistics()
    print(f"\n=== Statistics ===")
    print(f"Total solutions: {stats['total_solutions']}")
    print(f"Total successes: {stats['total_successes']}")
    print(f"Success rate: {stats['success_rate']:.1%}")

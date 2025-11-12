"""
ReviewAgent - Production-Ready Code Review Agent
Model: llama3.1:8b
Purpose: Code review, architecture validation, best practices enforcement
Memory: D:\\\\AGENT_MEMORY with SQLite persistence
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Optional

class ReviewAgent:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.name = "Reviewer"
        self.model = "llama3.1:8b"
        
        # Paths
        self.workspace = Path("D:/U3DAW")
        self.memory_root = Path("D:/AGENT_MEMORY")
        self.conversations_dir = self.memory_root / "conversations" / "reviewer"
        self.audit_dir = self.memory_root / "audit"
        self.db_path = self.memory_root / "memory.db"
        
        # Ensure directories exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self._init_db()
        
        # Context
        self.max_context_messages = 8
        self.conversation_history: List[Dict] = []
        
        # Review criteria
        self.review_categories = {
            "security": ["buffer overflows", "null pointer", "memory leaks", "unsafe casts"],
            "performance": ["unnecessary copies", "inefficient loops", "cache misses", "lock contention"],
            "style": ["naming conventions", "code clarity", "documentation", "modularity"],
            "correctness": ["logic errors", "edge cases", "error handling", "race conditions"]
        }
        
        self._load_recent_context()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_reviewed TEXT,
                review_type TEXT,
                prompt TEXT NOT NULL,
                findings TEXT NOT NULL,
                severity TEXT,
                approved INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                suggestion TEXT,
                resolved INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS architecture_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                review_notes TEXT NOT NULL,
                recommendations TEXT,
                approved INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_recent_context(self):
        """Load recent review sessions"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prompt, findings 
            FROM review_conversations 
            ORDER BY id DESC 
            LIMIT ?
        """, (self.max_context_messages,))
        
        rows = cursor.fetchall()
        conn.close()
        
        for prompt, findings in reversed(rows):
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": findings})
    
    def _save_to_db(self, file_path: str, review_type: str, prompt: str, 
                    findings: str, severity: str = "medium", approved: bool = False):
        """Save review session to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO review_conversations 
            (timestamp, file_reviewed, review_type, prompt, findings, severity, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            file_path,
            review_type,
            prompt,
            findings,
            severity,
            1 if approved else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _save_finding(self, file_path: str, category: str, severity: str, 
                     description: str, suggestion: str = ""):
        """Save individual finding for tracking"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO review_findings 
            (timestamp, file_path, category, severity, description, suggestion, resolved)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            datetime.now().isoformat(),
            file_path,
            category,
            severity,
            description,
            suggestion
        ))
        
        conn.commit()
        conn.close()
    
    def _save_architecture_review(self, component: str, notes: str, 
                                  recommendations: str, approved: bool = False):
        """Save architecture review"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO architecture_reviews 
            (timestamp, component, review_notes, recommendations, approved)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            component,
            notes,
            recommendations,
            1 if approved else 0
        ))
        
        conn.commit()
        conn.close()
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with llama3.1:8b for fast review"""
        
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes - fast model
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Update history
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                if len(self.conversation_history) > self.max_context_messages * 2:
                    self.conversation_history = self.conversation_history[-self.max_context_messages * 2:]
                
                return response
            else:
                return f"Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Review timed out"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def execute(self, intent: Dict) -> str:
        """Main review execution"""
        target = intent.get('target', '')
        review_type = intent.get('type', 'code')  # code, architecture, security, performance
        file_path = intent.get('file', '')
        
        print(f"[Reviewer] Reviewing: {target}")
        print(f"[Reviewer] Type: {review_type}")
        print(f"[Reviewer] Using model: {self.model} (fast)")
        
        # Route to appropriate review method
        if review_type == "architecture":
            return await self._review_architecture(target, file_path)
        else:
            return await self._review_code(target, file_path, review_type)
    
    async def _review_code(self, target: str, file_path: str, review_type: str) -> str:
        """Perform code review"""
        
        # Read code if file provided
        code_content = ""
        if file_path and Path(file_path).exists():
            code_content = Path(file_path).read_text(encoding='utf-8')
        
        # Build review prompt
        prompt = self._build_code_review_prompt(target, code_content, review_type)
        
        # Get review
        findings = await self._call_ollama(prompt)
        
        # Parse severity (simple heuristic)
        severity = "low"
        if any(word in findings.lower() for word in ["critical", "severe", "unsafe", "vulnerability"]):
            severity = "high"
        elif any(word in findings.lower() for word in ["warning", "issue", "problem"]):
            severity = "medium"
        
        # Determine approval
        approved = "approved" in findings.lower() or "looks good" in findings.lower()
        
        # Save to database
        self._save_to_db(file_path, review_type, prompt, findings, severity, approved)
        
        # Extract and save individual findings
        self._extract_and_save_findings(file_path, findings)
        
        # Save review report
        report_file = self.audit_dir / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_content = f"""Code Review Report
==================
File: {file_path}
Type: {review_type}
Timestamp: {datetime.now().isoformat()}
Severity: {severity}
Approved: {approved}

Findings:
{findings}
"""
        report_file.write_text(report_content, encoding='utf-8')
        
        # Build result
        result = f"✓ Review completed: {target}\n"
        result += f"  Type: {review_type}\n"
        result += f"  Severity: {severity.upper()}\n"
        result += f"  Status: {'✓ APPROVED' if approved else '⚠ NEEDS ATTENTION'}\n"
        result += f"  Report: {report_file}\n"
        result += f"\nFindings:\n{findings[:300]}..."
        
        return result
    
    async def _review_architecture(self, component: str, file_path: str) -> str:
        """Review system architecture"""
        
        context = ""
        if file_path and Path(file_path).exists():
            context = Path(file_path).read_text(encoding='utf-8')
        
        prompt = f"""Review the architecture of this U3DAW component: {component}

{f'Design/Code:{context[:2000]}' if context else ''}

Evaluate:
1. Component responsibilities (single responsibility?)
2. Interface design (clean boundaries?)
3. Dependencies (minimal coupling?)
4. Scalability (can it handle growth?)
5. Testability (easy to test?)

Provide architectural recommendations:"""

        findings = await self._call_ollama(prompt)
        
        approved = "sound architecture" in findings.lower() or "well-designed" in findings.lower()
        
        # Save architecture review
        self._save_architecture_review(component, findings, findings, approved)
        
        result = f"✓ Architecture review: {component}\n"
        result += f"  Status: {'✓ APPROVED' if approved else '⚠ NEEDS REVISION'}\n"
        result += f"\n{findings[:400]}..."
        
        return result
    
    def _build_code_review_prompt(self, target: str, code: str, review_type: str) -> str:
        """Build code review prompt"""
        
        focus_areas = {
            "code": "code quality, style, correctness",
            "security": "security vulnerabilities, unsafe operations, input validation",
            "performance": "performance bottlenecks, inefficient algorithms, memory usage",
            "style": "coding standards, naming conventions, documentation"
        }
        
        focus = focus_areas.get(review_type, "general code quality")
        
        prompt = f"""Review this C++/CUDA code for U3DAW audio workstation.

Target: {target}
Focus: {focus}

Code:
```cpp
{code[:3000]}  # First 3000 chars
```

Review Criteria:
1. Correctness - Does it work? Edge cases handled?
2. Performance - Efficient algorithms? Cache-friendly?
3. Safety - Memory leaks? Buffer overflows? Race conditions?
4. Style - Clear naming? Good comments? Maintainable?

Provide:
- ✓ What's good
- ⚠ Issues found (with severity: low/medium/high)
- 💡 Suggestions for improvement

Be concise and actionable."""

        return prompt
    
    def _extract_and_save_findings(self, file_path: str, findings_text: str):
        """Parse findings and save individually"""
        
        # Simple heuristic: lines with warning symbols
        lines = findings_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect findings
            if any(marker in line for marker in ['⚠', '❌', 'warning:', 'issue:', 'problem:']):
                
                # Classify category
                category = "general"
                for cat, keywords in self.review_categories.items():
                    if any(kw in line_lower for kw in keywords):
                        category = cat
                        break
                
                # Classify severity
                severity = "medium"
                if any(word in line_lower for word in ["critical", "severe", "dangerous"]):
                    severity = "high"
                elif any(word in line_lower for word in ["minor", "suggestion", "consider"]):
                    severity = "low"
                
                # Save finding
                self._save_finding(file_path, category, severity, line.strip())
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM review_conversations")
        total_reviews = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM review_conversations WHERE approved = 1")
        approved_reviews = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM review_findings 
            WHERE resolved = 0 
            GROUP BY category
        """)
        open_findings = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM architecture_reviews WHERE approved = 1")
        approved_architectures = cursor.fetchone()[0]
        
        conn.close()
        
        approval_rate = (approved_reviews / total_reviews * 100) if total_reviews > 0 else 0
        
        return {
            "agent": self.name,
            "model": self.model,
            "total_reviews": total_reviews,
            "approved_reviews": approved_reviews,
            "approval_rate": f"{approval_rate:.1f}%",
            "open_findings_by_category": open_findings,
            "approved_architectures": approved_architectures,
            "memory_location": str(self.memory_root)
        }

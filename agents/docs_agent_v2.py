"""
DocsAgent - Production-Ready Documentation Agent
Model: mistral:7b
Purpose: Technical documentation, API docs, README generation
Memory: D:\AGENT_MEMORY with SQLite persistence
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List

class DocsAgent:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.name = "Docs"
        self.model = "mistral:7b"
        
        # Paths
        self.workspace = Path("D:/U3DAW")
        self.memory_root = Path("D:/AGENT_MEMORY")
        self.conversations_dir = self.memory_root / "conversations" / "docs"
        self.db_path = self.memory_root / "memory.db"
        
        # Ensure directories exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self._init_db()
        
        # Context
        self.max_context_messages = 6
        self.conversation_history: List[Dict] = []
        
        self._load_recent_context()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS docs_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                doc_type TEXT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                files_created TEXT,
                success INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documentation_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                module TEXT NOT NULL,
                doc_file TEXT NOT NULL,
                doc_type TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_recent_context(self):
        """Load recent documentation sessions"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prompt, response 
            FROM docs_conversations 
            WHERE success = 1
            ORDER BY id DESC 
            LIMIT ?
        """, (self.max_context_messages,))
        
        rows = cursor.fetchall()
        conn.close()
        
        for prompt, response in reversed(rows):
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
    
    def _save_to_db(self, prompt: str, response: str, doc_type: str = "", 
                    files_created: List[str] = None, success: bool = True):
        """Save documentation session"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO docs_conversations 
            (timestamp, doc_type, prompt, response, files_created, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            doc_type,
            prompt,
            response,
            json.dumps(files_created or []),
            1 if success else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _index_documentation(self, module: str, doc_file: str, doc_type: str):
        """Index created documentation"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documentation_index 
            (timestamp, module, doc_file, doc_type)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), module, doc_file, doc_type))
        
        conn.commit()
        conn.close()
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with mistral:7b for documentation"""
        
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes
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
            return "Error: Documentation generation timed out"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _save_documentation(self, doc_type: str, module_name: str, content: str) -> Path:
        """Save generated documentation to workspace"""
        
        docs_dir = self.workspace / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        ext_map = {
            "readme": "README.md",
            "api": f"API_{module_name}.md",
            "tutorial": f"TUTORIAL_{module_name}.md",
            "architecture": f"ARCHITECTURE_{module_name}.md",
            "inline": f"{module_name}_comments.txt"
        }
        
        filename = ext_map.get(doc_type, f"{module_name}_docs.md")
        doc_file = docs_dir / filename
        
        doc_file.write_text(content, encoding='utf-8')
        
        return doc_file
    
    async def execute(self, intent: Dict) -> str:
        """Main documentation generation execution"""
        target = intent.get('target', '')
        doc_type = intent.get('type', 'api')  # readme, api, tutorial, architecture, inline
        source_file = intent.get('file', '')
        
        print(f"[Docs] Generating documentation for: {target}")
        print(f"[Docs] Doc type: {doc_type}")
        print(f"[Docs] Using model: {self.model}")
        
        # Build documentation prompt
        prompt = self._build_docs_prompt(target, doc_type, source_file)
        
        # Generate documentation
        response = await self._call_ollama(prompt)
        
        # Extract module name
        module_name = target.lower().replace(' ', '_').replace('-', '_')
        
        # Save documentation
        doc_file = self._save_documentation(doc_type, module_name, response)
        
        print(f"[Docs] ✓ Created: {doc_file}")
        
        # Index documentation
        self._index_documentation(module_name, str(doc_file), doc_type)
        
        # Save to database
        self._save_to_db(prompt, response, doc_type, [str(doc_file)], success=True)
        
        # Build result
        result = f"✓ Docs generated for: {target}\n"
        result += f"  Type: {doc_type}\n"
        result += f"  File: {doc_file}\n"
        result += f"  Size: {len(response)} chars\n"
        
        return result
    
    def _build_docs_prompt(self, target: str, doc_type: str, source_file: str) -> str:
        """Build documentation generation prompt"""
        
        source_context = ""
        if source_file and Path(source_file).exists():
            code = Path(source_file).read_text(encoding='utf-8')
            source_context = f"\n\nSource code:\n```cpp\n{code}\n```"
        
        prompts = {
            "readme": f"""Generate a comprehensive README.md for {target} in the U3DAW project.

Include:
- Overview and purpose
- Key features
- Installation/setup
- Basic usage examples
- Configuration options
- Known limitations

{source_context}

Write clear, professional Markdown documentation:""",

            "api": f"""Generate API documentation for {target} in the U3DAW project.

Include:
- Class/function signatures
- Parameter descriptions
- Return values
- Usage examples
- Notes and warnings

{source_context}

Write detailed API documentation in Markdown:""",

            "tutorial": f"""Generate a tutorial for {target} in the U3DAW project.

Include:
- Step-by-step guide
- Code examples
- Common pitfalls
- Best practices
- Expected results

{source_context}

Write a beginner-friendly tutorial:""",

            "architecture": f"""Generate architecture documentation for {target} in the U3DAW project.

Include:
- System overview
- Component interactions
- Data flow
- Design decisions
- Performance considerations

{source_context}

Write technical architecture documentation:""",

            "inline": f"""Generate inline code comments for {target}.

Requirements:
- Brief but clear
- Explain WHY not WHAT
- Document complex logic
- Note assumptions/constraints

{source_context}

Provide commented version of the code:"""
        }
        
        return prompts.get(doc_type, prompts["api"])
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM docs_conversations WHERE success = 1")
        sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documentation_index")
        docs_created = cursor.fetchone()[0]
        
        cursor.execute("SELECT doc_type, COUNT(*) FROM documentation_index GROUP BY doc_type")
        docs_by_type = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "agent": self.name,
            "model": self.model,
            "doc_sessions": sessions,
            "docs_created": docs_created,
            "by_type": docs_by_type,
            "memory_location": str(self.memory_root)
        }

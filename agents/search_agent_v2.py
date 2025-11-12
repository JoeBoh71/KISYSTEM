"""
KISYSTEM Search Agent V2
Web-Recherche mit Context-Aware Learning
Memory: D:/AGENT_MEMORY with SQLite persistence

Author: Jörg Bohne
Date: 2025-11-06
"""

import asyncio
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from learning_module_v2 import LearningModuleV2


class SearchAgent:
    """
    Web-Recherche Agent mit Learning
    Nutzt Brave Search API
    """
    
    def __init__(self, supervisor=None):
        """
        Initialize Search Agent
        
        Args:
            supervisor: Supervisor instance (optional)
        """
        # Get workspace from supervisor or use default
        if supervisor and hasattr(supervisor, 'workspace'):
            workspace = supervisor.workspace
        else:
            workspace = "D:/AGENT_MEMORY"
        
        self.workspace = Path(workspace) / "searches"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Store supervisor reference
        self.supervisor = supervisor
        
        # Initialize Learning Module V2
        self.learning = LearningModuleV2()
        
        # Load Brave API key - Priority: Environment Variable > Config File
        import os
        self.api_key = os.environ.get('BRAVE_API_KEY')
        
        if not self.api_key:
            # Fallback: Load from config file
            config_path = Path(__file__).parent.parent / 'config' / 'api_keys.json'
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    key_from_config = config.get('brave_search_api_key')
                    if key_from_config and key_from_config != 'YOUR_BRAVE_API_KEY_HERE':
                        self.api_key = key_from_config
            except Exception as e:
                print(f"[Search] ⚠️ Could not load API key from config: {e}")
        
        if not self.api_key:
            print(f"[Search] ⚠️ No API key configured. Set BRAVE_API_KEY environment variable.")
        
        self.model = "phi4:latest"  # Fast model for search
        
        print(f"[Search] Initialized with workspace: {self.workspace}")
        print(f"[Search] API Key: {'✓' if self.api_key else '✗'}")
        print(f"[Search] Learning V2 active")
    
    async def execute(self, task) -> str:
        """
        Execute task (Supervisor API wrapper)
        
        Args:
            task: Search query (str or dict)
            
        Returns:
            Search results
        """
        # Handle dict input from supervisor
        if isinstance(task, dict):
            query = task.get('target', str(task))
        else:
            query = str(task)
        
        # Remove common prefixes
        query = query.replace("search:", "").replace("suche:", "").strip()
        return await self.search(query)
    
    async def search(self, query: str, context: str = "") -> str:
        """
        Search web for information
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            Search results summary
        """
        print(f"\n[Search] 🔍 Searching: {query}")
        
        if not self.api_key:
            return "❌ Brave Search API key nicht konfiguriert"
        
        # Check if we've searched this before
        similar_searches = self.learning.find_similar_solutions(
            error=query,
            code=context or query,
            model_used=self.model,
            min_confidence=0.80  # Higher threshold for search
        )
        
        if similar_searches:
            best = similar_searches[0]
            confidence = best['confidence']
            
            if confidence >= 0.90:
                print(f"[Search] 🧠 Using cached search result ({confidence:.1%})")
                return best['solution']['solution']
            else:
                print(f"[Search] 💡 Similar search found ({confidence:.1%}), but refreshing")
        
        try:
            # Perform search
            results = await self._brave_search(query)
            
            if not results:
                self.learning.store_solution(
                    error=query,
                    solution="No results found",
                    code=context or query,
                    model_used=self.model,
                    success=False
                )
                return "❌ Keine Suchergebnisse gefunden"
            
            # Format results
            summary = self._format_results(results)
            
            # Save to workspace
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.workspace / f"search_{timestamp}.json"
            
            try:
                filepath.write_text(
                    json.dumps({
                        'query': query,
                        'context': context,
                        'results': results,
                        'timestamp': timestamp
                    }, indent=2),
                    encoding='utf-8'
                )
            except Exception as e:
                print(f"[Search] ⚠️ Could not save results: {e}")
            
            # Store in learning
            self.learning.store_solution(
                error=query,
                solution=summary,
                code=context or query,
                model_used=self.model,
                success=True,
                solve_time=1.0
            )
            print(f"[Search] 🧠 Cached search result")
            
            return summary
            
        except Exception as e:
            print(f"[Search] ✗ Error: {e}")
            
            self.learning.store_solution(
                error=query,
                solution=str(e),
                code=context or query,
                model_used=self.model,
                success=False
            )
            
            return f"❌ Suchfehler: {str(e)}"
    
    async def _brave_search(self, query: str, count: int = 5) -> List[Dict]:
        """Perform Brave search"""
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        
        params = {
            'q': query,
            'count': count
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract web results
            if 'web' in data and 'results' in data['web']:
                return data['web']['results']
            
            return []
            
        except Exception as e:
            print(f"[Search] Brave API error: {e}")
            return []
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format search results"""
        if not results:
            return "Keine Ergebnisse"
        
        formatted = "🔍 Suchergebnisse:\n\n"
        
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            description = result.get('description', 'No description')
            
            formatted += f"{i}. **{title}**\n"
            formatted += f"   {description[:150]}...\n"
            formatted += f"   {url}\n\n"
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Get search statistics"""
        stats = self.learning.get_statistics()
        
        # Count searches
        search_count = stats['total_solutions']
        cached_count = sum(
            1 for sol in stats.get('top_solutions', [])
            if sol.get('domain') == 'general'
        )
        
        return {
            'total_searches': search_count,
            'cached_results': cached_count,
            'cache_hit_rate': cached_count / search_count if search_count > 0 else 0
        }


if __name__ == '__main__':
    async def test():
        searcher = SearchAgentV2()
        
        result = await searcher.search("Python async best practices")
        print(result)
        
        stats = searcher.get_stats()
        print(f"\n=== Stats ===")
        print(json.dumps(stats, indent=2))
    
    asyncio.run(test())

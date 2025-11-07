import json
import logging
from pathlib import Path
from datetime import datetime
import sys

sys.path.append("C:/KISYSTEM")
from security.security_module import SecurityModule

class SecureSupervisor:
    def __init__(self):
        self.base_path = Path("C:/KISYSTEM")
        self.config = self.load_config()
        self.security = SecurityModule()
        self.agents = {}
        self.setup_logging()
        self.logger.info("=== KISYSTEM SUPERVISOR STARTING ===")
        self.initialize_agents()
        
    def setup_logging(self):
        log_path = Path("D:/AGENT_MEMORY/audit")
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"supervisor_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - [%(name)s] - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger("SUPERVISOR")
        
    def load_config(self):
        config_file = self.base_path / "config" / "kisystem_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def initialize_agents(self):
        # Dummy agents for now
        self.agents = {
            'builder': {'name': 'Builder', 'status': 'ready'},
            'tester': {'name': 'Tester', 'status': 'ready'},
            'fixer': {'name': 'Fixer', 'status': 'ready'},
            'searcher': {'name': 'Searcher', 'status': 'sandboxed'}
        }
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def receive_command(self, command):
        is_safe, msg = self.security.validate_command(command, "user")
        if not is_safe:
            return f"BLOCKED: {msg}"
        
        # Parse command
        if "status" in command.lower():
            return self.get_status()
        elif "hilfe" in command.lower() or "help" in command.lower():
            return self.get_help()
        else:
            return f"Processing: {command}"
    
    def get_status(self):
        status = "=== KISYSTEM STATUS ===\n"
        for agent_id, agent in self.agents.items():
            status += f"{agent['name']}: {agent['status']}\n"
        return status
    
    def get_help(self):
        return """
=== KISYSTEM BEFEHLE ===
- erstelle [component] - Builder Agent
- teste [component] - Tester Agent
- debugge [problem] - Fixer Agent
- suche [topic] - Search Agent
- status - System Status
- hilfe - Diese Hilfe
- exit - Beenden
        """

if __name__ == "__main__":
    supervisor = SecureSupervisor()
    print("KISYSTEM bereit. 'hilfe' für Befehle.")
    while True:
        try:
            cmd = input("> ")
            if cmd.lower() == "exit":
                break
            response = supervisor.receive_command(cmd)
            print(response)
        except KeyboardInterrupt:
            break
    print("KISYSTEM beendet.")

import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

class SecurityModule:
    def __init__(self):
        self.base_path = Path("C:/KISYSTEM")
        self.audit_path = Path("D:/AGENT_MEMORY/audit")
        self.audit_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        self.load_security_patterns()
        
    def setup_logging(self):
        """Konfiguriert Security Logging"""
        log_file = self.audit_path / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SECURITY")
        
    def load_security_patterns(self):
        """Lädt Sicherheitsmuster"""
        self.blocked_patterns = [
            r'rm\s+-rf',
            r'format\s+[c-z]:',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess\.',
            r'os\.system',
        ]
        
        self.allowed_domains = [
            "github.com",
            "stackoverflow.com",
            "docs.nvidia.com",
            "cppreference.com",
            "docs.microsoft.com"
        ]
        
    def validate_command(self, command, agent_id="unknown"):
        """Validiert einen Befehl auf Sicherheit"""
        try:
            for pattern in self.blocked_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    self.logger.warning(f"BLOCKED: {agent_id} - Pattern: {pattern}")
                    return False, f"Sicherheitsrisiko: {pattern}"
            return True, "OK"
        except Exception as e:
            self.logger.error(f"Validierungsfehler: {e}")
            return False, str(e)
    
    def sanitize_code(self, code):
        """Säubert Code von gefährlichen Elementen"""
        sanitized = code
        sanitized = re.sub(r'os\.system\([^)]+\)', '# REMOVED: os.system', sanitized)
        sanitized = re.sub(r'(exec|eval)\s*\([^)]+\)', '# REMOVED: exec/eval', sanitized)
        return sanitized

if __name__ == "__main__":
    print("Security Module Test")
    security = SecurityModule()
    valid, msg = security.validate_command("print('Hello')", "test")
    print(f"Test: {valid} - {msg}")

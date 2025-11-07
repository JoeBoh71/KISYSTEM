#!/usr/bin/env python3
"""
KISYSTEM - Secure Autonomous Agent System
"""

import sys
import os
from pathlib import Path

# Add KISYSTEM to path
sys.path.insert(0, "C:/KISYSTEM")

def check_requirements():
    """Prüft Voraussetzungen"""
    print("=== KISYSTEM Systemcheck ===\n")
    
    checks = {
        "Verzeichnisstruktur": Path("C:/KISYSTEM/core").exists(),
        "Konfiguration": Path("C:/KISYSTEM/config/kisystem_config.json").exists(),
        "Security Module": Path("C:/KISYSTEM/security/security_module.py").exists(),
        "Supervisor": Path("C:/KISYSTEM/core/supervisor.py").exists(),
    }
    
    all_ok = True
    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
        if not status:
            all_ok = False
    
    return all_ok

def main():
    print("""
╔══════════════════════════════════════════╗
║           KISYSTEM v1.0                  ║
║    Secure Autonomous Agent System        ║
║        FOR U3DAW DEVELOPMENT            ║
╚══════════════════════════════════════════╝
    """)
    
    if not check_requirements():
        print("\n❌ Nicht alle Voraussetzungen erfüllt!")
        return
    
    print("\n✓ Alle Checks bestanden!\n")
    
    # Import and start supervisor
    from core.supervisor_v2 import SecureSupervisor
    
    supervisor = SecureSupervisor()
    print("\n" + "="*50)
    print("KISYSTEM BEREIT - SICHERER MODUS AKTIV")
    print("="*50)
    print("Befehle: hilfe, status, exit\n")
    
    while True:
        try:
            command = input("[YOU→SUPERVISOR]: ").strip()
            
            if command.lower() == 'exit':
                print("[SUPERVISOR] Fahre sicher herunter...")
                break
            
            response = supervisor.receive_command(command)
            print(f"[SUPERVISOR→YOU]: {response}\n")
            
        except KeyboardInterrupt:
            print("\n[SUPERVISOR] Shutdown...")
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")

if __name__ == "__main__":
    main()

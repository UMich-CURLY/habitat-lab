#!/usr/bin/env python3

import time
import os
from pathlib import Path
import subprocess

def monitor_tensorboard():
    """Monitor tensorboard logs and display available sensors."""
    
    tb_dir = "./tb_logs/oracle_social_nav"
    
    print("=== Sensor & Measurement Monitor ===")
    print(f"Monitoring: {tb_dir}")
    print()
    
    # Start tensorboard in background
    try:
        tb_process = subprocess.Popen([
            "tensorboard", "--logdir", tb_dir, "--port", "6006", "--bind_all"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Tensorboard started on http://localhost:6006")
        print("Available visualizations will include:")
        print("  • Agent positions and trajectories")
        print("  • Sensor readings (depth, GPS, compass)")
        print("  • Measurements (success, distance to goal)")
        print("  • PDDL task progress")
        print("  • Multi-agent coordination metrics")
        print()
        
        # Monitor for log files
        while True:
            if Path(tb_dir).exists():
                log_files = list(Path(tb_dir).rglob("*.tfevents*"))
                if log_files:
                    print(f"Found {len(log_files)} tensorboard log files")
                    for log_file in log_files[:3]:  # Show first 3
                        print(f"  • {log_file.name}")
                    if len(log_files) > 3:
                        print(f"  • ... and {len(log_files) - 3} more")
                    break
            
            print("Waiting for logs to appear...")
            time.sleep(2)
        
        print("\nMonitoring active. Press Ctrl+C to stop.")
        
        # Keep monitoring
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    except FileNotFoundError:
        print("Error: tensorboard not found. Install with: pip install tensorboard")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if 'tb_process' in locals():
            tb_process.terminate()
    
    return 0

if __name__ == "__main__":
    exit(monitor_tensorboard())
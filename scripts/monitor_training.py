#!/usr/bin/env python3
"""
Monitor training progress from log file.
"""

import time
import sys
from pathlib import Path
import re
from datetime import datetime

def monitor_log(log_file: str, refresh_rate: int = 5):
    """Monitor training log file for progress.
    
    Args:
        log_file: Path to log file
        refresh_rate: Seconds between updates
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📊 Monitoring: {log_file}")
    print(f"🔄 Refresh rate: {refresh_rate}s")
    print("-" * 60)
    
    last_size = 0
    last_trial = -1
    start_time = datetime.now()
    
    while True:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Check if file grew
            current_size = len(content)
            if current_size > last_size:
                # Extract new content
                new_content = content[last_size:]
                
                # Look for trial progress
                trial_matches = re.findall(r'Trial (\d+) finished', new_content)
                if trial_matches:
                    last_trial = int(trial_matches[-1])
                
                # Look for best value
                best_match = re.search(r'Best is trial \d+ with value: ([\d.]+)', new_content)
                best_value = float(best_match.group(1)) if best_match else None
                
                # Look for completion
                if "✅ Otimização completa" in new_content:
                    print("\n🎉 TRAINING COMPLETE!")
                    
                    # Extract final metrics
                    metrics = re.findall(r'(\w+):\s+([\d.-]+)', new_content)
                    if metrics:
                        print("\n📊 Final Metrics:")
                        for name, value in metrics:
                            print(f"  • {name}: {value}")
                    break
                
                # Update display
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\r⏱️ Time: {elapsed:.0f}s | Trial: {last_trial+1}/100 | ", end="")
                if best_value:
                    print(f"Best Score: {best_value:.4f}", end="")
                
                last_size = current_size
            
            time.sleep(refresh_rate)
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.txt"
    monitor_log(log_file)
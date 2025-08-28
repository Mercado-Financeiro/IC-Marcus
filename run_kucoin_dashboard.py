#!/usr/bin/env python3
"""
Simple launcher for KuCoin Dashboard.
Uses direct Streamlit integration without WebSocket server.
"""

import subprocess
import sys
import time
from pathlib import Path

def start_dashboard():
    """Start Streamlit dashboard."""
    print("🚀 Starting KuCoin Dashboard...")
    
    dashboard_file = Path(__file__).parent / "src" / "dashboard" / "app_enhanced.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_file),
            "--server.port=8512",
            "--server.address=localhost",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def main():
    """Launch KuCoin dashboard system."""
    print("🟢 KuCoin Dashboard Launcher (Streamlit-Only)")
    print("=" * 50)
    
    print("\n📋 Instructions:")
    print("1. Dashboard will open in your browser")
    print("2. Select '🟢 KuCoin Trading' from the sidebar")  
    print("3. Choose your trading pair (BTC/USDT, ETH/USDT, etc.)")
    print("4. Real data if CCXT available, mock data otherwise")
    print("5. Press Ctrl+C to stop")
    
    print(f"\n🌐 Dashboard URL: http://localhost:8512")
    print("=" * 50)
    
    # Start dashboard directly
    start_dashboard()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Install script for KuCoin Dashboard dependencies.
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Install required dependencies for KuCoin Dashboard."""
    print("🔧 KuCoin Dashboard Dependency Installer")
    print("=" * 50)
    
    # Required packages
    packages = [
        ("streamlit", "streamlit>=1.28.0"),
        ("plotly", "plotly>=5.17.0"), 
        ("pandas", "pandas>=1.5.0"),
        ("numpy", "numpy>=1.24.0"),
        ("ccxt", "ccxt>=4.1.0"),
        ("websockets", "websockets>=12.0"),
        ("asyncio", None),  # Built-in
        ("json", None),     # Built-in
        ("threading", None) # Built-in
    ]
    
    missing_packages = []
    
    print("📋 Checking installed packages...")
    
    for package_name, pip_name in packages:
        if check_package(package_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} (missing)")
            if pip_name:
                missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
        
        print("\n🔄 Re-checking packages...")
        all_good = True
        
        for package_name, _ in packages:
            if check_package(package_name):
                print(f"✅ {package_name}")
            else:
                print(f"❌ {package_name} still missing")
                all_good = False
        
        if all_good:
            print("\n🎉 All dependencies installed successfully!")
        else:
            print("\n⚠️ Some packages failed to install. Please install manually:")
            print("pip install streamlit plotly pandas numpy ccxt websockets")
    
    else:
        print("\n🎉 All dependencies are already installed!")
    
    print("\n🚀 Ready to run KuCoin Dashboard!")
    print("Run: python run_kucoin_dashboard.py")

if __name__ == "__main__":
    main()
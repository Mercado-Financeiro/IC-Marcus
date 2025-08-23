"""Legacy dashboard wrapper for backward compatibility."""

# Import and run the new modular dashboard
from src.dashboard.main import main

if __name__ == "__main__":
    main()

# Note: This file maintains backward compatibility.
# The actual dashboard implementation is now in src.dashboard.main.py
# and related modules in src.dashboard/
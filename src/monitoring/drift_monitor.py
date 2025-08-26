"""Legacy drift monitor wrapper for backward compatibility."""

# Import everything from the new modular structure
from src.monitoring.drift import DriftConfig, DriftMonitor

# Re-export for backward compatibility
__all__ = ["DriftConfig", "DriftMonitor"]

# Note: This file maintains backward compatibility.
# New code should import directly from src.monitoring.drift
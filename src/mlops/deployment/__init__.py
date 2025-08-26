"""
Edge deployment module for lightweight ML inference.
"""

from .edge_deployment import EdgeDeploymentManager, EdgeConfig
from .model_optimizer import ModelOptimizer, OptimizationConfig
from .containerization import ContainerBuilder

__all__ = [
    'EdgeDeploymentManager',
    'EdgeConfig',
    'ModelOptimizer',
    'OptimizationConfig',
    'ContainerBuilder'
]
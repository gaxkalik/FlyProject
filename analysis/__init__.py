"""
Analysis package for FlyMesh.
"""

from .metrics import compute_signal_latency, compute_propagation_velocity
from .visualizer import plot_two_fly_dynamics, plot_spatial_trajectories

__all__ = [
    "compute_signal_latency",
    "compute_propagation_velocity",
    "plot_two_fly_dynamics",
    "plot_spatial_trajectories",
]

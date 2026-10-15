"""
Network and Information Metrics for Biological Swarm Mesh.
Computes signal latency, alert wave propagation velocity, branching ratio, and percolation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

def compute_signal_latency(
    alert_times: np.ndarray,
    weather_onset_time: float,
    source_index: int = 0
) -> Dict[str, float]:
    """
    Computes alert timing metrics across the agent population.
    alert_times: [N] array of timestamps when each fly first fired an alarm DN.
    """
    valid_alerts = alert_times[alert_times > 0.0]
    num_alerted = len(valid_alerts)
    total_agents = len(alert_times)

    source_latency = float(alert_times[source_index] - weather_onset_time) if alert_times[source_index] > 0 else -1.0

    if num_alerted > 1:
        # Latency from source to first neighbor
        sorted_times = np.sort(valid_alerts)
        hop1_latency = float(sorted_times[1] - sorted_times[0])
        total_cascade_time = float(sorted_times[-1] - sorted_times[0])
    else:
        hop1_latency = -1.0
        total_cascade_time = 0.0

    return {
        "source_latency_ms": source_latency * 1000.0,
        "inter_fly_hop_latency_ms": hop1_latency * 1000.0,
        "total_cascade_time_ms": total_cascade_time * 1000.0,
        "percent_swarm_alerted": (num_alerted / total_agents) * 100.0,
    }

def compute_propagation_velocity(
    positions: np.ndarray,
    alert_times: np.ndarray,
    source_index: int = 0
) -> Optional[float]:
    """
    Estimates the propagation velocity v_wave (m/s) of the biological warning wave.
    """
    if alert_times[source_index] <= 0:
        return None

    source_pos = positions[source_index]
    t0 = alert_times[source_index]

    alerted_indices = np.where((alert_times > 0) & (np.arange(len(alert_times)) != source_index))[0]
    if len(alerted_indices) == 0:
        return None

    dists = np.linalg.norm(positions[alerted_indices] - source_pos, axis=1)
    delays = alert_times[alerted_indices] - t0

    valid_mask = delays > 1e-4
    if not np.any(valid_mask):
        return None

    # Regression slope: dist = v * delay
    v_wave = np.mean(dists[valid_mask] / delays[valid_mask])
    return float(v_wave)

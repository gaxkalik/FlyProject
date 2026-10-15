"""
Percolation Analysis, Branching Ratio Dynamics, and Information Reach for Swarm Mesh.
Computes epidemic branching ratio (R0), spatial percolation threshold, and attenuation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

def compute_branching_ratio(
    alert_times: np.ndarray,      # [N] timestamps when each agent fired alarm DN (-1 if unalerted)
    positions: np.ndarray,        # [N, 3]
    interaction_radius: float = 0.25, # Max effective multi-modal communication distance (m)
    causal_window_ms: Tuple[float, float] = (15.0, 75.0) # Physiological min/max hop delay
) -> Dict[str, float]:
    """
    Computes the biological epidemic branching ratio R0:
    R0 = E[ Number of secondary alerted neighbors directly triggered by an alerted index fly ]
    R0 > 1.0: Supercritical runaway panic cascade
    R0 < 1.0: Subcritical dying signal
    R0 ~ 1.0: Self-Organized Criticality (optimal information reach without false-positive ringing)
    """
    valid_mask = alert_times > 0
    alert_indices = np.where(valid_mask)[0]
    num_alerted = len(alert_indices)

    if num_alerted <= 1:
        return {"r0": 0.0, "criticality_state": "subcritical", "num_generations": 0}

    # Sort alerts chronologically
    sorted_idx = alert_indices[np.argsort(alert_times[alert_indices])]

    # Compute pairwise spatial distances
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)

    causal_min_s = causal_window_ms[0] / 1000.0
    causal_max_s = causal_window_ms[1] / 1000.0

    secondary_offspring_counts = []
    parent_map = {}

    for i in sorted_idx:
        t_i = alert_times[i]
        offspring = 0
        for j in sorted_idx:
            if i == j:
                continue
            t_j = alert_times[j]
            delay = t_j - t_i

            # Check if j is within communication radius and within causal response window
            if dists[i, j] <= interaction_radius and causal_min_s <= delay <= causal_max_s:
                # If j hasn't been claimed by an earlier parent
                if j not in parent_map:
                    parent_map[j] = i
                    offspring += 1

        secondary_offspring_counts.append(offspring)

    # Average branching ratio across all parent agents that had an opportunity to transmit
    if len(secondary_offspring_counts) > 0:
        # Exclude the last tier if it had no remaining flies ahead of it
        r0 = float(np.mean(secondary_offspring_counts[:-1])) if len(secondary_offspring_counts) > 1 else float(secondary_offspring_counts[0])
    else:
        r0 = 0.0

    if 0.85 <= r0 <= 1.25:
        state = "critical (Self-Organized Criticality)"
    elif r0 > 1.25:
        state = "supercritical (explosive cascade)"
    else:
        state = "subcritical (attenuating)"

    return {
        "r0": r0,
        "criticality_state": state,
        "num_alerted_agents": num_alerted,
        "branching_counts": secondary_offspring_counts
    }

def compute_percolation_metrics(
    positions: np.ndarray,
    alert_times: np.ndarray,
    cluster_radius: float = 0.22
) -> Dict[str, float]:
    """
    Evaluates whether the warning cascade spans the geometric extent of the swarm (Percolation).
    """
    num_agents = len(positions)
    alerted_mask = alert_times > 0
    num_alerted = int(np.sum(alerted_mask))

    if num_alerted == 0:
        return {
            "percolation_achieved": 0.0,
            "max_cluster_span_m": 0.0,
            "swarm_span_m": float(np.max(np.linalg.norm(positions - positions[0], axis=1))),
            "span_ratio": 0.0
        }

    alerted_pos = positions[alerted_mask]
    source_pos = positions[0]

    max_dist_alerted = float(np.max(np.linalg.norm(alerted_pos - source_pos, axis=1)))
    total_swarm_span = float(np.max(np.linalg.norm(positions - source_pos, axis=1)))

    span_ratio = max_dist_alerted / max(1e-4, total_swarm_span)
    percolated = 1.0 if span_ratio >= 0.85 else 0.0

    return {
        "percolation_achieved": percolated,
        "max_cluster_span_m": max_dist_alerted,
        "swarm_span_m": total_swarm_span,
        "span_ratio": span_ratio
    }

"""
Visualization Suite for Drosophila Connectome Mesh Network.
Generates publication-quality figures:
1. Two-Fly Closed-Loop Neural Dynamics (Membrane potential traces and sensory inputs).
2. Spiking Raster across all connectome neurons for both sender and receiver flies.
3. 3D Spatial Trajectories with localized meteorological front and communication hops.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

def plot_two_fly_dynamics(
    time_series_ms: np.ndarray,
    v_mem_history: np.ndarray,       # [T, num_agents, num_neurons]
    spike_history: np.ndarray,       # [T, num_agents, num_neurons]
    i_ext_history: np.ndarray,       # [T, num_agents, num_neurons]
    positions_history: np.ndarray,   # [T, num_agents, 3]
    circuit_neurons: List[str],
    output_path: str = "milestone1_two_fly_dynamics.png"
):
    """
    Renders comprehensive 4-panel figure comparing Fly 1 (Weather Source) and Fly 2 (Receiver).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    neuron_to_idx = {name: i for i, name in enumerate(circuit_neurons)}

    # Selected key neurons for comparison
    tracked_neurons = [
        ("VP1l_Cold", "Cold Sensor (Arista)", "#1f77b4"),
        ("JON_C", "Wind/Pressure (JO-C)", "#17becf"),
        ("LC11", "Visual Conspecific (LC11)", "#ff7f0e"),
        ("JON_A", "Acoustic Flutter (JO-A)", "#bcbd22"),
        ("LHON_Aversive", "Aversive Valence (LH)", "#9467bd"),
        ("DNp01_AlarmSaccade", "Alarm Saccade (DNp01)", "#d62728"),
        ("bLN1_AcousticBuzz", "Acoustic Buzz (bLN1)", "#8c564b"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    # ------------------ FLY 1 (SENDER / WEATHER FRONT) ------------------
    # Panel 1: Sensory Injected Currents
    ax = axes[0, 0]
    ax.plot(time_series_ms, i_ext_history[:, 0, neuron_to_idx["VP1l_Cold"]], label="VP1l Cold (pA)", color="#1f77b4", lw=1.8)
    ax.plot(time_series_ms, i_ext_history[:, 0, neuron_to_idx["JON_C"]], label="JON_C Wind (pA)", color="#17becf", lw=1.8)
    ax.set_title("Fly 1 (Weather Sector) - Sensory Inputs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Injected Current (pA)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 2: Central Integration (LHON)
    ax = axes[1, 0]
    ax.plot(time_series_ms, v_mem_history[:, 0, neuron_to_idx["LHON_Aversive"]], color="#9467bd", lw=1.8, label="LHON Aversive Vm")
    ax.axhline(-50.0, color="gray", linestyle="--", alpha=0.6, label="Threshold V_th0")
    ax.set_title("Fly 1 - Central Valence Integration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Pot. (mV)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 3: Descending Motor Commands (DNp01 & bLN1)
    ax = axes[2, 0]
    ax.plot(time_series_ms, v_mem_history[:, 0, neuron_to_idx["DNp01_AlarmSaccade"]], color="#d62728", lw=1.8, label="DNp01 Saccade Vm")
    ax.plot(time_series_ms, v_mem_history[:, 0, neuron_to_idx["bLN1_AcousticBuzz"]], color="#8c564b", lw=1.5, linestyle=":", label="bLN1 Buzz Vm")
    ax.set_title("Fly 1 - Descending Command & Warning Broadcast", fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Pot. (mV)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 4: Spiking Raster for Fly 1
    ax = axes[3, 0]
    spike_indices = np.where(spike_history[:, 0, :] > 0.5)
    ax.scatter(time_series_ms[spike_indices[0]], spike_indices[1], s=8, color="#2ca02c", marker="|")
    ax.set_yticks(range(len(circuit_neurons)))
    ax.set_yticklabels(circuit_neurons, fontsize=7)
    ax.set_title("Fly 1 - Whole-Circuit Spike Raster", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Neuron Index", fontsize=10)
    ax.grid(True, alpha=0.2)

    # ------------------ FLY 2 (RECEIVER / MESH NEIGHBOR) ------------------
    # Panel 1: Inter-Fly Communication Inputs (Visual & Acoustic)
    ax = axes[0, 1]
    ax.plot(time_series_ms, i_ext_history[:, 1, neuron_to_idx["LC11"]], label="LC11 Visual Loom (pA)", color="#ff7f0e", lw=1.8)
    ax.plot(time_series_ms, i_ext_history[:, 1, neuron_to_idx["JON_A"]], label="JON_A Acoustic (pA)", color="#bcbd22", lw=1.8)
    ax.set_title("Fly 2 (Neighbor) - Received Warning Signals", fontsize=12, fontweight="bold")
    ax.set_ylabel("Injected Current (pA)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 2: Central Integration (LHON)
    ax = axes[1, 1]
    ax.plot(time_series_ms, v_mem_history[:, 1, neuron_to_idx["LHON_Aversive"]], color="#9467bd", lw=1.8, label="LHON Aversive Vm")
    ax.axhline(-50.0, color="gray", linestyle="--", alpha=0.6, label="Threshold V_th0")
    ax.set_title("Fly 2 - Alert State Activation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Pot. (mV)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 3: Descending Motor Commands (DNp01 Reactive Evasion)
    ax = axes[2, 1]
    ax.plot(time_series_ms, v_mem_history[:, 1, neuron_to_idx["DNp01_AlarmSaccade"]], color="#d62728", lw=1.8, label="DNp01 Evasive Saccade Vm")
    ax.plot(time_series_ms, v_mem_history[:, 1, neuron_to_idx["bLN1_AcousticBuzz"]], color="#8c564b", lw=1.5, linestyle=":", label="bLN1 Re-Broadcast Vm")
    ax.set_title("Fly 2 - Reactive Evasion & Re-Broadcast", fontsize=12, fontweight="bold")
    ax.set_ylabel("Membrane Pot. (mV)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 4: Spiking Raster for Fly 2
    ax = axes[3, 1]
    spike_indices_2 = np.where(spike_history[:, 1, :] > 0.5)
    ax.scatter(time_series_ms[spike_indices_2[0]], spike_indices_2[1], s=8, color="#1f77b4", marker="|")
    ax.set_yticks(range(len(circuit_neurons)))
    ax.set_yticklabels(circuit_neurons, fontsize=7)
    ax.set_title("Fly 2 - Whole-Circuit Spike Raster", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Neuron Index", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Drosophila Connectome Mesh: Milestone 1 Closed-Loop Neural Dynamics", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved neural dynamics figure to: {output_path}")

def plot_spatial_trajectories(
    positions_history: np.ndarray,  # [T, num_agents, 3]
    alert_times: np.ndarray,        # [num_agents]
    weather_center: np.ndarray,     # [3]
    weather_radius: float,
    output_path: str = "milestone1_spatial_trajectories.png"
):
    """
    Renders 3D physical trajectories of the flies relative to the meteorological shock front.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw weather front sphere wireframe
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_s = weather_center[0] + weather_radius * np.outer(np.cos(u), np.sin(v))
    y_s = weather_center[1] + weather_radius * np.outer(np.sin(u), np.sin(v))
    z_s = weather_center[2] + weather_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_s, y_s, z_s, color="deepskyblue", alpha=0.18, label="Weather Front Zone")

    # Plot Fly 1 trajectory
    pos1 = positions_history[:, 0, :]
    ax.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], label="Fly 1 (Weather Source)", color="#d62728", lw=2.5)
    ax.scatter(pos1[0, 0], pos1[0, 1], pos1[0, 2], color="darkred", s=60, marker="o", label="Fly 1 Start")
    ax.scatter(pos1[-1, 0], pos1[-1, 1], pos1[-1, 2], color="red", s=70, marker="^", label="Fly 1 End")

    # Plot Fly 2 trajectory
    pos2 = positions_history[:, 1, :]
    ax.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], label="Fly 2 (Mesh Receiver)", color="#1f77b4", lw=2.5)
    ax.scatter(pos2[0, 0], pos2[0, 1], pos2[0, 2], color="navy", s=60, marker="o", label="Fly 2 Start")
    ax.scatter(pos2[-1, 0], pos2[-1, 1], pos2[-1, 2], color="blue", s=70, marker="^", label="Fly 2 End")

    # Communication vector between start positions
    ax.plot(
        [pos1[0, 0], pos2[0, 0]],
        [pos1[0, 1], pos2[0, 1]],
        [pos1[0, 2], pos2[0, 2]],
        color="purple", linestyle="--", lw=1.8, label="Inter-Fly Mesh Channel"
    )

    ax.set_xlabel("X (meters)", fontsize=11)
    ax.set_ylabel("Y (meters)", fontsize=11)
    ax.set_zlabel("Z (meters)", fontsize=11)
    ax.set_title("3D Flight Trajectories & Weather Warning Transmission", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Saved 3D spatial trajectories figure to: {output_path}")

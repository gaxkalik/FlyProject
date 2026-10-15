"""
Milestone 2: Multi-Fly Micro-Cluster Mesh Network Simulation.
Demonstrates:
1. Decentralized biological mesh propagation across a multi-agent cluster (12 flies).
2. Multi-hop warning relay: Fly 0 (Weather Sector) -> Fly 1 -> Fly 2 -> ... -> Fly 11.
3. Multi-modal transmission (acoustic particle velocity, visual optical flow, pheromone plume).
4. Quantifies signal propagation wave velocity (v_wave), inter-hop latency, and branching stability.
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_fly"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fly_connectome import build_canonical_subcircuit
from neural_engine import SwarmLIFEngine, DrosophilaSensoryEncoder, LIFParameters
from environment import WeatherFrontParams, WeatherEnvironment, SpatialArena
from agents import FlyAgentSwarm
from analysis import compute_signal_latency, compute_propagation_velocity

def run_milestone2_cluster_simulation(
    num_agents: int = 12,
    sim_duration_s: float = 1.2,
    dt_ms: float = 0.5,
    weather_onset_s: float = 0.15,
    plot_results: bool = True
):
    print("=" * 70)
    print(f"RUNNING MILESTONE 2: MULTI-AGENT CLUSTER MESH RELAY ({num_agents} FLIES)")
    print("=" * 70)

    # 1. Initialize Connectome Sub-Circuit & Spiking Engine for N Agents
    circuit = build_canonical_subcircuit(scale_factor=5.0)
    lif_params = LIFParameters(
        v_rest=-70.0,
        v_reset=-75.0,
        v_th_0=-50.0,
        delta_th=12.0,
        tau_m=18.0,
        tau_syn=4.5,
        tau_th=70.0,
        r_m=10.0,
        refractory_period=2.0
    )
    lif_engine = SwarmLIFEngine(circuit=circuit, num_agents=num_agents, params=lif_params)
    sensory_encoder = DrosophilaSensoryEncoder(circuit=circuit)

    # 2. Setup Spatial Geometry: Natural Staggered Swarm Cluster (12 Flies)
    # Provides lateral and diagonal line-of-sight visual visibility (LC11 optimal field)
    # and near-field acoustic coupling between neighbor tiers.
    initial_positions = np.array([
        [0.00,  0.00, 1.00],   # Tier 0: Fly 0 (Front tip in weather sector)
        [0.08,  0.08, 1.00],   # Tier 1: Fly 1 (Right lateral wing)
        [0.08, -0.08, 1.00],   # Tier 1: Fly 2 (Left lateral wing)
        [0.16,  0.15, 1.02],   # Tier 2: Fly 3 (Outer right)
        [0.16,  0.00, 1.00],   # Tier 2: Fly 4 (Center core)
        [0.16, -0.15, 0.98],   # Tier 2: Fly 5 (Outer left)
        [0.24,  0.20, 1.01],   # Tier 3: Fly 6 (Flank right)
        [0.24,  0.07, 1.00],   # Tier 3: Fly 7 (Inner right)
        [0.24, -0.07, 1.00],   # Tier 3: Fly 8 (Inner left)
        [0.24, -0.20, 0.99],   # Tier 3: Fly 9 (Flank left)
        [0.32,  0.10, 1.01],   # Tier 4: Fly 10 (Rear right)
        [0.32, -0.10, 0.99]    # Tier 4: Fly 11 (Rear left)
    ], dtype=np.float32)

    initial_headings = np.zeros(num_agents, dtype=np.float32)  # Cruising +X

    # 3. Weather Front strictly localized to Fly 0 (Sector 0)
    weather_front = WeatherFrontParams(
        center=np.array([0.02, 0.0, 1.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        radius=0.06,            # Strictly confined to 6 cm around Fly 0
        temp_drop=9.0,          # 9 deg C cooling
        pressure_drop=2.8,      # 2.8 kPa drop
        humidity_spike=25.0,
        wind_speed=1.6,
        onset_time=weather_onset_s,
        ramp_duration=0.08
    )
    weather_env = WeatherEnvironment(front_params=weather_front)
    arena = SpatialArena(weather_env=weather_env)

    # 4. Initialize Swarm
    swarm = FlyAgentSwarm(
        num_agents=num_agents,
        circuit=circuit,
        initial_positions=initial_positions,
        initial_headings=initial_headings
    )

    # 5. Simulation Loop
    dt_s = dt_ms / 1000.0
    total_steps = int(sim_duration_s / dt_s)
    time_series_ms = np.linspace(0, sim_duration_s * 1000.0, total_steps)

    # Recording buffers
    positions_history = np.zeros((total_steps, num_agents, 3), dtype=np.float32)
    spike_history = np.zeros((total_steps, num_agents, circuit.num_neurons), dtype=np.float32)
    dnp01_idx = circuit.get_index("DNp01_AlarmSaccade")

    print(f"[Simulation] Executing multi-agent swarm: {num_agents} flies, {total_steps} steps...")
    start_wall = time.time()

    for step in range(total_steps):
        t_s = step * dt_s

        # A. Sample local weather
        temp_c, d_temp_dt, hum_pct, d_hum_dt, p_kpa, d_p_dt, wind_vectors = weather_env.sample_at_positions(
            swarm.positions, t_s
        )

        # B. Inter-fly physical signaling
        acoustic_emissions = swarm.get_acoustic_emissions()
        received_acoustic = arena.compute_pairwise_acoustic_signals(swarm.positions, acoustic_emissions)

        saccade_states = swarm.get_saccade_states()
        head_dirs = swarm.get_head_directions()
        conspecific_motion, looming_rate = arena.compute_pairwise_visual_signals(
            swarm.positions, swarm.velocities, head_dirs, saccade_states
        )

        new_odors = swarm.check_new_odor_releases()
        for i in range(num_agents):
            if new_odors[i]:
                arena.add_pheromone_puff(swarm.positions[i], t_s, mass=1.0)
        alarm_odors = arena.sample_pheromone_concentration(swarm.positions, t_s)

        v_rel = swarm.velocities - wind_vectors
        rel_airspeeds = np.linalg.norm(v_rel, axis=1)

        # C. Transduce sensory inputs
        i_ext = sensory_encoder.encode_sensory_inputs(
            num_agents=num_agents,
            temp_c=temp_c,
            d_temp_dt=d_temp_dt,
            humidity_pct=hum_pct,
            d_humidity_dt=d_hum_dt,
            pressure_kpa=p_kpa,
            d_pressure_dt=d_p_dt,
            rel_airspeed_ms=rel_airspeeds,
            acoustic_velocity=received_acoustic,
            visual_conspecific_motion=conspecific_motion,
            visual_looming_rate=looming_rate,
            alarm_odor_conc=alarm_odors
        )

        # D. Step LIF Spiking Engine
        spikes = lif_engine.step(dt=dt_ms, i_ext=i_ext)

        # E. Update Kinematics
        swarm.update_kinematics(dt=dt_s, spikes=spikes, wind_vectors=wind_vectors, current_time=t_s)

        # Record
        positions_history[step] = swarm.positions
        spike_history[step] = spikes

    elapsed_wall = time.time() - start_wall
    print(f"[Simulation] Completed in {elapsed_wall:.2f} s ({total_steps / elapsed_wall:.1f} steps/s).")

    # 6. Quantitative Network Metrics
    alert_times = swarm.first_alert_time
    valid_alert_mask = alert_times > 0
    num_alerted = int(np.sum(valid_alert_mask))

    metrics = compute_signal_latency(alert_times, weather_onset_s, source_index=0)
    v_wave = compute_propagation_velocity(initial_positions, alert_times, source_index=0)

    print("\n" + "=" * 50)
    print("MILESTONE 2: MULTI-AGENT CLUSTER METRICS:")
    print("=" * 50)
    print(f"• Total Agents in Mesh:        {num_agents}")
    print(f"• Number of Agents Alerted:    {num_alerted} / {num_agents} ({metrics['percent_swarm_alerted']:.1f}%)")
    print(f"• Fly 0 (Source) Detection:    {alert_times[0] * 1000.0:.1f} ms (Latency: {metrics['source_latency_ms']:.1f} ms)")

    for k in range(min(num_agents, 6)):
        t_k = alert_times[k] * 1000.0 if alert_times[k] > 0 else -1.0
        print(f"  - Fly {k:2d} Alert Time:       {t_k:6.1f} ms")

    if num_alerted > 1:
        print(f"• Mean Inter-Hop Latency:      {metrics['inter_fly_hop_latency_ms']:.1f} ms")
        print(f"• Total Cascade Duration:      {metrics['total_cascade_time_ms']:.1f} ms")
        if v_wave:
            print(f"• Warning Wave Speed (v_wave): {v_wave:.2f} m/s")
    print("=" * 50)

    # 7. Render Milestone 2 Figures
    if plot_results:
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Panel 1: Multi-Agent Alert Timing vs Distance
        dists_from_source = np.linalg.norm(initial_positions - initial_positions[0], axis=1)
        ax1.scatter(dists_from_source, alert_times * 1000.0, s=90, c=range(num_agents), cmap="turbo", edgecolors="black", zorder=3)
        for i in range(num_agents):
            if alert_times[i] > 0:
                ax1.annotate(f"Fly {i}", (dists_from_source[i], alert_times[i] * 1000.0 + 10), fontsize=9)
        if v_wave:
            # Linear trend
            x_line = np.linspace(0, max(dists_from_source), 50)
            y_line = (alert_times[0] + x_line / v_wave) * 1000.0
            ax1.plot(x_line, y_line, "r--", label=f"Wave Front Speed: {v_wave:.2f} m/s")

        ax1.set_xlabel("Distance from Weather Sector (m)", fontsize=11)
        ax1.set_ylabel("First Alert Timestamp (ms)", fontsize=11)
        ax1.set_title("Biological Mesh Warning Wave Propagation Across Swarm Cluster", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        # Panel 2: Multi-Agent DNp01 Spiking Cascade Raster
        for i in range(num_agents):
            spk_t = time_series_ms[spike_history[:, i, dnp01_idx] > 0.5]
            ax2.scatter(spk_t, np.full_like(spk_t, i), s=18, color=plt.cm.turbo(i / num_agents), marker="|")

        ax2.set_yticks(range(num_agents))
        ax2.set_yticklabels([f"Fly {i}" for i in range(num_agents)])
        ax2.set_xlabel("Time (ms)", fontsize=11)
        ax2.set_ylabel("Fly Agent Index", fontsize=11)
        ax2.set_title("Descending Alarm Neuron (DNp01) Multi-Agent Spiking Cascade", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_cluster = os.path.join(plots_dir, "milestone2_cluster_cascade.png")
        plt.savefig(out_cluster, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved cluster cascade figure to: {out_cluster}")

    return {
        "metrics": metrics,
        "alert_times": alert_times,
        "v_wave": v_wave,
        "success": (num_alerted >= num_agents * 0.75)
    }

if __name__ == "__main__":
    res = run_milestone2_cluster_simulation()
    if res["success"]:
        print("\n>>> MILESTONE 2 CLUSTER MESH RELAY SUCCESSFUL! <<<")
    else:
        print("\n>>> MILESTONE 2 INCOMPLETE PROPAGATION <<<")

"""
Milestone 1: Two-Fly Sensorimotor & Communication Proof-of-Concept.
Demonstrates:
1. Environmental weather front triggering sensory neurons in Fly 1.
2. Connectome-mediated descending motor activation (DNp01 / bLN1).
3. Inter-fly signaling via near-field acoustic flutter and visual conspecific tracking.
4. Receipt and translation of warning signal by Fly 2, driving reactive evasive state.
"""

import sys
import os
import time
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_fly"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fly_connectome import build_canonical_subcircuit
from neural_engine import SwarmLIFEngine, DrosophilaSensoryEncoder, LIFParameters
from environment import WeatherFrontParams, WeatherEnvironment, SpatialArena
from agents import FlyAgentSwarm
from analysis import compute_signal_latency, plot_two_fly_dynamics, plot_spatial_trajectories

def run_milestone1_simulation(
    sim_duration_s: float = 1.0,
    dt_ms: float = 0.5,
    weather_onset_s: float = 0.20,
    plot_results: bool = True
):
    print("=" * 70)
    print("RUNNING MILESTONE 1: TWO-FLY SENSORIMOTOR & SIGNAL TRANSFER POC")
    print("=" * 70)

    # 1. Initialize Connectome Sub-Circuit & Spiking Neural Engine
    circuit = build_canonical_subcircuit(scale_factor=5.0)
    print(f"[Connectome] Loaded canonical sub-circuit with {circuit.num_neurons} neurons.")
    print(f"             Synaptic connections count: {np.count_nonzero(circuit.adj_matrix)}")

    num_agents = 2
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

    # 2. Setup 3D Spatial Arena & Localized Weather Disturbance
    # Fly 1 is cruising along +X into the weather front centered at (0.08, 0.0, 1.0)
    # Fly 2 is cruising parallel at y=0.16 m in calm sector outside the front radius
    initial_positions = np.array([
        [0.0, 0.0, 1.0],      # Fly 1 (Direct path into weather sector)
        [0.0, 0.16, 1.0]      # Fly 2 (Parallel in calm sector, 16 cm separation)
    ], dtype=np.float32)

    initial_headings = np.array([0.0, 0.0], dtype=np.float32)  # Facing +X direction

    weather_front = WeatherFrontParams(
        center=np.array([0.08, 0.0, 1.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        radius=0.14,            # Confined shock radius (14 cm)
        temp_drop=9.0,          # 9 deg C cooling
        pressure_drop=2.8,      # 2.8 kPa drop
        humidity_spike=25.0,    # 25% humidity spike
        wind_speed=1.5,         # 1.5 m/s wind
        onset_time=weather_onset_s,
        ramp_duration=0.10
    )
    weather_env = WeatherEnvironment(front_params=weather_front)
    arena = SpatialArena(weather_env=weather_env)

    # 3. Setup Fly Agents
    swarm = FlyAgentSwarm(
        num_agents=num_agents,
        circuit=circuit,
        initial_positions=initial_positions,
        initial_headings=initial_headings
    )

    # 4. Simulation Time Loop
    dt_s = dt_ms / 1000.0
    total_steps = int(sim_duration_s / dt_s)
    time_series_ms = np.linspace(0, sim_duration_s * 1000.0, total_steps)

    # History recording buffers
    v_mem_history = np.zeros((total_steps, num_agents, circuit.num_neurons), dtype=np.float32)
    spike_history = np.zeros((total_steps, num_agents, circuit.num_neurons), dtype=np.float32)
    i_ext_history = np.zeros((total_steps, num_agents, circuit.num_neurons), dtype=np.float32)
    positions_history = np.zeros((total_steps, num_agents, 3), dtype=np.float32)

    print(f"[Simulation] Starting execution: {total_steps} steps, dt={dt_ms} ms, duration={sim_duration_s} s...")
    start_wall_time = time.time()

    for step in range(total_steps):
        t_s = step * dt_s

        # A. Sample local weather variables at agent positions
        temp_c, d_temp_dt, hum_pct, d_hum_dt, p_kpa, d_p_dt, wind_vectors = weather_env.sample_at_positions(
            swarm.positions, t_s
        )

        # B. Inter-fly physical signaling
        # 1. Near-field acoustic particle velocity (wing buzz from neighbors)
        acoustic_emissions = swarm.get_acoustic_emissions()
        received_acoustic = arena.compute_pairwise_acoustic_signals(swarm.positions, acoustic_emissions)

        # 2. Visual conspecific motion & optical looming
        saccade_states = swarm.get_saccade_states()
        head_dirs = swarm.get_head_directions()
        conspecific_motion, looming_rate = arena.compute_pairwise_visual_signals(
            swarm.positions, swarm.velocities, head_dirs, saccade_states
        )

        # 3. Volatile alarm pheromones
        new_odors = swarm.check_new_odor_releases()
        for i in range(num_agents):
            if new_odors[i]:
                arena.add_pheromone_puff(swarm.positions[i], t_s, mass=1.0)
        alarm_odors = arena.sample_pheromone_concentration(swarm.positions, t_s)

        # Compute relative airspeed for mechanosensory antennae
        v_rel = swarm.velocities - wind_vectors
        rel_airspeeds = np.linalg.norm(v_rel, axis=1)

        # C. Transduce sensory inputs into injected neural currents
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

        # D. Step Spiking Connectome Engine
        spikes = lif_engine.step(dt=dt_ms, i_ext=i_ext)

        # E. Update Kinematics & Motor Saccades
        swarm.update_kinematics(dt=dt_s, spikes=spikes, wind_vectors=wind_vectors, current_time=t_s)

        # Record histories
        v_mem_history[step] = lif_engine.v_mem
        spike_history[step] = spikes
        i_ext_history[step] = i_ext
        positions_history[step] = swarm.positions

    elapsed_wall = time.time() - start_wall_time
    print(f"[Simulation] Completed in {elapsed_wall:.2f} s ({total_steps / elapsed_wall:.1f} steps/s).")

    # 5. Quantitative Signal Latency Analysis
    metrics = compute_signal_latency(
        alert_times=swarm.first_alert_time,
        weather_onset_time=weather_onset_s,
        source_index=0
    )

    print("\n" + "=" * 50)
    print("QUANTITATIVE METRICS & LATENCY BENCHMARK:")
    print("=" * 50)
    print(f"• Weather Front Onset:               {weather_onset_s * 1000.0:.1f} ms")
    print(f"• Fly 1 (Weather Source) Alert Time: {swarm.first_alert_time[0] * 1000.0:.1f} ms")
    print(f"• Fly 1 Detection Latency:           {metrics['source_latency_ms']:.1f} ms")
    print(f"• Fly 2 (Mesh Neighbor) Alert Time:  {swarm.first_alert_time[1] * 1000.0:.1f} ms")
    print(f"• Inter-Fly Hop Signal Latency:      {metrics['inter_fly_hop_latency_ms']:.1f} ms")
    print(f"• Percentage Swarm Alerted:          {metrics['percent_swarm_alerted']:.0f}%")
    print("=" * 50)

    # 6. Render Artifact Figures
    if plot_results:
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        dyn_path = os.path.join(plots_dir, "milestone1_two_fly_dynamics.png")
        plot_two_fly_dynamics(
            time_series_ms=time_series_ms,
            v_mem_history=v_mem_history,
            spike_history=spike_history,
            i_ext_history=i_ext_history,
            positions_history=positions_history,
            circuit_neurons=circuit.neurons,
            output_path=dyn_path
        )

        traj_path = os.path.join(plots_dir, "milestone1_spatial_trajectories.png")
        plot_spatial_trajectories(
            positions_history=positions_history,
            alert_times=swarm.first_alert_time,
            weather_center=weather_front.center,
            weather_radius=weather_front.radius,
            output_path=traj_path
        )

    return {
        "metrics": metrics,
        "first_alert_times": swarm.first_alert_time,
        "circuit": circuit,
        "success": (swarm.first_alert_time[0] > 0 and swarm.first_alert_time[1] > 0)
    }

if __name__ == "__main__":
    results = run_milestone1_simulation()
    if results["success"]:
        print("\n>>> MILESTONE 1 PROOF-OF-CONCEPT SUCCESSFUL! Both flies demonstrated closed-loop alert transfer. <<<")
    else:
        print("\n>>> MILESTONE 1 FAILED TO ACHIEVE INTER-FLY PROPAGATION <<<")
        sys.exit(1)

"""
Milestone 3: MuJoCo Multi-Agent Biophysics Swarm Driven by Real-Time Detailed Weather Telemetry.
Combines:
1. MuJoCo 3.16.0 Continuous 6-DOF Aerodynamic Physics Engine.
2. Real-Time Meteorological Telemetry (Open-Meteo API) with Atmospheric Boundary-Layer Downscaling.
3. FlyWire Connectome Spiking LIF Neural Sub-Circuits.
4. Decentralized Peer-to-Peer Mesh Warning Propagation under Live Weather Conditions.
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
from environment import (
    MuJoCoSwarmArena,
    RealTimeWeatherClient,
    WeatherFrontParams,
    WeatherEnvironment,
    SpatialArena
)
from analysis import compute_signal_latency, compute_propagation_velocity

def run_realtime_weather_mujoco_swarm(
    num_agents: int = 12,
    sim_duration_s: float = 1.0,
    dt_ms: float = 1.0,  # 1 ms MuJoCo timestep
    weather_onset_s: float = 0.15,
    plot_results: bool = True
):
    print("=" * 75)
    print("RUNNING MILESTONE 3: MUJOCO BIOPHYSICS SWARM + REAL-TIME DETAILED WEATHER")
    print("=" * 75)

    # 1. Fetch Real-Time Detailed Weather Telemetry
    weather_client = RealTimeWeatherClient()
    reading = weather_client.fetch_live_weather()
    print("\n[Meteorological Telemetry - Live Ingestion]:")
    print(f"  • Station:            {reading.latitude}°N, {reading.longitude}°E (elevation: {reading.elevation_m}m)")
    print(f"  • Timestamp:          {reading.timestamp_iso}")
    print(f"  • Ambient Temp:       {reading.temperature_c:.1f} °C")
    print(f"  • Relative Humidity:  {reading.relative_humidity_pct:.1f} %")
    print(f"  • Surface Pressure:   {reading.surface_pressure_kpa:.2f} kPa")
    print(f"  • Macro Wind (10m):   {reading.wind_speed_ms:.2f} m/s (Gusts: {reading.wind_gust_ms:.2f} m/s)")
    print(f"  • Wind Heading:       {reading.wind_direction_deg:.0f}°")

    # 2. Setup Staggered Cluster Coordinates for 12 Flies in 3D Space
    # In nature, flies orient their headings into the ambient wind (positive anemotaxis)
    # Wind direction from reading is angle wind is blowing FROM. We face into it:
    upwind_yaw = np.radians(reading.wind_direction_deg)
    initial_headings = np.full(num_agents, upwind_yaw, dtype=np.float32)

    # Position swarm along upwind axis
    # Rotate canonical formation into the wind coordinate frame
    rot_matrix = np.array([
        [np.cos(upwind_yaw), -np.sin(upwind_yaw)],
        [np.sin(upwind_yaw),  np.cos(upwind_yaw)]
    ])

    canonical_2d = np.array([
        [0.00,  0.00],   # Tier 0: Fly 0 (Front windward tip in weather sector)
        [0.08,  0.08],   # Tier 1: Fly 1 (Right lateral wing)
        [0.08, -0.08],   # Tier 1: Fly 2 (Left lateral wing)
        [0.16,  0.15],   # Tier 2: Fly 3 (Outer right)
        [0.16,  0.00],   # Tier 2: Fly 4 (Center core)
        [0.16, -0.15],   # Tier 2: Fly 5 (Outer left)
        [0.24,  0.20],   # Tier 3: Fly 6 (Flank right)
        [0.24,  0.07],   # Tier 3: Fly 7 (Inner right)
        [0.24, -0.07],   # Tier 3: Fly 8 (Inner left)
        [0.24, -0.20],   # Tier 3: Fly 9 (Flank left)
        [0.32,  0.10],   # Tier 4: Fly 10 (Rear right)
        [0.32, -0.10]    # Tier 4: Fly 11 (Rear left)
    ], dtype=np.float32)

    rotated_2d = canonical_2d @ rot_matrix.T
    z_coords = np.array([0.80, 0.80, 0.80, 0.82, 0.80, 0.78, 0.81, 0.80, 0.80, 0.79, 0.81, 0.79], dtype=np.float32)
    initial_positions = np.column_stack([rotated_2d, z_coords])

    # 3. Initialize MuJoCo Biophysics Arena
    dt_s = dt_ms / 1000.0
    mujoco_arena = MuJoCoSwarmArena(
        num_agents=num_agents,
        initial_positions=initial_positions,
        initial_headings=initial_headings,
        timestep=dt_s
    )
    print(f"\n[MuJoCo Biophysics] Initialized 6-DOF arena with {num_agents} fly rigid bodies.")

    # 4. Initialize Connectome Sub-Circuit Engine
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

    # Calibrate sensory baseline adaptation to live ambient station reading
    macro_winds_init = weather_client.downscale_to_insect_boundary_layer(reading, np.array([0.80]))[0]
    ambient_insect_wind = float(np.linalg.norm(macro_winds_init))
    sensory_encoder.set_ambient_baselines(
        temp_c=reading.temperature_c,
        humidity_pct=reading.relative_humidity_pct,
        pressure_kpa=reading.surface_pressure_kpa,
        wind_speed_ms=ambient_insect_wind + 0.45
    )

    # 5. Localized Real-Time Meteorological Disturbance (Convective Gust Cell)
    # The micro-cell moves with the ambient fluid flow (fluid advection)
    live_gust_speed = max(2.5, reading.wind_gust_ms)
    weather_front = WeatherFrontParams(
        center=initial_positions[0].copy(),
        velocity=macro_winds_init.copy(),
        radius=0.06,            # Confined shock cell (6 cm around Fly 0)
        temp_drop=6.5,          # Rapid localized adiabatic drop (from 14.2°C to 7.7°C)
        pressure_drop=2.5,      # 2.5 kPa barometric depression
        humidity_spike=15.0,
        wind_speed=live_gust_speed,
        wind_direction=reading.wind_vector_10m / (np.linalg.norm(reading.wind_vector_10m) + 1e-6),
        onset_time=weather_onset_s,
        ramp_duration=0.08
    )
    weather_env = WeatherEnvironment(
        baseline_temp=reading.temperature_c,
        baseline_humidity=reading.relative_humidity_pct,
        baseline_pressure=reading.surface_pressure_kpa,
        front_params=weather_front
    )
    inter_fly_arena = SpatialArena(weather_env=weather_env)

    # State tracking
    total_steps = int(sim_duration_s / dt_s)
    time_series_ms = np.linspace(0, sim_duration_s * 1000.0, total_steps)

    positions_history = np.zeros((total_steps, num_agents, 3), dtype=np.float32)
    spike_history = np.zeros((total_steps, num_agents, circuit.num_neurons), dtype=np.float32)
    alert_times = np.full(num_agents, -1.0, dtype=np.float32)

    saccade_timer = np.zeros(num_agents, dtype=np.float32)
    acoustic_timer = np.zeros(num_agents, dtype=np.float32)

    idx_dnp01 = circuit.get_index("DNp01_AlarmSaccade")
    idx_dna02 = circuit.get_index("DNa02_Steering")
    idx_bln1 = circuit.get_index("bLN1_AcousticBuzz")

    print(f"\n[Simulation] Executing MuJoCo Swarm: {num_agents} flies, {total_steps} steps (dt={dt_ms} ms)...")
    start_wall = time.time()

    for step in range(total_steps):
        t_s = step * dt_s

        # A. Current MuJoCo State
        positions = mujoco_arena.get_positions()
        velocities = mujoco_arena.get_velocities()
        headings = mujoco_arena.get_headings()

        # B. Downscale Real-Time Weather to Micro-Scale Altitudes
        # 1. Macro boundary layer wind at insect heights (Prandtl-von Kármán)
        z_heights = positions[:, 2]
        macro_winds = weather_client.downscale_to_insect_boundary_layer(reading, z_heights)

        # 2. Localized convective front disturbance
        temp_c, d_temp_dt, hum_pct, d_hum_dt, p_kpa, d_p_dt, storm_winds = weather_env.sample_at_positions(
            positions, t_s
        )
        total_winds = macro_winds + storm_winds

        # C. Inter-Fly Acoustic & Visual Signaling
        acoustic_emissions = (acoustic_timer > 0.0).astype(np.float32)
        received_acoustic = inter_fly_arena.compute_pairwise_acoustic_signals(positions, acoustic_emissions)

        saccade_active_mask = (saccade_timer > 0.0).astype(np.float32)
        head_dirs = np.stack([np.cos(headings), np.sin(headings), np.zeros(num_agents, dtype=np.float32)], axis=1)
        conspecific_motion, looming_rate = inter_fly_arena.compute_pairwise_visual_signals(
            positions, velocities, head_dirs, saccade_active_mask
        )

        alarm_odors = inter_fly_arena.sample_pheromone_concentration(positions, t_s)
        rel_airspeeds = np.linalg.norm(velocities - total_winds, axis=1)

        # D. Transduce into Injected Connectome Currents
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

        # E. Step Spiking Connectome Engine
        spikes = lif_engine.step(dt=dt_ms, i_ext=i_ext)

        # F. Decode Motor Commands into Timers
        dnp01_spiked = (spikes[:, idx_dnp01] > 0.5)
        bln1_spiked = (spikes[:, idx_bln1] > 0.5)
        dna02_spiked = (spikes[:, idx_dna02] > 0.5)

        for i in range(num_agents):
            if dnp01_spiked[i]:
                saccade_timer[i] = 0.08
                if alert_times[i] < 0:
                    alert_times[i] = t_s
            if bln1_spiked[i]:
                acoustic_timer[i] = 0.06
                if alert_times[i] < 0:
                    alert_times[i] = t_s

        saccade_timer = np.maximum(0.0, saccade_timer - dt_s)
        acoustic_timer = np.maximum(0.0, acoustic_timer - dt_s)

        # G. Advance MuJoCo Physics with Aerodynamic Forces & Descending Torques
        mujoco_arena.step(
            wind_vectors=total_winds,
            saccade_active=(saccade_timer > 0.0).astype(np.float32),
            steering_active=dna02_spiked.astype(np.float32)
        )

        # Record
        positions_history[step] = positions
        spike_history[step] = spikes

    elapsed_wall = time.time() - start_wall
    print(f"\n[Simulation] Completed MuJoCo run in {elapsed_wall:.2f} s ({total_steps / elapsed_wall:.1f} steps/s).")

    # 6. Quantitative Metrics
    metrics = compute_signal_latency(alert_times, weather_onset_s, source_index=0)
    v_wave = compute_propagation_velocity(initial_positions, alert_times, source_index=0)
    num_alerted = int(np.sum(alert_times > 0))

    print("\n" + "=" * 55)
    print("MILESTONE 3: MUJOCO BIOPHYSICS SWARM BENCHMARK:")
    print("=" * 55)
    print(f"• Total Agents in MuJoCo:      {num_agents} Flies")
    print(f"• Swarm Alert Convergence:    {num_alerted} / {num_agents} ({metrics['percent_swarm_alerted']:.1f}%)")
    print(f"• Weather Front Onset:         {weather_onset_s * 1000.0:.1f} ms")
    print(f"• Fly 0 (Source) Detection:    {alert_times[0] * 1000.0:.1f} ms (Latency: {metrics['source_latency_ms']:.1f} ms)")
    for k in range(min(num_agents, 6)):
        print(f"  - Fly {k:2d} Alert Timestamp:   {alert_times[k] * 1000.0:6.1f} ms")
    if num_alerted > 1:
        print(f"• Mean Inter-Hop Latency:      {metrics['inter_fly_hop_latency_ms']:.1f} ms")
        print(f"• Total Cascade Duration:      {metrics['total_cascade_time_ms']:.1f} ms")
        if v_wave:
            print(f"• Warning Wave Speed (v_wave): {v_wave:.2f} m/s")
    print("=" * 55)

    # 7. Render Figures
    if plot_results:
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Figure 1: MuJoCo Cascade & Neural Timing
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Panel 1: Alert Times vs Distance
        dists = np.linalg.norm(initial_positions - initial_positions[0], axis=1)
        ax1.scatter(dists, alert_times * 1000.0, s=90, c=range(num_agents), cmap="plasma", edgecolors="black", zorder=3)
        for i in range(num_agents):
            if alert_times[i] > 0:
                ax1.annotate(f"Fly {i}", (dists[i], alert_times[i] * 1000.0 + 8), fontsize=9)
        if v_wave:
            x_line = np.linspace(0, max(dists), 50)
            y_line = (alert_times[0] + x_line / v_wave) * 1000.0
            ax1.plot(x_line, y_line, "r--", lw=2, label=f"Warning Wave Speed: {v_wave:.2f} m/s")

        ax1.set_xlabel("Distance from Sector 0 (meters)", fontsize=11)
        ax1.set_ylabel("Alert Timestamp (ms)", fontsize=11)
        ax1.set_title(
            f"MuJoCo Biophysics Swarm: Warning Propagation under Real-Time Weather\n"
            f"(Live Telemetry: T={reading.temperature_c:.1f}°C, H={reading.relative_humidity_pct:.0f}%, "
            f"P={reading.surface_pressure_kpa:.1f}kPa, Wind={reading.wind_speed_ms:.1f}m/s)",
            fontsize=12, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        # Panel 2: Multi-Agent DNp01 Spiking Cascade in MuJoCo
        for i in range(num_agents):
            spk_t = time_series_ms[spike_history[:, i, idx_dnp01] > 0.5]
            ax2.scatter(spk_t, np.full_like(spk_t, i), s=18, color=plt.cm.plasma(i / num_agents), marker="|")

        ax2.set_yticks(range(num_agents))
        ax2.set_yticklabels([f"Fly {i}" for i in range(num_agents)])
        ax2.set_xlabel("Time (ms)", fontsize=11)
        ax2.set_ylabel("MuJoCo Agent Index", fontsize=11)
        ax2.set_title("Descending Alarm Neuron (DNp01) Spikes Across MuJoCo Swarm", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_cascade = os.path.join(plots_dir, "milestone3_mujoco_realtime_weather_cascade.png")
        plt.savefig(out_cascade, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved MuJoCo cascade plot to: {out_cascade}")

        # Figure 2: 3D Spatial Trajectories in MuJoCo
        fig = plt.figure(figsize=(11, 9))
        ax3d = fig.add_subplot(111, projection='3d')

        for i in range(num_agents):
            traj = positions_history[:, i, :]
            color = plt.cm.plasma(i / num_agents)
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=2.0, alpha=0.85)
            ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=40, marker="o")
            ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=60, marker="^")
            ax3d.text(traj[0, 0], traj[0, 1], traj[0, 2] + 0.015, f"Fly {i}", fontsize=8)

        # Weather Front Sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        xs = weather_front.center[0] + weather_front.radius * np.outer(np.cos(u), np.sin(v))
        ys = weather_front.center[1] + weather_front.radius * np.outer(np.sin(u), np.sin(v))
        zs = weather_front.center[2] + weather_front.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax3d.plot_wireframe(xs, ys, zs, color="deepskyblue", alpha=0.25, label="Localized Shock Cell")

        ax3d.set_xlabel("X (m)", fontsize=10)
        ax3d.set_ylabel("Y (m)", fontsize=10)
        ax3d.set_zlabel("Altitude Z (m)", fontsize=10)
        ax3d.set_title("MuJoCo Continuous 6-DOF Flight Trajectories in Live Weather Field", fontsize=12, fontweight="bold")
        ax3d.legend(loc="upper right")
        plt.tight_layout()

        out_3d = os.path.join(plots_dir, "milestone3_mujoco_3d_flight_trajectories.png")
        plt.savefig(out_3d, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved MuJoCo 3D trajectories plot to: {out_3d}")

    return {
        "metrics": metrics,
        "alert_times": alert_times,
        "v_wave": v_wave,
        "reading": reading,
        "success": (num_alerted >= num_agents * 0.75)
    }

if __name__ == "__main__":
    res = run_realtime_weather_mujoco_swarm()
    if res["success"]:
        print("\n>>> MILESTONE 3: MUJOCO + REAL-TIME WEATHER SIMULATION SUCCESSFUL! <<<")
    else:
        print("\n>>> MILESTONE 3 INCOMPLETE PROPAGATION <<<")
        sys.exit(1)

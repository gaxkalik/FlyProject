"""
Milestone 4: Large-Scale Swarm Percolation, Turbulence Robustness, & Criticality (MuJoCo 40-Fly Swarm).
Demonstrates:
1. Scaling up to a 40-fly biological mesh network in 3D MuJoCo space.
2. Robustness under continuous Dryden atmospheric turbulence and sensory background noise.
3. Self-Organized Criticality analysis: Measuring epidemic branching ratio (R0 ~ 1.0).
4. Spatial percolation across the full swarm volume under live meteorological telemetry.
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
from environment.turbulence import AtmosphericTurbulenceModel, generate_ambient_acoustic_noise
from analysis import (
    compute_signal_latency,
    compute_propagation_velocity,
    compute_branching_ratio,
    compute_percolation_metrics
)

def run_large_scale_percolation_swarm(
    num_agents: int = 40,
    sim_duration_s: float = 1.2,
    dt_ms: float = 1.0,
    weather_onset_s: float = 0.15,
    plot_results: bool = True
):
    print("=" * 80)
    print(f"RUNNING MILESTONE 4: LARGE-SCALE SWARM PERCOLATION & CRITICALITY ({num_agents} FLIES)")
    print("=" * 80)

    # 1. Fetch Real-Time Detailed Weather Telemetry
    weather_client = RealTimeWeatherClient()
    reading = weather_client.fetch_live_weather()
    print("\n[Meteorological Telemetry - Live Ingestion]:")
    print(f"  • Station:            {reading.latitude}°N, {reading.longitude}°E (alt: {reading.elevation_m}m)")
    print(f"  • Ambient Temp:       {reading.temperature_c:.1f} °C, RH: {reading.relative_humidity_pct:.0f}%, Pressure: {reading.surface_pressure_kpa:.2f} kPa")
    print(f"  • Macro Wind (10m):   {reading.wind_speed_ms:.2f} m/s (Gusts: {reading.wind_gust_ms:.2f} m/s), Heading: {reading.wind_direction_deg:.0f}°")

    # 2. Generate 3D Swarm Formation (40 Flies in a Dispersed Tiered Swarm Volume)
    # Swarm extends along the windward axis, spanning ~1.2 meters
    rng = np.random.default_rng(2026)
    upwind_yaw = np.radians(reading.wind_direction_deg)
    initial_headings = np.full(num_agents, upwind_yaw, dtype=np.float32)

    # Generate tiered grid with jitter to emulate natural swarm spacing (8 - 14 cm nearest neighbor)
    num_tiers = 8
    agents_per_tier = num_agents // num_tiers  # 5 flies per tier

    pos_list = []
    # Tier 0 (Tip in Sector 0)
    for tier in range(num_tiers):
        x_base = tier * 0.12  # 12 cm between tiers along length
        for k in range(agents_per_tier):
            y_base = (k - (agents_per_tier - 1) / 2.0) * 0.09  # 9 cm lateral spacing
            z_base = 0.80 + 0.04 * np.sin(tier + k)
            # Add biological position jitter
            xj = x_base + rng.normal(0, 0.015)
            yj = y_base + rng.normal(0, 0.012)
            zj = z_base + rng.normal(0, 0.010)
            pos_list.append([xj, yj, zj])

    canonical_3d = np.array(pos_list, dtype=np.float32)

    # Rotate swarm coordinates into the upwind heading
    rot_matrix = np.array([
        [np.cos(upwind_yaw), -np.sin(upwind_yaw)],
        [np.sin(upwind_yaw),  np.cos(upwind_yaw)]
    ], dtype=np.float32)

    rotated_xy = canonical_3d[:, :2] @ rot_matrix.T
    initial_positions = np.column_stack([rotated_xy, canonical_3d[:, 2]])

    # 3. Initialize MuJoCo 6-DOF Biophysics Arena for 40 Flies
    dt_s = dt_ms / 1000.0
    mujoco_arena = MuJoCoSwarmArena(
        num_agents=num_agents,
        initial_positions=initial_positions,
        initial_headings=initial_headings,
        timestep=dt_s
    )
    print(f"\n[MuJoCo Physics] Initialized continuous 6-DOF swarm arena with {num_agents} fly rigid bodies.")

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

    # Calibrate sensory baseline adaptation to prevailing ambient weather + turbulence margin
    macro_winds_init = weather_client.downscale_to_insect_boundary_layer(reading, np.array([0.80]))[0]
    ambient_insect_wind = float(np.linalg.norm(macro_winds_init))
    sensory_encoder.set_ambient_baselines(
        temp_c=reading.temperature_c,
        humidity_pct=reading.relative_humidity_pct,
        pressure_kpa=reading.surface_pressure_kpa,
        wind_speed_ms=ambient_insect_wind + 0.85
    )

    # 5. Atmospheric Turbulence & Weather Disturbance
    turbulence_model = AtmosphericTurbulenceModel(
        turbulence_intensity=0.35,  # 0.35 m/s turbulent fluctuations
        integral_length_scale=0.30, # 30 cm correlation length
        time_scale_s=0.20
    )

    live_gust_speed = max(2.5, reading.wind_gust_ms)
    weather_front = WeatherFrontParams(
        center=initial_positions[0].copy(),
        velocity=macro_winds_init.copy(),
        radius=0.08,            # 8 cm localized convective cell around Fly 0
        temp_drop=6.5,          # Localized cooling
        pressure_drop=2.5,
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

    print(f"\n[Simulation] Executing 40-Fly MuJoCo Swarm: {total_steps} steps with Dryden turbulence...")
    start_wall = time.time()

    for step in range(total_steps):
        t_s = step * dt_s

        # A. Sample MuJoCo state
        positions = mujoco_arena.get_positions()
        velocities = mujoco_arena.get_velocities()
        headings = mujoco_arena.get_headings()

        # B. Macro Wind + Atmospheric Turbulence + Convective Cell
        macro_winds = weather_client.downscale_to_insect_boundary_layer(reading, positions[:, 2])
        turbulent_winds = turbulence_model.sample_turbulence(positions, dt=dt_s)
        temp_c, d_temp_dt, hum_pct, d_hum_dt, p_kpa, d_p_dt, storm_winds = weather_env.sample_at_positions(positions, t_s)
        total_winds = macro_winds + turbulent_winds + storm_winds

        # C. Inter-Fly Acoustic & Visual Signaling under Noise
        # Ambient acoustic noise floor (turbulence boundary layer noise)
        ambient_noise = generate_ambient_acoustic_noise(num_agents, snr_floor=0.08)
        acoustic_emissions = (acoustic_timer > 0.0).astype(np.float32)
        received_acoustic = inter_fly_arena.compute_pairwise_acoustic_signals(positions, acoustic_emissions) + ambient_noise

        saccade_active_mask = (saccade_timer > 0.0).astype(np.float32)
        head_dirs = np.stack([np.cos(headings), np.sin(headings), np.zeros(num_agents, dtype=np.float32)], axis=1)
        conspecific_motion, looming_rate = inter_fly_arena.compute_pairwise_visual_signals(
            positions, velocities, head_dirs, saccade_active_mask
        )

        alarm_odors = inter_fly_arena.sample_pheromone_concentration(positions, t_s)
        rel_airspeeds = np.linalg.norm(velocities - total_winds, axis=1)

        # D. Transduce into Injected Currents
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

        # E. Step Spiking Connectome
        spikes = lif_engine.step(dt=dt_ms, i_ext=i_ext)

        # F. Motor Decoders
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

        # G. Advance MuJoCo Biophysics
        mujoco_arena.step(
            wind_vectors=total_winds,
            saccade_active=(saccade_timer > 0.0).astype(np.float32),
            steering_active=dna02_spiked.astype(np.float32)
        )

        # Record
        positions_history[step] = positions
        spike_history[step] = spikes

    elapsed_wall = time.time() - start_wall
    print(f"\n[Simulation] Completed 40-fly MuJoCo run in {elapsed_wall:.2f} s ({total_steps / elapsed_wall:.1f} steps/s).")

    # 6. Quantitative Criticality, Branching, and Percolation Analysis
    num_alerted = int(np.sum(alert_times > 0))
    branching = compute_branching_ratio(alert_times, initial_positions, interaction_radius=0.25)
    percolation = compute_percolation_metrics(initial_positions, alert_times)
    metrics = compute_signal_latency(alert_times, weather_onset_s, source_index=0)
    v_wave = compute_propagation_velocity(initial_positions, alert_times, source_index=0)

    print("\n" + "=" * 60)
    print("MILESTONE 4: SWARM CRITICALITY & PERCOLATION BENCHMARK:")
    print("=" * 60)
    print(f"• Total Swarm Population:     {num_agents} Flies (MuJoCo 6-DOF)")
    print(f"• Swarm Alert Convergence:    {num_alerted} / {num_agents} ({num_alerted / num_agents * 100.0:.1f}%)")
    print(f"• Branching Ratio (R0):       {branching['r0']:.2f} ({branching['criticality_state']})")
    print(f"• Spatial Percolation Span:   {percolation['span_ratio'] * 100.0:.1f}% ({percolation['max_cluster_span_m']:.2f}m / {percolation['swarm_span_m']:.2f}m)")
    print(f"• Percolation Achieved:       {'YES (Full Spanning Cluster)' if percolation['percolation_achieved'] else 'NO'}")
    print(f"• Fly 0 (Source) Latency:     {metrics['source_latency_ms']:.1f} ms")
    print(f"• Mean Inter-Hop Latency:     {metrics['inter_fly_hop_latency_ms']:.1f} ms")
    if v_wave:
        print(f"• Warning Wave Speed:         {v_wave:.2f} m/s")
    print("=" * 60)

    # 7. Render Milestone 4 Figures
    if plot_results:
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Figure 1: Percolation Curve, Branching Dynamics, and Spiking Cascade
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)

        # Panel 1: Alert Time vs Distance (Wavefront Propagation)
        ax1 = fig.add_subplot(gs[0, :])
        dists = np.linalg.norm(initial_positions - initial_positions[0], axis=1)
        scatter = ax1.scatter(dists, alert_times * 1000.0, s=80, c=alert_times * 1000.0, cmap="viridis", edgecolors="black", zorder=3)
        if v_wave:
            x_line = np.linspace(0, max(dists), 50)
            y_line = (alert_times[0] + x_line / v_wave) * 1000.0
            ax1.plot(x_line, y_line, "r--", lw=2, label=f"Wavefront Speed: {v_wave:.2f} m/s")
        ax1.set_xlabel("Distance from Weather Sector 0 (meters)", fontsize=11)
        ax1.set_ylabel("Alert Timestamp (ms)", fontsize=11)
        ax1.set_title(f"Warning Wavefront Propagation Across 40-Fly Swarm (R0={branching['r0']:.2f}, Wave Speed={v_wave:.2f} m/s)", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
        fig.colorbar(scatter, ax=ax1, label="Alert Time (ms)")

        # Panel 2: Spiking Cascade Raster for all 40 Flies
        ax2 = fig.add_subplot(gs[1, :])
        for i in range(num_agents):
            spk_t = time_series_ms[spike_history[:, i, idx_dnp01] > 0.5]
            ax2.scatter(spk_t, np.full_like(spk_t, i), s=10, color=plt.cm.viridis(i / num_agents), marker="|")
        ax2.set_xlabel("Time (ms)", fontsize=11)
        ax2.set_ylabel("Agent Index (0 to 39)", fontsize=11)
        ax2.set_title("Descending Alarm Neuron (DNp01) Spikes Across 40 Flies under Dryden Wind Turbulence", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Branching Distribution (Offspring histogram)
        ax3 = fig.add_subplot(gs[2, 0])
        counts = branching.get("branching_counts", [])
        if len(counts) > 0:
            bins = np.arange(0, max(counts) + 2) - 0.5
            ax3.hist(counts, bins=bins, color="teal", edgecolor="black", rwidth=0.7)
            ax3.axvline(branching["r0"], color="red", linestyle="--", lw=2, label=f"Mean R0 = {branching['r0']:.2f}")
        ax3.set_xlabel("Secondary Alerts Triggered per Agent (k)", fontsize=11)
        ax3.set_ylabel("Number of Agents", fontsize=11)
        ax3.set_title(f"Branching Distribution (Epidemic Ratio R0 = {branching['r0']:.2f})", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Spatial Percolation Span
        ax4 = fig.add_subplot(gs[2, 1])
        categories = ["Percolation Span Ratio", "Swarm Alert %"]
        values = [percolation["span_ratio"] * 100.0, (num_alerted / num_agents) * 100.0]
        bars = ax4.bar(categories, values, color=["#1f77b4", "#2ca02c"], edgecolor="black", width=0.45)
        ax4.set_ylim(0, 115)
        ax4.set_ylabel("Percentage (%)", fontsize=11)
        ax4.set_title("Spatial Percolation & Swarm Reach", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}%", ha='center', fontsize=10, fontweight="bold")

        plt.tight_layout()
        out_percolation = os.path.join(plots_dir, "milestone4_percolation_and_branching.png")
        plt.savefig(out_percolation, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved percolation & branching figure to: {out_percolation}")

        # Figure 2: 3D Flight Trajectories & Spatial Mesh Connectivity
        fig = plt.figure(figsize=(12, 10))
        ax3d = fig.add_subplot(111, projection='3d')

        for i in range(num_agents):
            traj = positions_history[:, i, :]
            color = plt.cm.viridis(i / num_agents)
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.5, alpha=0.8)
            ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=35, marker="o")
            ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=45, marker="^")

        # Draw peer-to-peer transmission mesh links (distance <= 0.16m between neighbor tiers)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                d = np.linalg.norm(initial_positions[i] - initial_positions[j])
                if d <= 0.16:
                    ax3d.plot(
                        [initial_positions[i, 0], initial_positions[j, 0]],
                        [initial_positions[i, 1], initial_positions[j, 1]],
                        [initial_positions[i, 2], initial_positions[j, 2]],
                        color="gray", linestyle=":", alpha=0.35, lw=0.9
                    )

        # Weather Front Sphere at origin
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        xs = weather_front.center[0] + weather_front.radius * np.outer(np.cos(u), np.sin(v))
        ys = weather_front.center[1] + weather_front.radius * np.outer(np.sin(u), np.sin(v))
        zs = weather_front.center[2] + weather_front.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax3d.plot_wireframe(xs, ys, zs, color="deepskyblue", alpha=0.3, label="Convective Shock Front")

        ax3d.set_xlabel("X (m)", fontsize=10)
        ax3d.set_ylabel("Y (m)", fontsize=10)
        ax3d.set_zlabel("Altitude Z (m)", fontsize=10)
        ax3d.set_title(f"40-Fly MuJoCo Mesh Network under Dryden Wind Turbulence\n(Live Station Wind={reading.wind_speed_ms:.1f} m/s, Heading={reading.wind_direction_deg:.0f}°)", fontsize=12, fontweight="bold")
        ax3d.legend(loc="upper right")
        plt.tight_layout()

        out_mesh = os.path.join(plots_dir, "milestone4_large_swarm_spatial_mesh.png")
        plt.savefig(out_mesh, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved 3D spatial mesh figure to: {out_mesh}")

    return {
        "metrics": metrics,
        "branching": branching,
        "percolation": percolation,
        "v_wave": v_wave,
        "num_alerted": num_alerted,
        "success": (percolation["percolation_achieved"] > 0.5)
    }

if __name__ == "__main__":
    res = run_large_scale_percolation_swarm()
    if res["success"]:
        print("\n>>> MILESTONE 4: LARGE-SCALE PERCOLATION & CRITICALITY EXPERIMENT SUCCESSFUL! <<<")
    else:
        print("\n>>> MILESTONE 4: INCOMPLETE PERCOLATION <<<")
        sys.exit(1)

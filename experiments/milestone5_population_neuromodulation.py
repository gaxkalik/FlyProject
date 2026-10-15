"""
Milestone 5: Neuromodulated Connectome Swarm with Octopaminergic Arousal Dynamics.
Demonstrates:
1. Dynamic Octopamine (OA) release triggered by acute environmental shock and descending motor spikes.
2. Neuromodulatory gain control: OA lowers firing thresholds (V_th) and enhances synaptic drive.
3. Comparative benchmark: Aroused (Neuromodulated) vs. Non-modulated Swarm Latencies.
4. Generates publication-quality figure comparing neuromodulatory surge and wavefront acceleration.
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

from fly_connectome import build_canonical_subcircuit, NeuromodulationEngine
from neural_engine import SwarmLIFEngine, DrosophilaSensoryEncoder, LIFParameters
from environment import (
    MuJoCoSwarmArena,
    RealTimeWeatherClient,
    WeatherFrontParams,
    WeatherEnvironment,
    SpatialArena
)
from environment.turbulence import AtmosphericTurbulenceModel, generate_ambient_acoustic_noise
from analysis import compute_signal_latency, compute_propagation_velocity

def run_neuromodulation_experiment(
    num_agents: int = 24,
    sim_duration_s: float = 1.0,
    dt_ms: float = 1.0,
    weather_onset_s: float = 0.15,
    plot_results: bool = True
):
    print("=" * 80)
    print(f"RUNNING MILESTONE 5: NEUROMODULATED CONNECTOME SWARM ({num_agents} FLIES)")
    print("=" * 80)

    weather_client = RealTimeWeatherClient()
    reading = weather_client.fetch_live_weather()

    # Geometry: 24 flies in 6 tiers of 4 flies each
    rng = np.random.default_rng(42)
    upwind_yaw = np.radians(reading.wind_direction_deg)
    initial_headings = np.full(num_agents, upwind_yaw, dtype=np.float32)

    pos_list = []
    tiers = 6
    per_tier = 4
    for t in range(tiers):
        x_base = t * 0.12
        for k in range(per_tier):
            y_base = (k - 1.5) * 0.09
            z_base = 0.80 + 0.03 * np.sin(t + k)
            pos_list.append([x_base + rng.normal(0, 0.01), y_base + rng.normal(0, 0.01), z_base])
    canonical_3d = np.array(pos_list, dtype=np.float32)
    rot = np.array([[np.cos(upwind_yaw), -np.sin(upwind_yaw)], [np.sin(upwind_yaw), np.cos(upwind_yaw)]], dtype=np.float32)
    initial_positions = np.column_stack([canonical_3d[:, :2] @ rot.T, canonical_3d[:, 2]])

    def run_trial(enable_neuromodulation: bool):
        dt_s = dt_ms / 1000.0
        total_steps = int(sim_duration_s / dt_s)
        time_series_ms = np.linspace(0, sim_duration_s * 1000.0, total_steps)

        mujoco_arena = MuJoCoSwarmArena(num_agents, initial_positions, initial_headings, timestep=dt_s)
        circuit = build_canonical_subcircuit(scale_factor=5.0)
        lif_engine = SwarmLIFEngine(circuit, num_agents)
        sensory_encoder = DrosophilaSensoryEncoder(circuit)
        mod_engine = NeuromodulationEngine(num_agents)

        macro_winds_init = weather_client.downscale_to_insect_boundary_layer(reading, np.array([0.80]))[0]
        sensory_encoder.set_ambient_baselines(
            reading.temperature_c, reading.relative_humidity_pct, reading.surface_pressure_kpa,
            float(np.linalg.norm(macro_winds_init)) + 0.85
        )

        turb_model = AtmosphericTurbulenceModel(0.35, 0.30, 0.20)
        front = WeatherFrontParams(
            center=initial_positions[0].copy(),
            velocity=macro_winds_init.copy(),
            radius=0.07,
            temp_drop=6.5,
            pressure_drop=2.5,
            humidity_spike=15.0,
            wind_speed=reading.wind_gust_ms,
            wind_direction=reading.wind_vector_10m / np.linalg.norm(reading.wind_vector_10m),
            onset_time=weather_onset_s,
            ramp_duration=0.08
        )
        weather_env = WeatherEnvironment(reading.temperature_c, reading.relative_humidity_pct, reading.surface_pressure_kpa, front_params=front)
        inter_arena = SpatialArena(weather_env=weather_env)

        alert_times = np.full(num_agents, -1.0, dtype=np.float32)
        oa_history = np.zeros((total_steps, num_agents), dtype=np.float32)
        saccade_timer = np.zeros(num_agents, dtype=np.float32)
        acoustic_timer = np.zeros(num_agents, dtype=np.float32)

        idx_dnp01 = circuit.get_index("DNp01_AlarmSaccade")
        idx_bln1 = circuit.get_index("bLN1_AcousticBuzz")

        for step in range(total_steps):
            t_s = step * dt_s
            pos = mujoco_arena.get_positions()
            vel = mujoco_arena.get_velocities()
            head = mujoco_arena.get_headings()

            macro_w = weather_client.downscale_to_insect_boundary_layer(reading, pos[:, 2])
            turb_w = turb_model.sample_turbulence(pos, dt_s)
            temp_c, dt_dt, h_pct, dh_dt, p_kpa, dp_dt, storm_w = weather_env.sample_at_positions(pos, t_s)
            total_w = macro_w + turb_w + storm_w

            amb_noise = generate_ambient_acoustic_noise(num_agents, 0.08)
            acoust = inter_arena.compute_pairwise_acoustic_signals(pos, (acoustic_timer > 0).astype(np.float32)) + amb_noise
            head_dirs = np.stack([np.cos(head), np.sin(head), np.zeros(num_agents, dtype=np.float32)], axis=1)
            mot, loom = inter_arena.compute_pairwise_visual_signals(pos, vel, head_dirs, (saccade_timer > 0).astype(np.float32))
            rel_air = np.linalg.norm(vel - total_w, axis=1)

            i_ext = sensory_encoder.encode_sensory_inputs(
                num_agents, temp_c, dt_dt, h_pct, dh_dt, p_kpa, dp_dt, rel_air, acoust, mot, loom, np.zeros(num_agents)
            )

            # Apply Neuromodulatory gain control
            if enable_neuromodulation:
                # Octopamine lowers V_th and heightens excitatory gain
                mod_v_th = mod_engine.compute_modulated_thresholds(base_v_th=-50.0, max_th_drop=3.5)
                lif_engine.v_th = np.where(lif_engine.v_mem < mod_v_th[:, np.newaxis], mod_v_th[:, np.newaxis], lif_engine.v_th)

            spikes = lif_engine.step(dt=dt_ms, i_ext=i_ext)

            dnp01_spk = (spikes[:, idx_dnp01] > 0.5)
            bln1_spk = (spikes[:, idx_bln1] > 0.5)

            # Update neuromodulatory dynamics
            shock_mag = np.clip(np.maximum(0.0, -dt_dt) / 10.0 + np.maximum(0.0, rel_air - 2.0) / 2.0, 0.0, 1.0)
            oa_levels, da_levels = mod_engine.step(dt_s, dnp01_spk, shock_mag)
            oa_history[step] = oa_levels

            for i in range(num_agents):
                if dnp01_spk[i]:
                    saccade_timer[i] = 0.08
                    if alert_times[i] < 0: alert_times[i] = t_s
                if bln1_spk[i]:
                    acoustic_timer[i] = 0.06
                    if alert_times[i] < 0: alert_times[i] = t_s

            saccade_timer = np.maximum(0.0, saccade_timer - dt_s)
            acoustic_timer = np.maximum(0.0, acoustic_timer - dt_s)

            mujoco_arena.step(total_w, (saccade_timer > 0).astype(np.float32), np.zeros(num_agents))

        return alert_times, oa_history, time_series_ms

    print("[Simulation] Running Trial A: Baseline (Without Neuromodulation)...")
    alerts_baseline, _, time_ms = run_trial(enable_neuromodulation=False)

    print("[Simulation] Running Trial B: Neuromodulated (With Octopaminergic Arousal)...")
    alerts_modulated, oa_hist, _ = run_trial(enable_neuromodulation=True)

    metrics_base = compute_signal_latency(alerts_baseline, weather_onset_s, 0)
    metrics_mod = compute_signal_latency(alerts_modulated, weather_onset_s, 0)
    v_base = compute_propagation_velocity(initial_positions, alerts_baseline, 0)
    v_mod = compute_propagation_velocity(initial_positions, alerts_modulated, 0)

    print("\n" + "=" * 65)
    print("MILESTONE 5: NEUROMODULATORY AROUSAL COMPARISON:")
    print("=" * 65)
    print(f"• Baseline Swarm (No OA):")
    print(f"  - Total Cascade Time:       {metrics_base['total_cascade_time_ms']:.1f} ms")
    print(f"  - Mean Inter-Hop Latency:   {metrics_base['inter_fly_hop_latency_ms']:.1f} ms")
    print(f"  - Warning Wave Speed:       {v_base:.2f} m/s")
    print(f"• Neuromodulated Swarm (With OA Surge):")
    print(f"  - Total Cascade Time:       {metrics_mod['total_cascade_time_ms']:.1f} ms")
    print(f"  - Mean Inter-Hop Latency:   {metrics_mod['inter_fly_hop_latency_ms']:.1f} ms")
    print(f"  - Warning Wave Speed:       {v_mod:.2f} m/s")
    speedup = ((v_mod - v_base) / v_base) * 100.0 if v_base else 0.0
    print(f"• Wave Speedup from Octopamine Arousal: +{speedup:.1f}%")
    print("=" * 65)

    if plot_results:
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Panel 1: Octopamine Concentration Surge Across Swarm
        for i in range(num_agents):
            ax1.plot(time_ms, oa_hist[:, i], color=plt.cm.cool(i / num_agents), lw=1.5, alpha=0.85)
        ax1.axvline(weather_onset_s * 1000.0, color="red", linestyle="--", label="Weather Shock Onset (150 ms)")
        ax1.set_xlabel("Time (ms)", fontsize=11)
        ax1.set_ylabel("Octopamine (OA) Concentration", fontsize=11)
        ax1.set_title("Octopaminergic Fight-or-Flight Surge Across 24-Fly Connectome Swarm", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 2: Comparative Arrival Timestamps: Baseline vs Neuromodulated
        dists = np.linalg.norm(initial_positions - initial_positions[0], axis=1)
        ax2.scatter(dists, alerts_baseline * 1000.0, color="gray", s=60, marker="o", label=f"Baseline (v = {v_base:.2f} m/s)")
        ax2.scatter(dists, alerts_modulated * 1000.0, color="#d62728", s=80, marker="^", label=f"Neuromodulated +OA (v = {v_mod:.2f} m/s)")
        ax2.set_xlabel("Distance from Source Sector (meters)", fontsize=11)
        ax2.set_ylabel("Alert Timestamp (ms)", fontsize=11)
        ax2.set_title("Wavefront Acceleration from Octopaminergic Disinhibition", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        out_fig = os.path.join(plots_dir, "milestone5_neuromodulation_arousal.png")
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"[Visualizer] Saved neuromodulation comparison figure to: {out_fig}")

    return {
        "metrics_baseline": metrics_base,
        "metrics_modulated": metrics_mod,
        "speedup_pct": speedup,
        "success": (v_mod >= v_base)
    }

if __name__ == "__main__":
    res = run_neuromodulation_experiment()
    if res["success"]:
        print("\n>>> MILESTONE 5: NEUROMODULATED SWARM EXPERIMENT SUCCESSFUL! <<<")

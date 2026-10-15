#!/usr/bin/env python3
"""
FlyProject: Automated Test Suite & Milestone Verification Runner
================================================================
Comprehensive test runner for the Drosophila Connectome Swarm Mesh Network.

Usage:
  python3 run_tests.py                   # Run fast unit test suite (< 3s)
  python3 run_tests.py --quick           # Run fast unit test suite
  python3 run_tests.py --milestone <N>   # Run specific milestone (1 to 6)
  python3 run_tests.py --all             # Run unit tests and all 6 milestones

Author: Principal Computational Neuroscientist & Swarm Intelligence Architect
"""

import os
import sys
import time
import argparse
import unittest
import numpy as np

# Ensure MPLCONFIGDIR is set to avoid cache warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_fly")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# =====================================================================
# UNIT TESTS
# =====================================================================

class TestDrosophilaConnectome(unittest.TestCase):
    """Verifies FlyWire connectome circuit loading and synapse matrix structure."""

    def test_canonical_circuit_topology(self):
        from fly_connectome.circuit_builder import build_canonical_subcircuit
        circuit = build_canonical_subcircuit(scale_factor=5.0)

        # 23 distinct functional neuron types
        self.assertEqual(circuit.num_neurons, 23)
        
        # Verify sensory neurons exist
        expected_sensory = ["VP1l_Cold", "VP1m_Warm", "VP2_Dry", "VP3_Moist", 
                            "JON_C", "JON_E", "JON_A", "JON_B", "LC11", "LC4", "Glomerulus_V_CO2"]
        for name in expected_sensory:
            self.assertIn(name, circuit.neurons)
            idx = circuit.get_index(name)
            self.assertGreaterEqual(idx, 0)

        # Verify motor outputs exist
        self.assertIn("DNp01_AlarmSaccade", circuit.neurons)
        self.assertIn("DNa02_Steering", circuit.neurons)
        self.assertIn("bLN1_AcousticBuzz", circuit.neurons)

        # Verify synaptic weight matrix contains signed weights (both excitatory and inhibitory)
        self.assertEqual(circuit.adj_matrix.shape, (23, 23))
        self.assertTrue(np.any(circuit.adj_matrix > 0.0), "No excitatory synapses found")
        self.assertTrue(np.any(circuit.adj_matrix < 0.0), "No inhibitory synapses found")
        self.assertGreater(np.sum(np.abs(circuit.adj_matrix)), 10.0)

    def test_lif_dynamics(self):
        from fly_connectome.circuit_builder import build_canonical_subcircuit
        from neural_engine.lif_network import SwarmLIFEngine
        circuit = build_canonical_subcircuit(scale_factor=5.0)
        engine = SwarmLIFEngine(circuit, num_agents=2)

        # At rest with zero input, membrane potential should hover near V_rest (-70 mV +/- 3 mV noise)
        v_rest = engine.params.v_rest
        spikes = engine.step(dt=1.0, i_ext=np.zeros((2, circuit.num_neurons)))
        self.assertEqual(np.sum(spikes), 0.0)
        np.testing.assert_allclose(engine.v_mem, v_rest, atol=3.5)

        # Strong external injection into Cold sensor should trigger spikes downstream
        idx_cold = circuit.get_index("VP1l_Cold")
        i_ext = np.zeros((2, circuit.num_neurons), dtype=np.float32)
        i_ext[0, idx_cold] = 80.0  # 80 pA input to Fly 0 only

        fired = False
        for _ in range(50):
            spk = engine.step(dt=1.0, i_ext=i_ext)
            if spk[0, idx_cold] > 0.5:
                fired = True
                break
        self.assertTrue(fired, "VP1l_Cold neuron failed to spike under strong current injection")


class TestSensoryEncoders(unittest.TestCase):
    """Verifies sensory transduction: thresholding, noise filtering, and dynamic derivatives."""

    def setUp(self):
        from fly_connectome.circuit_builder import build_canonical_subcircuit
        from neural_engine.sensory_encoders import DrosophilaSensoryEncoder
        self.circuit = build_canonical_subcircuit()
        self.encoder = DrosophilaSensoryEncoder(self.circuit)
        self.encoder.set_ambient_baselines(temp_c=15.0, humidity_pct=80.0, pressure_kpa=90.0, wind_speed_ms=1.5)

    def test_turbulence_noise_rejection(self):
        # Normal turbulent buffeting (< 7.5 rad/s) must NOT activate LC11 visual motion detector
        num_agents = 4
        i_ext = self.encoder.encode_sensory_inputs(
            num_agents=num_agents,
            temp_c=np.full(num_agents, 15.0),
            d_temp_dt=np.zeros(num_agents),
            humidity_pct=np.full(num_agents, 80.0),
            d_humidity_dt=np.zeros(num_agents),
            pressure_kpa=np.full(num_agents, 90.0),
            d_pressure_dt=np.zeros(num_agents),
            rel_airspeed_ms=np.full(num_agents, 1.5),
            acoustic_velocity=np.zeros(num_agents),
            visual_conspecific_motion=np.full(num_agents, 4.0),  # 4 rad/s turbulence buffeting
            visual_looming_rate=np.zeros(num_agents),
            alarm_odor_conc=np.zeros(num_agents)
        )
        idx_lc11 = self.circuit.get_index("LC11")
        # Subtract background noise (< 0.5 pA)
        self.assertLess(np.max(i_ext[:, idx_lc11]), 1.0, "LC11 failed to filter out turbulent motion")

    def test_saccade_activation(self):
        # Emergency saccade (> 20 rad/s) must strongly activate LC11
        num_agents = 2
        i_ext = self.encoder.encode_sensory_inputs(
            num_agents=num_agents,
            temp_c=np.full(num_agents, 15.0),
            d_temp_dt=np.zeros(num_agents),
            humidity_pct=np.full(num_agents, 80.0),
            d_humidity_dt=np.zeros(num_agents),
            pressure_kpa=np.full(num_agents, 90.0),
            d_pressure_dt=np.zeros(num_agents),
            rel_airspeed_ms=np.full(num_agents, 1.5),
            acoustic_velocity=np.zeros(num_agents),
            visual_conspecific_motion=np.array([25.0, 0.0]),  # Fly 0 sees saccade
            visual_looming_rate=np.zeros(num_agents),
            alarm_odor_conc=np.zeros(num_agents)
        )
        idx_lc11 = self.circuit.get_index("LC11")
        self.assertGreater(i_ext[0, idx_lc11], 20.0, "LC11 failed to activate under visual saccade")
        self.assertLess(i_ext[1, idx_lc11], 1.0)


class TestMuJoCoBiophysics(unittest.TestCase):
    """Verifies continuous 6-DOF rigid body physics in MuJoCo 3.16."""

    def test_mujoco_stepping_and_heading(self):
        from environment.mujoco_arena import MuJoCoSwarmArena
        num_agents = 2
        init_pos = np.array([[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]], dtype=np.float32)
        init_head = np.array([0.0, 0.5], dtype=np.float32)

        arena = MuJoCoSwarmArena(num_agents, init_pos, init_head, timestep=0.001)
        self.assertEqual(arena.num_agents, 2)

        # Step 20 ms
        w = np.zeros((num_agents, 3), dtype=np.float32)
        saccade = np.zeros(num_agents, dtype=np.float32)
        steer = np.zeros(num_agents, dtype=np.float32)
        for _ in range(20):
            arena.step(w, saccade, steer)

        pos = arena.get_positions()
        vel = arena.get_velocities()
        head = arena.get_headings()

        self.assertEqual(pos.shape, (2, 3))
        self.assertEqual(vel.shape, (2, 3))
        self.assertEqual(head.shape, (2,))
        # Z altitude should remain stable (~0.8m)
        self.assertTrue(np.all(pos[:, 2] > 0.5) and np.all(pos[:, 2] < 1.2))


class TestRealTimeMeteorology(unittest.TestCase):
    """Verifies Open-Meteo telemetry client, caching, and boundary-layer downscaling."""

    def test_live_weather_parser_and_boundary_layer(self):
        from environment.realtime_weather import RealTimeWeatherClient
        client = RealTimeWeatherClient(cache_file="data/live_weather_cache.json")
        reading = client.fetch_live_weather(force_live=False)

        # Plausible meteorological ranges
        self.assertGreater(reading.temperature_c, -30.0)
        self.assertLess(reading.temperature_c, 50.0)
        self.assertGreater(reading.relative_humidity_pct, 0.0)
        self.assertLessEqual(reading.relative_humidity_pct, 100.0)
        self.assertGreater(reading.surface_pressure_kpa, 70.0)
        self.assertLess(reading.surface_pressure_kpa, 110.0)

        # Boundary layer wind downscaling: wind at z=0.8m must be less than 10m wind
        z_fly = np.array([0.80], dtype=np.float32)
        wind_fly = client.downscale_to_insect_boundary_layer(reading, z_fly)
        speed_fly = np.linalg.norm(wind_fly[0])
        speed_10m = np.linalg.norm(reading.wind_vector_10m)

        self.assertLess(speed_fly, speed_10m + 1e-3, "Insect boundary layer wind speed exceeded 10m reference wind")
        self.assertGreater(speed_fly, 0.0)


class TestMeshCommunicationChannels(unittest.TestCase):
    """Verifies physical acoustic, visual, and chemical signal transmission."""

    def test_acoustic_inverse_square(self):
        from environment.spatial_arena import SpatialArena
        arena = SpatialArena()
        # Fly 0 at origin, Fly 1 at 5cm, Fly 2 at 20cm
        pos = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32)
        # Fly 0 buzzing
        buzz = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        acoust = arena.compute_pairwise_acoustic_signals(pos, buzz)

        # Fly 1 should hear much louder signal than Fly 2
        self.assertGreater(acoust[1], acoust[2] * 2.0)

    def test_pheromone_diffusion(self):
        from environment.spatial_arena import SpatialArena
        from environment.weather_front import WeatherEnvironment
        env = WeatherEnvironment()
        arena = SpatialArena(weather_env=env)

        arena.add_pheromone_puff(position=np.array([0.0, 0.0, 0.8], dtype=np.float32), release_time=0.0, mass=100.0)
        pos = np.array([[0.02, 0.0, 0.8], [0.50, 0.0, 0.8]], dtype=np.float32)
        c = arena.sample_pheromone_concentration(pos, current_time=0.1)

        self.assertGreater(c[0], c[1], "Pheromone concentration was not higher near emission source")


# =====================================================================
# MILESTONE EXECUTORS
# =====================================================================

def run_milestone(num: int) -> bool:
    """Executes a specific milestone script and reports status."""
    print(f"\n{Color.HEADER}{'=' * 75}{Color.ENDC}")
    print(f"{Color.BOLD}EXECUTING MILESTONE {num}{Color.ENDC}")
    print(f"{Color.HEADER}{'=' * 75}{Color.ENDC}\n")

    t0 = time.time()
    try:
        if num == 1:
            from experiments.milestone1_two_fly_poc import run_milestone1_simulation
            run_milestone1_simulation()
        elif num == 2:
            from experiments.milestone2_cluster_mesh import run_milestone2_cluster_simulation
            run_milestone2_cluster_simulation()
        elif num == 3:
            from experiments.milestone3_realtime_weather_mujoco import run_realtime_weather_mujoco_swarm
            run_realtime_weather_mujoco_swarm()
        elif num == 4:
            from experiments.milestone4_large_scale_percolation import run_large_scale_percolation_swarm
            run_large_scale_percolation_swarm()
        elif num == 5:
            from experiments.milestone5_population_neuromodulation import run_neuromodulation_experiment
            run_neuromodulation_experiment()
        elif num == 6:
            from experiments.milestone6_fault_tolerance_and_jamming import run_fault_tolerance_experiment
            run_fault_tolerance_experiment()
        else:
            print(f"{Color.FAIL}Error: Invalid milestone number {num}. Valid options: 1 to 6.{Color.ENDC}")
            return False

        elapsed = time.time() - t0
        print(f"\n{Color.OKGREEN}>>> MILESTONE {num} PASSED SUCCESSFULLY ({elapsed:.2f}s) <<<{Color.ENDC}\n")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n{Color.FAIL}>>> MILESTONE {num} FAILED ({elapsed:.2f}s): {e} <<<{Color.ENDC}\n")
        import traceback
        traceback.print_exc()
        return False


def run_unit_tests() -> bool:
    """Executes all unit tests."""
    print(f"\n{Color.HEADER}{'=' * 75}{Color.ENDC}")
    print(f"{Color.BOLD}RUNNING UNIT TEST SUITE (Fast Physics, Connectome & Sensory Checks){Color.ENDC}")
    print(f"{Color.HEADER}{'=' * 75}{Color.ENDC}\n")

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestDrosophilaConnectome))
    suite.addTest(loader.loadTestsFromTestCase(TestSensoryEncoders))
    suite.addTest(loader.loadTestsFromTestCase(TestMuJoCoBiophysics))
    suite.addTest(loader.loadTestsFromTestCase(TestRealTimeMeteorology))
    suite.addTest(loader.loadTestsFromTestCase(TestMeshCommunicationChannels))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FlyProject: Drosophila Connectome Swarm Mesh Network Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_tests.py                 # Run fast unit tests (< 2s)
  python3 run_tests.py --milestone 1   # Run Milestone 1: 2-Fly Sensorimotor POC
  python3 run_tests.py --milestone 3   # Run Milestone 3: MuJoCo 6-DOF + Live Weather
  python3 run_tests.py --milestone 6   # Run Milestone 6: Fault Tolerance & Fog Blackout
  python3 run_tests.py --all           # Run full suite (Unit tests + Milestones 1 to 6)
        """
    )
    parser.add_argument("--quick", action="store_true", help="Run fast unit tests only")
    parser.add_argument("--milestone", type=int, choices=range(1, 7), help="Run a specific milestone (1-6)")
    parser.add_argument("--all", action="store_true", help="Run unit tests and all 6 milestones sequentially")

    args = parser.parse_args()

    # Default to unit tests if no arguments provided
    if len(sys.argv) == 1 or args.quick:
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    if args.milestone:
        success = run_milestone(args.milestone)
        sys.exit(0 if success else 1)

    if args.all:
        print(f"\n{Color.BOLD}{Color.OKBLUE}STARTING COMPREHENSIVE BENCHMARK: UNIT TESTS + ALL 6 MILESTONES{Color.ENDC}\n")
        all_passed = True
        unit_ok = run_unit_tests()
        all_passed = all_passed and unit_ok

        milestone_results = {}
        for m in range(1, 7):
            ok = run_milestone(m)
            milestone_results[m] = ok
            all_passed = all_passed and ok

        print(f"\n{Color.HEADER}{'=' * 75}{Color.ENDC}")
        print(f"{Color.BOLD}FINAL SUMMARY REPORT{Color.ENDC}")
        print(f"{Color.HEADER}{'=' * 75}{Color.ENDC}")
        print(f"  • Unit Tests:    {'[ PASS ]' if unit_ok else '[ FAIL ]'}")
        for m in range(1, 7):
            status = '[ PASS ]' if milestone_results[m] else '[ FAIL ]'
            print(f"  • Milestone {m}:  {status}")
        print(f"{Color.HEADER}{'=' * 75}{Color.ENDC}")

        if all_passed:
            print(f"{Color.OKGREEN}{Color.BOLD}ALL TESTS & MILESTONES COMPLETED SUCCESSFULLY!{Color.ENDC}\n")
            sys.exit(0)
        else:
            print(f"{Color.FAIL}{Color.BOLD}SOME TESTS OR MILESTONES FAILED.{Color.ENDC}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()

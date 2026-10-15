"""
Sensory Transduction and Neural Encoding Models for Drosophila Connectome.
Translates physical ambient weather variables and inter-fly signaling vectors
into injected depolarizing currents (pA) into primary sensory neurons.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from fly_connectome.circuit_builder import ConnectomeSubCircuit

class DrosophilaSensoryEncoder:
    """
    Translates biophysical sensory stimuli into synaptic injection currents (I_ext).
    """

    def __init__(self, circuit: ConnectomeSubCircuit):
        self.circuit = circuit
        self.idx_jon_c = circuit.get_index("JON_C")
        self.idx_jon_e = circuit.get_index("JON_E")
        self.idx_jon_a = circuit.get_index("JON_A")
        self.idx_jon_b = circuit.get_index("JON_B")
        self.idx_vp1l = circuit.get_index("VP1l_Cold")
        self.idx_vp1m = circuit.get_index("VP1m_Warm")
        self.idx_vp2 = circuit.get_index("VP2_Dry")
        self.idx_vp3 = circuit.get_index("VP3_Moist")
        self.idx_lc11 = circuit.get_index("LC11")
        self.idx_lc4 = circuit.get_index("LC4")
        self.idx_co2 = circuit.get_index("Glomerulus_V_CO2")

        # Baseline physiological setpoints
        self.T_prefer = 24.0      # Preferred temperature (Celsius)
        self.H_prefer = 65.0      # Preferred relative humidity (%)
        self.P_baseline = 101.325 # Standard barometric pressure (kPa)

    def encode_sensory_inputs(
        self,
        num_agents: int,
        temp_c: np.ndarray,              # [num_agents] (Celsius)
        d_temp_dt: np.ndarray,           # [num_agents] (deg/s)
        humidity_pct: np.ndarray,        # [num_agents] (%)
        d_humidity_dt: np.ndarray,       # [num_agents] (%/s)
        pressure_kpa: np.ndarray,        # [num_agents] (kPa)
        d_pressure_dt: np.ndarray,       # [num_agents] (kPa/s)
        rel_airspeed_ms: np.ndarray,     # [num_agents] (m/s)
        acoustic_velocity: np.ndarray,   # [num_agents] (mm/s near-field flutter)
        visual_conspecific_motion: np.ndarray, # [num_agents] (rad/s neighbor angular speed)
        visual_looming_rate: np.ndarray, # [num_agents] (rad/s expansion rate)
        alarm_odor_conc: np.ndarray,     # [num_agents] (ppm dSO/CO2)
    ) -> np.ndarray:
        """
        Computes the I_ext matrix of shape [num_agents, num_neurons] in pA.
        """
        i_ext = np.zeros((num_agents, self.circuit.num_neurons), dtype=np.float32)

        # 1. Mechanosensation: Johnston's Organ C & E (Wind & Pressure drop)
        # Steady-state drag deflection + dynamic pressure drop rate
        wind_drive = 2.5 * np.maximum(0.0, rel_airspeed_ms)
        press_drop_drive = 4.0 * np.maximum(0.0, -d_pressure_dt)  # Positive for falling barometric pressure
        i_ext[:, self.idx_jon_c] = wind_drive + 0.6 * press_drop_drive
        i_ext[:, self.idx_jon_e] = 0.5 * wind_drive + press_drop_drive

        # 2. Thermosensation: Arista & Sacculus Cold Cells (VP1l) & Warm Cells (VP1m)
        # Cold receptor (IR21a/25a) depolarizes during cooling (negative d_temp) or low temperature
        cold_stat = np.maximum(0.0, self.T_prefer - temp_c)
        cold_dyn = np.maximum(0.0, -d_temp_dt)
        i_ext[:, self.idx_vp1l] = 0.8 * cold_stat + 3.0 * cold_dyn

        # Warm receptor (TrpA1) depolarizes during sudden warming
        warm_stat = np.maximum(0.0, temp_c - 28.0)
        warm_dyn = np.maximum(0.0, d_temp_dt)
        i_ext[:, self.idx_vp1m] = 0.8 * warm_stat + 2.5 * warm_dyn

        # 3. Hygrosensation: Dry Cells (VP2) & Moist Cells (VP3)
        dry_stat = np.maximum(0.0, self.H_prefer - humidity_pct)
        dry_dyn = np.maximum(0.0, -d_humidity_dt)
        i_ext[:, self.idx_vp2] = 0.1 * dry_stat + 2.0 * dry_dyn

        moist_stat = np.maximum(0.0, humidity_pct - 85.0)
        moist_dyn = np.maximum(0.0, d_humidity_dt)
        i_ext[:, self.idx_vp3] = 0.1 * moist_stat + 2.0 * moist_dyn

        # 4. Acoustic Near-Field Particle Velocity (JON-A / JON-B)
        # Threshold: normal cruising wingbeat (<0.25 mm/s) is filtered; alarm flutter (>0.5 mm/s) activates
        excess_acoustic = np.maximum(0.0, acoustic_velocity - 0.25)
        acoustic_drive = 6.0 * excess_acoustic
        i_ext[:, self.idx_jon_a] = acoustic_drive
        i_ext[:, self.idx_jon_b] = 1.2 * acoustic_drive

        # 5. Visual Conspecific Motion & Looming
        # Threshold: normal cruising optic flow (<1.5 rad/s) is filtered out;
        # High-g evasive saccades (>4 rad/s) activate LC11
        excess_motion = np.maximum(0.0, visual_conspecific_motion - 1.5)
        i_ext[:, self.idx_lc11] = 4.5 * excess_motion

        # LC4 responds to looming expansion (imminent collision / fast approach > 0.01 rad/s)
        excess_loom = np.maximum(0.0, visual_looming_rate - 0.008)
        i_ext[:, self.idx_lc4] = 8.0 * excess_loom

        # 6. Olfactory Alarm Pheromone (CO2 / dSO)
        # Hill activation curve: I = I_max * C^n / (K_d^n + C^n)
        kd = 5.0  # ppm
        c_pow = np.power(np.maximum(0.0, alarm_odor_conc), 2.0)
        i_ext[:, self.idx_co2] = 5.0 * (c_pow / (np.power(kd, 2.0) + c_pow + 1e-6))

        # Add modest biological background synaptic noise (0.1 pA standard deviation)
        rng = np.random.default_rng()
        i_ext += rng.normal(0.0, 0.1, size=i_ext.shape).astype(np.float32)

        return i_ext

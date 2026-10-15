"""
Vectorized Spiking Leaky Integrate-and-Fire (LIF) Neural Engine with Adaptive Thresholding.
Supports single-fly and batched multi-agent simulation using vectorized NumPy operations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
from fly_connectome.circuit_builder import ConnectomeSubCircuit

@dataclass
class LIFParameters:
    v_rest: float = -70.0        # Resting membrane potential (mV)
    v_reset: float = -75.0       # Reset potential post-spike (mV)
    v_th_0: float = -50.0        # Baseline firing threshold (mV)
    delta_th: float = 15.0       # Threshold jump on spike (spike-frequency adaptation) (mV)
    tau_m: float = 20.0          # Membrane time constant (ms)
    tau_syn: float = 5.0         # Synaptic current decay time constant (ms)
    tau_th: float = 80.0         # Threshold adaptation relaxation time constant (ms)
    r_m: float = 10.0            # Membrane resistance (MOhm)
    refractory_period: float = 2.0 # Absolute refractory period (ms)

class SwarmLIFEngine:
    """
    Simulates N fly connectomes in parallel.
    State shapes:
        v_mem: [num_agents, num_neurons] (mV)
        v_th:  [num_agents, num_neurons] (mV)
        i_syn: [num_agents, num_neurons] (pA / arbitrary synaptic units)
        spikes: [num_agents, num_neurons] (bool/float)
        refractory_timer: [num_agents, num_neurons] (ms remaining)
    """

    def __init__(self, circuit: ConnectomeSubCircuit, num_agents: int = 1, params: Optional[LIFParameters] = None):
        self.circuit = circuit
        self.num_agents = num_agents
        self.num_neurons = circuit.num_neurons
        self.params = params or LIFParameters()

        # Connectome synaptic weight matrix: W[pre, post]
        self.W = circuit.adj_matrix.astype(np.float32)  # [num_neurons, num_neurons]

        # Precompute decay factors for Euler integration
        self.reset_states()

    def reset_states(self):
        """Initializes all agent neural states to resting equilibrium with small biological noise."""
        rng = np.random.default_rng()
        self.v_mem = np.full((self.num_agents, self.num_neurons), self.params.v_rest, dtype=np.float32)
        # Add 1.0 mV random noise to initial membrane potentials to prevent artificial synchronous locking
        self.v_mem += rng.normal(0.0, 1.0, size=(self.num_agents, self.num_neurons)).astype(np.float32)

        self.v_th = np.full((self.num_agents, self.num_neurons), self.params.v_th_0, dtype=np.float32)
        self.i_syn = np.zeros((self.num_agents, self.num_neurons), dtype=np.float32)
        self.spikes = np.zeros((self.num_agents, self.num_neurons), dtype=np.float32)
        self.refractory_timer = np.zeros((self.num_agents, self.num_neurons), dtype=np.float32)
        self.time_ms = 0.0

    def step(self, dt: float, i_ext: np.ndarray) -> np.ndarray:
        """
        Advances the network state by dt milliseconds.
        i_ext: [num_agents, num_neurons] external injected currents from sensory encoders.
        Returns: spikes [num_agents, num_neurons]
        """
        dt = float(dt)
        self.time_ms += dt

        # 1. Propagate last spikes across the connectome:
        # synaptic_input[agent, post] = sum_pre (spikes[agent, pre] * W[pre, post])
        # In matrix multiplication: [num_agents, num_neurons] @ [num_neurons, num_neurons]
        synaptic_input = np.dot(self.spikes, self.W)

        # 2. Update Synaptic Currents (Exponential decay + incoming spikes)
        decay_syn = np.exp(-dt / self.params.tau_syn)
        self.i_syn = self.i_syn * decay_syn + synaptic_input

        # 3. Decrement Refractory Timers
        self.refractory_timer = np.maximum(0.0, self.refractory_timer - dt)
        non_refractory = (self.refractory_timer <= 0.0)

        # 4. Membrane Potential Update (LIF equation)
        # dv/dt = (-(v - v_rest) + R_m * (i_syn + i_ext)) / tau_m
        total_current = self.i_syn + i_ext
        dv = (-(self.v_mem - self.params.v_rest) + self.params.r_m * total_current) * (dt / self.params.tau_m)

        # Apply update only to non-refractory neurons
        self.v_mem = np.where(non_refractory, self.v_mem + dv, self.params.v_reset)

        # 5. Threshold Adaptation Relaxation
        # dv_th/dt = -(v_th - v_th_0) / tau_th
        dv_th = (-(self.v_th - self.params.v_th_0)) * (dt / self.params.tau_th)
        self.v_th = self.v_th + dv_th

        # 6. Spike Generation
        spike_mask = (self.v_mem >= self.v_th) & non_refractory
        self.spikes = spike_mask.astype(np.float32)

        # 7. Post-Spike Reset & Adaptation Jump
        self.v_mem = np.where(spike_mask, self.params.v_reset, self.v_mem)
        self.v_th = np.where(spike_mask, self.v_th + self.params.delta_th, self.v_th)
        self.refractory_timer = np.where(spike_mask, self.params.refractory_period, self.refractory_timer)

        return self.spikes

    def get_firing_rates(self, spike_history: np.ndarray, window_steps: int) -> np.ndarray:
        """
        Computes instantaneous firing rate (Hz) over sliding window.
        spike_history: [time_steps, num_agents, num_neurons]
        """
        if len(spike_history) == 0:
            return np.zeros((self.num_agents, self.num_neurons), dtype=np.float32)
        window = spike_history[-window_steps:]
        mean_spikes_per_step = np.mean(window, axis=0)
        dt_s = (window_steps * 0.5) / 1000.0  # Assumes dt=0.5 ms
        return mean_spikes_per_step / (0.5 / 1000.0)

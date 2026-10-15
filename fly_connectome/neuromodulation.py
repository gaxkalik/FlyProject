"""
Neuromodulation Engine for Drosophila Connectome: Octopaminergic & Dopaminergic Dynamics.
Models fight-or-flight behavioral state switching, sensory gain control, and arousal.
References:
- Octopamine in Drosophila flight and arousal (Suver et al., 2012; Strother et al., 2018)
- Dopaminergic salience and valence gating in Mushroom Body / Lateral Horn (Aso et al., 2014)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class NeuromodulatoryState:
    octopamine: np.ndarray    # [N] OA concentration [0.0 to 1.0] (Arousal / gain boost)
    dopamine: np.ndarray      # [N] DA concentration [0.0 to 1.0] (Aversive valence memory)

class NeuromodulationEngine:
    """
    Simulates dynamic release and clearance of Octopamine (OA) and Dopamine (DA).
    OA release is triggered by mechanosensory wind shear, descending motor alerts (DNp01),
    and sudden temperature plunges, boosting synaptic sensitivity across the connectome.
    """

    def __init__(
        self,
        num_agents: int,
        tau_oa_decay: float = 1.2,    # Octopamine clearance time constant (seconds)
        tau_da_decay: float = 3.5,    # Dopamine clearance time constant (seconds)
        baseline_oa: float = 0.05,
        baseline_da: float = 0.02
    ):
        self.num_agents = num_agents
        self.tau_oa = tau_oa_decay
        self.tau_da = tau_da_decay
        self.base_oa = baseline_oa
        self.base_da = baseline_da

        self.oa = np.full(num_agents, baseline_oa, dtype=np.float32)
        self.da = np.full(num_agents, baseline_da, dtype=np.float32)

    def step(
        self,
        dt_s: float,
        dnp01_spikes: np.ndarray,       # [N] binary flags
        sensory_shock_intensity: np.ndarray # [N] continuous shock magnitude [0.0 to 1.0]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advances neuromodulatory state by dt_s seconds.
        Returns: (octopamine [N], dopamine [N])
        """
        # OA Innovation: burst release upon acute descending motor activation and sensory shock
        oa_release = 0.45 * dnp01_spikes.astype(np.float32) + 0.35 * sensory_shock_intensity
        # Exponential decay towards baseline
        decay_oa = np.exp(-dt_s / self.tau_oa)
        self.oa = self.base_oa + (self.oa - self.base_oa) * decay_oa + oa_release
        self.oa = np.clip(self.oa, 0.0, 1.0)

        # DA Innovation: slower accumulation tracking persistent threat valence
        da_release = 0.15 * dnp01_spikes.astype(np.float32) + 0.20 * sensory_shock_intensity
        decay_da = np.exp(-dt_s / self.tau_da)
        self.da = self.base_da + (self.da - self.base_da) * decay_da + da_release
        self.da = np.clip(self.da, 0.0, 1.0)

        return self.oa, self.da

    def compute_modulated_thresholds(
        self,
        base_v_th: float = -50.0,
        max_th_drop: float = 4.0
    ) -> np.ndarray:
        """
        Octopamine lowers membrane firing thresholds (heightens global neuronal excitability).
        V_th(OA) = V_th0 - max_drop * OA
        """
        return base_v_th - max_th_drop * self.oa

    def compute_modulated_synaptic_gain(
        self,
        max_gain_boost: float = 0.35
    ) -> np.ndarray:
        """
        Octopamine enhances excitatory synaptic transmission gain: W_eff = W * (1 + boost * OA).
        """
        return 1.0 + max_gain_boost * self.oa

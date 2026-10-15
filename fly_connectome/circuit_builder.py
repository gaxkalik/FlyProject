"""
Connectome Sub-Circuit Graph Builder for Drosophila Swarm Mesh.
Constructs synaptic adjacency matrices with physiological neurotransmitter polarities,
synaptic weights, and topological routing from sensory inputs to descending outputs.
"""

from typing import Dict, List, Tuple
import numpy as np
from .cell_types import CORE_CELL_CATALOG, CellTypeSpec, Neurotransmitter

class ConnectomeSubCircuit:
    def __init__(self, neurons: List[str], adj_matrix: np.ndarray, cell_specs: Dict[str, CellTypeSpec]):
        self.neurons = neurons
        self.adj_matrix = adj_matrix  # Shape: [N_neurons, N_neurons], where adj[i, j] is weight from i to j
        self.cell_specs = cell_specs
        self.neuron_to_idx = {name: idx for idx, name in enumerate(neurons)}
        self.num_neurons = len(neurons)

    def get_index(self, name: str) -> int:
        return self.neuron_to_idx[name]

    def get_synaptic_weights(self) -> np.ndarray:
        """Returns the signed synaptic weight matrix W where W[i, j] is pre-synaptic i to post-synaptic j."""
        return self.adj_matrix

def build_canonical_subcircuit(scale_factor: float = 5.0) -> ConnectomeSubCircuit:
    """
    Constructs the canonical core sensory-motor integration sub-circuit.
    Scale factor 5.0 calibrates single canonical nodes to represent the collective
    synaptic drive of whole FlyWire projection neuron populations (~20-50 PNs per glomerulus).
    """
    neuron_keys = list(CORE_CELL_CATALOG.keys())
    n = len(neuron_keys)
    adj = np.zeros((n, n), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(neuron_keys)}

    def add_synapse(pre: str, post: str, weight: float, transmitter: Neurotransmitter):
        if pre in name_to_idx and post in name_to_idx:
            u = name_to_idx[pre]
            v = name_to_idx[post]
            sign = -1.0 if transmitter == Neurotransmitter.GABA else 1.0
            adj[u, v] = sign * weight * scale_factor

    # 1. Weather Inputs: Mechanosensory (Wind / Delta P) -> AMMC -> WED -> CX Steering
    add_synapse("JON_C", "AMMC_Int1", 2.6, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("JON_E", "AMMC_Int1", 2.4, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("AMMC_Int1", "WED_PN1", 2.2, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("WED_PN1", "CX_FB_Steering", 2.5, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("WED_PN1", "CX_EPG_Compass", 1.8, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("CX_EPG_Compass", "CX_FB_Steering", 1.5, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("CX_FB_Steering", "DNa02_Steering", 3.0, Neurotransmitter.ACETYLCHOLINE)

    # 2. Weather Inputs: Thermal & Hygric & Olfactory -> Lateral Horn Aversive Valence
    add_synapse("VP1l_Cold", "LHON_Aversive", 3.2, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("VP1m_Warm", "LHON_Aversive", 2.8, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("VP2_Dry", "LHON_Aversive", 2.4, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("VP3_Moist", "LHON_Aversive", 2.0, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("Glomerulus_V_CO2", "LHON_Aversive", 2.7, Neurotransmitter.ACETYLCHOLINE)

    # 3. Inter-Fly Communication: Visual Conspecific Motion & Looming
    add_synapse("LC11", "LAL_Interneuron", 3.0, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("LC4", "DNp01_AlarmSaccade", 3.8, Neurotransmitter.ACETYLCHOLINE)

    # 4. Inter-Fly Communication: Acoustic Wing Buzz (Near-field particle velocity)
    add_synapse("JON_A", "WED_Acoustic_PN", 2.5, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("JON_B", "WED_Acoustic_PN", 2.8, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("WED_Acoustic_PN", "LHON_Aversive", 2.2, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("WED_Acoustic_PN", "LAL_Interneuron", 2.0, Neurotransmitter.ACETYLCHOLINE)

    # 5. Central Convergence into Descending Motor Commands
    add_synapse("LHON_Aversive", "LAL_Interneuron", 2.8, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("LAL_Interneuron", "DNp01_AlarmSaccade", 3.5, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("LAL_Interneuron", "bLN1_AcousticBuzz", 3.2, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("LAL_Interneuron", "DN_OdorRelease", 2.2, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("LHON_Aversive", "DNa02_Steering", 2.0, Neurotransmitter.ACETYLCHOLINE)

    # 6. Negative Feedback & Post-Alarm Refractory Loop (GABAergic Veto)
    # Prevents infinite positive feedback loops between communicating neighbors
    add_synapse("DNp01_AlarmSaccade", "GABA_Refractory_LN", 4.0, Neurotransmitter.ACETYLCHOLINE)
    add_synapse("GABA_Refractory_LN", "LAL_Interneuron", 4.5, Neurotransmitter.GABA)
    add_synapse("GABA_Refractory_LN", "LHON_Aversive", 3.8, Neurotransmitter.GABA)
    add_synapse("GABA_Refractory_LN", "DNp01_AlarmSaccade", 5.0, Neurotransmitter.GABA)

    return ConnectomeSubCircuit(
        neurons=neuron_keys,
        adj_matrix=adj,
        cell_specs=CORE_CELL_CATALOG
    )

def build_population_subcircuit(neurons_per_type: int = 10, noise_std: float = 0.05) -> ConnectomeSubCircuit:
    """
    Expands the canonical circuit into a population-based network where each cell type
    is represented by a pool of neurons with heterogeneous synaptic connection probabilities
    and weights (FlyWire connectome population realism).
    """
    canonical = build_canonical_subcircuit()
    base_names = canonical.neurons
    n_base = len(base_names)
    total_neurons = n_base * neurons_per_type

    expanded_names = []
    expanded_specs = {}
    for base_name in base_names:
        for k in range(neurons_per_type):
            p_name = f"{base_name}_{k}"
            expanded_names.append(p_name)
            expanded_specs[p_name] = canonical.cell_specs[base_name]

    adj_pop = np.zeros((total_neurons, total_neurons), dtype=np.float32)
    rng = np.random.default_rng(42)

    for i, pre_base in enumerate(base_names):
        for j, post_base in enumerate(base_names):
            w_base = canonical.adj_matrix[i, j]
            if w_base != 0.0:
                # Distribute connections across the population pool with random sparsity and weight jitter
                p_conn = 0.7  # 70% connection density within pool
                for k_pre in range(neurons_per_type):
                    u = i * neurons_per_type + k_pre
                    for k_post in range(neurons_per_type):
                        v = j * neurons_per_type + k_post
                        if rng.random() < p_conn:
                            jitter = rng.normal(1.0, noise_std)
                            adj_pop[u, v] = (w_base / neurons_per_type) * jitter

    return ConnectomeSubCircuit(
        neurons=expanded_names,
        adj_matrix=adj_pop,
        cell_specs=expanded_specs
    )

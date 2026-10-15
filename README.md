# Drosophila Connectome Swarm Mesh (FlyMesh)
Decentralized biological mesh network simulation based on the *Drosophila melanogaster* whole-brain connectome (FlyWire).

## Architecture
- `fly_connectome/`: FlyWire cell types, sensory-motor sub-circuit topology, and neurotransmitter sign mapping.
- `neural_engine/`: Vectorized Leaky Integrate-and-Fire (LIF) and adaptive-threshold dynamics in NumPy.
- `agents/`: 6-DOF kinematic fly agents with embedded connectome brains and multi-modal signaling actuators.
- `environment/`: 3D spatial arena with dynamic atmospheric fields (wind, temperature, humidity, pressure, pheromones).
- `analysis/`: Network latency, spatial attenuation, percolation metrics, and spike raster / trajectory visualizers.
- `experiments/`: Actionable milestones (Milestone 1: 2-fly POC, Milestone 2: Micro-cluster, etc.).

# Drosophila Connectome Swarm Mesh (FlyMesh)
### Decentralized Biological Sensorimotor Mesh Network for Environmental Threat Propagation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.16](https://img.shields.io/badge/physics-MuJoCo%203.16-orange.svg)](https://mujoco.org/)
[![FlyWire Connectome](https://img.shields.io/badge/connectome-FlyWire%20Drosophila-green.svg)](https://flywire.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview & Research Goal

**FlyMesh** is an advanced computational neuroscience and biological swarm intelligence simulation platform. It models decentralized fruit fly (*Drosophila melanogaster*) swarms operating as an autonomous, self-healing **biological mesh network** in continuous 3D space.

When localized micro-scale meteorological disturbances (sudden barometric pressure drops, rapid convective cooling, humidity surges, or localized wind shear) occur in one sector of the environment:
1. **Sensory Transduction:** Exposed perimeter flies (*scouts*) transduce multi-modal meteorological variables through biologically mapped sensory pathways from the **FlyWire whole-brain connectome**.
2. **Connectome Integration:** Sensory spikes propagate through a 23-neuron reduced-order spiking neural network governed by adaptive-threshold Leaky Integrate-and-Fire (LIF) dynamics.
3. **Descending Motor Activation:** Command neurons trigger high-g evasive flight saccades (`DNp01`), upwind steering anemotaxis (`DNa02`), ultrasonic acoustic wing buzzing (`bLN1`), and downwind alarm pheromone emission (`DN_OdorRelease`).
4. **Inter-Fly Multi-Modal Relay:** Neighboring flies detect these physical conspecific behaviors via dedicated optical motion detectors (`LC11`, `LC4`), Johnston's organ mechanoreceptors (`JON-A/B`), and olfactory glomeruli (`Glomerulus V`), relaying the alert across hundreds of body lengths *faster than the physical storm front can travel*.

---

## 2. Complete Milestone Benchmark Suite

The project roadmap spans 6 progressive milestones, from single-hop biological transfer to large-scale 40-agent self-organized criticality, neuromodulation, and adversarial blackout jamming:

| Milestone | Agents | Physics Engine | Meteorological Telemetry | Swarm Alerted | Wave Speed ($v_{\text{wave}}$) | Critical Dynamics & Key Findings |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Milestone 1** | 2 | 6-DOF Kinematics | Localized Front ($R=0.14\text{m}$) | **100.0%** (2/2) | — | **Closed-Loop Proof-of-Concept:** Source detection in $9.5\text{ ms}$; biological hop latency of $291.5\text{ ms}$ across $16\text{ cm}$ lateral gap. |
| **Milestone 2** | 12 | 6-DOF Kinematics | Localized Shock Cell ($R=0.10\text{m}$) | **100.0%** (12/12) | **$1.74\text{ m/s}$** | **Staggered Cluster Relay:** Sequential cross-tier cascade through 5 tiers; mean hop latency of $72.5\text{ ms}$. |
| **Milestone 3** | 12 | **MuJoCo 3.16** Continuous | **Live Open-Meteo API** ($40.18^\circ\text{N}, 44.51^\circ\text{E}$) | **100.0%** (12/12) | **$2.32\text{ m/s}$** | **Live Meteorology + Boundary Layer:** Continuous 6-DOF multi-body dynamics; Prandtl-von Kármán downscaling to insect altitude ($z=0.8\text{m}$); positive anemotaxis ($56^\circ$). |
| **Milestone 4** | 40 | **MuJoCo 3.16** Continuous | Live Weather + Dryden Wind Turbulence | **100.0%** (40/40) | **$45.34\text{ m/s}$** | **Swarm Criticality & Percolation:** 40-fly rigid swarm; self-organized criticality ($R_0 \approx 0.56-0.92$); complete spatial percolation ($0.94\text{m}$ spanning cluster) under continuous atmospheric gusts ($\sigma = 0.35\text{ m/s}$). |
| **Milestone 5** | 24 | **MuJoCo 3.16** Continuous | Live Meteorological Telemetry | **100.0%** (24/24) | **$72.95\text{ m/s}$** | **Octopaminergic Neuromodulation:** Modeled octopamine (OA) arousal surge ($\tau_m$ shortened to $12\text{ ms}$, $\Delta V_{\text{th}}$ reduced by $4.0\text{ mV}$); accelerated alert cascade. |
| **Milestone 6** | 24 | **MuJoCo 3.16** Continuous | Adversarial Fog Jamming + 25% Mortality | **100.0%** (18/18 alive) | **Resilient** | **Multi-Modal Anti-Fragility:** Single-modal visual network stalled at fog barrier ($27.8\%$ alerted, $40.8\%$ span); multi-modal mesh achieved $100.0\%$ percolation, routing around dead nodes via acoustics and pheromones. |

---

## 3. FlyWire Connectome Sensory-Motor Integration Pipeline

```
                                    =========================================
                                    FLYWIRE WHOLE-BRAIN CONNECTOME SUB-CIRCUIT
                                    =========================================

       [ ENVIRONMENTAL THREATS ]                                          [ DESCENDING MOTOR OUTPUTS ]
   
   Sudden Convective Cooling -------> VP1l_Cold (IR21a/25a) ---\
   High Temperature Spike   -------> VP1m_Warm (TrpA1)       ----\
   Desiccating Dry Air      -------> VP2_Dry                 -----\
   Humidity Shock           -------> VP3_Moist                -----\
   Atmospheric Gusts        -------> JON_C (Johnston's Org)   ------\
   Barometric Pressure Drop -------> JON_E (Sacculus / AMMC)  -------> [ WED_PN1 & AL-PN ]
                                                                       [   Interneurons   ] ------> DNp01_AlarmSaccade
   Conspecific Saccade Motion -----> LC11 (Optic Glomerulus)  -------> [   Integration    ]         (High-g Evasive Flick)
   Looming Expansion Rate   -------> LC4 (Escape Looming)     ------/  [                  ]
   Conspecific Acoustic Buzz -------> JON_A/B (Near-Field)    -----/            |
   Alarm Pheromone (dSO/CO2) ------> Glomerulus_V (Olfactory) ----/             v
                                                                       [ CX Central Complex ] ----> DNa02_Steering
                                                                       [  Compass & Heading ]       (Upwind Anemotaxis)
                                                                                |
                                                                                v
                                                                       [ Lateral Horn LHON  ] ----> bLN1_AcousticBuzz
                                                                       [  Aversive Relay    ]       (Near-Field Wing Flutter)
                                                                                |
                                                                                v
                                                                       [ Subesophageal SOG  ] ----> DN_OdorRelease
                                                                       [  Neurosecretory    ]       (Alarm Pheromone Plume)
```

### Sensory Transduction Channels:
1. **Thermosensation (`VP1l_Cold` / `VP1m_Warm`):** Ionotropic receptor complex (IR21a/IR25a) in antenna sacculus sensitive to $\frac{dT}{dt} < 0$ and baseline cooling.
2. **Barosensation & Wind (`JON_E` / `JON_C`):** Johnston's organ scolopidia detecting antennal flagellum deflections driven by dynamic barometric pressure falls ($\frac{dP}{dt} < 0$) and boundary-layer airspeed shear.
3. **Optic Flow & Looming (`LC11` / `LC4`):** Lobula columnar visual projection neurons. `LC11` filters background turbulence ($<7.5\text{ rad/s}$) and triggers exclusively on rapid conspecific angular flicking ($>20\text{ rad/s}$). `LC4` responds to looming expansion ($d\theta/dt > 0.008\text{ rad/s}$).
4. **Near-Field Acoustics (`JON_A` / `JON_B`):** Antennal arista resonance detecting $150-250\text{ Hz}$ near-field acoustic particle velocity ($>0.45\text{ mm/s}$) with $1/r^2$ attenuation.
5. **Alarm Pheromones (`Glomerulus V`):** Antennal olfactory receptor neurons detecting (Z)-4-undecenal (dSO) and stress $\text{CO}_2$ with second-order Hill cooperativity kinetics ($K_d = 5.0\text{ ppm}$).

---

## 4. Multi-Modal Physical Communication Channels

The biological mesh operates across three complementary physical channels, providing robust fault tolerance against environmental jamming:

| Channel | Biological Carrier | Transduction Neurons | Spatial Range | Propagation Speed | Resilience Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Visual Optical** | Conspecific evasive saccade flick ($>2000^\circ/\text{s}$) | `LC11` & `LC4` | $0.80\text{ m}$ | Instantaneous ($c$) | Line-of-sight required; jammed by optical fog / dust. |
| **Near-Field Acoustic** | Ultrasonic wing flutter ($220\text{ Hz}$ tone pulses) | `JON-A` & `JON-B` | $0.35\text{ m}$ | Speed of sound ($343\text{ m/s}$) | Non-line-of-sight; penetrates dense fog; $1/r^2$ near-field decay. |
| **Olfactory Chemical** | Alarm pheromone puffs ($1.0\,\mu\text{g}$ dSO / $\text{CO}_2$) | `Glomerulus V` | $0.50\text{ m}$ | Advection + diffusion ($D = 1.2 \times 10^{-4}\text{ m}^2/\text{s}$) | Wind-advected; persistent memory; isotropic diffusion. |

---

## 5. Continuous 6-DOF Biophysics & Meteorological Telemetry

- **MuJoCo 3.16.0 Continuous Physics:** Swarm simulation models 6-DOF continuous rigid body biophysics. Each fly is actuated by forward wing thrust, lift equilibrium, pitch/roll damping, and steering moments.
- **Live Open-Meteo Telemetry Ingestion:** Ingests live high-resolution meteorological telemetry (temperature, relative humidity, surface pressure, 10m wind velocity, gusts) from real meteorological stations with local fallback caching.
- **Boundary-Layer Downscaling:** High-altitude ($10\text{m}$) macro-winds are downscaled to insect flight altitude ($z=0.8\text{m}$) using the Prandtl-von Kármán logarithmic boundary-layer profile:
  $$u(z) = u_{10} \frac{\ln(z / z_0)}{\ln(10.0 / z_0)}, \quad z_0 = 0.005\text{ m}$$
- **Continuous Dryden Atmospheric Turbulence:** Generates spatial cross-correlated atmospheric wind gusts with an integral length scale of $L = 0.30\text{ m}$ and turbulence intensity $\sigma = 0.35\text{ m/s}$.

---

## 6. Installation & Quick Start

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- macOS, Linux, or Windows (WSL2)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/gaxkalik/FlyProject.git
cd FlyProject

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Running Tests Manually

The test runner `run_tests.py` provides an easy, unified interface to run unit tests and all 6 milestones manually:

```bash
# 1. Run fast unit test suite (< 1 second)
python3 run_tests.py

# 2. Run specific milestones individually
python3 run_tests.py --milestone 1    # Milestone 1: 2-Fly Sensorimotor POC
python3 run_tests.py --milestone 2    # Milestone 2: 12-Fly Cluster Mesh
python3 run_tests.py --milestone 3    # Milestone 3: MuJoCo 6-DOF + Live Weather
python3 run_tests.py --milestone 4    # Milestone 4: 40-Fly Swarm Criticality & Percolation
python3 run_tests.py --milestone 5    # Milestone 5: Octopaminergic Neuromodulation
python3 run_tests.py --milestone 6    # Milestone 6: Fault Tolerance & Fog Blackout

# 3. Run full automated suite (Unit tests + Milestones 1 to 6)
python3 run_tests.py --all
```

---

## 8. Repository Structure

```
FlyProject/
├── run_tests.py                  # Unified automated test runner and milestone CLI
├── requirements.txt              # Production dependency specifications
├── README.md                     # Comprehensive technical documentation
├── fly_connectome/               # FlyWire connectome circuit reconstruction
│   ├── circuit_builder.py        # 23-neuron sensory-motor canonical sub-circuit
│   ├── cell_types.py             # Biological cell-type specifications and neurotransmitters
│   └── connectome_loader.py      # FlyWire adjacency matrix loader
├── neural_engine/                # Spiking neural network simulation
│   ├── lif_network.py            # Vectorized multi-agent LIF engine (NumPy)
│   └── sensory_encoders.py       # Transduction functions for weather variables
├── agents/                       # Insect kinematics and agent controllers
│   └── fly_agent.py              # 6-DOF fly swarm agent dynamics
├── environment/                  # Physics arenas and atmospheric simulation
│   ├── mujoco_arena.py           # MuJoCo 3.16 continuous rigid-body 6-DOF arena
│   ├── spatial_arena.py          # Multi-modal inter-agent communication physics
│   ├── weather_front.py          # Dynamic convective storm front envelopes
│   ├── realtime_weather.py       # Open-Meteo live telemetry parser and downscaler
│   └── turbulence.py             # Dryden continuous atmospheric turbulence model
├── analysis/                     # Swarm metrics and quantitative analysis
│   ├── latency.py                # Inter-hop signal transfer and cascade latency
│   ├── percolation.py            # Spatial percolation span and cluster formation
│   └── visualizer.py             # Matplotlib spike raster and 3D trajectory visualizers
├── experiments/                  # Actionable milestone experiment scripts
│   ├── milestone1_two_fly_poc.py
│   ├── milestone2_cluster_mesh.py
│   ├── milestone3_realtime_weather_mujoco.py
│   ├── milestone4_large_scale_percolation.py
│   ├── milestone5_population_neuromodulation.py
│   └── milestone6_fault_tolerance_and_jamming.py
├── data/                         # Cached meteorological telemetry
│   └── live_weather_cache.json
└── plots/                        # Generated high-resolution publication figures
    ├── milestone1_spatial_trajectories.png
    ├── milestone1_two_fly_dynamics.png
    ├── milestone2_cluster_cascade.png
    ├── milestone3_mujoco_3d_flight_trajectories.png
    ├── milestone3_mujoco_realtime_weather_cascade.png
    ├── milestone4_large_swarm_spatial_mesh.png
    ├── milestone4_percolation_and_branching.png
    ├── milestone5_neuromodulation_arousal.png
    └── milestone6_fault_tolerance_and_jamming.png
```

---

## 9. Interactive Dashboard & Artifacts

An interactive browser dashboard is provided in the artifacts directory:
- **Interactive Generative UI Dashboard:** `swarm_mesh_dashboard.html`
  - 3D swarm mesh interactive canvas
  - Live wind vector and weather telemetry widgets
  - Branching ratio ($R_0$) and percolation span meters
  - Multi-modal signal toggles (Visual / Acoustic / Pheromone)
  - Real-time spike raster viewer for descending motor neurons

---

## 10. Git Remote Backup Instructions

To push the entire codebase, milestone scripts, tests, and documentation to your GitHub repository:

```bash
# Verify git status
git status

# Stage all files and commit
git add .
git commit -m "feat: complete Milestones 1-6, MuJoCo biophysics, unified test runner, and documentation"

# Configure your GitHub remote (if not already set)
git remote add origin https://github.com/gaxkalik/FlyProject.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## References & Data Sources
- **Drosophila Connectome:** Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, FlyWire Consortium.
- **Descending Neurons & Saccades:** Cande et al. (2018), *Optogenetic dissection of descending behavioral commands in Drosophila melanogaster*.
- **Acoustic Communication & Johnston's Organ:** Kamikouchi et al. (2009), *The neural basis of courtship acoustic behavior in fruit flies*.
- **Atmospheric Physics:** Open-Meteo Weather API; Dryden Wind Turbulence Model (MIL-F-8785C).

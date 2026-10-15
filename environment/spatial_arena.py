"""
3D Spatial Arena for Drosophila Swarm Mesh Simulation.
Manages spatial bounds, physical agent positions, line-of-sight visual visibility,
acoustic near-field particle velocity propagation, and volatile pheromone diffusion.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .weather_front import WeatherEnvironment

@dataclass
class ArenaBounds:
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.0
    z_max: float = 3.0

class SpatialArena:
    """
    3D Continuous Arena providing environmental field queries and inter-agent physical signals.
    """

    def __init__(self, bounds: Optional[ArenaBounds] = None, weather_env: Optional[WeatherEnvironment] = None):
        self.bounds = bounds or ArenaBounds()
        self.weather = weather_env or WeatherEnvironment()

        # Acoustic transmission constants
        # In Drosophila, near-field sound particle velocity decays sharply (~ 1/r^2 or 1/r^3)
        self.acoustic_r0 = 0.05  # Reference near-field distance (5 cm)
        self.acoustic_max_range = 0.35 # Cutoff range (35 cm)

        # Visual tracking parameters
        # Fruit flies have ~360 deg panoramic vision with minor blind spot directly behind (ventroposterior ~30 deg)
        self.visual_max_range = 0.8  # Cutoff range for conspecific tracking (80 cm)

        # Alarm pheromone plumes (tracked as expanding Gaussian puffs)
        # Each puff: [x, y, z, release_time, peak_mass]
        self.pheromone_puffs: List[Tuple[np.ndarray, float, float]] = []
        self.pheromone_diffusivity = 1.2e-4 # m^2/s in air

    def add_pheromone_puff(self, position: np.ndarray, release_time: float, mass: float = 1.0):
        """Records an alarm pheromone (dSO/CO2) emission event."""
        self.pheromone_puffs.append((position.copy(), release_time, mass))

    def sample_pheromone_concentration(self, positions: np.ndarray, current_time: float) -> np.ndarray:
        """
        Calculates local pheromone concentration (ppm) at agent positions via superposition
        of diffusive Gaussian puffs.
        C(r, t) = M / (4 pi D (t - t0))^(3/2) * exp(-r^2 / (4 D (t - t0)))
        """
        num_agents = len(positions)
        conc = np.zeros(num_agents, dtype=np.float32)

        # Purge puffs older than 10 seconds to maintain performance
        self.pheromone_puffs = [p for p in self.pheromone_puffs if (current_time - p[1]) < 10.0]

        for puff_pos, release_t, mass in self.pheromone_puffs:
            dt = current_time - release_t
            if dt <= 0.005:
                continue

            sigma_sq = 2.0 * self.pheromone_diffusivity * dt + 1e-4
            deltas = positions - puff_pos  # [N, 3]
            dist_sq = np.sum(deltas ** 2, axis=1)

            # 3D Gaussian diffusion kernel
            amp = mass / (np.power(2.0 * np.pi * sigma_sq, 1.5))
            puff_c = amp * np.exp(-0.5 * dist_sq / sigma_sq)
            conc += puff_c.astype(np.float32)

        return conc

    def compute_pairwise_acoustic_signals(
        self,
        positions: np.ndarray,       # [N, 3]
        acoustic_emissions: np.ndarray # [N] (e.g. wing buzz amplitude 0.0 to 1.0)
    ) -> np.ndarray:
        """
        Computes the received near-field acoustic particle velocity at each agent's antenna.
        Returns: received_velocity [N] (mm/s)
        """
        num_agents = len(positions)
        received_velocity = np.zeros(num_agents, dtype=np.float32)

        for i in range(num_agents):
            total_v = 0.0
            for j in range(num_agents):
                if i == j or acoustic_emissions[j] <= 1e-3:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.acoustic_max_range:
                    # Near-field dipole particle velocity model: v ~ v_source / (1 + (r/r0)^2)
                    attenuation = 1.0 / (1.0 + (dist / self.acoustic_r0) ** 2)
                    total_v += acoustic_emissions[j] * 2.5 * attenuation  # Scaling to mm/s
            received_velocity[i] = total_v

        return received_velocity

    def compute_pairwise_visual_signals(
        self,
        positions: np.ndarray,       # [N, 3]
        velocities: np.ndarray,      # [N, 3]
        head_directions: np.ndarray, # [N, 3]
        saccade_states: np.ndarray   # [N] (1.0 if performing sharp saccade)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes:
        1. visual_conspecific_motion [N]: angular speed of moving neighbors (rad/s)
        2. visual_looming_rate [N]: optical expansion of approaching neighbors (rad/s)
        """
        num_agents = len(positions)
        conspecific_motion = np.zeros(num_agents, dtype=np.float32)
        looming_rate = np.zeros(num_agents, dtype=np.float32)

        for i in range(num_agents):
            max_ang_speed = 0.0
            max_loom = 0.0

            for j in range(num_agents):
                if i == j:
                    continue

                rel_pos = positions[j] - positions[i]
                dist = np.linalg.norm(rel_pos)
                if dist > self.visual_max_range or dist < 1e-4:
                    continue

                # Check if j is within i's field of view
                unit_rel = rel_pos / dist
                cos_angle = np.dot(head_directions[i], unit_rel)
                if cos_angle < -0.85:  # In posterior blind spot (> ~150 deg behind)
                    continue

                # Relative velocity of neighbor j relative to i
                rel_vel = velocities[j] - velocities[i]

                # Orthogonal velocity component (subtending angular motion)
                v_ortho = rel_vel - np.dot(rel_vel, unit_rel) * unit_rel
                ang_speed = np.linalg.norm(v_ortho) / dist

                # If neighbor j is actively saccading / flaring wings, visual salience multiplies
                if saccade_states[j] > 0.5:
                    ang_speed *= 3.5

                if ang_speed > max_ang_speed:
                    max_ang_speed = ang_speed

                # Looming expansion rate: d theta / dt = - (d * v_parallel) / (d^2 + (R_fly)^2)
                v_parallel = np.dot(rel_vel, unit_rel)  # Negative if approaching
                if v_parallel < 0:
                    r_fly = 0.0025  # ~2.5 mm fly radius
                    loom = (-v_parallel * r_fly) / (dist ** 2 + r_fly ** 2)
                    if loom > max_loom:
                        max_loom = loom

            conspecific_motion[i] = max_ang_speed
            looming_rate[i] = max_loom

        return conspecific_motion, looming_rate

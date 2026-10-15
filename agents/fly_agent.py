"""
Drosophila Agent Model: Connectome Brain + 6-DOF Kinematic Flight Dynamics.
Bridges descending neuron spike activity into physical flight maneuvers
and signaling emissions (visual saccades, acoustic wing-buzz, pheromones).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from fly_connectome.circuit_builder import ConnectomeSubCircuit

@dataclass
class FlyState:
    position: np.ndarray      # [x, y, z] (m)
    velocity: np.ndarray      # [vx, vy, vz] (m/s)
    heading_yaw: float        # Yaw angle (rad)
    pitch: float              # Pitch angle (rad)
    roll: float               # Roll angle (rad)
    is_alerted: bool          # Cognitive alert state flag
    saccade_timer: float      # Seconds remaining in active saccade
    acoustic_timer: float     # Seconds remaining in acoustic buzz
    odor_timer: float         # Seconds remaining in odor release

class FlyAgentSwarm:
    """
    Manages the physical kinematic states and motor decoders for N flies.
    """

    def __init__(
        self,
        num_agents: int,
        circuit: ConnectomeSubCircuit,
        initial_positions: Optional[np.ndarray] = None,
        initial_headings: Optional[np.ndarray] = None
    ):
        self.num_agents = num_agents
        self.circuit = circuit

        # Indices of Descending Neurons
        self.idx_dnp01 = circuit.get_index("DNp01_AlarmSaccade")
        self.idx_dna02 = circuit.get_index("DNa02_Steering")
        self.idx_bln1 = circuit.get_index("bLN1_AcousticBuzz")
        self.idx_odor = circuit.get_index("DN_OdorRelease")

        # Kinematic states
        if initial_positions is not None:
            self.positions = initial_positions.astype(np.float32).copy()
        else:
            self.positions = np.zeros((num_agents, 3), dtype=np.float32)

        if initial_headings is not None:
            self.headings = initial_headings.astype(np.float32).copy()
        else:
            self.headings = np.zeros(num_agents, dtype=np.float32)

        self.velocities = np.zeros((num_agents, 3), dtype=np.float32)
        # Initialize cruising velocities along heading
        cruising_speed = 0.4  # m/s
        self.velocities[:, 0] = cruising_speed * np.cos(self.headings)
        self.velocities[:, 1] = cruising_speed * np.sin(self.headings)

        # Internal timers for active motor maneuvers
        self.saccade_timer = np.zeros(num_agents, dtype=np.float32)
        self.acoustic_timer = np.zeros(num_agents, dtype=np.float32)
        self.odor_timer = np.zeros(num_agents, dtype=np.float32)
        self.just_released_odor = np.zeros(num_agents, dtype=bool)

        # Swarm alert tracking
        self.alert_states = np.zeros(num_agents, dtype=bool)
        self.first_alert_time = np.full(num_agents, -1.0, dtype=np.float32)

        # Behavioral parameters
        self.cruising_speed = 0.45       # m/s
        self.saccade_speed_boost = 1.2   # m/s during evasive dive
        self.saccade_duration = 0.08     # 80 ms sharp maneuver
        self.acoustic_duration = 0.06    # 60 ms wing flutter pulse
        self.drag_coeff = 2.5            # Aerodynamic body damping

    def get_head_directions(self) -> np.ndarray:
        """Computes 3D unit vectors pointing in the forward gaze direction."""
        head_dirs = np.zeros((self.num_agents, 3), dtype=np.float32)
        head_dirs[:, 0] = np.cos(self.headings)
        head_dirs[:, 1] = np.sin(self.headings)
        head_dirs[:, 2] = 0.0
        return head_dirs

    def update_kinematics(self, dt: float, spikes: np.ndarray, wind_vectors: np.ndarray, current_time: float):
        """
        Advances the 6-DOF aerodynamic state according to descending neuron firing.
        spikes: [num_agents, num_neurons]
        wind_vectors: [num_agents, 3]
        """
        dt = float(dt)

        # 1. Decode Descending Neuron Spikes
        dnp01_fired = (spikes[:, self.idx_dnp01] > 0.5)
        dna02_fired = (spikes[:, self.idx_dna02] > 0.5)
        bln1_fired = (spikes[:, self.idx_bln1] > 0.5)
        odor_fired = (spikes[:, self.idx_odor] > 0.5)

        # Trigger timers
        for i in range(self.num_agents):
            if dnp01_fired[i]:
                self.saccade_timer[i] = self.saccade_duration
                if not self.alert_states[i]:
                    self.alert_states[i] = True
                    self.first_alert_time[i] = current_time

            if bln1_fired[i]:
                self.acoustic_timer[i] = self.acoustic_duration
                if not self.alert_states[i]:
                    self.alert_states[i] = True
                    self.first_alert_time[i] = current_time

            if odor_fired[i] and self.odor_timer[i] <= 0.0:
                self.odor_timer[i] = 0.5
                self.just_released_odor[i] = True

        # 2. Update Timers
        self.saccade_timer = np.maximum(0.0, self.saccade_timer - dt)
        self.acoustic_timer = np.maximum(0.0, self.acoustic_timer - dt)
        self.odor_timer = np.maximum(0.0, self.odor_timer - dt)

        # 3. Apply Steering & Heading Updates
        # DNa02 commands gradual yaw turn (e.g. 5 rad/s)
        yaw_rate = np.zeros(self.num_agents, dtype=np.float32)
        yaw_rate += dna02_fired.astype(np.float32) * 5.0

        # DNp01 saccades command high-speed angular flick (> 35 rad/s ~ 2000 deg/s)
        in_saccade = (self.saccade_timer > 0.0)
        yaw_rate += in_saccade.astype(np.float32) * 35.0

        self.headings += yaw_rate * dt
        # Keep heading in [-pi, pi]
        self.headings = (self.headings + np.pi) % (2.0 * np.pi) - np.pi

        # 4. Compute Aerodynamic Forces & Accelerations
        # Forward thrust vector
        target_speed = np.where(in_saccade, self.saccade_speed_boost, self.cruising_speed)
        thrust = np.zeros((self.num_agents, 3), dtype=np.float32)
        thrust[:, 0] = target_speed * np.cos(self.headings)
        thrust[:, 1] = target_speed * np.sin(self.headings)
        # Downward evasive dive during saccade
        thrust[:, 2] = np.where(in_saccade, -0.4, 0.0)

        # Relative airspeed vector: v_rel = v_fly - v_wind
        v_rel = self.velocities - wind_vectors
        aerodynamic_drag = -self.drag_coeff * (self.velocities - thrust)

        # Acceleration
        accel = aerodynamic_drag

        # Update Velocity & Position
        self.velocities += accel * dt
        self.positions += self.velocities * dt

        # Prevent flies from plunging below ground (z=0)
        self.positions[:, 2] = np.maximum(0.05, self.positions[:, 2])

    def get_acoustic_emissions(self) -> np.ndarray:
        """Returns acoustic pulse amplitude [N] (0.0 to 1.0)."""
        return (self.acoustic_timer > 0.0).astype(np.float32)

    def get_saccade_states(self) -> np.ndarray:
        """Returns binary flag whether fly is currently performing sharp saccade."""
        return (self.saccade_timer > 0.0).astype(np.float32)

    def check_new_odor_releases(self) -> np.ndarray:
        """Returns boolean mask of agents that just triggered a pheromone release."""
        releases = self.just_released_odor.copy()
        self.just_released_odor[:] = False
        return releases

"""
MuJoCo Biophysics Arena for Drosophila Swarm Mesh Simulation.
Provides state-of-the-art continuous rigid-body 6-DOF physics, collision handling,
and biological aerodynamics integrated with real-time meteorological wind fields.
"""

from typing import List, Tuple, Optional
import numpy as np
import mujoco

class MuJoCoSwarmArena:
    """
    Manages multi-agent fruit fly physics in MuJoCo.
    Each fly is modeled as an aerodynamic ellipsoidal body with 6-DOF freejoint
    driven by biological lift, drag, and steering torques commanded by descending neurons.
    """

    def __init__(
        self,
        num_agents: int,
        initial_positions: np.ndarray,
        initial_headings: Optional[np.ndarray] = None,
        timestep: float = 0.001
    ):
        self.num_agents = num_agents
        self.dt = timestep
        self.initial_positions = initial_positions.astype(np.float32)

        if initial_headings is not None:
            self.initial_headings = initial_headings.astype(np.float32)
        else:
            self.initial_headings = np.zeros(num_agents, dtype=np.float32)

        # Biological constants for Drosophila melanogaster
        self.fly_mass = 1.2e-6        # kg (1.2 mg)
        self.fly_length = 0.0025      # 2.5 mm
        self.fly_width = 0.0010       # 1.0 mm
        self.gravity = 9.81           # m/s^2
        self.hover_thrust = self.fly_mass * self.gravity  # Base thrust to offset gravity

        # Aerodynamic drag coefficients
        self.c_drag_trans = 1.2e-5    # N / (m/s) translational drag
        self.c_drag_rot = 2.0e-9      # N*m / (rad/s) rotational damping

        # Build and compile MJCF model
        self.mjcf_xml = self._generate_mjcf()
        self.model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        self.data = mujoco.MjData(self.model)

        # Set initial positions & velocities
        self._initialize_state()

    def _generate_mjcf(self) -> str:
        """Dynamically generates the MJCF XML specification for N fly agents."""
        body_xmls = []
        for i in range(self.num_agents):
            pos_str = f"{self.initial_positions[i, 0]} {self.initial_positions[i, 1]} {self.initial_positions[i, 2]}"
            # Quaternion from initial heading yaw
            yaw = self.initial_headings[i]
            qw = np.cos(yaw / 2.0)
            qz = np.sin(yaw / 2.0)
            quat_str = f"{qw:.6f} 0 0 {qz:.6f}"

            color = "0.85 0.25 0.25 1" if i == 0 else "0.25 0.55 0.85 1"

            body_xml = f"""
            <body name="fly_{i}" pos="{pos_str}" quat="{quat_str}">
                <freejoint name="joint_{i}"/>
                <geom name="geom_{i}" type="ellipsoid" size="{self.fly_length} {self.fly_width} {self.fly_width}"
                      mass="{self.fly_mass}" rgba="{color}" friction="0.8 0.1 0.1"/>
            </body>"""
            body_xmls.append(body_xml)

        xml = f"""
        <mujoco model="drosophila_swarm_arena">
            <option gravity="0 0 -9.81" timestep="{self.dt}" integrator="Euler"/>
            <visual>
                <headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4"/>
            </visual>
            <worldbody>
                <geom name="floor" type="plane" size="5.0 5.0 0.1" rgba="0.95 0.95 0.95 1"/>
                {''.join(body_xmls)}
            </worldbody>
        </mujoco>
        """
        return xml

    def _initialize_state(self):
        """Sets qpos and qvel for all free joints."""
        mujoco.mj_resetData(self.model, self.data)
        cruising_speed = 0.45  # m/s

        for i in range(self.num_agents):
            # qpos has 7 values per freejoint: [x, y, z, qw, qx, qy, qz]
            qpos_idx = i * 7
            qvel_idx = i * 6

            self.data.qpos[qpos_idx:qpos_idx + 3] = self.initial_positions[i]
            yaw = self.initial_headings[i]
            self.data.qpos[qpos_idx + 3] = np.cos(yaw / 2.0)
            self.data.qpos[qpos_idx + 4] = 0.0
            self.data.qpos[qpos_idx + 5] = 0.0
            self.data.qpos[qpos_idx + 6] = np.sin(yaw / 2.0)

            # Velocity: vx, vy, vz
            self.data.qvel[qvel_idx] = cruising_speed * np.cos(yaw)
            self.data.qvel[qvel_idx + 1] = cruising_speed * np.sin(yaw)
            self.data.qvel[qvel_idx + 2] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_positions(self) -> np.ndarray:
        """Returns [N, 3] positions from MuJoCo qpos."""
        positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        for i in range(self.num_agents):
            positions[i] = self.data.qpos[i * 7: i * 7 + 3]
        return positions

    def get_velocities(self) -> np.ndarray:
        """Returns [N, 3] linear velocities from MuJoCo qvel."""
        velocities = np.zeros((self.num_agents, 3), dtype=np.float32)
        for i in range(self.num_agents):
            velocities[i] = self.data.qvel[i * 6: i * 6 + 3]
        return velocities

    def get_headings(self) -> np.ndarray:
        """Returns [N] yaw angles (radians) computed from MuJoCo quaternions."""
        headings = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            qw = self.data.qpos[i * 7 + 3]
            qz = self.data.qpos[i * 7 + 6]
            # Yaw for z-axis rotation: 2 * atan2(qz, qw)
            headings[i] = 2.0 * np.arctan2(qz, qw)
        return headings

    def step(
        self,
        wind_vectors: np.ndarray,      # [N, 3] downscaled micro-wind velocity
        saccade_active: np.ndarray,    # [N] 1.0 if DNp01 saccade active
        steering_active: np.ndarray    # [N] 1.0 if DNa02 steering active
    ):
        """
        Applies aerodynamic forces and biological motor torques, then advances MuJoCo physics.
        """
        # Clear applied generalized forces
        self.data.qfrc_applied[:] = 0.0

        positions = self.get_positions()
        velocities = self.get_velocities()
        headings = self.get_headings()

        for i in range(self.num_agents):
            qvel_idx = i * 6
            v_fly = velocities[i]
            v_wind = wind_vectors[i]
            yaw = headings[i]

            # 1. Forward thrust along heading to maintain cruising speed
            target_speed = 1.1 if saccade_active[i] > 0.5 else 0.45
            f_forward = np.array([
                target_speed * np.cos(yaw),
                target_speed * np.sin(yaw),
                0.0
            ], dtype=np.float32)

            # 2. Upward lift force to offset gravity
            f_lift = np.array([0.0, 0.0, self.hover_thrust], dtype=np.float32)
            if saccade_active[i] > 0.5:
                # Evasive dive during alarm saccade
                f_lift[2] -= 0.4 * self.hover_thrust

            # 3. Aerodynamic drag against relative airspeed: v_rel = v_fly - v_wind
            v_rel = v_fly - v_wind
            f_drag = -self.c_drag_trans * v_rel

            total_force = (f_forward * self.fly_mass * 10.0) + f_lift + f_drag

            # 4. Motor Steering & Saccadic Torques (applied around Z-axis)
            torque_z = 0.0
            # Rotational damping
            torque_z -= self.c_drag_rot * self.data.qvel[qvel_idx + 5]

            # DNa02 steering command (moderate turn ~ 5 rad/s)
            if steering_active[i] > 0.5:
                torque_z += 1.5e-8

            # DNp01 sharp evasive saccade (> 35 rad/s ~ 2000 deg/s)
            if saccade_active[i] > 0.5:
                torque_z += 1.2e-7

            # Apply forces: [fx, fy, fz, tx, ty, tz]
            self.data.qfrc_applied[qvel_idx:qvel_idx + 3] = total_force
            self.data.qfrc_applied[qvel_idx + 5] = torque_z

        # Advance MuJoCo biophysics
        mujoco.mj_step(self.model, self.data)

        # Enforce ground floor boundary
        for i in range(self.num_agents):
            if self.data.qpos[i * 7 + 2] < 0.02:
                self.data.qpos[i * 7 + 2] = 0.02
                self.data.qvel[i * 6 + 2] = 0.0

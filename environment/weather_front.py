"""
Dynamic Meteorological Weather Fronts for Spatial Swarm Arena.
Generates localized, propagating atmospheric disturbances:
- Barometric pressure drops (d P / dt < 0)
- Rapid adiabatic cooling (d T / dt < 0)
- Humidity saturation spikes (d H / dt > 0)
- Turbulent wind gust vectors (v_wind)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class WeatherFrontParams:
    center: np.ndarray             # Initial center [x, y, z] in meters
    velocity: np.ndarray           # Drift velocity [vx, vy, vz] in m/s
    radius: float = 0.4            # Spatial radius of disturbance (meters)
    temp_drop: float = 8.0         # Temperature drop within front (deg C)
    humidity_spike: float = 25.0   # Relative humidity rise (%)
    pressure_drop: float = 2.5     # Pressure drop (kPa)
    wind_speed: float = 1.8        # Peak wind gust speed (m/s)
    wind_direction: np.ndarray = None # Normalized direction vector
    onset_time: float = 0.5        # Seconds after simulation start when storm begins
    ramp_duration: float = 0.3     # Seconds over which weather ramps to peak

    def __post_init__(self):
        if self.wind_direction is None:
            self.wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.wind_direction = np.array(self.wind_direction, dtype=np.float32)
            norm = np.linalg.norm(self.wind_direction)
            if norm > 1e-6:
                self.wind_direction /= norm

class WeatherEnvironment:
    """
    Manages continuous 3D ambient fields and moving localized weather disturbances.
    """

    def __init__(
        self,
        baseline_temp: float = 24.0,       # Celsius
        baseline_humidity: float = 65.0,   # %
        baseline_pressure: float = 101.325,# kPa
        front_params: Optional[WeatherFrontParams] = None
    ):
        self.t_base = baseline_temp
        self.h_base = baseline_humidity
        self.p_base = baseline_pressure
        self.front = front_params or WeatherFrontParams(
            center=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            velocity=np.array([0.1, 0.0, 0.0], dtype=np.float32)
        )

        # Cache previous state for dynamic derivative calculation
        self._prev_time = 0.0
        self._prev_t = None
        self._prev_h = None
        self._prev_p = None

    def sample_at_positions(self, positions: np.ndarray, time_s: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Samples weather state at agent positions [N, 3] at time_s.
        Returns:
            temp_c: [N]
            d_temp_dt: [N]
            humidity_pct: [N]
            d_humidity_dt: [N]
            pressure_kpa: [N]
            d_pressure_dt: [N]
            wind_vector: [N, 3] (m/s)
        """
        num_agents = len(positions)
        dt = max(1e-4, time_s - self._prev_time)

        # Storm movement: center(t) = center_0 + velocity * max(0, t - onset_time)
        active_t = max(0.0, time_s - self.front.onset_time)
        current_center = self.front.center + self.front.velocity * active_t

        # Ramp-up factor [0.0 to 1.0]
        if time_s < self.front.onset_time:
            intensity = 0.0
        else:
            intensity = min(1.0, active_t / max(1e-3, self.front.ramp_duration))

        # Spatial compact envelope around the storm front (smooth cosine taper, identically 0 for dist > radius)
        # Prevents unphysical Gaussian tail leakage into neighboring calm sectors
        deltas = positions - current_center
        dists = np.linalg.norm(deltas, axis=1)
        r_ratio = dists / max(1e-4, self.front.radius)
        spatial_factor = np.where(
            r_ratio <= 1.0,
            0.5 * (1.0 + np.cos(np.pi * r_ratio)) * intensity,
            0.0
        )

        # Local instantaneous values
        temp_c = self.t_base - self.front.temp_drop * spatial_factor
        humidity_pct = np.clip(self.h_base + self.front.humidity_spike * spatial_factor, 0.0, 100.0)
        pressure_kpa = self.p_base - self.front.pressure_drop * spatial_factor

        # Wind velocity vectors: [N, 3]
        wind_magnitude = self.front.wind_speed * spatial_factor[:, np.newaxis]
        wind_vector = wind_magnitude * self.front.wind_direction[np.newaxis, :]

        # Derivatives
        if self._prev_t is None or len(self._prev_t) != num_agents:
            d_temp_dt = np.zeros(num_agents, dtype=np.float32)
            d_humidity_dt = np.zeros(num_agents, dtype=np.float32)
            d_pressure_dt = np.zeros(num_agents, dtype=np.float32)
        else:
            d_temp_dt = (temp_c - self._prev_t) / dt
            d_humidity_dt = (humidity_pct - self._prev_h) / dt
            d_pressure_dt = (pressure_kpa - self._prev_p) / dt

        self._prev_time = time_s
        self._prev_t = temp_c.copy()
        self._prev_h = humidity_pct.copy()
        self._prev_p = pressure_kpa.copy()

        return (
            temp_c.astype(np.float32),
            d_temp_dt.astype(np.float32),
            humidity_pct.astype(np.float32),
            d_humidity_dt.astype(np.float32),
            pressure_kpa.astype(np.float32),
            d_pressure_dt.astype(np.float32),
            wind_vector.astype(np.float32)
        )

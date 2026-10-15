"""
Atmospheric Turbulence and Sensory Noise Injection Models.
Implements Dryden/von Kármán continuous spatio-temporal wind turbulence
and environmental acoustic background noise for robust swarm stress-testing.
"""

from typing import Tuple, Optional
import numpy as np

class AtmosphericTurbulenceModel:
    """
    Spatio-temporally correlated atmospheric wind turbulence based on the Dryden spectral formulation.
    Generates dynamic gust perturbations: u'(x, t) = [u'_x, u'_y, u'_z]
    """

    def __init__(
        self,
        turbulence_intensity: float = 0.45,  # Standard deviation of turbulent fluctuations (m/s)
        integral_length_scale: float = 0.35, # Spatial correlation length scale L (meters)
        time_scale_s: float = 0.25           # Temporal relaxation scale tau (seconds)
    ):
        self.sigma = turbulence_intensity
        self.L = integral_length_scale
        self.tau = time_scale_s
        self.prev_turb = None

    def sample_turbulence(
        self,
        positions: np.ndarray,   # [N, 3] in meters
        dt: float
    ) -> np.ndarray:
        """
        Computes 3D turbulent velocity vectors [N, 3] at agent positions.
        Uses an Ornstein-Uhlenbeck stochastic differential equation with spatial cross-correlation:
        du' = - (u' / tau) * dt + sigma * sqrt(2 * dt / tau) * dW
        """
        num_agents = len(positions)
        dt = float(dt)

        # Decay factor
        alpha = np.exp(-dt / self.tau)

        # Stochastic Wiener increment
        rng = np.random.default_rng()
        dW = rng.normal(0.0, 1.0, size=(num_agents, 3)).astype(np.float32)

        # Spatial cross-correlation between neighbor agents: C(r) = exp(-r / L)
        if num_agents > 1:
            dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
            corr_matrix = np.exp(-dists / self.L).astype(np.float32)
            # Regularize for numerical stability
            corr_matrix += 1e-4 * np.eye(num_agents, dtype=np.float32)
            # Cholesky factor to spatially correlate noise vectors
            try:
                chol = np.linalg.cholesky(corr_matrix)
                dW = np.dot(chol, dW)
            except np.linalg.LinAlgError:
                pass  # Fall back to uncorrelated noise if ill-conditioned

        innovation = self.sigma * np.sqrt(max(1e-5, 1.0 - alpha ** 2)) * dW

        if self.prev_turb is None or len(self.prev_turb) != num_agents:
            u_turb = self.sigma * dW
        else:
            u_turb = alpha * self.prev_turb + innovation

        self.prev_turb = u_turb.copy()
        return u_turb.astype(np.float32)

def generate_ambient_acoustic_noise(num_agents: int, snr_floor: float = 0.15) -> np.ndarray:
    """
    Generates background environmental acoustic particle velocity noise (mm/s)
    caused by ambient leaves, turbulent boundary shear, and distant conspecific chatter.
    """
    rng = np.random.default_rng()
    # Rayleigh distributed particle velocity noise
    noise = rng.rayleigh(scale=snr_floor, size=num_agents).astype(np.float32)
    return noise

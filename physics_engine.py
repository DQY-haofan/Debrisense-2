# ==============================================================================
# physics_engine.py - THz-ISL Diffraction Channel Physics Engine
# ==============================================================================
# Version: 2.0 (Paper Version)
#
# Description:
#   Implements the Fresnel diffraction physics for THz forward scatter detection:
#   - Lommel function computation for circular aperture
#   - Narrowband and wideband (DFS) diffraction patterns
#   - Broadband chirp signal generation
#
# References:
#   - Born & Wolf, Principles of Optics (Fresnel diffraction)
#   - Paper equations for forward scatter geometry
# ==============================================================================

import numpy as np
import scipy.special as sp
from typing import Tuple, Optional, Dict, Any


class DiffractionChannel:
    """
    THz-ISL Diffraction Channel Model.
    
    Models the Fresnel diffraction of a THz carrier wave by spherical
    debris, producing characteristic amplitude and phase modulation
    that encodes debris size and velocity information.
    
    Attributes:
        fc: Carrier frequency (Hz)
        B: Signal bandwidth (Hz)
        L_eff: Effective link length (m)
        a: Debris radius (m)
        v_rel: Default relative velocity (m/s)
        c: Speed of light (m/s)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diffraction channel with physical parameters.
        
        Args:
            config: Dictionary with physical parameters:
                - fc: Carrier frequency (Hz)
                - B: Bandwidth (Hz)
                - L_eff: Effective link length (m)
                - a: Debris radius (m)
                - v_rel: Relative velocity (m/s), optional
        """
        self.fc = float(config.get('fc', 300e9))
        self.B = float(config.get('B', 10e9))
        self.L_eff = float(config.get('L_eff', 50e3))
        self.a = float(config.get('a', 0.05))
        self.v_rel = float(config.get('v_rel', 15000.0))
        self.c = 299792458.0
        
        # Derived parameters
        self.lambda_c = self.c / self.fc  # Central wavelength
        self.fresnel_number = self.a ** 2 / (self.lambda_c * self.L_eff)
    
    def _lommel_series(
        self, 
        u: np.ndarray, 
        v: np.ndarray, 
        max_terms: int = 40
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lommel functions U1 and U2 for Fresnel diffraction.
        
        The Lommel functions arise in the Fresnel diffraction integral
        for a circular aperture/obstacle:
        
            U1(u,v) = Σ_{m=0}^∞ (-1)^m (u/v)^{1+2m} J_{1+2m}(v)
            U2(u,v) = Σ_{m=0}^∞ (-1)^m (u/v)^{2+2m} J_{2+2m}(v)
        
        where:
            u = π a² / (λ L_eff)  (Fresnel zone parameter)
            v = π a ρ / (λ L_eff) (off-axis parameter, ρ = lateral distance)
        
        Args:
            u: Fresnel zone parameter (may be array)
            v: Off-axis parameter (array, same shape as u)
            max_terms: Series truncation for convergence
            
        Returns:
            U1: First Lommel function
            U2: Second Lommel function
        """
        # Prevent division by zero for v → 0
        v_safe = np.maximum(v, 1e-5)
        ratio = u / v_safe
        
        U1 = np.zeros_like(v, dtype=np.complex128)
        U2 = np.zeros_like(v, dtype=np.complex128)
        
        for m in range(max_terms):
            sign = (-1.0) ** m
            
            # Bessel functions
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)
            
            # Power terms
            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)
            
            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2
        
        return U1, U2
    
    def compute_fresnel_parameters(
        self, 
        wavelength: float, 
        rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fresnel parameters u and v.
        
        Args:
            wavelength: Wavelength (m)
            rho: Lateral distance from optical axis (m)
            
        Returns:
            u: Fresnel zone parameter
            v: Off-axis parameter
        """
        u = (2 * np.pi * self.a ** 2) / (wavelength * self.L_eff)
        v = (2 * np.pi * self.a * rho) / (wavelength * self.L_eff)
        return u, v
    
    def generate_diffraction_pattern(
        self, 
        t_axis: np.ndarray, 
        freqs: np.ndarray,
        v_rel: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate diffraction pattern for specified frequencies.
        
        Computes the complex diffraction coefficient d_k(t) for each
        frequency in the grid, returning a matrix of shape (N_freqs, N_time).
        
        The diffraction field is:
            d_k(t) = exp(-j u_k/2) · [U1(u_k, v_k) + j U2(u_k, v_k)] · exp(j v_k²/(2u_k))
        
        Args:
            t_axis: Time axis (s), centered at crossing
            freqs: Frequency array (Hz)
            v_rel: Relative velocity (m/s), uses default if None
            
        Returns:
            d_k: Complex diffraction matrix (N_freqs x N_time)
        """
        if v_rel is None:
            v_rel = self.v_rel
        
        freqs = np.atleast_1d(freqs)
        
        # Matrix computation for efficiency
        freqs_col = freqs.reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        
        # Wavelengths
        lam_col = self.c / freqs_col
        
        # Lateral distance: ρ(t) = v_rel · |t|
        rho_t = v_rel * np.abs(t_row)
        
        # Fresnel parameters
        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)
        
        # Compute Lommel functions
        U1, U2 = self._lommel_series(u_k, v_k)
        
        # Compose diffraction field
        quad_phase = np.exp(1j * (v_k ** 2) / (2 * u_k))
        phase_term = np.exp(-1j * u_k / 2)
        
        d_k_matrix = phase_term * (U1 + 1j * U2) * quad_phase
        
        return d_k_matrix
    
    def generate_narrowband_pattern(
        self, 
        t_axis: np.ndarray,
        freq: Optional[float] = None,
        v_rel: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate narrowband (single frequency) diffraction pattern.
        
        Args:
            t_axis: Time axis (s)
            freq: Carrier frequency (Hz), uses fc if None
            v_rel: Relative velocity (m/s)
            
        Returns:
            d: Complex diffraction signal (N_time,)
        """
        if freq is None:
            freq = self.fc
        
        d_k = self.generate_diffraction_pattern(
            t_axis, 
            np.array([freq]),
            v_rel
        )
        
        return d_k[0, :]  # Single frequency
    
    def generate_broadband_chirp(
        self, 
        t_axis: np.ndarray, 
        N_sub: int = 32,
        v_rel: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate wideband Discrete Frequency Summation (DFS) signal.
        
        The broadband signal is formed by coherent summation across
        the bandwidth:
            d_wb(t) = (1/N_sub) Σ_k d_k(t) · exp(j 2π Δf_k t)
        
        where Δf_k = f_k - f_c is the frequency offset from center.
        
        Args:
            t_axis: Time axis (s), centered at crossing
            N_sub: Number of sub-carriers for DFS
            v_rel: Relative velocity (m/s)
            
        Returns:
            d_wb: Complex broadband diffraction signal (N_time,)
        """
        if v_rel is None:
            v_rel = self.v_rel
        
        # Frequency grid spanning bandwidth
        freqs = np.linspace(
            self.fc - self.B / 2,
            self.fc + self.B / 2,
            N_sub
        )
        
        # Compute diffraction for all frequencies
        d_k_matrix = self.generate_diffraction_pattern(t_axis, freqs, v_rel)
        
        # Baseband rotation: multiply by exp(j 2π Δf_k t)
        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)
        
        # Coherent summation
        broadband_signal = np.sum(d_k_matrix * time_phase_matrix, axis=0) / N_sub
        
        return broadband_signal
    
    def generate_received_signal(
        self,
        t_axis: np.ndarray,
        N_sub: int = 32,
        v_rel: Optional[float] = None,
        baseline: complex = 1.0
    ) -> np.ndarray:
        """
        Generate complete received signal with diffraction modulation.
        
        The received signal is:
            y(t) = baseline · (1 - d_wb(t))
        
        where d_wb is the broadband diffraction pattern causing the
        characteristic "dip" in received amplitude.
        
        Args:
            t_axis: Time axis (s)
            N_sub: Number of DFS sub-carriers
            v_rel: Relative velocity (m/s)
            baseline: Baseline received signal level
            
        Returns:
            y: Complex received signal (N_time,)
        """
        d_wb = self.generate_broadband_chirp(t_axis, N_sub, v_rel)
        y = baseline * (1.0 - d_wb)
        return y
    
    def compute_diffraction_depth(
        self,
        t_axis: np.ndarray,
        N_sub: int = 32,
        v_rel: Optional[float] = None
    ) -> float:
        """
        Compute the maximum diffraction depth (amplitude dip).
        
        Depth = 1 - min(|1 - d_wb(t)|)
        
        Args:
            t_axis: Time axis (s)
            N_sub: Number of DFS sub-carriers
            v_rel: Relative velocity (m/s)
            
        Returns:
            depth: Maximum diffraction depth as fraction
        """
        d_wb = self.generate_broadband_chirp(t_axis, N_sub, v_rel)
        envelope = np.abs(1.0 - d_wb)
        depth = 1.0 - np.min(envelope)
        return depth
    
    def get_physical_summary(self) -> Dict[str, Any]:
        """
        Get summary of physical parameters and derived quantities.
        
        Returns:
            Dictionary with physical parameters
        """
        return {
            'fc': self.fc,
            'fc_GHz': self.fc / 1e9,
            'B': self.B,
            'B_GHz': self.B / 1e9,
            'L_eff': self.L_eff,
            'L_eff_km': self.L_eff / 1e3,
            'a': self.a,
            'a_cm': self.a * 100,
            'v_rel': self.v_rel,
            'lambda_c': self.lambda_c,
            'lambda_c_mm': self.lambda_c * 1e3,
            'fresnel_number': self.fresnel_number,
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_channel_from_config(config) -> DiffractionChannel:
    """
    Create DiffractionChannel from ConfigManager.
    
    Args:
        config: ConfigManager instance
        
    Returns:
        Configured DiffractionChannel instance
    """
    return DiffractionChannel(config.get_physics_config())


def create_channel_with_params(
    L_eff: float = None,
    a: float = None,
    v_rel: float = None,
    base_config: Dict[str, Any] = None
) -> DiffractionChannel:
    """
    Create channel with specific parameter overrides.
    
    Args:
        L_eff: Override effective length (m)
        a: Override debris radius (m)
        v_rel: Override velocity (m/s)
        base_config: Base configuration to modify
        
    Returns:
        DiffractionChannel with specified parameters
    """
    if base_config is None:
        base_config = {
            'fc': 300e9,
            'B': 10e9,
            'L_eff': 50e3,
            'a': 0.05,
            'v_rel': 15000.0,
        }
    
    config = base_config.copy()
    if L_eff is not None:
        config['L_eff'] = L_eff
    if a is not None:
        config['a'] = a
    if v_rel is not None:
        config['v_rel'] = v_rel
    
    return DiffractionChannel(config)

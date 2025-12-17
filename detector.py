# ==============================================================================
# detector.py - THz-ISL Debris Detection Engine (Paper Version)
# ==============================================================================
# Version: 2.0 (Paper Pipeline Implementation)
# 
# Description:
#   Implementation of the paper's detection pipeline:
#   Step 1: Log-envelope transform
#   Step 2: Survival-space (DCT low-frequency subspace projection)
#   Step 3: Optional whitening
#   Step 4: GLRT / Matched Filter statistics
#
# CRITICAL: This implements the PAPER VERSION of the detection chain.
#           The "dip/peak" detector is DEPRECATED and only available for debug.
#
# References:
#   - DR_algo_01: Survival space theory with DCT projection
#   - Paper Eq. (X): GLRT statistic formulation
# ==============================================================================

import numpy as np
import scipy.special as sp
import scipy.signal as sig_proc
from typing import Optional, Tuple, Dict, Any, Union
import warnings


class TerahertzDebrisDetector:
    """
    THz-ISL Forward Scatter Debris Detector (Paper Implementation)
    
    Implements the complete detection pipeline:
    1. Log-envelope transform: x[n] = log(|y[n]| + ε)
    2. Survival-space projection: z = P_perp @ x
    3. Optional whitening: z_w = L^{-1} @ z
    4. GLRT statistic: T = (s_perp^T z)^2 / (s_perp^T s_perp)
    
    Attributes:
        fs: Sampling frequency (Hz)
        N: Window length (samples)
        f_cut: DCT projection cutoff frequency (Hz)
        P_perp: Orthogonal projection matrix (N x N)
        H: DCT basis matrix (N x k_max)
    """
    
    def __init__(
        self,
        fs: float,
        N_window: int,
        cutoff_freq: float = 300.0,
        L_eff: float = 50e3,
        fc: float = 300e9,
        a: float = 0.05,
        B: float = 10e9,
        N_sub: int = 32,
        epsilon_log: float = 1e-12,
        delta_reg: float = 1e-20,
    ):
        """
        Initialize detector with physical and processing parameters.
        
        Args:
            fs: Sampling frequency (Hz)
            N_window: Number of samples in observation window
            cutoff_freq: DCT projection cutoff frequency (Hz), baseline = 300 Hz
            L_eff: Effective link length (m)
            fc: Carrier frequency (Hz)
            a: Debris radius (m)
            B: Signal bandwidth (Hz)
            N_sub: Number of sub-carriers for DFS
            epsilon_log: Small constant to prevent log(0)
            delta_reg: Regularization for division
        """
        self.fs = float(fs)
        self.N = int(N_window)
        self.f_cut = float(cutoff_freq)
        self.L_eff = float(L_eff)
        self.fc = float(fc)
        self.B = float(B)
        self.a = float(a)
        self.c = 299792458.0
        self.N_sub = int(N_sub)
        self.epsilon_log = epsilon_log
        self.delta_reg = delta_reg
        
        # Build DCT projection matrix (PAPER VERSION)
        self.H, self.k_max = self._build_dct_basis()
        self.P_perp = self._build_projection_matrix()
        
        # Cache for templates
        self._template_cache: Dict[float, np.ndarray] = {}
        
    def _build_dct_basis(self) -> Tuple[np.ndarray, int]:
        """
        Build DCT-II orthonormal basis vectors.
        
        The DCT basis vectors are:
            h_k[n] = s_k * cos(π k (2n+1) / (2N))
        where:
            s_0 = sqrt(1/N)
            s_k = sqrt(2/N) for k >= 1
        
        Returns:
            H: DCT basis matrix (N x k_max)
            k_max: Number of basis vectors
        """
        # Frequency resolution
        f_res = self.fs / (2 * self.N)
        
        # Number of low-frequency basis vectors to keep
        k_max = int(np.ceil(self.f_cut / f_res))
        k_max = max(k_max, 1)  # At least 1
        
        # Build basis matrix
        H = np.zeros((self.N, k_max), dtype=np.float64)
        n_idx = np.arange(self.N)
        
        for k in range(k_max):
            # Scaling factor for orthonormality
            scale = np.sqrt(1.0 / self.N) if k == 0 else np.sqrt(2.0 / self.N)
            # DCT-II basis vector
            H[:, k] = scale * np.cos(np.pi * k * (2 * n_idx + 1) / (2 * self.N))
        
        return H, k_max
    
    def _build_projection_matrix(self) -> np.ndarray:
        """
        Build orthogonal complement projection matrix.
        
        P_perp = I - H @ H^T
        
        This projects onto the subspace orthogonal to low-frequency DCT components,
        effectively removing slow variations (jitter, drift) while preserving
        the diffraction signal's high-frequency structure.
        
        Returns:
            P_perp: Projection matrix (N x N)
        """
        P_perp = np.eye(self.N) - (self.H @ self.H.T)
        return P_perp
    
    # =========================================================================
    # STEP 0: Automatic Gain Control (AGC)
    # =========================================================================
    
    def apply_agc(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply Automatic Gain Control (AGC).
        
        Normalizes signal power to unity, which is critical for
        consistent detection performance across different signal levels.
        
        Args:
            signal_in: Input signal (complex)
            
        Returns:
            Power-normalized signal
        """
        p_sig = np.mean(np.abs(signal_in) ** 2)
        if p_sig < 1e-20:
            return signal_in
        gain = 1.0 / np.sqrt(p_sig)
        return signal_in * gain
    
    # =========================================================================
    # STEP 1: Log-Envelope Transform
    # =========================================================================
    
    def log_envelope_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Compute log-envelope of complex baseband signal.
        
        x[n] = log(|y[n]| + ε)
        
        Args:
            y: Complex baseband signal (N,)
            
        Returns:
            x: Log-envelope (N,)
            
        Note:
            The epsilon prevents log(0) for samples with very low amplitude.
        """
        envelope = np.abs(y)
        x = np.log(envelope + self.epsilon_log)
        return x
    
    # =========================================================================
    # STEP 2: Survival-Space Projection
    # =========================================================================
    
    def apply_projection(self, x: np.ndarray) -> np.ndarray:
        """
        Apply survival-space (DCT) projection.
        
        z = P_perp @ x
        
        This removes the low-frequency subspace (k < k_max) from the signal,
        preserving the "survival space" where the diffraction chirp resides.
        
        Args:
            x: Log-envelope signal (N,)
            
        Returns:
            z: Projected signal in survival space (N,)
        """
        if len(x) != self.N:
            x = np.reshape(x, (self.N,))
        return self.P_perp @ x
    
    def compute_energy_retention(self, signal: np.ndarray) -> float:
        """
        Compute energy retention ratio after projection.
        
        η = ||P_perp @ s||^2 / ||s||^2
        
        This is a CRITICAL sanity check to ensure the signal is not
        "self-eliminated" by the projection.
        
        Args:
            signal: Signal to check (template or observation)
            
        Returns:
            η: Energy retention ratio (0 to 1)
        """
        signal_energy = np.sum(signal ** 2)
        if signal_energy < self.delta_reg:
            return 0.0
        
        projected = self.P_perp @ signal
        projected_energy = np.sum(projected ** 2)
        
        return projected_energy / signal_energy
    
    # =========================================================================
    # STEP 3: Optional Whitening
    # =========================================================================
    
    def estimate_noise_covariance(
        self, 
        noise_samples: np.ndarray,
        diagonal_loading: float = 1e-10
    ) -> np.ndarray:
        """
        Estimate noise covariance matrix from H0 samples.
        
        Args:
            noise_samples: Array of shape (n_trials, N) containing noise-only observations
            diagonal_loading: Regularization for matrix conditioning
            
        Returns:
            R: Estimated covariance matrix (N x N)
        """
        if noise_samples.ndim == 1:
            noise_samples = noise_samples.reshape(1, -1)
        
        # Sample covariance
        R = np.cov(noise_samples, rowvar=False)
        
        # Diagonal loading for numerical stability
        R += diagonal_loading * np.eye(self.N)
        
        return R
    
    def compute_whitening_matrix(
        self, 
        R: np.ndarray
    ) -> np.ndarray:
        """
        Compute whitening matrix from covariance.
        
        L @ L^T = R (Cholesky decomposition)
        Returns L^{-1} for whitening: z_w = L^{-1} @ z
        
        Args:
            R: Covariance matrix (N x N)
            
        Returns:
            L_inv: Whitening matrix (N x N)
        """
        try:
            L = np.linalg.cholesky(R)
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition
            warnings.warn("Cholesky failed, using eigendecomposition")
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, 1e-10)
            L_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        return L_inv
    
    def apply_whitening(
        self, 
        z: np.ndarray, 
        L_inv: np.ndarray
    ) -> np.ndarray:
        """
        Apply whitening transform.
        
        z_w = L^{-1} @ z
        
        Args:
            z: Signal in survival space (N,)
            L_inv: Whitening matrix (N x N)
            
        Returns:
            z_w: Whitened signal (N,)
        """
        return L_inv @ z
    
    # =========================================================================
    # STEP 4: GLRT / Matched Filter Statistics
    # =========================================================================
    
    def compute_glrt_statistic(
        self,
        z_perp: np.ndarray,
        s_perp: np.ndarray,
    ) -> float:
        """
        Compute GLRT statistic for detection.
        
        T = (s_perp^T @ z_perp)^2 / (s_perp^T @ s_perp + δ)
        
        Under H0 (noise only), T follows a chi-squared distribution.
        Under H1 (signal present), T has a non-central chi-squared distribution.
        
        Args:
            z_perp: Projected observation (N,)
            s_perp: Projected template (N,)
            
        Returns:
            T: GLRT statistic (scalar)
        """
        template_energy = np.sum(s_perp ** 2)
        correlation = np.dot(s_perp, z_perp)
        
        T = (correlation ** 2) / (template_energy + self.delta_reg)
        return T
    
    def compute_normalized_correlation(
        self,
        z_perp: np.ndarray,
        s_perp: np.ndarray,
    ) -> float:
        """
        Compute normalized correlation (for matched filter).
        
        ρ = (s_perp^T @ z_perp) / (||s_perp|| * ||z_perp||)
        
        Args:
            z_perp: Projected observation
            s_perp: Projected template
            
        Returns:
            ρ: Normalized correlation in [-1, 1]
        """
        s_norm = np.linalg.norm(s_perp)
        z_norm = np.linalg.norm(z_perp)
        
        if s_norm < self.delta_reg or z_norm < self.delta_reg:
            return 0.0
        
        return np.dot(s_perp, z_perp) / (s_norm * z_norm)
    
    # =========================================================================
    # Template Generation (Physics-based)
    # =========================================================================
    
    def _lommel_series(
        self, 
        u: np.ndarray, 
        v: np.ndarray, 
        max_terms: int = 40
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lommel functions U1 and U2 for Fresnel diffraction.
        
        U1(u,v) = Σ (-1)^m (u/v)^{1+2m} J_{1+2m}(v)
        U2(u,v) = Σ (-1)^m (u/v)^{2+2m} J_{2+2m}(v)
        
        Args:
            u: Fresnel parameter (related to aperture)
            v: Fresnel parameter (related to observation point)
            max_terms: Series truncation
            
        Returns:
            U1, U2: Lommel function values
        """
        v_safe = np.maximum(v, 1e-5)
        ratio = u / v_safe
        
        U1 = np.zeros_like(v, dtype=np.complex128)
        U2 = np.zeros_like(v, dtype=np.complex128)
        
        for m in range(max_terms):
            sign = (-1.0) ** m
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)
            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)
            
            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2
        
        return U1, U2
    
    def _generate_diffraction_pattern(
        self,
        t_axis: np.ndarray,
        freqs: np.ndarray,
        v_rel: float
    ) -> np.ndarray:
        """
        Generate diffraction pattern for given frequencies.
        
        Args:
            t_axis: Time axis (N,)
            freqs: Frequency array (N_sub,)
            v_rel: Relative velocity (m/s)
            
        Returns:
            d_k: Diffraction matrix (N_sub x N)
        """
        freqs = np.atleast_1d(freqs)
        freqs_col = freqs.reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        
        lam_col = self.c / freqs_col
        rho_t = v_rel * np.abs(t_row)
        
        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)
        
        U1, U2 = self._lommel_series(u_k, v_k)
        
        quad_phase = np.exp(1j * (v_k ** 2) / (2 * u_k))
        phase_term = np.exp(-1j * u_k / 2)
        
        d_k = phase_term * (U1 + 1j * U2) * quad_phase
        return d_k
    
    def generate_template(
        self, 
        v_rel: float,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate broadband DFS (Discrete Frequency Summation) template.
        
        The template represents the expected log-envelope signature
        for a debris crossing at velocity v_rel.
        
        Args:
            v_rel: Relative velocity (m/s)
            use_cache: Whether to use cached templates
            
        Returns:
            s_template: Template signal in log-envelope domain (N,)
        """
        # Check cache
        if use_cache and v_rel in self._template_cache:
            return self._template_cache[v_rel].copy()
        
        # Time axis centered at t=0 (crossing time)
        t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs
        
        # Frequency grid for DFS
        freqs = np.linspace(
            self.fc - self.B / 2,
            self.fc + self.B / 2,
            self.N_sub
        )
        
        # Compute diffraction pattern
        d_k_matrix = self._generate_diffraction_pattern(t_axis, freqs, v_rel)
        
        # Baseband rotation and coherent summation
        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)
        
        broadband_d = np.sum(d_k_matrix * time_phase_matrix, axis=0) / self.N_sub
        
        # Template in log-envelope domain
        # Note: We use -real part because diffraction causes amplitude dip
        s_template = -np.real(broadband_d)
        
        # Cache
        if use_cache:
            self._template_cache[v_rel] = s_template.copy()
        
        return s_template
    
    def clear_template_cache(self):
        """Clear the template cache"""
        self._template_cache.clear()
    
    # =========================================================================
    # PAPER PIPELINE: Complete Detection Chain
    # =========================================================================
    
    def paper_pipeline_detect(
        self,
        y: np.ndarray,
        v_rel: float,
        whiten_mode: str = 'none',
        L_inv: Optional[np.ndarray] = None,
        return_intermediate: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Execute complete paper detection pipeline.
        
        Pipeline:
        1. Log-envelope: x = log(|y| + ε)
        2. Survival-space projection: z = P_perp @ x
        3. Optional whitening: z_w = L^{-1} @ z
        4. Template projection: s_perp = P_perp @ s
        5. GLRT statistic: T = (s_perp^T z)^2 / ||s_perp||^2
        
        Args:
            y: Complex baseband observation (N,)
            v_rel: Assumed relative velocity for template (m/s)
            whiten_mode: 'none', 'provided' (use L_inv), or 'auto' (estimate)
            L_inv: Whitening matrix if whiten_mode='provided'
            return_intermediate: Return all intermediate results
            
        Returns:
            T: GLRT statistic (if return_intermediate=False)
            results: Dict with all intermediate values (if return_intermediate=True)
        """
        # Step 1: Log-envelope
        x = self.log_envelope_transform(y)
        
        # Step 2: Survival-space projection
        z_perp = self.apply_projection(x)
        
        # Step 3: Optional whitening
        if whiten_mode == 'provided' and L_inv is not None:
            z_final = self.apply_whitening(z_perp, L_inv)
        elif whiten_mode == 'auto':
            warnings.warn("Auto whitening requires H0 samples, using no whitening")
            z_final = z_perp
        else:
            z_final = z_perp
        
        # Step 4: Generate and project template
        s_raw = self.generate_template(v_rel)
        s_perp = self.apply_projection(s_raw)
        
        # Apply same whitening to template if used
        if whiten_mode == 'provided' and L_inv is not None:
            s_final = self.apply_whitening(s_perp, L_inv)
        else:
            s_final = s_perp
        
        # Step 5: GLRT statistic
        T = self.compute_glrt_statistic(z_final, s_final)
        
        if return_intermediate:
            return {
                'T': T,
                'x': x,
                'z_perp': z_perp,
                'z_final': z_final,
                's_raw': s_raw,
                's_perp': s_perp,
                's_final': s_final,
                'template_energy': np.sum(s_final ** 2),
                'correlation': np.dot(s_final, z_final),
                'eta': self.compute_energy_retention(s_raw),
            }
        
        return T
    
    def glrt_scan(
        self,
        z_perp: np.ndarray,
        velocity_grid: np.ndarray
    ) -> np.ndarray:
        """
        Scan GLRT statistic over velocity grid.
        
        Args:
            z_perp: Projected observation (already in survival space)
            velocity_grid: Array of velocities to test
            
        Returns:
            glrt_stats: GLRT statistic for each velocity
        """
        glrt_stats = np.zeros(len(velocity_grid))
        
        for i, v in enumerate(velocity_grid):
            s_raw = self.generate_template(v)
            s_perp = self.P_perp @ s_raw
            
            energy = np.sum(s_perp ** 2)
            if energy < self.delta_reg:
                glrt_stats[i] = 0.0
            else:
                correlation = np.dot(s_perp, z_perp)
                glrt_stats[i] = (correlation ** 2) / energy
        
        return glrt_stats
    
    # =========================================================================
    # SANITY CHECK: Energy Retention Analysis
    # =========================================================================
    
    def sanity_check_energy_retention(
        self,
        f_cut_values: np.ndarray,
        v_rel_values: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Perform energy retention sanity check across parameters.
        
        This is CRITICAL to verify that the projection does not eliminate
        the signal ("self-cancellation" issue).
        
        Args:
            f_cut_values: Array of cutoff frequencies to test
            v_rel_values: Array of velocities to test
            
        Returns:
            results: Dictionary with eta matrix and metadata
        """
        n_fcut = len(f_cut_values)
        n_v = len(v_rel_values)
        
        eta_matrix = np.zeros((n_fcut, n_v))
        
        original_fcut = self.f_cut
        
        for i, f_cut in enumerate(f_cut_values):
            # Rebuild projection for this f_cut
            self.f_cut = f_cut
            self.H, self.k_max = self._build_dct_basis()
            self.P_perp = self._build_projection_matrix()
            
            for j, v_rel in enumerate(v_rel_values):
                # Generate template and compute retention
                s_raw = self.generate_template(v_rel, use_cache=False)
                eta_matrix[i, j] = self.compute_energy_retention(s_raw)
        
        # Restore original
        self.f_cut = original_fcut
        self.H, self.k_max = self._build_dct_basis()
        self.P_perp = self._build_projection_matrix()
        self.clear_template_cache()
        
        return {
            'eta_matrix': eta_matrix,
            'f_cut_values': f_cut_values,
            'v_rel_values': v_rel_values,
        }
    
    # =========================================================================
    # DEPRECATED: Peak/Dip Detection (Debug Only)
    # =========================================================================
    
    def _deprecated_detect_dip_peak(
        self, 
        rx_signal: np.ndarray,
        return_stats: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        [DEPRECATED] Peak detection method - for DEBUG ONLY.
        
        WARNING: This method is NOT the paper pipeline and should NOT be used
        for generating paper figures. It is retained only for debugging.
        
        The paper uses the survival-space projection + GLRT approach.
        """
        warnings.warn(
            "detect_dip_peak is DEPRECATED and not the paper method. "
            "Use paper_pipeline_detect() instead.",
            DeprecationWarning
        )
        
        rx_amp = np.abs(rx_signal)
        rx_mean = np.mean(rx_amp)
        rx_ac = rx_amp - rx_mean
        
        stat_peak = -np.min(rx_ac)
        rx_std = np.std(rx_ac) + self.epsilon_log
        stat_peak_normalized = stat_peak / rx_std
        
        if return_stats:
            return {
                'peak': stat_peak,
                'peak_normalized': stat_peak_normalized,
                'dip_location': np.argmin(rx_ac),
            }
        
        return stat_peak_normalized
    
    # =========================================================================
    # Energy Detector (Baseline Comparison)
    # =========================================================================
    
    def energy_detector(self, z_perp: np.ndarray) -> float:
        """
        Simple energy detector in survival space.
        
        T_ED = ||z_perp||^2
        
        This is the simplest detector and serves as a baseline.
        
        Args:
            z_perp: Projected observation
            
        Returns:
            T_ED: Energy statistic
        """
        return np.sum(z_perp ** 2)


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_detector(
    fs: float,
    N: int,
    cutoff_freq: float = 300.0,
    **kwargs
) -> TerahertzDebrisDetector:
    """
    Create detector instance with specified parameters.
    
    Args:
        fs: Sampling frequency (Hz)
        N: Window length (samples)
        cutoff_freq: DCT cutoff frequency (Hz)
        **kwargs: Additional detector parameters
        
    Returns:
        Detector instance
    """
    return TerahertzDebrisDetector(fs, N, cutoff_freq=cutoff_freq, **kwargs)


def create_detector_from_config(config) -> TerahertzDebrisDetector:
    """
    Create detector from ConfigManager instance.
    
    Args:
        config: ConfigManager instance
        
    Returns:
        Configured detector instance
    """
    return TerahertzDebrisDetector(
        fs=config.sampling.fs,
        N_window=config.sampling.N,
        cutoff_freq=config.survival_space.f_cut,
        L_eff=config.scenario.L_eff,
        fc=config.physics.fc,
        a=config.scenario.a,
        B=config.physics.B,
        N_sub=config.physics.N_sub,
        epsilon_log=config.detection.epsilon_log,
        delta_reg=config.detection.delta_regularization,
    )

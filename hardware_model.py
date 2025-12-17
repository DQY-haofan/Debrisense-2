# ==============================================================================
# hardware_model.py - Hardware Impairments Simulation Engine (Paper Version)
# ==============================================================================
# Version: 2.0 (Parameterized PSD)
#
# Description:
#   IEEE TWC level hardware impairment simulation including:
#   - Saleh PA model (AM-AM, AM-PM)
#   - Colored jitter with parameterized 1/f^α PSD
#   - Phase noise with PLL suppression
#
# Jitter Model (Paper Definition):
#   ξ_J(t) is dimensionless log-gain jitter
#   PSD: S_J(f) ∝ 1/f^α · 1/(1 + (f/f_knee)^p)
#   where α is configurable (baseline α=0.5, comparison α=1.0)
#
# ==============================================================================

import numpy as np
from typing import Dict, Any, Tuple, Optional


class HardwareImpairments:
    """
    Hardware impairment simulation engine.
    
    Implements realistic hardware non-idealities for THz-ISL link simulation:
    - Power Amplifier (Saleh model)
    - Colored jitter (parameterized 1/f^α family)
    - Phase noise (Leeson model with PLL)
    
    Attributes:
        jitter_rms: RMS of log-gain jitter (dimensionless)
        psd_alpha: PSD exponent α (0.5 for baseline, 1.0 for comparison)
        f_knee: Knee frequency for PSD roll-off (Hz)
        roll_off_order: Roll-off filter order p
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hardware model from configuration.
        
        Args:
            config: Dictionary with hardware parameters
        """
        # PA Parameters (Saleh Model)
        self.alpha_a = config.get('alpha_a', 10.127)
        self.beta_a = config.get('beta_a', 5995.0)
        self.alpha_phi = config.get('alpha_phi', 4.0033)
        self.beta_phi = config.get('beta_phi', 9.1040)
        
        # Jitter Parameters (Log-Gain Jitter)
        self.jitter_rms = config.get('jitter_rms', 1.0e-6)  # σ_j (dimensionless)
        self.psd_alpha = config.get('psd_alpha', 0.5)       # α (baseline = 0.5)
        self.f_knee = config.get('f_knee', 200.0)           # f_knee (Hz)
        self.roll_off_order = config.get('roll_off_order', 8)  # p
        
        # Phase Noise Parameters (Leeson Model)
        self.L_1MHz = config.get('L_1MHz', -95.0)           # dBc/Hz at 1 MHz
        self.L_floor = config.get('L_floor', -120.0)        # Noise floor (dBc/Hz)
        self.f_corner = config.get('f_corner', 100e3)       # 1/f corner (Hz)
        self.pll_bw = config.get('pll_bw', 50e3)            # PLL bandwidth (Hz)
    
    # =========================================================================
    # Colored Jitter Generation (Log-Gain Jitter)
    # =========================================================================
    
    def generate_colored_jitter(
        self, 
        N_samples: int, 
        fs: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate colored log-gain jitter with parameterized 1/f^α PSD.
        
        The jitter model follows:
            S_J(f) ∝ 1/f^α · 1/(1 + (f/f_knee)^p)
        
        where:
            α = psd_alpha (configurable, baseline 0.5)
            f_knee = knee frequency for high-frequency roll-off
            p = roll_off_order
        
        Output is dimensionless log-gain jitter ξ_J[n] with RMS = σ_j.
        
        Args:
            N_samples: Number of samples to generate
            fs: Sampling frequency (Hz)
            seed: Random seed for reproducibility
            
        Returns:
            jitter: Dimensionless log-gain jitter (N_samples,)
            
        Note:
            The output is ξ_J(t), where the received signal amplitude is
            modulated as: A_rx = A_tx * exp(ξ_J(t))
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Frequency grid for RFFT
        freqs = np.fft.rfftfreq(N_samples, d=1.0/fs)
        
        # Avoid division by zero at DC
        safe_freqs = np.maximum(freqs, 1e-6)
        
        # Parameterized PSD shape: 1/f^α with high-frequency roll-off
        # S(f) = 1/f^α · 1/(1 + (f/f_knee)^p)
        psd_shape = (1.0 / np.power(safe_freqs, self.psd_alpha)) * \
                    (1.0 / (1.0 + np.power(safe_freqs / self.f_knee, self.roll_off_order)))
        
        # Zero DC component (no static offset)
        psd_shape[0] = 0.0
        
        # Amplitude spectrum
        amplitude = np.sqrt(psd_shape)
        
        # Random phase
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        
        # Generate time-domain signal
        spectrum = amplitude * random_phase
        jitter_raw = np.fft.irfft(spectrum, n=N_samples)
        
        # RMS normalization to achieve target σ_j
        current_rms = np.std(jitter_raw)
        if current_rms > 1e-20:
            jitter_scaled = jitter_raw * (self.jitter_rms / current_rms)
        else:
            jitter_scaled = jitter_raw
        
        return jitter_scaled
    
    def get_jitter_psd(
        self, 
        N_samples: int, 
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get theoretical jitter PSD for plotting/analysis.
        
        Args:
            N_samples: Number of frequency points
            fs: Sampling frequency (Hz)
            
        Returns:
            freqs: Frequency array (Hz)
            psd: Theoretical PSD shape
        """
        freqs = np.fft.rfftfreq(N_samples, d=1.0/fs)
        safe_freqs = np.maximum(freqs, 1e-6)
        
        psd_shape = (1.0 / np.power(safe_freqs, self.psd_alpha)) * \
                    (1.0 / (1.0 + np.power(safe_freqs / self.f_knee, self.roll_off_order)))
        psd_shape[0] = 0.0
        
        # Scale to match RMS
        # For proper normalization, integrate PSD
        df = fs / N_samples
        total_power = np.sum(psd_shape) * df
        if total_power > 0:
            psd_normalized = psd_shape * (self.jitter_rms ** 2) / total_power
        else:
            psd_normalized = psd_shape
        
        return freqs, psd_normalized
    
    # =========================================================================
    # Phase Noise Generation (with PLL suppression)
    # =========================================================================
    
    def generate_phase_noise(
        self, 
        N_samples: int, 
        fs: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate residual phase noise after PLL tracking.
        
        Models Leeson phase noise with PLL high-pass suppression:
            S_φ(f) = S_open_loop(f) · |H_hp(f)|^2
        
        where H_hp is the PLL error transfer function (high-pass).
        
        Args:
            N_samples: Number of samples
            fs: Sampling frequency (Hz)
            seed: Random seed
            
        Returns:
            theta: Phase noise samples (radians)
        """
        if seed is not None:
            np.random.seed(seed)
        
        freqs = np.fft.rfftfreq(N_samples, d=1.0/fs)
        safe_freqs = np.maximum(freqs, 1.0)
        
        # Convert dB values to linear
        S_floor = 10 ** (self.L_floor / 10.0)
        S_1MHz = 10 ** (self.L_1MHz / 10.0)
        
        # Leeson model: white FM + flicker FM + floor
        S_white_fm = S_1MHz * (1e6 / safe_freqs) ** 2
        S_flicker = S_white_fm * (self.f_corner / safe_freqs)
        
        psd_open_loop = S_white_fm + S_floor + \
                        np.where(safe_freqs < self.f_corner, S_flicker, 0)
        
        # PLL suppression: 2nd-order high-pass
        # |H_hp(f)|^2 = (f/f_pll)^4 / (1 + (f/f_pll)^4)
        pll_suppression = np.power(safe_freqs / self.pll_bw, 4) / \
                          (1 + np.power(safe_freqs / self.pll_bw, 4))
        
        # Residual PSD after PLL tracking
        psd_residual = psd_open_loop * pll_suppression
        
        # Generate phase noise via IFFT
        amplitude = np.sqrt(psd_residual * fs * N_samples / 2)
        amplitude[0] = 0.0  # No DC
        
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        theta_pn = np.fft.irfft(amplitude * phase, n=N_samples)
        
        return theta_pn
    
    # =========================================================================
    # Power Amplifier Model (Saleh)
    # =========================================================================
    
    def apply_saleh_pa(
        self, 
        input_signal: np.ndarray, 
        ibo_dB: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply Saleh PA model with specified input back-off.
        
        AM-AM: A_out = (α_a · r) / (1 + β_a · r²)
        AM-PM: φ_out = (α_φ · r²) / (1 + β_φ · r²)
        
        Args:
            input_signal: Complex input signal (normalized)
            ibo_dB: Input back-off in dB
            
        Returns:
            output: Complex output signal
            scr: Signal-to-compression ratio
            v_op_peak: Operating voltage peak
        """
        # Compute operating point
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)
        v_op_peak = v_sat_phys * (10 ** (-ibo_dB / 20.0))
        
        # Input envelope and phase
        r_norm = np.abs(input_signal)
        r_phys = r_norm * v_op_peak
        phase_in = np.angle(input_signal)
        
        # AM-AM conversion
        denominator_am = 1.0 + self.beta_a * (r_phys ** 2)
        a_out_phys = (self.alpha_a * r_phys) / denominator_am
        
        # AM-PM conversion
        denominator_pm = 1.0 + self.beta_phi * (r_phys ** 2)
        phi_distortion = (self.alpha_phi * (r_phys ** 2)) / denominator_pm
        
        # Output signal
        output_phys = a_out_phys * np.exp(1j * (phase_in + phi_distortion))
        
        # Signal-to-compression ratio
        beta_r2 = self.beta_a * (r_phys ** 2)
        scr = (1.0 - beta_r2) / (1.0 + beta_r2 + 1e-12)
        
        return output_phys, scr, v_op_peak
    
    def get_pa_curves(
        self, 
        n_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get PA AM-AM and SCR curves for characterization.
        
        Args:
            n_points: Number of curve points
            
        Returns:
            pin_db: Input power (dB)
            am_am_norm: Normalized output amplitude
            scr: Signal-to-compression ratio
        """
        r_norm = np.linspace(0, 2.0, n_points)
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)
        r_phys = r_norm * v_sat_phys
        
        # AM-AM
        denom = 1.0 + self.beta_a * (r_phys ** 2)
        a_out = (self.alpha_a * r_phys) / denom
        a_out_norm = a_out / (self.alpha_a * v_sat_phys)
        
        # SCR
        beta_r2 = self.beta_a * (r_phys ** 2)
        scr = (1.0 - beta_r2) / (1.0 + beta_r2)
        
        # Input power in dB
        pin_db = 20 * np.log10(r_norm + 1e-9)
        
        return pin_db, a_out_norm, scr
    
    # =========================================================================
    # Complete Signal Corruption
    # =========================================================================
    
    def corrupt_signal(
        self,
        clean_signal: np.ndarray,
        fs: float,
        ibo_dB: float = 10.0,
        include_jitter: bool = True,
        include_phase_noise: bool = True,
        include_pa: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply all hardware impairments to clean signal.
        
        Args:
            clean_signal: Clean complex baseband signal
            fs: Sampling frequency (Hz)
            ibo_dB: PA input back-off (dB)
            include_jitter: Apply colored jitter
            include_phase_noise: Apply phase noise
            include_pa: Apply PA nonlinearity
            seed: Random seed
            
        Returns:
            corrupted: Corrupted signal
            diagnostics: Dictionary with impairment diagnostics
        """
        N = len(clean_signal)
        signal = clean_signal.copy()
        diagnostics = {}
        
        if seed is not None:
            np.random.seed(seed)
        
        # Apply jitter (amplitude modulation)
        if include_jitter:
            jitter = self.generate_colored_jitter(N, fs)
            jitter_multiplier = np.exp(jitter)  # Log-gain -> linear
            signal = signal * jitter_multiplier
            diagnostics['jitter'] = jitter
            diagnostics['jitter_rms'] = np.std(jitter)
        
        # Apply phase noise
        if include_phase_noise:
            phase_noise = self.generate_phase_noise(N, fs)
            pn_multiplier = np.exp(1j * phase_noise)
            signal = signal * pn_multiplier
            diagnostics['phase_noise'] = phase_noise
            diagnostics['pn_rms'] = np.std(phase_noise)
        
        # Apply PA
        if include_pa:
            signal, scr, v_op = self.apply_saleh_pa(signal, ibo_dB)
            diagnostics['scr'] = scr
            diagnostics['mean_scr'] = np.mean(scr)
            diagnostics['v_op_peak'] = v_op
        
        return signal, diagnostics


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_hardware_from_config(config) -> HardwareImpairments:
    """
    Create HardwareImpairments from ConfigManager.
    
    Args:
        config: ConfigManager instance
        
    Returns:
        Configured HardwareImpairments instance
    """
    return HardwareImpairments(config.get_hardware_config())


def create_hardware_with_alpha(
    base_config: Dict[str, Any],
    alpha: float
) -> HardwareImpairments:
    """
    Create hardware model with specific PSD alpha for comparison.
    
    Args:
        base_config: Base configuration dictionary
        alpha: PSD exponent to use
        
    Returns:
        HardwareImpairments with specified alpha
    """
    modified_config = base_config.copy()
    modified_config['psd_alpha'] = alpha
    return HardwareImpairments(modified_config)


# ==============================================================================
# Sensitivity Analysis Functions
# ==============================================================================

def compare_alpha_sensitivity(
    base_config: Dict[str, Any],
    alpha_values: list,
    N_samples: int,
    fs: float,
    n_trials: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compare jitter statistics for different α values.
    
    Args:
        base_config: Base hardware configuration
        alpha_values: List of α values to compare
        N_samples: Number of samples per realization
        fs: Sampling frequency
        n_trials: Number of MC trials
        
    Returns:
        Dictionary with comparison statistics
    """
    results = {
        'alpha': np.array(alpha_values),
        'mean_rms': np.zeros(len(alpha_values)),
        'std_rms': np.zeros(len(alpha_values)),
    }
    
    for i, alpha in enumerate(alpha_values):
        hw = create_hardware_with_alpha(base_config, alpha)
        
        rms_values = []
        for trial in range(n_trials):
            jitter = hw.generate_colored_jitter(N_samples, fs, seed=trial)
            rms_values.append(np.std(jitter))
        
        results['mean_rms'][i] = np.mean(rms_values)
        results['std_rms'][i] = np.std(rms_values)
    
    return results

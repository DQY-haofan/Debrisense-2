# ==============================================================================
# estimator.py - 2D Maximum Likelihood Grid Search Estimator
# ==============================================================================
# Version: 1.0
# 
# Description:
#   Implements 2D ML grid search for joint estimation of:
#   - v_rel: Relative velocity (m/s)
#   - t0: Crossing time offset (s)
#
#   Uses FFT-based correlation for efficient computation across time shifts.
#
# Usage:
#   estimator = MLGridEstimator(detector, config)
#   v_hat, t0_hat, score_map = estimator.estimate(y)
# ==============================================================================

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import warnings

from detector import TerahertzDebrisDetector


@dataclass
class EstimationResult:
    """Result of 2D ML estimation"""
    v_hat: float              # Estimated velocity (m/s)
    t0_hat: float             # Estimated time offset (s)
    v_hat_refined: float      # Parabolic-refined velocity
    t0_hat_refined: float     # Parabolic-refined time
    max_statistic: float      # Maximum GLRT statistic
    score_map: np.ndarray     # Full 2D score map
    v_grid: np.ndarray        # Velocity grid used
    t0_grid: np.ndarray       # Time offset grid used
    v_idx: int                # Index of max in velocity
    t0_idx: int               # Index of max in time


class MLGridEstimator:
    """
    2D Maximum Likelihood Grid Search Estimator
    
    Performs joint estimation of velocity and crossing time by:
    1. Generating template bank for velocity grid
    2. Using FFT-based correlation for efficient time shift search
    3. Computing GLRT statistic over full (v, t0) grid
    4. Optionally refining with parabolic interpolation
    
    Attributes:
        detector: TerahertzDebrisDetector instance
        v_grid: Velocity search grid
        t0_grid: Time offset search grid
        use_fft: Whether to use FFT for time correlation
    """
    
    def __init__(
        self,
        detector: TerahertzDebrisDetector,
        v_range: Tuple[float, float] = (13500.0, 16500.0),
        v_points: int = 61,
        t0_range: Tuple[float, float] = (-0.005, 0.005),
        t0_points: int = 101,
        use_fft: bool = True,
    ):
        """
        Initialize estimator.
        
        Args:
            detector: Configured detector instance
            v_range: (v_min, v_max) velocity search range (m/s)
            v_points: Number of velocity grid points
            t0_range: (t0_min, t0_max) time offset range (s)
            t0_points: Number of time grid points
            use_fft: Use FFT for fast correlation computation
        """
        self.detector = detector
        self.fs = detector.fs
        self.N = detector.N
        
        # Build grids
        self.v_grid = np.linspace(v_range[0], v_range[1], v_points)
        self.t0_grid = np.linspace(t0_range[0], t0_range[1], t0_points)
        
        self.use_fft = use_fft
        
        # Pre-compute projected templates for velocity grid
        self._template_bank: Dict[float, np.ndarray] = {}
        self._template_energy: Dict[float, float] = {}
        
    def _precompute_templates(self):
        """Pre-compute and project all templates for velocity grid"""
        for v in self.v_grid:
            if v not in self._template_bank:
                s_raw = self.detector.generate_template(v)
                s_perp = self.detector.P_perp @ s_raw
                self._template_bank[v] = s_perp
                self._template_energy[v] = np.sum(s_perp ** 2) + 1e-20
    
    def _get_template(self, v: float) -> Tuple[np.ndarray, float]:
        """Get projected template and its energy"""
        if v not in self._template_bank:
            s_raw = self.detector.generate_template(v)
            s_perp = self.detector.P_perp @ s_raw
            self._template_bank[v] = s_perp
            self._template_energy[v] = np.sum(s_perp ** 2) + 1e-20
        return self._template_bank[v], self._template_energy[v]
    
    def _apply_time_shift(
        self, 
        template: np.ndarray, 
        shift_samples: int
    ) -> np.ndarray:
        """
        Apply time shift to template.
        
        Args:
            template: Template signal
            shift_samples: Number of samples to shift (positive = delay)
            
        Returns:
            Shifted template with zero-padding
        """
        shifted = np.roll(template, shift_samples)
        
        # Zero out wrapped samples
        if shift_samples > 0:
            shifted[:shift_samples] = 0
        elif shift_samples < 0:
            shifted[shift_samples:] = 0
            
        return shifted
    
    def _fft_correlation(
        self, 
        z_perp: np.ndarray, 
        s_perp: np.ndarray
    ) -> np.ndarray:
        """
        Compute correlation using FFT for all time shifts.
        
        correlation[τ] = Σ_n z_perp[n] * s_perp[n - τ]
        
        Args:
            z_perp: Projected observation
            s_perp: Projected template
            
        Returns:
            correlation: Correlation for all shifts
        """
        # Zero-pad to avoid circular correlation artifacts
        n_fft = 2 * self.N
        
        Z = np.fft.fft(z_perp, n_fft)
        S = np.fft.fft(s_perp, n_fft)
        
        # Cross-correlation
        corr = np.fft.ifft(Z * np.conj(S))
        
        # Return only valid part
        return np.real(corr[:self.N])
    
    def estimate(
        self,
        y: np.ndarray,
        refine: bool = True,
        return_full: bool = True
    ) -> EstimationResult:
        """
        Perform 2D ML grid search estimation.
        
        Args:
            y: Complex baseband observation (N,)
            refine: Apply parabolic refinement
            return_full: Include full score map in result
            
        Returns:
            EstimationResult with estimated parameters
        """
        # Step 1: Log-envelope and projection
        x = self.detector.log_envelope_transform(y)
        z_perp = self.detector.apply_projection(x)
        
        # Step 2: Initialize score map
        n_v = len(self.v_grid)
        n_t = len(self.t0_grid)
        score_map = np.zeros((n_v, n_t))
        
        # Convert t0 grid to sample shifts
        t0_samples = np.round(self.t0_grid * self.fs).astype(int)
        
        # Step 3: Compute GLRT over grid
        if self.use_fft:
            # FFT-based approach: compute full correlation once per velocity
            for i, v in enumerate(self.v_grid):
                s_perp, s_energy = self._get_template(v)
                
                # Full correlation using FFT
                full_corr = self._fft_correlation(z_perp, s_perp)
                
                # Sample correlation at t0 grid points
                for j, shift in enumerate(t0_samples):
                    # Get correlation at this shift
                    idx = shift % self.N
                    corr = full_corr[idx]
                    
                    # GLRT statistic (need to adjust energy for shift)
                    score_map[i, j] = (corr ** 2) / s_energy
        else:
            # Direct approach: compute each (v, t0) separately
            for i, v in enumerate(self.v_grid):
                s_perp, s_energy = self._get_template(v)
                
                for j, shift in enumerate(t0_samples):
                    s_shifted = self._apply_time_shift(s_perp, shift)
                    shifted_energy = np.sum(s_shifted ** 2)
                    
                    if shifted_energy > 1e-20:
                        corr = np.dot(z_perp, s_shifted)
                        score_map[i, j] = (corr ** 2) / shifted_energy
                    else:
                        score_map[i, j] = 0.0
        
        # Step 4: Find maximum
        max_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
        v_idx, t0_idx = max_idx
        
        v_hat = self.v_grid[v_idx]
        t0_hat = self.t0_grid[t0_idx]
        max_stat = score_map[v_idx, t0_idx]
        
        # Step 5: Parabolic refinement
        v_hat_refined = v_hat
        t0_hat_refined = t0_hat
        
        if refine:
            v_hat_refined = self._parabolic_refine_1d(
                self.v_grid, score_map[:, t0_idx], v_idx
            )
            t0_hat_refined = self._parabolic_refine_1d(
                self.t0_grid, score_map[v_idx, :], t0_idx
            )
        
        return EstimationResult(
            v_hat=v_hat,
            t0_hat=t0_hat,
            v_hat_refined=v_hat_refined,
            t0_hat_refined=t0_hat_refined,
            max_statistic=max_stat,
            score_map=score_map if return_full else np.array([]),
            v_grid=self.v_grid,
            t0_grid=self.t0_grid,
            v_idx=v_idx,
            t0_idx=t0_idx,
        )
    
    def _parabolic_refine_1d(
        self,
        grid: np.ndarray,
        values: np.ndarray,
        peak_idx: int
    ) -> float:
        """
        Parabolic interpolation for sub-grid refinement.
        
        Given three points around peak, fit parabola and find maximum.
        
        Args:
            grid: Parameter grid
            values: Function values at grid points
            peak_idx: Index of discrete maximum
            
        Returns:
            Refined parameter estimate
        """
        if peak_idx <= 0 or peak_idx >= len(grid) - 1:
            return grid[peak_idx]
        
        alpha = values[peak_idx - 1]
        beta = values[peak_idx]
        gamma = values[peak_idx + 1]
        
        denom = alpha - 2 * beta + gamma
        if abs(denom) < 1e-10:
            return grid[peak_idx]
        
        p = 0.5 * (alpha - gamma) / denom
        delta = grid[1] - grid[0]
        
        return grid[peak_idx] + p * delta
    
    def estimate_velocity_only(
        self,
        y: np.ndarray,
        refine: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate velocity only (assume t0 = 0).
        
        Faster than full 2D search when timing is known.
        
        Args:
            y: Complex baseband observation
            refine: Apply parabolic refinement
            
        Returns:
            v_hat: Estimated velocity
            glrt_stats: GLRT statistics for each velocity
        """
        x = self.detector.log_envelope_transform(y)
        z_perp = self.detector.apply_projection(x)
        
        glrt_stats = np.zeros(len(self.v_grid))
        
        for i, v in enumerate(self.v_grid):
            s_perp, s_energy = self._get_template(v)
            corr = np.dot(z_perp, s_perp)
            glrt_stats[i] = (corr ** 2) / s_energy
        
        peak_idx = np.argmax(glrt_stats)
        v_hat = self.v_grid[peak_idx]
        
        if refine:
            v_hat = self._parabolic_refine_1d(self.v_grid, glrt_stats, peak_idx)
        
        return v_hat, glrt_stats
    
    def estimate_timing_only(
        self,
        y: np.ndarray,
        v_rel: float,
        refine: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate timing only (given velocity).
        
        Args:
            y: Complex baseband observation
            v_rel: Known velocity
            refine: Apply parabolic refinement
            
        Returns:
            t0_hat: Estimated time offset
            corr_stats: Correlation statistics for each time
        """
        x = self.detector.log_envelope_transform(y)
        z_perp = self.detector.apply_projection(x)
        
        s_raw = self.detector.generate_template(v_rel)
        s_perp = self.detector.P_perp @ s_raw
        s_energy = np.sum(s_perp ** 2) + 1e-20
        
        t0_samples = np.round(self.t0_grid * self.fs).astype(int)
        corr_stats = np.zeros(len(self.t0_grid))
        
        if self.use_fft:
            full_corr = self._fft_correlation(z_perp, s_perp)
            for j, shift in enumerate(t0_samples):
                idx = shift % self.N
                corr_stats[j] = (full_corr[idx] ** 2) / s_energy
        else:
            for j, shift in enumerate(t0_samples):
                s_shifted = self._apply_time_shift(s_perp, shift)
                corr = np.dot(z_perp, s_shifted)
                corr_stats[j] = (corr ** 2) / s_energy
        
        peak_idx = np.argmax(corr_stats)
        t0_hat = self.t0_grid[peak_idx]
        
        if refine:
            t0_hat = self._parabolic_refine_1d(self.t0_grid, corr_stats, peak_idx)
        
        return t0_hat, corr_stats


class MCEstimationAnalyzer:
    """
    Monte Carlo analysis for estimation performance.
    
    Computes:
    - RMSE for velocity and timing estimates
    - Bias analysis
    - Error histograms
    - Comparison between ideal and hardware cases
    """
    
    def __init__(self, estimator: MLGridEstimator):
        self.estimator = estimator
    
    def run_mc_trials(
        self,
        signal_generator,
        true_v: float,
        true_t0: float,
        n_trials: int,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo trials for estimation analysis.
        
        Args:
            signal_generator: Callable that generates noisy observation
            true_v: True velocity
            true_t0: True time offset
            n_trials: Number of MC trials
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with MC analysis results
        """
        np.random.seed(seed)
        
        v_estimates = np.zeros(n_trials)
        t0_estimates = np.zeros(n_trials)
        v_estimates_refined = np.zeros(n_trials)
        t0_estimates_refined = np.zeros(n_trials)
        max_stats = np.zeros(n_trials)
        
        for i in range(n_trials):
            # Generate observation
            y = signal_generator(seed + i)
            
            # Estimate
            result = self.estimator.estimate(y, refine=True, return_full=False)
            
            v_estimates[i] = result.v_hat
            t0_estimates[i] = result.t0_hat
            v_estimates_refined[i] = result.v_hat_refined
            t0_estimates_refined[i] = result.t0_hat_refined
            max_stats[i] = result.max_statistic
        
        # Compute statistics
        v_error = v_estimates_refined - true_v
        t0_error = t0_estimates_refined - true_t0
        
        return {
            'v_estimates': v_estimates_refined,
            't0_estimates': t0_estimates_refined,
            'v_error': v_error,
            't0_error': t0_error,
            'v_rmse': np.sqrt(np.mean(v_error ** 2)),
            't0_rmse': np.sqrt(np.mean(t0_error ** 2)),
            'v_bias': np.mean(v_error),
            't0_bias': np.mean(t0_error),
            'v_std': np.std(v_error),
            't0_std': np.std(t0_error),
            'max_stats': max_stats,
            'true_v': true_v,
            'true_t0': true_t0,
            'n_trials': n_trials,
        }


def create_estimator_from_config(
    detector: TerahertzDebrisDetector,
    config
) -> MLGridEstimator:
    """
    Create estimator from ConfigManager.
    
    Args:
        detector: Configured detector
        config: ConfigManager instance
        
    Returns:
        Configured estimator
    """
    est_cfg = config.estimation
    return MLGridEstimator(
        detector=detector,
        v_range=tuple(est_cfg.v_grid_range),
        v_points=est_cfg.v_grid_points,
        t0_range=tuple(est_cfg.t0_grid_range),
        t0_points=est_cfg.t0_grid_points,
        use_fft=est_cfg.use_fft_correlation,
    )

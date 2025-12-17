# ==============================================================================
# config_manager.py - Single Source of Truth Configuration Manager
# ==============================================================================
# Version: 1.0
# Description: Centralized configuration management for THz-ISL project
#              Ensures all parameters come from config file, no scattered constants
# ==============================================================================

import yaml
import json
import os
import hashlib
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class PhysicsConfig:
    """Physics parameters"""
    fc: float = 300e9
    B: float = 10e9
    c: float = 299792458.0
    N_sub: int = 32


@dataclass
class ScenarioConfig:
    """Scenario parameters"""
    L_eff: float = 50e3
    a: float = 0.05
    v_default: float = 15000.0


@dataclass
class SamplingConfig:
    """Sampling parameters"""
    fs: float = 200e3
    T_span: float = 0.02
    
    @property
    def N(self) -> int:
        """Number of samples (computed)"""
        return int(self.fs * self.T_span)


@dataclass
class SurvivalSpaceConfig:
    """Survival space / DCT projection parameters"""
    f_cut: float = 300.0


@dataclass
class JitterConfig:
    """Jitter model parameters"""
    model: str = "log_gain"
    sigma_j: float = 1e-6
    psd_alpha: float = 0.5
    f_knee: float = 200.0
    roll_off_order: int = 8


@dataclass
class HardwareConfig:
    """Hardware impairment parameters"""
    alpha_a: float = 10.127
    beta_a: float = 5995.0
    alpha_phi: float = 4.0033
    beta_phi: float = 9.1040
    L_1MHz: float = -95.0
    L_floor: float = -120.0
    f_corner: float = 100e3
    pll_bw: float = 50e3


@dataclass
class DetectionConfig:
    """Detection parameters"""
    delta_regularization: float = 1e-20
    epsilon_log: float = 1e-12
    whitening_enabled: bool = False
    whitening_diagonal_loading: float = 1e-10


@dataclass
class EstimationConfig:
    """2D ML estimation parameters"""
    v_grid_range: List[float] = field(default_factory=lambda: [13500.0, 16500.0])
    v_grid_points: int = 61
    t0_grid_range: List[float] = field(default_factory=lambda: [-0.005, 0.005])
    t0_grid_points: int = 101
    use_fft_correlation: bool = True


@dataclass  
class ReproducibilityConfig:
    """Reproducibility settings"""
    master_seed: int = 42
    numpy_seed_offset: int = 1000
    mc_seed_method: str = "deterministic"


class ConfigManager:
    """
    Centralized configuration manager for THz-ISL project.
    
    Features:
    - Single source of truth from YAML/JSON config
    - Figure-specific parameter overrides
    - Config snapshot export for reproducibility
    - Automatic consistency validation
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML/JSON config file. If None, uses default.
        """
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}
        self._load_time = None
        self._config_hash = None
        
        # Initialize sub-configs
        self.physics = PhysicsConfig()
        self.scenario = ScenarioConfig()
        self.sampling = SamplingConfig()
        self.survival_space = SurvivalSpaceConfig()
        self.jitter = JitterConfig()
        self.hardware = HardwareConfig()
        self.detection = DetectionConfig()
        self.estimation = EstimationConfig()
        self.reproducibility = ReproducibilityConfig()
        
        self.mode = "paper"
        self.figures_config: Dict[str, Dict] = {}
        self.sanity_check_config: Dict[str, Any] = {}
        self.output_config: Dict[str, Any] = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> 'ConfigManager':
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Self for method chaining
        """
        self.config_path = config_path
        self._load_time = datetime.now().isoformat()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.raw_config = yaml.safe_load(f)
            else:
                self.raw_config = json.load(f)
        
        # Compute config hash for tracking
        config_str = json.dumps(self.raw_config, sort_keys=True)
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Parse into structured configs
        self._parse_config()
        
        return self
    
    def _parse_config(self):
        """Parse raw config into structured dataclasses"""
        cfg = self.raw_config
        
        self.mode = cfg.get('mode', 'paper')
        
        # Physics
        if 'physics' in cfg:
            p = cfg['physics']
            self.physics = PhysicsConfig(
                fc=float(p.get('fc', 300e9)),
                B=float(p.get('B', 10e9)),
                c=float(p.get('c', 299792458.0)),
                N_sub=int(p.get('N_sub', 32))
            )
        
        # Scenario
        if 'scenario' in cfg:
            s = cfg['scenario']
            self.scenario = ScenarioConfig(
                L_eff=float(s.get('L_eff', 50e3)),
                a=float(s.get('a', 0.05)),
                v_default=float(s.get('v_default', 15000.0))
            )
        
        # Sampling
        if 'sampling' in cfg:
            s = cfg['sampling']
            self.sampling = SamplingConfig(
                fs=float(s.get('fs', 200e3)),
                T_span=float(s.get('T_span', 0.02))
            )
        
        # Survival space
        if 'survival_space' in cfg:
            ss = cfg['survival_space']
            self.survival_space = SurvivalSpaceConfig(
                f_cut=float(ss.get('f_cut', 300.0))
            )
        
        # Jitter
        if 'jitter' in cfg:
            j = cfg['jitter']
            self.jitter = JitterConfig(
                model=j.get('model', 'log_gain'),
                sigma_j=float(j.get('sigma_j', 1e-6)),
                psd_alpha=float(j.get('psd_alpha', 0.5)),
                f_knee=float(j.get('f_knee', 200.0)),
                roll_off_order=int(j.get('roll_off_order', 8))
            )
        
        # Hardware
        if 'hardware' in cfg:
            h = cfg['hardware']
            self.hardware = HardwareConfig(
                alpha_a=float(h.get('alpha_a', 10.127)),
                beta_a=float(h.get('beta_a', 5995.0)),
                alpha_phi=float(h.get('alpha_phi', 4.0033)),
                beta_phi=float(h.get('beta_phi', 9.1040)),
                L_1MHz=float(h.get('L_1MHz', -95.0)),
                L_floor=float(h.get('L_floor', -120.0)),
                f_corner=float(h.get('f_corner', 100e3)),
                pll_bw=float(h.get('pll_bw', 50e3))
            )
        
        # Detection
        if 'detection' in cfg:
            d = cfg['detection']
            self.detection = DetectionConfig(
                delta_regularization=float(d.get('delta_regularization', 1e-20)),
                epsilon_log=float(d.get('epsilon_log', 1e-12)),
                whitening_enabled=d.get('whitening_enabled', False),
                whitening_diagonal_loading=float(d.get('whitening_diagonal_loading', 1e-10))
            )
        
        # Estimation
        if 'estimation' in cfg:
            e = cfg['estimation']
            self.estimation = EstimationConfig(
                v_grid_range=list(e.get('v_grid_range', [13500.0, 16500.0])),
                v_grid_points=int(e.get('v_grid_points', 61)),
                t0_grid_range=list(e.get('t0_grid_range', [-0.005, 0.005])),
                t0_grid_points=int(e.get('t0_grid_points', 101)),
                use_fft_correlation=e.get('use_fft_correlation', True)
            )
        
        # Reproducibility
        if 'reproducibility' in cfg:
            r = cfg['reproducibility']
            self.reproducibility = ReproducibilityConfig(
                master_seed=int(r.get('master_seed', 42)),
                numpy_seed_offset=int(r.get('numpy_seed_offset', 1000)),
                mc_seed_method=r.get('mc_seed_method', 'deterministic')
            )
        
        # Figures config
        self.figures_config = cfg.get('figures', {})
        
        # Sanity check config
        self.sanity_check_config = cfg.get('sanity_check', {})
        
        # Output config
        self.output_config = cfg.get('output', {})
    
    def get_figure_config(self, fig_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific figure with overrides applied.
        
        Args:
            fig_name: Figure name (e.g., 'fig6', 'fig7')
            
        Returns:
            Dictionary with merged baseline + figure-specific config
        """
        base_config = self.to_dict()
        fig_config = self.figures_config.get(fig_name, {})
        
        # Apply changes from figure config
        changes = fig_config.get('changes', {})
        for key, value in changes.items():
            base_config[key] = value
        
        # Add figure-specific settings
        base_config['fig_settings'] = fig_config
        
        return base_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current config to flat dictionary for legacy compatibility"""
        return {
            # Physics
            'fc': self.physics.fc,
            'B': self.physics.B,
            'c': self.physics.c,
            'N_sub': self.physics.N_sub,
            
            # Scenario
            'L_eff': self.scenario.L_eff,
            'a': self.scenario.a,
            'v_default': self.scenario.v_default,
            
            # Sampling
            'fs': self.sampling.fs,
            'T_span': self.sampling.T_span,
            'N': self.sampling.N,
            
            # Survival space
            'f_cut': self.survival_space.f_cut,
            
            # Jitter
            'jitter_model': self.jitter.model,
            'sigma_j': self.jitter.sigma_j,
            'psd_alpha': self.jitter.psd_alpha,
            'f_knee': self.jitter.f_knee,
            'roll_off_order': self.jitter.roll_off_order,
            
            # Hardware
            'alpha_a': self.hardware.alpha_a,
            'beta_a': self.hardware.beta_a,
            'alpha_phi': self.hardware.alpha_phi,
            'beta_phi': self.hardware.beta_phi,
            'L_1MHz': self.hardware.L_1MHz,
            'L_floor': self.hardware.L_floor,
            'f_corner': self.hardware.f_corner,
            'pll_bw': self.hardware.pll_bw,
            
            # Detection
            'delta_reg': self.detection.delta_regularization,
            'epsilon_log': self.detection.epsilon_log,
            'whitening_enabled': self.detection.whitening_enabled,
            
            # Mode
            'mode': self.mode,
        }
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware config dictionary for HardwareImpairments class"""
        return {
            'jitter_rms': self.jitter.sigma_j,
            'f_knee': self.jitter.f_knee,
            'psd_alpha': self.jitter.psd_alpha,
            'roll_off_order': self.jitter.roll_off_order,
            'beta_a': self.hardware.beta_a,
            'alpha_a': self.hardware.alpha_a,
            'alpha_phi': self.hardware.alpha_phi,
            'beta_phi': self.hardware.beta_phi,
            'L_1MHz': self.hardware.L_1MHz,
            'L_floor': self.hardware.L_floor,
            'f_corner': self.hardware.f_corner,
            'pll_bw': self.hardware.pll_bw,
        }
    
    def get_physics_config(self) -> Dict[str, Any]:
        """Get physics config dictionary for DiffractionChannel class"""
        return {
            'fc': self.physics.fc,
            'B': self.physics.B,
            'L_eff': self.scenario.L_eff,
            'a': self.scenario.a,
            'v_rel': self.scenario.v_default,
        }
    
    def get_detector_config(self) -> Dict[str, Any]:
        """Get detector config dictionary for TerahertzDebrisDetector class"""
        return {
            'cutoff_freq': self.survival_space.f_cut,
            'L_eff': self.scenario.L_eff,
            'fc': self.physics.fc,
            'a': self.scenario.a,
            'B': self.physics.B,
            'N_sub': self.physics.N_sub,
        }
    
    def export_snapshot(self, output_path: str):
        """
        Export current configuration as snapshot for reproducibility.
        
        Args:
            output_path: Path to save snapshot YAML
        """
        snapshot = {
            'snapshot_time': datetime.now().isoformat(),
            'source_config': self.config_path,
            'config_hash': self._config_hash,
            'config': self.raw_config,
        }
        
        # Try to get git info
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            snapshot['git_commit'] = git_hash
        except:
            snapshot['git_commit'] = 'unavailable'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)
    
    def get_seed_for_trial(self, fig_name: str, trial_idx: int) -> int:
        """
        Get deterministic seed for a specific trial.
        
        Args:
            fig_name: Figure name
            trial_idx: Trial index
            
        Returns:
            Deterministic seed value
        """
        # Create hash-based seed from fig_name + trial_idx + master_seed
        seed_str = f"{self.reproducibility.master_seed}_{fig_name}_{trial_idx}"
        seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return seed_hash % (2**31 - 1)
    
    def validate(self) -> List[str]:
        """
        Validate configuration for consistency.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check mode
        if self.mode not in ['paper', 'debug']:
            errors.append(f"Invalid mode '{self.mode}', must be 'paper' or 'debug'")
        
        # Check physics constraints
        if self.physics.fc <= 0:
            errors.append(f"Invalid fc={self.physics.fc}, must be positive")
        if self.physics.B <= 0:
            errors.append(f"Invalid B={self.physics.B}, must be positive")
        
        # Check sampling constraints
        if self.sampling.fs <= 0:
            errors.append(f"Invalid fs={self.sampling.fs}, must be positive")
        if self.sampling.T_span <= 0:
            errors.append(f"Invalid T_span={self.sampling.T_span}, must be positive")
        
        # Check survival space
        nyquist = self.sampling.fs / 2
        if self.survival_space.f_cut >= nyquist:
            errors.append(f"f_cut={self.survival_space.f_cut} must be < Nyquist={nyquist}")
        
        # Check scenario
        if self.scenario.L_eff <= 0:
            errors.append(f"Invalid L_eff={self.scenario.L_eff}, must be positive")
        if self.scenario.a <= 0:
            errors.append(f"Invalid a={self.scenario.a}, must be positive")
        
        return errors
    
    def print_summary(self):
        """Print configuration summary to console"""
        print("=" * 70)
        print("THz-ISL Configuration Summary")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Config Hash: {self._config_hash}")
        print("-" * 70)
        print("Physics:")
        print(f"  fc = {self.physics.fc/1e9:.1f} GHz, B = {self.physics.B/1e9:.1f} GHz")
        print(f"  N_sub = {self.physics.N_sub}")
        print("Scenario:")
        print(f"  L_eff = {self.scenario.L_eff/1e3:.1f} km, a = {self.scenario.a*100:.1f} cm")
        print(f"  v_default = {self.scenario.v_default:.0f} m/s")
        print("Sampling:")
        print(f"  fs = {self.sampling.fs/1e3:.1f} kHz, T_span = {self.sampling.T_span*1e3:.1f} ms")
        print(f"  N = {self.sampling.N} samples")
        print("Survival Space:")
        print(f"  f_cut = {self.survival_space.f_cut:.1f} Hz")
        print("Jitter:")
        print(f"  model = {self.jitter.model}, σ_j = {self.jitter.sigma_j:.2e}")
        print(f"  PSD: 1/f^{self.jitter.psd_alpha}, f_knee = {self.jitter.f_knee:.1f} Hz")
        print("=" * 70)


def load_config(config_path: str = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Loaded ConfigManager instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            'config/paper_baseline.yaml',
            '../config/paper_baseline.yaml',
            'paper_baseline.yaml',
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise FileNotFoundError("Could not find config file in default locations")
    
    return ConfigManager(config_path)


# Global config instance (lazy loaded)
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global config instance, loading default if needed"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: ConfigManager):
    """Set global config instance"""
    global _global_config
    _global_config = config

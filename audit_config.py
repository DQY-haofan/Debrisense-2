#!/usr/bin/env python3
# ==============================================================================
# audit_config.py - Configuration Consistency Audit Tool
# ==============================================================================
# Version: 1.0
# Description: Validates configuration consistency and prints parameter table
#              Returns exit code != 0 if any issues found (CI-style failure)
# Usage: python audit_config.py [config_path]
# ==============================================================================

import sys
import os
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager, load_config


class ConfigAuditor:
    """
    Configuration auditor for THz-ISL project.
    
    Performs comprehensive validation and generates audit report.
    """
    
    # Critical parameters that MUST be checked
    CRITICAL_PARAMS = [
        ('L_eff', 'm', 'Effective link length'),
        ('a', 'm', 'Debris radius'),
        ('fs', 'Hz', 'Sampling frequency'),
        ('T_span', 's', 'Observation window'),
        ('f_cut', 'Hz', 'DCT cutoff frequency'),
        ('psd_alpha', '-', 'Jitter PSD exponent'),
        ('sigma_j', '-', 'Jitter RMS (dimensionless)'),
        ('fc', 'Hz', 'Carrier frequency'),
        ('B', 'Hz', 'Bandwidth'),
        ('v_default', 'm/s', 'Default velocity'),
    ]
    
    # Expected baseline values for paper mode
    BASELINE_VALUES = {
        'L_eff': 50e3,
        'a': 0.05,
        'fs': 200e3,
        'T_span': 0.02,
        'f_cut': 300.0,
        'psd_alpha': 0.5,
        'sigma_j': 1e-6,
        'fc': 300e9,
        'B': 10e9,
        'v_default': 15000.0,
    }
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def audit(self) -> bool:
        """
        Run full configuration audit.
        
        Returns:
            True if audit passes, False otherwise
        """
        print("=" * 80)
        print("THz-ISL Configuration Audit Report")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"Config: {self.config.config_path}")
        print("=" * 80)
        
        # Run all checks
        self._check_mode()
        self._check_critical_params()
        self._check_derived_params()
        self._check_survival_space()
        self._check_jitter_model()
        self._check_figures_config()
        self._check_reproducibility()
        
        # Print parameter table
        self._print_parameter_table()
        
        # Print results
        self._print_results()
        
        # Return pass/fail
        return len(self.errors) == 0
    
    def _check_mode(self):
        """Check execution mode"""
        if self.config.mode != 'paper':
            self.warnings.append(
                f"Mode is '{self.config.mode}', not 'paper'. "
                "Paper figures should use mode='paper'."
            )
        else:
            self.info.append("Mode: 'paper' (correct for publication)")
    
    def _check_critical_params(self):
        """Check all critical parameters exist and have valid values"""
        cfg_dict = self.config.to_dict()
        
        for param_name, unit, desc in self.CRITICAL_PARAMS:
            # Check existence
            if param_name not in cfg_dict:
                self.errors.append(f"MISSING: {param_name} ({desc})")
                continue
            
            value = cfg_dict[param_name]
            
            # Check type and validity
            if value is None:
                self.errors.append(f"NULL VALUE: {param_name} = None")
            elif isinstance(value, (int, float)) and value <= 0:
                if param_name not in ['psd_alpha']:  # alpha can be any positive value
                    self.errors.append(f"INVALID: {param_name} = {value} (must be positive)")
            
            # Check against baseline (warning if different)
            if param_name in self.BASELINE_VALUES:
                baseline = self.BASELINE_VALUES[param_name]
                if abs(value - baseline) / (abs(baseline) + 1e-20) > 0.01:
                    self.warnings.append(
                        f"NON-BASELINE: {param_name} = {value:.6g} "
                        f"(baseline: {baseline:.6g})"
                    )
    
    def _check_derived_params(self):
        """Check derived parameters for consistency"""
        fs = self.config.sampling.fs
        T_span = self.config.sampling.T_span
        N = self.config.sampling.N
        
        expected_N = int(fs * T_span)
        if N != expected_N:
            self.errors.append(
                f"INCONSISTENT: N = {N}, but fs*T_span = {expected_N}"
            )
        else:
            self.info.append(f"N = {N} samples (consistent with fs*T_span)")
    
    def _check_survival_space(self):
        """Check survival space parameters"""
        fs = self.config.sampling.fs
        N = self.config.sampling.N
        f_cut = self.config.survival_space.f_cut
        
        # Check Nyquist
        nyquist = fs / 2
        if f_cut >= nyquist:
            self.errors.append(
                f"INVALID: f_cut = {f_cut} Hz >= Nyquist = {nyquist} Hz"
            )
        
        # Check k_max computation
        f_res = fs / (2 * N)
        k_max = int(np.ceil(f_cut / f_res))
        
        self.info.append(
            f"Survival space: f_res = {f_res:.2f} Hz, k_max = {k_max}"
        )
        
        # Warn if k_max is very small (might cause issues)
        if k_max < 5:
            self.warnings.append(
                f"LOW k_max = {k_max}: DCT projection has few basis vectors"
            )
    
    def _check_jitter_model(self):
        """Check jitter model configuration"""
        jitter = self.config.jitter
        
        # Check model type
        if jitter.model != 'log_gain':
            self.warnings.append(
                f"Jitter model is '{jitter.model}', expected 'log_gain'"
            )
        
        # Check PSD alpha
        if jitter.psd_alpha not in [0.5, 1.0]:
            self.warnings.append(
                f"psd_alpha = {jitter.psd_alpha}, "
                "paper recommends 0.5 (baseline) or 1.0 (comparison)"
            )
        
        # Check sigma_j is dimensionless
        if jitter.sigma_j > 1.0:
            self.warnings.append(
                f"sigma_j = {jitter.sigma_j} > 1.0: "
                "This should be dimensionless RMS, typically << 1"
            )
        
        self.info.append(
            f"Jitter PSD: 1/f^{jitter.psd_alpha} with f_knee = {jitter.f_knee} Hz"
        )
    
    def _check_figures_config(self):
        """Check figure-specific configurations"""
        for fig_name, fig_cfg in self.config.figures_config.items():
            changes = fig_cfg.get('changes', {})
            
            if len(changes) > 1:
                self.warnings.append(
                    f"{fig_name}: Changes more than one variable from baseline: "
                    f"{list(changes.keys())}"
                )
    
    def _check_reproducibility(self):
        """Check reproducibility settings"""
        repro = self.config.reproducibility
        
        if repro.master_seed is None:
            self.errors.append("MISSING: master_seed for reproducibility")
        else:
            self.info.append(f"Master seed: {repro.master_seed}")
        
        if repro.mc_seed_method != 'deterministic':
            self.warnings.append(
                f"MC seed method is '{repro.mc_seed_method}', "
                "recommend 'deterministic' for reproducibility"
            )
    
    def _print_parameter_table(self):
        """Print formatted parameter table"""
        print("\n" + "-" * 80)
        print("Critical Parameter Table")
        print("-" * 80)
        print(f"{'Parameter':<15} {'Value':>15} {'Unit':<10} {'Source':<12} {'Status':<10}")
        print("-" * 80)
        
        cfg_dict = self.config.to_dict()
        
        for param_name, unit, desc in self.CRITICAL_PARAMS:
            value = cfg_dict.get(param_name, 'MISSING')
            
            if isinstance(value, float):
                if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.6g}"
            else:
                value_str = str(value)
            
            # Determine status
            if value == 'MISSING':
                status = "ERROR"
            elif param_name in self.BASELINE_VALUES:
                baseline = self.BASELINE_VALUES[param_name]
                if abs(cfg_dict[param_name] - baseline) / (abs(baseline) + 1e-20) < 0.01:
                    status = "OK"
                else:
                    status = "CHANGED"
            else:
                status = "OK"
            
            print(f"{param_name:<15} {value_str:>15} {unit:<10} {'config':<12} {status:<10}")
        
        print("-" * 80)
    
    def _print_results(self):
        """Print audit results summary"""
        print("\n" + "=" * 80)
        print("Audit Results")
        print("=" * 80)
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for err in self.errors:
                print(f"   • {err}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"   • {warn}")
        
        # Info
        if self.info:
            print(f"\nℹ️  INFO ({len(self.info)}):")
            for info in self.info:
                print(f"   • {info}")
        
        # Final status
        print("\n" + "=" * 80)
        if self.errors:
            print("❌ AUDIT FAILED - Configuration has errors that must be fixed")
            print("=" * 80)
        elif self.warnings:
            print("⚠️  AUDIT PASSED WITH WARNINGS - Review warnings before publication")
            print("=" * 80)
        else:
            print("✅ AUDIT PASSED - Configuration is consistent")
            print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='THz-ISL Configuration Audit Tool'
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        default='config/paper_baseline.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )
    
    args = parser.parse_args()
    
    # Check config file exists
    if not os.path.exists(args.config_path):
        print(f"ERROR: Config file not found: {args.config_path}")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Run audit
    import numpy as np  # Needed for calculations
    auditor = ConfigAuditor(config)
    passed = auditor.audit()
    
    # Determine exit code
    if not passed:
        sys.exit(1)
    elif args.strict and auditor.warnings:
        print("\n--strict mode: Treating warnings as errors")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    import numpy as np
    main()

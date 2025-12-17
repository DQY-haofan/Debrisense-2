#!/usr/bin/env python3
# ==============================================================================
# run_all_figures.py - Paper Figure Generation Runner
# ==============================================================================
# Version: 1.0
#
# Description:
#   Master script for generating all paper figures with full reproducibility.
#   - Reads configuration from YAML
#   - Generates all figures with proper seeding
#   - Exports CSV, PNG, PDF, and config snapshots
#   - Creates run log for traceability
#
# Usage:
#   python run_all_figures.py --config config/paper_baseline.yaml --seed 42
#   python run_all_figures.py --figures fig6 fig7 fig8
#   python run_all_figures.py --sanity-check
#
# ==============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import scipy.signal as signal
from joblib import Parallel, delayed
import multiprocessing
import os
import sys
import argparse
import hashlib
import subprocess
import platform
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import yaml
from tqdm import tqdm
import pandas as pd

# Import project modules
from config_manager import ConfigManager, load_config, set_config
from physics_engine import DiffractionChannel, create_channel_from_config
from hardware_model import HardwareImpairments, create_hardware_from_config
from detector import TerahertzDebrisDetector, create_detector_from_config
from estimator import MLGridEstimator, create_estimator_from_config


# ==============================================================================
# Plot Style Configuration
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
    'figure.figsize': (5, 4),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
})


class OutputManager:
    """
    Manages output files, directories, and reproducibility artifacts.
    """
    
    def __init__(self, config: ConfigManager, base_dir: str = None):
        self.config = config
        
        out_cfg = config.output_config
        self.base_dir = base_dir or out_cfg.get('base_dir', 'outputs')
        self.paper_id = out_cfg.get('paper_id', 'thz_isl')
        
        self.output_root = os.path.join(self.base_dir, self.paper_id)
        os.makedirs(self.output_root, exist_ok=True)
        
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def get_figure_dir(self, fig_name: str) -> str:
        """Get output directory for a specific figure"""
        fig_dir = os.path.join(self.output_root, fig_name)
        os.makedirs(fig_dir, exist_ok=True)
        return fig_dir
    
    def save_figure(self, fig_name: str, fig: plt.Figure = None):
        """Save figure as PNG and PDF"""
        fig_dir = self.get_figure_dir(fig_name)
        
        if fig is None:
            fig = plt.gcf()
        
        plt.tight_layout(pad=0.5)
        fig.savefig(os.path.join(fig_dir, 'figure.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'figure.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        print(f"   [Saved] {fig_name}/figure.png, figure.pdf")
    
    def save_data(self, fig_name: str, data: Dict[str, Any], filename: str = 'data.csv'):
        """Save data as CSV"""
        fig_dir = self.get_figure_dir(fig_name)
        
        # Convert to DataFrame-compatible format
        max_len = 1
        for v in data.values():
            if hasattr(v, '__len__') and not isinstance(v, str):
                max_len = max(max_len, len(v))
        
        aligned = {}
        for k, v in data.items():
            if hasattr(v, '__len__') and not isinstance(v, str):
                v_arr = np.array(v)
                if len(v_arr) < max_len:
                    padded = np.full(max_len, np.nan)
                    padded[:len(v_arr)] = v_arr
                    aligned[k] = padded
                else:
                    aligned[k] = v_arr
            else:
                aligned[k] = np.full(max_len, v)
        
        df = pd.DataFrame(aligned)
        df.to_csv(os.path.join(fig_dir, filename), index=False)
        print(f"   [Saved] {fig_name}/{filename}")
    
    def save_config_snapshot(self, fig_name: str, fig_config: Dict[str, Any]):
        """Save configuration snapshot for this figure"""
        fig_dir = self.get_figure_dir(fig_name)
        
        snapshot = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'figure': fig_name,
            'config': fig_config,
        }
        
        with open(os.path.join(fig_dir, 'config_snapshot.yaml'), 'w') as f:
            yaml.dump(snapshot, f, default_flow_style=False)
    
    def save_run_log(self, fig_name: str, seed: int, mc_trials: int, extra: Dict = None):
        """Save run log with environment info"""
        fig_dir = self.get_figure_dir(fig_name)
        
        log = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'mc_trials': mc_trials,
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'platform': platform.platform(),
        }
        
        # Try to get git hash
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            log['git_commit'] = git_hash
        except:
            log['git_commit'] = 'unavailable'
        
        if extra:
            log.update(extra)
        
        with open(os.path.join(fig_dir, 'run.log'), 'w') as f:
            json.dump(log, f, indent=2)


class PaperFigureGenerator:
    """
    Generates all paper figures with full reproducibility support.
    """
    
    def __init__(self, config: ConfigManager, seed: int = 42):
        self.config = config
        self.master_seed = seed
        
        # Initialize output manager
        self.output = OutputManager(config)
        
        # Get sampling parameters
        self.fs = config.sampling.fs
        self.N = config.sampling.N
        self.T_span = config.sampling.T_span
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs
        
        # Initialize physics engine
        self.physics = create_channel_from_config(config)
        
        # Initialize hardware model
        self.hardware = create_hardware_from_config(config)
        
        # Initialize detector
        self.detector = create_detector_from_config(config)
        
        # Parallel processing
        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        
        print(f"[Init] PaperFigureGenerator")
        print(f"   Config: {config.config_path}")
        print(f"   Mode: {config.mode}")
        print(f"   Seed: {seed}")
        print(f"   N_jobs: {self.n_jobs}")
        print(f"   Output: {self.output.output_root}")
    
    def get_seed(self, fig_name: str, trial_idx: int = 0) -> int:
        """Get deterministic seed for a specific figure and trial"""
        seed_str = f"{self.master_seed}_{fig_name}_{trial_idx}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31 - 1)
    
    # =========================================================================
    # Sanity Check: Energy Retention
    # =========================================================================
    
    def generate_sanity_check(self):
        """
        Generate sanity check figures for energy retention.
        
        This is CRITICAL to verify the survival-space projection
        does not "self-eliminate" the signal.
        """
        print("\n" + "="*60)
        print("SANITY CHECK: Energy Retention Analysis")
        print("="*60)
        
        sanity_cfg = self.config.sanity_check_config
        f_cut_values = np.array(sanity_cfg.get('f_cut_sweep', [50, 100, 200, 300, 500, 1000]))
        v_rel_values = np.array(sanity_cfg.get('v_rel_sweep', [10000, 15000, 20000]))
        eta_threshold = sanity_cfg.get('eta_threshold', 0.01)
        
        # Run energy retention check
        results = self.detector.sanity_check_energy_retention(f_cut_values, v_rel_values)
        eta_matrix = results['eta_matrix']
        
        # Save data
        data = {
            'f_cut': f_cut_values,
        }
        for j, v in enumerate(v_rel_values):
            data[f'eta_v{int(v)}'] = eta_matrix[:, j]
        
        self.output.save_data('sanity_check', data, 'eta_vs_fcut.csv')
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(v_rel_values)))
        for j, v in enumerate(v_rel_values):
            ax.semilogy(f_cut_values, eta_matrix[:, j], 
                       'o-', color=colors[j], 
                       label=f'v = {v/1000:.0f} km/s')
        
        # Mark baseline f_cut
        baseline_fcut = self.config.survival_space.f_cut
        ax.axvline(baseline_fcut, color='red', linestyle='--', 
                  label=f'Baseline f_cut = {baseline_fcut} Hz')
        
        # Mark threshold
        ax.axhline(eta_threshold, color='orange', linestyle=':', 
                  label=f'Threshold η = {eta_threshold}')
        
        ax.set_xlabel('Cutoff Frequency f_cut (Hz)')
        ax.set_ylabel('Energy Retention Ratio η')
        ax.set_title('Sanity Check: Signal Energy Retention')
        ax.legend()
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        self.output.save_figure('sanity_check', fig)
        
        # Check baseline
        baseline_idx = np.argmin(np.abs(f_cut_values - baseline_fcut))
        baseline_eta = eta_matrix[baseline_idx, :]
        
        print(f"\n   Baseline f_cut = {baseline_fcut} Hz")
        print(f"   Energy retention at baseline:")
        for j, v in enumerate(v_rel_values):
            status = "✓ OK" if baseline_eta[j] > eta_threshold else "✗ FAIL"
            print(f"      v = {v/1000:.0f} km/s: η = {baseline_eta[j]:.4f} {status}")
        
        if np.all(baseline_eta > eta_threshold):
            print(f"\n   ✓ SANITY CHECK PASSED: All η > {eta_threshold}")
            return True
        else:
            print(f"\n   ✗ SANITY CHECK FAILED: Some η < {eta_threshold}")
            return False
    
    # =========================================================================
    # Figure 2: Hardware Mechanisms
    # =========================================================================
    
    def generate_fig2_mechanisms(self):
        """Generate Fig 2: Hardware impairment mechanisms"""
        print("\n--- Fig 2: Hardware Mechanisms ---")
        
        seed = self.get_seed('fig2')
        np.random.seed(seed)
        
        # Generate jitter for PSD analysis
        jitter = self.hardware.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jitter, self.fs, nperseg=2048)
        psd_log = 10 * np.log10(psd + 1e-20)
        
        # PA curves
        pin_db, am_am, scr = self.hardware.get_pa_curves()
        
        # Save data
        self.output.save_data('fig2', {
            'freq_hz': f, 'jitter_psd_db': psd_log,
            'pa_pin_db': pin_db, 'pa_out': am_am, 'pa_scr': scr
        })
        
        # Plot jitter PSD
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.semilogx(f[1:], psd_log[1:], 'b-')  # Skip DC
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_title(f'Jitter PSD (α = {self.config.jitter.psd_alpha})')
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        self.output.save_figure('fig2a_jitter', fig)
        
        # Plot PA curves
        fig, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(pin_db, am_am, 'k-', label='AM-AM')
        ax1.set_xlabel('Input Power (dB)')
        ax1.set_ylabel('Normalized Output')
        
        ax2 = ax1.twinx()
        ax2.plot(pin_db, np.maximum(scr, -0.2), 'r--', label='SCR')
        ax2.set_ylabel('SCR', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('PA Characteristics (Saleh Model)')
        self.output.save_figure('fig2b_pa', fig)
        
        self.output.save_config_snapshot('fig2', self.config.get_figure_config('fig2'))
        self.output.save_run_log('fig2', seed, 1)
    
    # =========================================================================
    # Figure 3: Dispersion (NB vs WB)
    # =========================================================================
    
    def generate_fig3_dispersion(self):
        """Generate Fig 3: Narrowband vs Wideband dispersion"""
        print("\n--- Fig 3: Dispersion ---")
        
        # Narrowband pattern (single frequency)
        d_nb = np.abs(self.physics.generate_narrowband_pattern(self.t_axis))
        
        # Wideband pattern (DFS)
        d_wb = np.abs(self.physics.generate_broadband_chirp(self.t_axis, self.config.physics.N_sub))
        
        # Save data
        self.output.save_data('fig3', {
            'time_ms': self.t_axis * 1000,
            'amp_nb': d_nb,
            'amp_wb': d_wb
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.t_axis * 1000, d_nb, 'r--', label='Narrowband')
        ax.plot(self.t_axis * 1000, d_wb, 'b-', label='Wideband (DFS)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|d(t)|')
        ax.set_title('Diffraction Pattern: NB vs WB')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        
        self.output.save_figure('fig3', fig)
        self.output.save_config_snapshot('fig3', self.config.get_figure_config('fig3'))
        self.output.save_run_log('fig3', self.get_seed('fig3'), 1)
    
    # =========================================================================
    # Figure 7: ROC Comparison
    # =========================================================================
    
    def generate_fig7_roc(self):
        """Generate Fig 7: ROC curves comparison"""
        print("\n--- Fig 7: ROC Comparison ---")
        
        fig_cfg = self.config.figures_config.get('fig7', {})
        snr_db = fig_cfg.get('changes', {}).get('snr_db', 50.0)
        jitter_sigma = fig_cfg.get('changes', {}).get('jitter_sigma', 2.0e-3)
        mc_trials = fig_cfg.get('mc_trials', 1500)
        
        print(f"   SNR = {snr_db} dB, Jitter σ = {jitter_sigma}")
        
        seed = self.get_seed('fig7')
        np.random.seed(seed)
        
        # Generate clean signal
        d_wb = self.physics.generate_broadband_chirp(self.t_axis, self.config.physics.N_sub)
        sig_h1 = 1.0 - d_wb
        sig_h0 = np.ones(self.N, dtype=complex)
        
        # Templates
        s_raw = self.detector.generate_template(self.config.scenario.v_default)
        s_prop = self.detector.P_perp @ s_raw
        s_prop_energy = np.sum(s_prop ** 2) + 1e-20
        
        s_naive = s_raw - np.mean(s_raw)
        s_naive_energy = np.sum(s_naive ** 2) + 1e-20
        
        def run_trial(sig, trial_seed, is_ideal):
            np.random.seed(trial_seed)
            
            if is_ideal:
                pa_out = sig.copy()
                p_ref = np.mean(np.abs(sig) ** 2)
            else:
                hw_cfg = self.config.get_hardware_config()
                hw_cfg['jitter_rms'] = jitter_sigma
                hw = HardwareImpairments(hw_cfg)
                jitter = np.exp(hw.generate_colored_jitter(self.N, self.fs))
                pa_out, _, _ = hw.apply_saleh_pa(sig * jitter, ibo_dB=10.0)
                p_ref = np.mean(np.abs(pa_out) ** 2)
            
            # Add noise
            noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
            noise = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) * noise_std / np.sqrt(2)
            rx = pa_out + noise
            
            # Log envelope
            x = self.detector.log_envelope_transform(rx)
            z_perp = self.detector.P_perp @ x
            
            # Proposed (GLRT in survival space)
            stat_prop = (np.dot(s_prop, z_perp) ** 2) / s_prop_energy
            
            # Naive (mean-subtracted)
            x_ac = x - np.mean(x)
            stat_naive = (np.dot(s_naive, x_ac) ** 2) / s_naive_energy
            
            # Energy detector
            stat_ed = np.sum(z_perp ** 2)
            
            return stat_prop, stat_naive, stat_ed
        
        # Run MC trials
        print("   Running H0 trials...")
        h0_results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_trial)(sig_h0, seed + i, False) 
            for i in tqdm(range(mc_trials), desc="H0")
        )
        h0_results = np.array(h0_results)
        
        print("   Running H1 trials...")
        h1_results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_trial)(sig_h1, seed + mc_trials + i, False) 
            for i in tqdm(range(mc_trials), desc="H1")
        )
        h1_results = np.array(h1_results)
        
        # Ideal case
        print("   Running Ideal trials...")
        h0_ideal = Parallel(n_jobs=self.n_jobs)(
            delayed(run_trial)(sig_h0, seed + 2*mc_trials + i, True) 
            for i in range(mc_trials)
        )
        h1_ideal = Parallel(n_jobs=self.n_jobs)(
            delayed(run_trial)(sig_h1, seed + 3*mc_trials + i, True) 
            for i in range(mc_trials)
        )
        h0_ideal = np.array(h0_ideal)
        h1_ideal = np.array(h1_ideal)
        
        # Compute ROC curves
        def get_roc(h0_stats, h1_stats):
            all_stats = np.concatenate([h0_stats, h1_stats])
            thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 500)
            pfa = np.array([np.mean(h0_stats > th) for th in thresholds])
            pd = np.array([np.mean(h1_stats > th) for th in thresholds])
            auc = -np.trapz(pd, pfa)
            return pfa, pd, auc
        
        pfa_prop, pd_prop, auc_prop = get_roc(h0_results[:, 0], h1_results[:, 0])
        pfa_naive, pd_naive, auc_naive = get_roc(h0_results[:, 1], h1_results[:, 1])
        pfa_ed, pd_ed, auc_ed = get_roc(h0_results[:, 2], h1_results[:, 2])
        pfa_ideal, pd_ideal, auc_ideal = get_roc(h0_ideal[:, 0], h1_ideal[:, 0])
        
        print(f"   AUC: Ideal={auc_ideal:.3f}, Proposed={auc_prop:.3f}, "
              f"Naive={auc_naive:.3f}, ED={auc_ed:.3f}")
        
        # Save data
        self.output.save_data('fig7', {
            'pfa_ideal': pfa_ideal, 'pd_ideal': pd_ideal,
            'pfa_prop': pfa_prop, 'pd_prop': pd_prop,
            'pfa_naive': pfa_naive, 'pd_naive': pd_naive,
            'pfa_ed': pfa_ed, 'pd_ed': pd_ed,
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(pfa_ideal, pd_ideal, 'g-', label=f'Ideal ({auc_ideal:.2f})')
        ax.plot(pfa_prop, pd_prop, 'b-', label=f'Proposed ({auc_prop:.2f})')
        ax.plot(pfa_naive, pd_naive, '--', color='orange', label=f'Naive MF ({auc_naive:.2f})')
        ax.plot(pfa_ed, pd_ed, 'r:', label=f'Energy Det ({auc_ed:.2f})')
        ax.plot([0, 1], [0, 1], 'k-', alpha=0.2)
        
        ax.set_xlabel('Probability of False Alarm')
        ax.set_ylabel('Probability of Detection')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        self.output.save_figure('fig7', fig)
        self.output.save_config_snapshot('fig7', self.config.get_figure_config('fig7'))
        self.output.save_run_log('fig7', seed, mc_trials, {'snr_db': snr_db, 'jitter_sigma': jitter_sigma})
    
    # =========================================================================
    # Figure 10: Ambiguity Function
    # =========================================================================
    
    def generate_fig10_ambiguity(self):
        """Generate Fig 10: Ambiguity function"""
        print("\n--- Fig 10: Ambiguity Function ---")
        
        fig_cfg = self.config.figures_config.get('fig10', {})
        v_range = fig_cfg.get('v_shift_range', [-600, 600])
        t_range = fig_cfg.get('t_shift_range', [-0.0015, 0.0015])
        n_points = fig_cfg.get('grid_points', 41)
        
        v_ref = self.config.scenario.v_default
        s0_raw = self.detector.generate_template(v_ref)
        s0 = self.detector.P_perp @ s0_raw
        s0_energy = np.sum(s0 ** 2) + 1e-20
        
        v_shifts = np.linspace(v_range[0], v_range[1], n_points)
        t_shifts = np.linspace(t_range[0], t_range[1], n_points)
        
        ambiguity = np.zeros((n_points, n_points))
        
        for i, dv in enumerate(v_shifts):
            s_v_raw = self.detector.generate_template(v_ref + dv)
            s_v = self.detector.P_perp @ s_v_raw
            
            for j, dt in enumerate(t_shifts):
                shift_samples = int(np.round(dt * self.fs))
                s_shifted = np.roll(s_v, shift_samples)
                
                # Zero pad
                if shift_samples > 0:
                    s_shifted[:shift_samples] = 0
                elif shift_samples < 0:
                    s_shifted[shift_samples:] = 0
                
                corr = np.abs(np.dot(s0, s_shifted)) ** 2
                ambiguity[i, j] = corr / s0_energy
        
        # Normalize
        ambiguity_norm = ambiguity / np.max(ambiguity)
        
        # Save data
        self.output.save_data('fig10', {'ambiguity_matrix': ambiguity_norm.flatten()}, 'ambiguity.csv')
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        cf = ax.contourf(t_shifts * 1000, v_shifts, ambiguity_norm, 20, cmap='viridis')
        ax.axhline(0, color='w', linestyle=':', alpha=0.5)
        ax.axvline(0, color='w', linestyle=':', alpha=0.5)
        ax.set_xlabel('Delay Mismatch (ms)')
        ax.set_ylabel('Velocity Mismatch (m/s)')
        plt.colorbar(cf, ax=ax, label='Normalized Correlation')
        ax.set_title('Ambiguity Function')
        
        self.output.save_figure('fig10', fig)
        self.output.save_config_snapshot('fig10', self.config.get_figure_config('fig10'))
        self.output.save_run_log('fig10', self.get_seed('fig10'), 1)
    
    # =========================================================================
    # Alpha Sensitivity Comparison
    # =========================================================================
    
    def generate_alpha_sensitivity(self):
        """Generate alpha sensitivity comparison figure"""
        print("\n--- Alpha Sensitivity Analysis ---")
        
        seed = self.get_seed('alpha_sensitivity')
        np.random.seed(seed)
        
        alpha_values = [0.5, 1.0]
        mc_trials = 100
        
        # Generate clean signal
        d_wb = self.physics.generate_broadband_chirp(self.t_axis, self.config.physics.N_sub)
        sig_h1 = 1.0 - d_wb
        
        snr_db = 50.0
        
        results = {a: [] for a in alpha_values}
        
        for alpha in alpha_values:
            print(f"   Testing α = {alpha}...")
            
            hw_cfg = self.config.get_hardware_config()
            hw_cfg['psd_alpha'] = alpha
            hw_cfg['jitter_rms'] = 2.0e-3
            
            for trial in tqdm(range(mc_trials)):
                hw = HardwareImpairments(hw_cfg)
                
                jitter = np.exp(hw.generate_colored_jitter(self.N, self.fs, seed=seed + trial))
                pa_out, _, _ = hw.apply_saleh_pa(sig_h1 * jitter, ibo_dB=10.0)
                p_ref = np.mean(np.abs(pa_out) ** 2)
                
                noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
                noise = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) * noise_std / np.sqrt(2)
                rx = pa_out + noise
                
                stat = self.detector.paper_pipeline_detect(rx, self.config.scenario.v_default)
                results[alpha].append(stat)
        
        # Save data
        data = {'trial': list(range(mc_trials))}
        for alpha in alpha_values:
            data[f'stat_alpha_{alpha}'] = results[alpha]
        self.output.save_data('alpha_sensitivity', data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        for alpha in alpha_values:
            ax.hist(results[alpha], bins=30, alpha=0.5, label=f'α = {alpha}', density=True)
        ax.set_xlabel('GLRT Statistic')
        ax.set_ylabel('Density')
        ax.set_title('Detection Statistic Distribution by α')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        
        self.output.save_figure('alpha_sensitivity', fig)
        self.output.save_config_snapshot('alpha_sensitivity', {'alpha_values': alpha_values})
        self.output.save_run_log('alpha_sensitivity', seed, mc_trials)
    
    # =========================================================================
    # PD vs f_cut (导师要求的第二张验证图)
    # =========================================================================
    
    def generate_pd_vs_fcut(self):
        """
        Generate PD (at fixed PFA) vs f_cut figure.
        
        导师要求：这张图与 η vs f_cut 一起，能让审稿人"基本无话可说"。
        显示在不同 f_cut 下，检测概率如何变化。
        """
        print("\n--- PD vs f_cut Analysis ---")
        
        seed = self.get_seed('pd_vs_fcut')
        np.random.seed(seed)
        
        # Parameters
        f_cut_values = np.array([50, 100, 150, 200, 250, 300, 400, 500])
        target_pfa = 0.01  # Fixed PFA
        mc_trials = 200
        snr_db = 50.0
        jitter_sigma = 2.0e-3
        
        print(f"   Target PFA = {target_pfa}, MC trials = {mc_trials}")
        print(f"   SNR = {snr_db} dB, Jitter σ = {jitter_sigma}")
        
        # Generate clean signals
        d_wb = self.physics.generate_broadband_chirp(self.t_axis, self.config.physics.N_sub)
        sig_h1 = 1.0 - d_wb
        sig_h0 = np.ones(self.N, dtype=complex)
        
        # Results storage
        pd_values = np.zeros(len(f_cut_values))
        eta_values = np.zeros(len(f_cut_values))
        
        for i, f_cut in enumerate(tqdm(f_cut_values, desc="f_cut sweep")):
            # Create detector with this f_cut
            det_cfg = self.config.get_detection_config()
            det_cfg['f_cut'] = f_cut
            det_cfg['fs'] = self.fs
            det_cfg['N'] = self.N
            det_cfg['T_span'] = self.T_span
            
            from detector import TerahertzDebrisDetector
            detector_i = TerahertzDebrisDetector(det_cfg, self.N)
            
            # Compute energy retention
            s_template = detector_i.generate_template(self.config.scenario.v_default)
            eta_values[i] = detector_i.compute_energy_retention(s_template)
            
            # Projected template
            s_proj = detector_i.P_perp @ s_template
            s_proj_energy = np.sum(s_proj ** 2) + 1e-20
            
            # MC trials for H0 (to determine threshold)
            h0_stats = []
            for trial in range(mc_trials):
                trial_seed = seed + i * mc_trials * 2 + trial
                np.random.seed(trial_seed)
                
                # Hardware corruption
                hw_cfg = self.config.get_hardware_config()
                hw_cfg['jitter_rms'] = jitter_sigma
                hw = HardwareImpairments(hw_cfg)
                jitter = np.exp(hw.generate_colored_jitter(self.N, self.fs))
                pa_out, _, _ = hw.apply_saleh_pa(sig_h0 * jitter, ibo_dB=10.0)
                
                # Add noise
                p_ref = np.mean(np.abs(pa_out) ** 2)
                noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
                noise = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) * noise_std / np.sqrt(2)
                rx = pa_out + noise
                
                # Detection statistic
                x = detector_i.log_envelope_transform(rx)
                z_perp = detector_i.P_perp @ x
                stat = (np.dot(s_proj, z_perp) ** 2) / s_proj_energy
                h0_stats.append(stat)
            
            h0_stats = np.array(h0_stats)
            
            # Determine threshold for target PFA
            threshold = np.percentile(h0_stats, 100 * (1 - target_pfa))
            
            # MC trials for H1 (to measure PD)
            h1_stats = []
            for trial in range(mc_trials):
                trial_seed = seed + i * mc_trials * 2 + mc_trials + trial
                np.random.seed(trial_seed)
                
                # Hardware corruption
                hw_cfg = self.config.get_hardware_config()
                hw_cfg['jitter_rms'] = jitter_sigma
                hw = HardwareImpairments(hw_cfg)
                jitter = np.exp(hw.generate_colored_jitter(self.N, self.fs))
                pa_out, _, _ = hw.apply_saleh_pa(sig_h1 * jitter, ibo_dB=10.0)
                
                # Add noise
                p_ref = np.mean(np.abs(pa_out) ** 2)
                noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
                noise = (np.random.randn(self.N) + 1j * np.random.randn(self.N)) * noise_std / np.sqrt(2)
                rx = pa_out + noise
                
                # Detection statistic
                x = detector_i.log_envelope_transform(rx)
                z_perp = detector_i.P_perp @ x
                stat = (np.dot(s_proj, z_perp) ** 2) / s_proj_energy
                h1_stats.append(stat)
            
            h1_stats = np.array(h1_stats)
            pd_values[i] = np.mean(h1_stats > threshold)
            
            print(f"   f_cut={f_cut:3d} Hz: η={eta_values[i]:.4f}, PD={pd_values[i]:.3f}")
        
        # Save data
        self.output.save_data('pd_vs_fcut', {
            'f_cut_hz': f_cut_values,
            'eta': eta_values,
            'pd': pd_values,
            'target_pfa': np.full(len(f_cut_values), target_pfa),
        })
        
        # Plot: Dual y-axis (η and PD vs f_cut)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Cutoff Frequency $f_{cut}$ (Hz)')
        ax1.set_ylabel('Energy Retention $\\eta$', color=color1)
        ax1.plot(f_cut_values, eta_values, 'o-', color=color1, label='$\\eta$')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 1.05])
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(f'Detection Probability (PFA={target_pfa})', color=color2)
        ax2.plot(f_cut_values, pd_values, 's--', color=color2, label='$P_D$')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([0, 1.05])
        
        # Mark baseline f_cut
        baseline_fcut = self.config.survival_space.f_cut
        ax1.axvline(baseline_fcut, color='green', linestyle=':', alpha=0.7,
                   label=f'Baseline $f_{{cut}}$={baseline_fcut} Hz')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        self.output.save_figure('pd_vs_fcut', fig)
        self.output.save_config_snapshot('pd_vs_fcut', {
            'f_cut_values': f_cut_values.tolist(),
            'target_pfa': target_pfa,
            'snr_db': snr_db,
            'jitter_sigma': jitter_sigma,
            'mc_trials': mc_trials,
        })
        self.output.save_run_log('pd_vs_fcut', seed, mc_trials)
        
        print(f"\n   Baseline f_cut={baseline_fcut} Hz: η={eta_values[np.argmin(np.abs(f_cut_values - baseline_fcut))]:.4f}")
    
    # =========================================================================
    # Unit Tests Runner (导师要求的验证)
    # =========================================================================
    
    def run_unit_tests(self):
        """
        Run estimator unit tests.
        
        导师要求的验证项：
        - SC2: FFT vs direct 一致性
        - SC3: 负时移对称性  
        - Toy case 回归
        """
        print("\n" + "="*60)
        print("Running Estimator Unit Tests")
        print("="*60)
        
        from estimator import run_unit_tests
        results = run_unit_tests(self.detector, verbose=True)
        
        # Save results
        self.output.save_data('unit_tests', {
            'test_name': list(results.keys()),
            'passed': [1 if v else 0 for v in results.values()],
        })
        
        return all(results.values())
    
    # =========================================================================
    # Run All Figures
    # =========================================================================
    
    def run_all(self, figures: List[str] = None, include_sanity: bool = True):
        """
        Run all figure generation.
        
        Args:
            figures: List of figure names to generate. If None, generates all.
            include_sanity: Include sanity check
        """
        print("\n" + "="*60)
        print("THz-ISL Paper Figure Generation")
        print("="*60)
        print(f"Start time: {datetime.now().isoformat()}")
        
        # Available figure generators
        all_figures = {
            'sanity_check': self.generate_sanity_check,
            'fig2': self.generate_fig2_mechanisms,
            'fig3': self.generate_fig3_dispersion,
            'fig7': self.generate_fig7_roc,
            'fig10': self.generate_fig10_ambiguity,
            'alpha_sensitivity': self.generate_alpha_sensitivity,
            'pd_vs_fcut': self.generate_pd_vs_fcut,  # 导师要求的第二张验证图
            'unit_tests': self.run_unit_tests,       # 导师要求的单元测试
        }
        
        if figures is None:
            figures = list(all_figures.keys())
        
        if include_sanity and 'sanity_check' not in figures:
            figures = ['sanity_check'] + figures
        
        # Run generators
        for fig_name in figures:
            if fig_name in all_figures:
                try:
                    all_figures[fig_name]()
                except Exception as e:
                    print(f"   [ERROR] {fig_name}: {e}")
                    raise
            else:
                print(f"   [SKIP] Unknown figure: {fig_name}")
        
        print("\n" + "="*60)
        print(f"Complete! Output: {self.output.output_root}")
        print(f"End time: {datetime.now().isoformat()}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='THz-ISL Paper Figure Generator')
    parser.add_argument('--config', default='config/paper_baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Master random seed')
    parser.add_argument('--figures', nargs='+', default=None,
                       help='Specific figures to generate')
    parser.add_argument('--sanity-check', action='store_true',
                       help='Run only sanity check')
    parser.add_argument('--unit-test', action='store_true',
                       help='Run estimator unit tests')
    parser.add_argument('--output-dir', default=None,
                       help='Override output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        set_config(config)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    # Print configuration summary
    config.print_summary()
    
    # Create generator
    generator = PaperFigureGenerator(config, seed=args.seed)
    
    # Run
    if args.sanity_check:
        passed = generator.generate_sanity_check()
        sys.exit(0 if passed else 1)
    elif args.unit_test:
        passed = generator.run_unit_tests()
        sys.exit(0 if passed else 1)
    else:
        generator.run_all(figures=args.figures)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

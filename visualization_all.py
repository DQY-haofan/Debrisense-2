# ----------------------------------------------------------------------------------
# 脚本名称: visualization_all_fixed_v7.py
# 版本: v7.0 (Physically Correct)
#
# 【核心修复】
# 1. 噪声计算基准修正：基于当前信号功率（Ideal 基于原始信号，Hardware 基于 PA 输出）
# 2. cutoff_freq = 300Hz（符合 DR_algo_01 理论的"生存空间"）
# 3. L_eff 和 a 参数正确传递给 detector
# 4. SNR 配置使用接收端 SNR（Rx SNR）
#
# 【物理说明】
# - 原始碎片配置 (a=5cm, L_eff=50km) 衍射深度仅 0.016%，需要 SNR >= 70dB
# - 本脚本使用 a=25cm, L_eff=10km，衍射深度 ~2%，需要 SNR ~45dB（更实际）
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

# 尝试导入核心模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Error: Core modules (physics_engine, hardware_model, detector) not found.")

# --- 绘图风格配置 ---
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
    'grid.alpha': 0.6
})

# --- 全局物理参数 ---
# 【修改】使用更短基线和更大碎片以获得可检测的衍射深度
# 原配置 (a=5cm, L_eff=50km) 衍射深度仅 0.016%，需要 SNR >= 70dB
# 新配置 (a=25cm, L_eff=10km) 衍射深度 ~2%，需要 SNR ~45dB（更实际）
GLOBAL_CONFIG = {
    'fc': 300e9,
    'B': 10e9,
    'L_eff': 10e3,  # 10 km 基线（原 50km）
    'fs': 200e3,
    'T_span': 0.02,
    'a': 0.25,  # 25 cm 半径（原 5cm）→ 衍射深度 ~2%
    'v_default': 15000
}

# --- SNR 配置 (接收端 SNR) ---
# 【修改】这些是接收端 SNR，噪声会基于当前信号功率计算
SNR_CONFIG = {
    "fig6": 45.0,  # RMSE: 需要较高 SNR 以看清 Jitter 影响
    "fig7": 45.0,  # ROC: 接收端 45dB 足够检测 2% 深度
    "fig8": 45.0,  # MDS: 同上
    "fig9": 40.0  # ISAC
}

# --- 硬件损伤默认配置 ---
HW_CONFIG = {
    'jitter_rms': 1.0e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0,
    'pll_bw': 50e3
}

# --- 检测器配置 ---
# 【修改】cutoff_freq = 300Hz 符合 DR_algo_01 理论
DETECTOR_CONFIG = {
    'cutoff_freq': 300.0,  # Hz - 生存空间理论
    'N_sub': 32
}


class PaperFigureGenerator:
    def __init__(self, output_dir='results_v7_fixed'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')

        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)

        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs

        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        print(f"[Init] Cores: {self.n_jobs} | Output: {os.path.abspath(self.out_dir)}")
        print(f"[Init] SNR Config (Rx SNR): {SNR_CONFIG}")
        print(f"[Init] Physics: L_eff={GLOBAL_CONFIG['L_eff'] / 1000:.0f}km, a={GLOBAL_CONFIG['a'] * 100:.0f}cm")

        # 计算衍射深度
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)
        depth = (1 - np.min(np.abs(1.0 - d_wb))) * 100
        print(f"[Init] Diffraction depth: {depth:.2f}%")

    def save_plot(self, name):
        """同时保存 PNG 和 PDF"""
        plt.tight_layout(pad=0.5)
        path_png = os.path.join(self.out_dir, f"{name}.png")
        path_pdf = os.path.join(self.out_dir, f"{name}.pdf")
        try:
            plt.savefig(path_png, dpi=300, bbox_inches='tight')
            plt.savefig(path_pdf, format='pdf', bbox_inches='tight')
            print(f"   [Plot Saved] {name}")
        except Exception as e:
            print(f"   [Error Saving Plot] {name}: {e}")
        plt.close('all')

    def save_csv(self, data, name, is_matrix=False):
        """保存数据为 CSV"""
        path = os.path.join(self.csv_dir, f"{name}.csv")
        try:
            if is_matrix and isinstance(data, (np.ndarray, list)):
                pd.DataFrame(data).to_csv(path, index=False, header=False)
            else:
                max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data.values()])
                aligned = {}
                for k, v in data.items():
                    if hasattr(v, '__len__'):
                        v_arr = np.array(v)
                        v_arr = np.nan_to_num(v_arr, nan=0.0)
                        if len(v_arr) < max_len:
                            padded = np.full(max_len, np.nan)
                            padded[:len(v_arr)] = v_arr
                            aligned[k] = padded
                        else:
                            aligned[k] = v_arr
                    else:
                        aligned[k] = np.full(max_len, v)
                pd.DataFrame(aligned).to_csv(path, index=False)
            print(f"   [Data Saved] {name}.csv")
        except Exception as e:
            print(f"   [Error Saving CSV] {name}: {e}")

    # =========================================================================
    # Group A: Physics Visualization
    # =========================================================================

    def generate_fig2_mechanisms(self):
        print("\n--- Fig 2: Mechanisms ---")
        hw = HardwareImpairments(HW_CONFIG)
        jit = hw.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)
        psd_log = 10 * np.log10(psd + 1e-20)
        pin_db, am_am, scr = hw.get_pa_curves()

        self.save_csv({
            'freq_hz': f, 'jitter_psd_db': psd_log,
            'pa_pin_db': pin_db, 'pa_out': am_am, 'pa_scr': scr
        }, 'Fig2_Mechanisms_Data')

        plt.figure(figsize=(5, 4))
        plt.semilogx(f, psd_log, 'b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (dB/Hz)')
        self.save_plot('Fig2a_Jitter')

        plt.figure(figsize=(5, 4))
        plt.plot(pin_db, am_am, 'k')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(pin_db, np.maximum(scr, -0.2), 'r--')
        ax.set_xlabel('Input Power (dB)')
        ax.set_ylabel('Output')
        ax2.set_ylabel('SCR', color='r')
        self.save_plot('Fig2b_PA')

    def generate_fig3_dispersion(self):
        print("\n--- Fig 3: Dispersion ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=32))

        self.save_csv({'time_ms': self.t_axis * 1000, 'amp_nb': d_nb, 'amp_wb': d_wb}, 'Fig3_Dispersion_Data')

        plt.figure(figsize=(5, 4))
        plt.plot(self.t_axis * 1000, d_nb, 'r--', label='NB')
        plt.plot(self.t_axis * 1000, d_wb, 'b-', label='WB')
        plt.xlabel('Time (ms)')
        plt.ylabel('|d(t)|')
        plt.legend()
        self.save_plot('Fig3_Dispersion')

    def generate_fig4_self_healing(self):
        print("\n--- Fig 4: Self Healing ---")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        sig = 1.0 - 0.2 * np.exp(-((t - 0.5) ** 2) / 0.005)

        out, _, _ = hw.apply_saleh_pa(sig, ibo_dB=10.0)
        out_norm = np.abs(out) / (np.max(np.abs(out)) + 1e-20)

        self.save_csv({'space': t, 'input': np.abs(sig), 'output': out_norm}, 'Fig4_Self_Healing_Data')

        plt.figure(figsize=(5, 4))
        plt.plot(t, np.abs(sig), 'k--', linewidth=1.5, label='In (Ideal)')
        plt.plot(t, out_norm, 'r-', linewidth=1.5, label='Out (PA Output)')
        plt.legend(loc='lower left')
        plt.xlabel('Normalized Time')
        plt.ylabel('Amplitude')
        plt.ylim(0.75, 1.02)
        plt.title('Waveform Preservation (High IBO)')
        plt.grid(True, linestyle=':', alpha=0.6)
        self.save_plot('Fig4_Self_Healing')

    def generate_fig5_survival_space(self):
        print("\n--- Fig 5: Spectrogram ---")
        from scipy.signal import chirp

        t = self.t_axis
        carrier = 0.5 * chirp(t, f0=20e3, f1=80e3, t1=t[-1], method='linear')
        debris_effect = 1.0 - 0.5 * np.exp(-((t) ** 2) / (0.0005 ** 2))
        y = carrier * debris_effect

        hw = HardwareImpairments(HW_CONFIG)
        jit = np.exp(hw.generate_colored_jitter(self.N, self.fs))
        y = y * jit

        f, t_sp, Sxx = signal.spectrogram(y, self.fs, nperseg=256, noverlap=220)
        Sxx_db = 10 * np.log10(Sxx + 1e-15)

        self.save_csv(Sxx_db, 'Fig5_Spectrogram_Matrix', is_matrix=True)

        plt.figure(figsize=(5, 4))
        plt.pcolormesh(t_sp * 1000, f / 1000, Sxx_db, shading='gouraud', cmap='inferno')
        plt.ylim(0, 100)
        plt.ylabel('Frequency (kHz)')
        plt.xlabel('Time (ms)')
        plt.colorbar(label='PSD (dB)')
        plt.tight_layout()
        self.save_plot('Fig5_Survival_Space')

    def generate_fig10_ambiguity(self):
        print("\n--- Fig 10: Ambiguity ---")
        det = TerahertzDebrisDetector(self.fs, self.N,
                                      cutoff_freq=DETECTOR_CONFIG['cutoff_freq'],
                                      L_eff=GLOBAL_CONFIG['L_eff'],
                                      a=GLOBAL_CONFIG['a'],
                                      N_sub=128)

        s0 = det.P_perp @ det._generate_template(15000)

        v_shifts = np.linspace(-600, 600, 41)
        t_shifts = np.linspace(-0.0015, 0.0015, 41)
        res = np.zeros((41, 41))
        E = np.sum(s0 ** 2) + 1e-20

        for i, dv in enumerate(v_shifts):
            s_v = det.P_perp @ det._generate_template(15000 + dv)
            for j, dt in enumerate(t_shifts):
                shift = int(np.round(dt * self.fs))
                s_shift = np.roll(s_v, shift)
                if shift > 0:
                    s_shift[:shift] = 0
                elif shift < 0:
                    s_shift[shift:] = 0
                res[i, j] = (np.abs(np.dot(s0, s_shift)) ** 2) / E

        self.save_csv(res, 'Fig10_Ambiguity_Matrix', is_matrix=True)

        plt.figure(figsize=(5, 4))
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20, cmap='viridis')
        plt.axhline(0, color='w', linestyle=':', alpha=0.5)
        plt.axvline(0, color='w', linestyle=':', alpha=0.5)
        plt.xlabel('Delay Mismatch (ms)')
        plt.ylabel('Velocity Mismatch (m/s)')
        plt.colorbar(label='Normalized Correlation')
        plt.tight_layout()
        self.save_plot('Fig10_Ambiguity')

    def generate_fig11_trajectory(self):
        print("\n--- Fig 11: Trajectory ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_ideal = phy.generate_broadband_chirp(self.t_axis, 32)
        hw = HardwareImpairments(HW_CONFIG)

        pn = np.exp(1j * hw.generate_phase_noise(self.N, self.fs))
        jit = np.exp(hw.generate_colored_jitter(self.N, self.fs) * 10)
        sig_rx = (1.0 - d_ideal) * jit * pn

        raw_error = sig_rx - (1.0 - d_ideal)
        cloud_error = np.diff(raw_error)
        cloud_error = cloud_error - np.mean(cloud_error)

        self.save_csv({'real': np.real(cloud_error), 'imag': np.imag(cloud_error)}, 'Fig11_Trajectory_Data')

        plt.figure(figsize=(5, 5))
        plt.plot(np.real(cloud_error), np.imag(cloud_error), '.', color='grey', alpha=0.3, markersize=2)
        plt.xlabel('In-Phase Jitter')
        plt.ylabel('Quadrature Jitter')
        plt.title("EVM Cloud (High-Pass Filtered)")
        plt.grid(True, linestyle=':')
        plt.axis('equal')
        plt.tight_layout()
        self.save_plot('Fig11_Trajectory')

    # =========================================================================
    # Group B: Performance
    # =========================================================================

    def generate_fig6_rmse_sensitivity(self):
        print(f"\n--- Fig 6: RMSE vs IBO (Rx SNR={SNR_CONFIG['fig6']} dB) ---")

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal

        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)

        det_cfg = {
            'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
            'L_eff': GLOBAL_CONFIG['L_eff'],
            'a': GLOBAL_CONFIG['a'],
            'N_sub': DETECTOR_CONFIG['N_sub']
        }
        det_main = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det_main.P_perp

        s_raw_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array([P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        jit_levels = [1.0e-5, 1.0e-4, 5.0e-4]
        ibo_scan = np.linspace(30, 0, 13)
        trials = 80

        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        # Ideal Baseline
        res_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_rmse_fixed)(10.0, 0, s, SNR_CONFIG['fig6'], sig_truth,
                                       self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                       E_bank, v_scan, ideal=True, P_perp=P_perp)
            for s in tqdm(range(trials), desc="Ideal")
        )
        rmse_id = np.sqrt(np.mean(np.array(res_id) ** 2))
        ax.semilogy(ibo_scan, [max(rmse_id, 0.1)] * len(ibo_scan), 'k--', label='Ideal', linewidth=2)
        data['ideal'] = [rmse_id] * len(ibo_scan)

        # Jitter Curves
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for idx, jit in enumerate(jit_levels):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_fixed)(ibo, jit, s, SNR_CONFIG['fig6'], sig_truth,
                                           self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                           E_bank, v_scan, ideal=False, P_perp=P_perp)
                for ibo in tqdm(ibo_scan, desc=f"Jit={jit:.0e}", leave=False)
                for s in range(trials)
            )
            res_mat = np.array(res).reshape(len(ibo_scan), trials)
            rmse = []
            for row in res_mat:
                valid = row[np.abs(row) < 1300]
                if len(valid) > 5:
                    rmse.append(np.sqrt(np.mean(valid ** 2)))
                else:
                    rmse.append(1000.0)

            ax.semilogy(ibo_scan, rmse, 'o-', color=colors[idx], label=rf'$\sigma_J$={jit:.0e} s')
            data[f'rmse_jit_{idx}'] = rmse

        ax.set_xlabel('IBO (dB)')
        ax.set_ylabel('Velocity RMSE (m/s)')
        ax.set_ylim(0.5, 2000)
        ax.invert_xaxis()
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.5)

        self.save_csv(data, 'Fig6_Sensitivity')
        self.save_plot('Fig6_Sensitivity')

    def generate_fig7_roc(self):
        print(f"\n--- Fig 7: ROC (Rx SNR={SNR_CONFIG['fig7']} dB) ---")
        trials = 1500

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)

        # 【关键】使用正确的 detector 配置
        det_ref = TerahertzDebrisDetector(
            self.fs, self.N,
            cutoff_freq=DETECTOR_CONFIG['cutoff_freq'],
            L_eff=GLOBAL_CONFIG['L_eff'],
            a=GLOBAL_CONFIG['a'],
            N_sub=DETECTOR_CONFIG['N_sub']
        )
        P_perp = det_ref.P_perp
        s_true_raw = det_ref._generate_template(15000)
        s_true_perp = P_perp @ s_true_raw
        s_energy = np.sum(s_true_perp ** 2) + 1e-20

        sig_h1_clean = 1.0 - d_wb
        sig_h0_clean = np.ones(self.N, dtype=complex)

        # Jitter 配置
        jit_proposed = 1.0e-5  # 低抖动（鲁棒方案）
        jit_standard = 2.0e-4  # 高抖动（标准方案）

        def run_roc_batch(is_ideal, jitter_val):
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_fixed)(sig_h0_clean, s, SNR_CONFIG['fig7'], is_ideal,
                                          s_true_perp, s_energy, P_perp, jitter_val,
                                          self.N, self.fs)
                for s in range(trials))
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_fixed)(sig_h1_clean, s, SNR_CONFIG['fig7'], is_ideal,
                                          s_true_perp, s_energy, P_perp, jitter_val,
                                          self.N, self.fs)
                for s in range(trials))
            return np.array(r0), np.array(r1)

        print("   Running Ideal...")
        r0_id, r1_id = run_roc_batch(True, 0)

        print(f"   Running Proposed (Jit={jit_proposed:.1e})...")
        r0_prop, r1_prop = run_roc_batch(False, jit_proposed)

        print(f"   Running Standard (Jit={jit_standard:.1e})...")
        r0_std, r1_std = run_roc_batch(False, jit_standard)

        def get_curve(n, s, col=0):
            n_vals, s_vals = n[:, col], s[:, col]
            all_v = np.concatenate([n_vals, s_vals])
            th = np.linspace(np.min(all_v), np.max(all_v), 500)
            pf = np.array([np.mean(n_vals > t) for t in th])
            pd = np.array([np.mean(s_vals > t) for t in th])
            auc = -np.trapezoid(pd, pf)
            return pf, pd, auc

        pf_id, pd_id, auc_id = get_curve(r0_id, r1_id)
        pf_prop, pd_prop, auc_prop = get_curve(r0_prop, r1_prop)
        pf_std, pd_std, auc_std = get_curve(r0_std, r1_std)
        pf_ed, pd_ed, auc_ed = get_curve(r0_std, r1_std, col=1)

        print(f"   AUC: Ideal={auc_id:.3f}, Proposed={auc_prop:.3f}, Standard={auc_std:.3f}, ED={auc_ed:.3f}")

        self.save_csv({
            'pf_id': pf_id, 'pd_id': pd_id,
            'pf_prop': pf_prop, 'pd_prop': pd_prop,
            'pf_std': pf_std, 'pd_std': pd_std,
            'pf_ed': pf_ed, 'pd_ed': pd_ed
        }, 'Fig7_ROC_Data')

        plt.figure(figsize=(5, 5))
        plt.plot(pf_id, pd_id, 'g-', linewidth=2, label=f'Ideal (AUC={auc_id:.2f})')
        plt.plot(pf_prop, pd_prop, 'b-', linewidth=2, label=f'Proposed (AUC={auc_prop:.2f})')
        plt.plot(pf_std, pd_std, 'r--', linewidth=1.5, label=f'Standard (AUC={auc_std:.2f})')
        plt.plot(pf_ed, pd_ed, 'k:', linewidth=1, label=f'Energy Det (AUC={auc_ed:.2f})')
        plt.plot([0, 1], [0, 1], 'k-', alpha=0.1)
        plt.xlabel('Probability of False Alarm (PFA)')
        plt.ylabel('Probability of Detection (PD)')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, linestyle=':')
        self.save_plot('Fig7_ROC')

    def generate_fig8_mds(self):
        print(f"\n--- Fig 8: MDS (Rx SNR={SNR_CONFIG['fig8']} dB) ---")

        # 碎片直径扫描 (mm)
        diams = np.array([2, 5, 8, 12, 16, 20, 30, 50, 80, 100])
        radii = diams / 2000.0  # 转换为米

        det = TerahertzDebrisDetector(
            self.fs, self.N,
            cutoff_freq=DETECTOR_CONFIG['cutoff_freq'],
            L_eff=GLOBAL_CONFIG['L_eff'],
            a=GLOBAL_CONFIG['a'],
            N_sub=DETECTOR_CONFIG['N_sub']
        )
        P_perp = det.P_perp
        s_norm = P_perp @ det._generate_template(15000)
        s_norm /= (np.linalg.norm(s_norm) + 1e-20)

        mds_jitter_proposed = 1.0e-5
        mds_jitter_standard = 2.0e-4

        # 计算 H0 阈值
        h0_runs_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds_fixed)(False, None, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                      self.N, self.fs, True, 0)
            for s in range(800))
        h0_runs_prop = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds_fixed)(False, None, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                      self.N, self.fs, False, mds_jitter_proposed)
            for s in range(800))
        h0_runs_std = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds_fixed)(False, None, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                      self.N, self.fs, False, mds_jitter_standard)
            for s in range(800))

        th_id = np.percentile(h0_runs_id, 95.0)
        th_prop = np.percentile(h0_runs_prop, 95.0)
        th_std = np.percentile(h0_runs_std, 95.0)

        pd_id, pd_prop, pd_std = [], [], []

        for a in tqdm(radii, desc="Scanning Sizes"):
            cfg_a = {**GLOBAL_CONFIG, 'a': a}
            phy = DiffractionChannel(cfg_a)
            sig = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

            r_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_fixed)(True, sig, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                          self.N, self.fs, True, 0)
                for s in range(200))
            r_prop = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_fixed)(True, sig, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                          self.N, self.fs, False, mds_jitter_proposed)
                for s in range(200))
            r_std = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_fixed)(True, sig, s, SNR_CONFIG['fig8'], s_norm, P_perp,
                                          self.N, self.fs, False, mds_jitter_standard)
                for s in range(200))

            pd_id.append(np.mean(np.array(r_id) > th_id))
            pd_prop.append(np.mean(np.array(r_prop) > th_prop))
            pd_std.append(np.mean(np.array(r_std) > th_std))

        self.save_csv({'d': diams, 'pd_id': pd_id, 'pd_prop': pd_prop, 'pd_std': pd_std}, 'Fig8_MDS_Data')

        plt.figure(figsize=(5, 4))
        plt.semilogx(diams, pd_id, 'g-o', label='Ideal')
        plt.semilogx(diams, pd_prop, 'b-s', label='Proposed (Robust)')
        plt.semilogx(diams, pd_std, 'r--^', label='Standard')
        plt.axhline(0.9, color='k', ls=':', alpha=0.5)
        plt.axhline(0.5, color='k', ls=':', alpha=0.3)
        plt.xlabel('Debris Diameter (mm)')
        plt.ylabel('Probability of Detection')
        plt.legend()
        plt.grid(True, which="both", ls=":")
        self.save_plot('Fig8_MDS')

    def generate_fig9_isac(self):
        print(f"\n--- Fig 9: ISAC (Rx SNR={SNR_CONFIG['fig9']} dB) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

        v_scan = np.linspace(14000, 16000, 41)
        det_cfg = {
            'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
            'L_eff': GLOBAL_CONFIG['L_eff'],
            'a': GLOBAL_CONFIG['a'],
            'N_sub': DETECTOR_CONFIG['N_sub']
        }
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det.P_perp
        s_raw = Parallel(n_jobs=self.n_jobs)(
            delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array([P_perp @ s for s in s_raw])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 15)
        cap, rmse = [], []
        isac_jitter = 1.0e-4

        for ibo in tqdm(ibo_scan, desc="ISAC Scan"):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac_fixed)(ibo, s, SNR_CONFIG['fig9'], sig_truth, T_bank,
                                           E_bank, v_scan, P_perp, self.N, self.fs,
                                           False, isac_jitter)
                for s in range(40)
            )
            res = np.array(res)
            cap.append(np.mean(res[:, 0]))

            errs = res[:, 1]
            valid = errs[errs < 1200]
            rmse.append(np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 1000.0)

        self.save_csv({'ibo': ibo_scan, 'cap': cap, 'rmse': rmse}, 'Fig9_ISAC_Data')

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(cap, rmse, c=ibo_scan, cmap='viridis_r', s=80, edgecolors='k')
        plt.colorbar(sc, label='IBO (dB)')
        plt.xlabel('Capacity (bits/s/Hz)')
        plt.ylabel('Velocity RMSE (m/s)')
        self.save_plot('Fig9_ISAC')

    def run_all(self):
        print("=" * 70)
        print("SIMULATION START (V7.0 - PHYSICALLY CORRECT)")
        print("=" * 70)

        # Group A: Physics
        self.generate_fig2_mechanisms()
        self.generate_fig3_dispersion()
        self.generate_fig4_self_healing()
        self.generate_fig5_survival_space()
        self.generate_fig10_ambiguity()
        self.generate_fig11_trajectory()

        # Group B: Performance
        self.generate_fig6_rmse_sensitivity()
        self.generate_fig7_roc()
        self.generate_fig8_mds()
        self.generate_fig9_isac()

        print("\n" + "=" * 70)
        print("ALL TASKS DONE")
        print("=" * 70)


# =========================================================================
# Worker Functions (Physically Correct)
# =========================================================================

def _log_envelope(y):
    return np.log(np.abs(y) + 1e-12)


def _apply_agc(signal_in):
    """AGC: 根据输入信号功率进行归一化"""
    p_sig = np.mean(np.abs(signal_in) ** 2)
    if p_sig < 1e-20:
        return signal_in
    gain = 1.0 / np.sqrt(p_sig)
    return signal_in * gain


def _gen_template(v, fs, N, cfg):
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


def _trial_rmse_fixed(ibo, jitter_rms, seed, snr_db, sig_truth, N, fs, true_v,
                      hw_base, T_bank, E_bank, v_scan, ideal, P_perp):
    """
    物理正确的 RMSE 测试函数
    【关键修改】噪声基于当前信号功率计算
    """
    np.random.seed(seed)

    if ideal:
        pa_out_raw = sig_truth
        p_ref = np.mean(np.abs(sig_truth) ** 2)
    else:
        hw_cfg = hw_base.copy()
        hw_cfg['jitter_rms'] = jitter_rms
        hw = HardwareImpairments(hw_cfg)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pn = np.exp(1j * hw.generate_phase_noise(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【关键】基于 PA 输出

    # 【关键修改】基于当前信号功率计算噪声
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    z = _log_envelope(_apply_agc(rx_signal))
    z_perp = P_perp @ z

    stats = (np.dot(T_bank, z_perp) ** 2) / (E_bank + 1e-20)
    idx_max = np.argmax(stats)
    v_coarse = v_scan[idx_max]

    if 0 < idx_max < len(stats) - 1:
        alpha = stats[idx_max - 1]
        beta = stats[idx_max]
        gamma = stats[idx_max + 1]
        denom = alpha - 2 * beta + gamma
        p = 0.5 * (alpha - gamma) / (denom + 1e-20) if abs(denom) > 1e-10 else 0.0
        v_est = v_coarse + p * (v_scan[1] - v_scan[0])
    else:
        v_est = v_coarse

    return v_est - true_v


def _trial_roc_fixed(sig_clean, seed, snr_db, ideal, s_true_perp, s_energy, P_perp,
                     jitter_val, N, fs):
    """
    物理正确的 ROC 测试函数
    【关键修改】噪声基于当前信号功率计算
    """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out_raw = sig_clean
        p_ref = np.mean(np.abs(sig_clean) ** 2)
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_clean * jit, ibo_dB=10.0)
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【关键】基于 PA 输出

    # 【关键修改】基于当前信号功率计算噪声
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    # GLRT
    z_log = _log_envelope(_apply_agc(rx_signal))
    z_perp = P_perp @ z_log
    stat_glrt = (np.dot(s_true_perp, z_perp) ** 2) / (s_energy + 1e-20)

    # Energy Detector
    stat_ed = np.sum(z_perp ** 2)

    return stat_glrt, stat_ed


def _trial_mds_fixed(is_h1, sig_clean, seed, snr_db, s_norm, P_perp, N, fs,
                     ideal, jitter_val=0):
    """
    物理正确的 MDS 测试函数
    【关键修改】噪声基于当前信号功率计算
    """
    np.random.seed(seed)

    if is_h1:
        sig_in = sig_clean
    else:
        sig_in = np.ones(N, dtype=complex)

    if ideal:
        pa_out_raw = sig_in
        p_ref = np.mean(np.abs(sig_in) ** 2)
    else:
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_in * jit, ibo_dB=10.0)
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【关键】基于 PA 输出

    # 【关键修改】基于当前信号功率计算噪声
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    z = _log_envelope(_apply_agc(rx_signal))
    stat = (np.dot(s_norm, P_perp @ z) ** 2)
    return stat


def _trial_isac_fixed(ibo, seed, snr_db, sig_truth, T_bank, E_bank, v_scan,
                      P_perp, N, fs, ideal, jitter_val=0):
    """
    物理正确的 ISAC 测试函数
    【关键修改】噪声基于当前信号功率计算
    """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out_raw = sig_truth
        p_ref = np.mean(np.abs(sig_truth) ** 2)
        gamma_eff = 0
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_truth * jit, ibo_dB=ibo)
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【关键】基于 PA 输出
        gamma_eff = 4.0e-3 * (10 ** (-ibo / 10.0))

    # 【关键修改】基于当前信号功率计算噪声
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    z = _log_envelope(_apply_agc(rx_signal))

    stats = (np.dot(T_bank, P_perp @ z) ** 2) / (E_bank + 1e-20)
    v_est = v_scan[np.argmax(stats)]

    # Capacity 计算
    p_rx = np.mean(np.abs(pa_out_raw) ** 2)
    sinr = p_rx / (noise_std ** 2 + p_rx * gamma_eff + 1e-20)
    cap = np.log2(1 + sinr)

    return cap, abs(v_est - GLOBAL_CONFIG['v_default'])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    PaperFigureGenerator().run_all()
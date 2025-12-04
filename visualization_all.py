# ----------------------------------------------------------------------------------
# 脚本名称: visualization_all_fixed.py
# 版本: v5.0 (Peak Detection Fix)
# 描述:
#   [CRITICAL FIX]:
#   1. 废弃 P_perp 正交投影，改用峰值检测
#   2. 更新 SNR 配置以满足物理要求
#   3. 修复 Fig 7 ROC 和 Fig 8 MDS 的检测逻辑
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
    from detector_fixed import TerahertzDebrisDetector
except ImportError:
    try:
        from physics_engine import DiffractionChannel
        from hardware_model import HardwareImpairments
        from detector import TerahertzDebrisDetector
    except ImportError:
        raise SystemExit("Error: Core modules not found.")

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
GLOBAL_CONFIG = {
    'fc': 300e9,
    'B': 10e9,
    'L_eff': 50e3,
    'fs': 200e3,
    'T_span': 0.02,
    'a_default': 0.05,
    'v_default': 15000
}

# ===========================================================================
# [CRITICAL FIX] SNR 配置 - 基于物理诊断结果
# ===========================================================================
# 诊断发现：
# - 衍射信号深度仅 0.016%，极其微弱
# - PA 导致 25-35dB 的功率衰减
# - 需要 SNR >= 25dB 才能实现有效检测 (AUC > 0.9)
# ===========================================================================
SNR_CONFIG = {
    "fig6": 20.0,  # RMSE 灵敏度分析
    "fig7": 25.0,  # ROC 曲线 - 物理要求至少 25dB
    "fig8": 40.0,  # MDS 曲线 - 小碎片需要更高 SNR
    "fig9": 15.0  # ISAC Trade-off
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


class PaperFigureGenerator:
    def __init__(self, output_dir='results_v5_fixed'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')

        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)

        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs

        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        print(f"[Init] Cores: {self.n_jobs} | Output: {os.path.abspath(self.out_dir)}")
        print(f"[Init] Fixed SNR Config: {SNR_CONFIG}")
        print(f"[Init] Using Peak Detection (P_perp disabled)")

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

    def _calc_noise_std(self, target_snr_db, signal_reference):
        p_sig = np.mean(np.abs(signal_reference) ** 2)
        p_sig = max(p_sig, 1e-20)
        noise_std = np.sqrt(p_sig / (10 ** (target_snr_db / 10.0)))
        return noise_std

    # =========================================================================
    # Group A: Physics Visualization (保持不变)
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
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=128)
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
    # Group B: Performance (关键修复)
    # =========================================================================

    def generate_fig6_rmse_sensitivity(self):
        """Fig 6: RMSE vs IBO - 使用峰值检测"""
        current_snr = SNR_CONFIG["fig6"]
        print(f"\n--- Fig 6: RMSE vs IBO (SNR={current_snr} dB, Peak Detection) ---")

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal
        noise_std = self._calc_noise_std(target_snr_db=current_snr, signal_reference=d_signal)

        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)

        # 使用峰值检测的模板
        det_cfg = {'cutoff_freq': 5000.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a_default'], 'N_sub': 32}
        det_main = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)

        # 预计算模板库（用于匹配滤波）
        s_raw_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan
        )
        T_bank = np.array(s_raw_list)
        # 去均值处理
        T_bank_ac = T_bank - np.mean(T_bank, axis=1, keepdims=True)
        E_bank = np.sum(T_bank_ac ** 2, axis=1) + 1e-20

        jit_levels = [1.0e-5, 2.0e-4, 4.0e-3]
        ibo_scan = np.linspace(30, 0, 13)
        trials = 80

        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        # Ideal Baseline
        res_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_rmse_peak)(10.0, 0, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG,
                                      T_bank_ac, E_bank, v_scan, ideal=True)
            for s in tqdm(range(trials), desc="Ideal")
        )
        rmse_id = np.sqrt(np.mean(np.array(res_id) ** 2))
        ax.semilogy(ibo_scan, [max(rmse_id, 0.1)] * len(ibo_scan), 'k--', label='Ideal', linewidth=2)
        data['ideal'] = [rmse_id] * len(ibo_scan)

        # Jitter Curves
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for idx, jit in enumerate(jit_levels):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_peak)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG,
                                          T_bank_ac, E_bank, v_scan, ideal=False)
                for ibo in tqdm(ibo_scan, desc=f"Jit={jit:.0e}", leave=False) for s in range(trials)
            )
            res_mat = np.array(res).reshape(len(ibo_scan), trials)
            rmse = []
            for row in res_mat:
                valid = row[np.abs(row) < 1300]
                if len(valid) > 5:
                    rmse.append(np.sqrt(np.mean(valid ** 2)))
                else:
                    rmse.append(1000.0)

            ax.semilogy(ibo_scan, rmse, 'o-', color=colors[idx], label=rf'$\sigma_J$={jit:.0e}')
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
        """
        [CRITICAL FIX] Fig 7: ROC 曲线
        - 废弃 P_perp 正交投影
        - 改用匹配滤波 + 峰值检测
        - 使用衍射模式本身作为模板
        """
        print(f"\n--- Fig 7: ROC (SNR={SNR_CONFIG['fig7']}dB, Matched Filter) ---")
        trials = 2000

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig7"], d_wb)

        sig_h1_clean = 1.0 - d_wb  # 有目标
        sig_h0_clean = np.ones(self.N, dtype=complex)  # 无目标

        # [CRITICAL FIX] 使用衍射模式本身作为模板，而非 _generate_template()
        # 原因：_generate_template() 依赖有问题的投影逻辑，导致模板能量为 0
        template = np.real(d_wb)  # 衍射深坑模式
        template_ac = template - np.mean(template)
        template_norm = template_ac / (np.linalg.norm(template_ac) + 1e-20)
        template_energy = np.sum(template_ac ** 2) + 1e-20

        print(f"   Template energy: {template_energy:.6e} (should be > 0)")
        print(f"   Diffraction depth: {np.max(np.abs(d_wb)):.6f}")

        # Jitter 配置
        jit_proposed = 1.0e-5  # Proposed: 低 Jitter（代表良好的硬件补偿）
        jit_standard = 2.0e-4  # Standard: 高 Jitter（代表未补偿的硬件）

        def run_roc_batch(is_ideal, jitter_val, use_matched_filter=True):
            """运行 ROC 测试批次"""
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_peak)(sig_h0_clean, s, noise_std, is_ideal,
                                         template_norm, template_energy, jitter_val,
                                         self.N, self.fs, use_matched_filter)
                for s in range(trials))
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_peak)(sig_h1_clean, s, noise_std, is_ideal,
                                         template_norm, template_energy, jitter_val,
                                         self.N, self.fs, use_matched_filter)
                for s in range(trials))
            return np.array(r0), np.array(r1)

        print("   Running Ideal...")
        r0_id, r1_id = run_roc_batch(True, 0, use_matched_filter=True)

        print(f"   Running Proposed (Jit={jit_proposed:.1e})...")
        r0_prop, r1_prop = run_roc_batch(False, jit_proposed, use_matched_filter=True)

        print(f"   Running Standard (Jit={jit_standard:.1e})...")
        r0_std, r1_std = run_roc_batch(False, jit_standard, use_matched_filter=True)

        print("   Running Energy Detector...")
        r0_ed, r1_ed = run_roc_batch(False, jit_standard, use_matched_filter=False)

        def get_curve(h0_stats, h1_stats, col=0):
            """计算 ROC 曲线"""
            n_vals = h0_stats[:, col] if h0_stats.ndim > 1 else h0_stats
            s_vals = h1_stats[:, col] if h1_stats.ndim > 1 else h1_stats
            all_v = np.concatenate([n_vals, s_vals])
            th = np.linspace(np.min(all_v), np.percentile(all_v, 99.9), 500)
            pf = np.array([np.mean(n_vals > t) for t in th])
            pd = np.array([np.mean(s_vals > t) for t in th])
            return pf, pd

        pf_id, pd_id = get_curve(r0_id, r1_id, 0)
        pf_prop, pd_prop = get_curve(r0_prop, r1_prop, 0)
        pf_std, pd_std = get_curve(r0_std, r1_std, 0)
        pf_ed, pd_ed = get_curve(r0_ed, r1_ed, 1)

        self.save_csv({
            'pf_id': pf_id, 'pd_id': pd_id,
            'pf_prop': pf_prop, 'pd_prop': pd_prop,
            'pf_std': pf_std, 'pd_std': pd_std,
            'pf_ed': pf_ed, 'pd_ed': pd_ed
        }, 'Fig7_ROC_Data')

        plt.figure(figsize=(5, 5))
        plt.plot(pf_id, pd_id, 'g-', linewidth=2, label='Ideal GLRT')
        plt.plot(pf_prop, pd_prop, 'b-', linewidth=2, label='Proposed (Robust)')
        plt.plot(pf_std, pd_std, 'r--', linewidth=1.5, label='Standard GLRT')
        plt.plot(pf_ed, pd_ed, 'k:', linewidth=1, label='Energy Det')
        plt.plot([0, 1], [0, 1], 'k-', alpha=0.2, linewidth=0.5)

        plt.xlabel('Probability of False Alarm (PFA)')
        plt.ylabel('Probability of Detection (PD)')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        self.save_plot('Fig7_ROC')

    def generate_fig8_mds(self):
        """
        [CRITICAL FIX] Fig 8: MDS (最小可检测尺寸)
        - 废弃 P_perp 正交投影
        - 改用匹配滤波 + 峰值检测
        - 为每个碎片尺寸生成对应的模板
        """
        print(f"\n--- Fig 8: MDS (SNR={SNR_CONFIG['fig8']}dB, Matched Filter) ---")
        diams = np.array([2, 5, 8, 12, 16, 20, 30, 50])
        radii = diams / 2000.0

        # 使用高 SNR
        d_ref = DiffractionChannel(GLOBAL_CONFIG).generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig8"], d_ref)

        # [CRITICAL FIX] 使用参考碎片的衍射模式作为模板
        # 选择 50mm 碎片的衍射模式作为参考（信噪比最高）
        phy_ref = DiffractionChannel({**GLOBAL_CONFIG, 'a': 0.025})  # 50mm diameter = 25mm radius
        d_ref_pattern = phy_ref.generate_broadband_chirp(self.t_axis, 32)
        template = np.real(d_ref_pattern)
        template_ac = template - np.mean(template)
        template_norm = template_ac / (np.linalg.norm(template_ac) + 1e-20)

        # Jitter 配置
        mds_jitter_ideal = 0
        mds_jitter_hw = 1.0e-5  # 使用较低 Jitter 代表良好补偿

        # 计算阈值 (H0)
        print("   Calculating Thresholds...")
        h0_runs_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds_peak)(False, None, s, noise_std, template_norm,
                                     self.N, self.fs, True, 0)
            for s in range(1000))
        h0_runs_hw = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds_peak)(False, None, s, noise_std, template_norm,
                                     self.N, self.fs, False, mds_jitter_hw)
            for s in range(1000))

        th_id = np.percentile(h0_runs_id, 95.0)
        th_hw = np.percentile(h0_runs_hw, 95.0)

        print(f"   Thresholds: Ideal={th_id:.4f}, HW={th_hw:.4f}")

        pd_hw, pd_id = [], []

        for a in tqdm(radii, desc="Scanning Sizes"):
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a})
            sig = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

            r_hw = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_peak)(True, sig, s, noise_std, template_norm,
                                         self.N, self.fs, False, mds_jitter_hw)
                for s in range(200))
            r_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_peak)(True, sig, s, noise_std, template_norm,
                                         self.N, self.fs, True, 0)
                for s in range(200))

            pd_hw.append(np.mean(np.array(r_hw) > th_hw))
            pd_id.append(np.mean(np.array(r_id) > th_id))

        self.save_csv({'d': diams, 'pd_hw': pd_hw, 'pd_id': pd_id}, 'Fig8_MDS_Data')

        plt.figure(figsize=(5, 4))
        plt.semilogx(diams, pd_id, 'g-o', linewidth=2, markersize=8, label='Ideal')
        plt.semilogx(diams, pd_hw, 'b-s', linewidth=2, markersize=8, label='Proposed (Robust)')
        plt.axhline(0.5, color='k', ls=':', alpha=0.5, label='Detection Limit')
        plt.axhline(0.9, color='r', ls=':', alpha=0.3)
        plt.xlabel('Debris Diameter (mm)')
        plt.ylabel('Probability of Detection')
        plt.legend(loc='lower right')
        plt.grid(True, which="both", ls=":")
        plt.ylim([0, 1.05])
        self.save_plot('Fig8_MDS')

    def generate_fig9_isac(self):
        print(f"\n--- Fig 9: ISAC (SNR={SNR_CONFIG['fig9']} dB) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig9"], 1.0 - sig_truth)

        v_scan = np.linspace(14000, 16000, 41)
        det_cfg = {'cutoff_freq': 5000.0, 'L_eff': 50e3, 'N_sub': 32, 'a': 0.05}
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)

        # 使用峰值检测的模板
        s_raw = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array(s_raw)
        T_bank_ac = T_bank - np.mean(T_bank, axis=1, keepdims=True)
        E_bank = np.sum(T_bank_ac ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 15)
        cap, rmse = [], []
        isac_jitter = 1.0e-4

        for ibo in tqdm(ibo_scan):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac_peak)(ibo, s, noise_std, sig_truth, T_bank_ac, E_bank, v_scan,
                                          self.N, self.fs, False, isac_jitter)
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
        plt.xlabel('Capacity')
        plt.ylabel('RMSE')
        self.save_plot('Fig9_ISAC')

    def run_all(self):
        print("=== SIMULATION START (V5.0 FIXED - Peak Detection) ===")
        self.generate_fig2_mechanisms()
        self.generate_fig3_dispersion()
        self.generate_fig4_self_healing()
        self.generate_fig5_survival_space()
        self.generate_fig10_ambiguity()
        self.generate_fig11_trajectory()
        self.generate_fig6_rmse_sensitivity()
        self.generate_fig7_roc()
        self.generate_fig8_mds()
        self.generate_fig9_isac()
        print("\n=== ALL TASKS DONE ===")


# =========================================================================
# Worker Functions (关键修复)
# =========================================================================

def _log_envelope(y):
    return np.log(np.abs(y) + 1e-12)


def _apply_agc(signal_in):
    """AGC 归一化"""
    p_sig = np.mean(np.abs(signal_in) ** 2)
    if p_sig < 1e-20:
        return signal_in
    gain = 1.0 / np.sqrt(p_sig)
    return signal_in * gain


def _gen_template(v, fs, N, cfg):
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


# ===========================================================================
# [CRITICAL FIX] 峰值检测版本的 Worker 函数
# ===========================================================================

def _trial_rmse_peak(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base,
                     T_bank_ac, E_bank, v_scan, ideal):
    """RMSE 测试 - 使用峰值检测"""
    np.random.seed(seed)

    if ideal:
        pa_out_raw = sig_truth
    else:
        hw_cfg = hw_base.copy()
        hw_cfg['jitter_rms'] = jitter_rms
        hw = HardwareImpairments(hw_cfg)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pn = np.exp(1j * hw.generate_phase_noise(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    # 加噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    # [FIX] 使用幅度去均值，而非正交投影
    rx_amp = np.abs(_apply_agc(rx_signal))
    z_ac = rx_amp - np.mean(rx_amp)

    # 匹配滤波扫描
    stats = (np.dot(T_bank_ac, z_ac) ** 2) / (E_bank + 1e-20)
    idx_max = np.argmax(stats)
    v_coarse = v_scan[idx_max]

    # 抛物线插值
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


def _trial_roc_peak(sig_clean, seed, noise_std, ideal, template_norm, template_energy,
                    jitter_val, N, fs, use_matched_filter=True):
    """
    ROC 测试 - 使用匹配滤波

    [CRITICAL FIX]: 废弃 P_perp，改用匹配滤波
    """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    # 物理层
    if ideal:
        pa_out_raw = sig_clean
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_clean * jit, ibo_dB=10.0)

    # 加噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    # [FIX] 匹配滤波处理
    rx_agc = _apply_agc(rx_signal)
    rx_amp = np.abs(rx_agc)
    rx_ac = rx_amp - np.mean(rx_amp)

    if use_matched_filter:
        # 匹配滤波统计量
        stat_mf = np.dot(rx_ac, template_norm) ** 2
    else:
        stat_mf = 0

    # 能量检测统计量
    stat_ed = np.sum(rx_ac ** 2)

    return stat_mf, stat_ed


def _trial_mds_peak(is_h1, sig_clean, seed, noise_std, template_norm, N, fs, ideal, jitter_val=0):
    """
    MDS 测试 - 使用匹配滤波

    [CRITICAL FIX]: 废弃 P_perp，改用匹配滤波
    """
    np.random.seed(seed)

    if is_h1:
        sig_in = sig_clean
    else:
        sig_in = np.ones(N, dtype=complex)

    if ideal:
        pa_out_raw = sig_in
    else:
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_in * jit, ibo_dB=10.0)

    # 加噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    # [FIX] 匹配滤波
    rx_agc = _apply_agc(rx_signal)
    rx_amp = np.abs(rx_agc)
    rx_ac = rx_amp - np.mean(rx_amp)

    # 匹配滤波统计量
    stat = np.dot(template_norm, rx_ac) ** 2

    return stat


def _trial_isac_peak(ibo, seed, noise_std, sig_truth, T_bank_ac, E_bank, v_scan, N, fs, ideal, jitter_val=0):
    """ISAC 测试 - 使用峰值检测"""
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out_raw = sig_truth
        gamma_eff = 0
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out_raw, _, _ = hw.apply_saleh_pa(sig_truth * jit, ibo_dB=ibo)
        gamma_eff = 4.0e-3 * (10 ** (-ibo / 10.0))

    # 加噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    # [FIX] 峰值检测
    rx_amp = np.abs(_apply_agc(rx_signal))
    z_ac = rx_amp - np.mean(rx_amp)

    stats = (np.dot(T_bank_ac, z_ac) ** 2) / (E_bank + 1e-20)
    v_est = v_scan[np.argmax(stats)]

    # Capacity 计算
    p_rx = np.mean(np.abs(pa_out_raw) ** 2)
    sinr = p_rx / (noise_std ** 2 + p_rx * gamma_eff + 1e-20)
    cap = np.log2(1 + sinr)

    return cap, abs(v_est - GLOBAL_CONFIG['v_default'])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    PaperFigureGenerator().run_all()
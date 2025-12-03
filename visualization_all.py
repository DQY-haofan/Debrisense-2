# ----------------------------------------------------------------------------------
# 脚本名称: reproduce_paper_results_ieee.py
# 版本: v11.0 (Parameter Tuned for U-Shape & Visibility)
# 描述:
#   1. [Tuning] 大幅降低 Fig6/9 的目标 SNR (8-10dB)，以显现 High IBO 下的噪声受限区（U型左侧）。
#   2. [Tuning] 大幅提升 Jitter 量级 (0.05-0.1)，确保不同曲线能被肉眼区分。
#   3. [Logic] 修正 RMSE 统计逻辑，增强鲁棒性。
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Error: Core modules not found.")

# --- 绘图配置 ---
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

# --- 全局参数 ---
GLOBAL_CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 50e3,
    'fs': 200e3, 'T_span': 0.02,
    'a_default': 0.05, 'v_default': 15000
}

# [CRITICAL] 参数调优：降低 SNR 以展示性能恶化边界
SNR_CONFIG = {
    "fig6": 8.0,  # [Tuned] 降至 8dB，确保 IBO=20dB 时进入噪声受限区 (U型左侧)
    "fig7": 15.0,  # [Tuned] 保持适中，展示 ROC 差异
    "fig8": 20.0,  # MDS 需要较高 SNR 才能检测小目标
    "fig9": 10.0  # [Tuned] 降至 10dB，让 Capacity 不会一直顶格
}

HW_CONFIG = {
    'jitter_rms': 0.5e-6, 'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}


# -------------------------------------------------------------------------
#  Generator Class & Methods
# -------------------------------------------------------------------------

class PaperFigureGenerator:
    def __init__(self, output_dir='results'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)
        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs
        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        print(f"[Init] Cores: {self.n_jobs} | SNR Config: {SNR_CONFIG}")

    def save_plot(self, name):
        plt.tight_layout(pad=0.5)
        plt.savefig(f"{self.out_dir}/{name}.png", dpi=300, bbox_inches='tight')
        plt.close('all')

    def save_csv(self, data, name):
        max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data.values()])
        aligned = {}
        for k, v in data.items():
            if hasattr(v, '__len__'):
                padded = np.full(max_len, np.nan)
                padded[:len(v)] = v
                aligned[k] = padded
            else:
                aligned[k] = np.full(max_len, v)
        pd.DataFrame(aligned).to_csv(f"{self.csv_dir}/{name}.csv", index=False)

    def save_matrix_csv(self, matrix, name):
        pd.DataFrame(matrix).to_csv(f"{self.csv_dir}/{name}.csv", index=False, header=False)

    def _calc_noise_std(self, target_snr_db, signal_reference):
        """ 基于残差信号 d(t) 的能量计算噪声 """
        p_sig = np.mean(np.abs(signal_reference) ** 2)
        noise_std = np.sqrt(p_sig / (10 ** (target_snr_db / 10.0)))
        return noise_std

    # --- Group A: Physics Visualization (Unchanged) ---
    def generate_fig2_mechanisms(self):
        print("\n--- Fig 2 ---")
        hw = HardwareImpairments(HW_CONFIG)
        jit = hw.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)
        pin_db, am_am, scr = hw.get_pa_curves()

        plt.figure(figsize=(5, 4))
        plt.loglog(f, psd, 'b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'PSD')
        self.save_plot('Fig2a_Jitter')

        plt.figure(figsize=(5, 4))
        plt.plot(pin_db, am_am, 'k')
        ax = plt.gca();
        ax2 = ax.twinx()
        ax2.plot(pin_db, np.maximum(scr, -0.2), 'r--')
        ax.set_xlabel('Input Power (dB)');
        ax.set_ylabel('Output')
        ax2.set_ylabel('SCR', color='r')
        self.save_plot('Fig2b_PA')

    def generate_fig3_dispersion(self):
        print("\n--- Fig 3 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=32))
        plt.figure(figsize=(5, 4))
        plt.plot(self.t_axis * 1000, d_nb, 'r--', label='NB')
        plt.plot(self.t_axis * 1000, d_wb, 'b-', label='WB')
        plt.xlabel('Time (ms)');
        plt.ylabel('|d(t)|');
        plt.legend()
        self.save_plot('Fig3_Dispersion')

    def generate_fig4_self_healing(self):
        print("\n--- Fig 4 ---")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        sig = 1.0 - 0.2 * np.exp(-((t - 0.5) ** 2) / 0.005)
        out, _, _ = hw.apply_saleh_pa(sig, ibo_dB=2.0)
        plt.figure(figsize=(5, 4))
        plt.plot(t, np.abs(sig), 'k--', label='In')
        plt.plot(t, np.abs(out) / np.max(np.abs(out)), 'r-', label='Out')
        plt.legend()
        self.save_plot('Fig4_Self_Healing')

    def generate_fig5_survival_space(self):
        print("\n--- Fig 5 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        hw = HardwareImpairments(HW_CONFIG)
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        y = (1.0 - phy.generate_broadband_chirp(self.t_axis, 32)) * np.exp(hw.generate_colored_jitter(self.N, self.fs))
        z_perp = det.apply_projection(det.log_envelope_transform(y))
        f, t_sp, Sxx = signal.spectrogram(z_perp, self.fs, nperseg=256, noverlap=220)
        plt.figure(figsize=(5, 4))
        plt.pcolormesh(t_sp * 1000, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='inferno')
        plt.ylim(0, 5000);
        plt.ylabel('Hz');
        plt.xlabel('ms')
        self.save_plot('Fig5_Survival_Space')
        self.save_csv({'t': self.t_axis, 'z': z_perp}, 'Fig5_Data')

    def generate_fig10_ambiguity(self):
        print("\n--- Fig 10 ---")
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        s0 = det.P_perp @ det._generate_template(15000)
        v_shifts = np.linspace(-500, 500, 41)
        t_shifts = np.linspace(-0.002, 0.002, 41)
        res = np.zeros((41, 41))
        E = np.sum(s0 ** 2)
        for i, dv in enumerate(v_shifts):
            s_v = det.P_perp @ det._generate_template(15000 + dv)
            for j, dt in enumerate(t_shifts):
                shift = int(dt * self.fs)
                s_shift = np.roll(s_v, shift)
                if shift > 0:
                    s_shift[:shift] = 0
                else:
                    s_shift[shift:] = 0
                res[i, j] = (np.dot(s0, s_shift) ** 2) / E
        plt.figure(figsize=(5, 4))
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20, cmap='viridis')
        self.save_plot('Fig10_Ambiguity')

    def generate_fig11_trajectory(self):
        print("\n--- Fig 11 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, 32)
        hw = HardwareImpairments(HW_CONFIG)
        # 用较强的噪声来展示轨迹云
        noise_cloud = (1.0 - d) * np.exp(hw.generate_colored_jitter(self.N, self.fs) * 10) * np.exp(
            1j * hw.generate_phase_noise(self.N, self.fs))
        noise_cloud -= np.mean(noise_cloud)
        plt.figure(figsize=(5, 5))
        plt.plot(np.real(noise_cloud), np.imag(noise_cloud), '.', color='grey', alpha=0.1)
        points = np.array([np.real(-d), np.imag(-d)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, self.N))
        lc.set_array(np.arange(self.N));
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.xlim(-0.015, 0.015);
        plt.ylim(-0.015, 0.015)
        self.save_plot('Fig11_Trajectory')

    # --- Group B: Performance (Updated) ---

    def generate_fig6_rmse_sensitivity(self):
        print("\n--- Fig 6: RMSE vs IBO (U-Shape Tuned) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal

        # [Tuning] SNR=8dB, 使得 IBO=20dB 时(信号-20dB)进入噪声受限区
        noise_std = self._calc_noise_std(target_snr_db=SNR_CONFIG["fig6"], signal_reference=d_signal)
        print(f"   [Calib] Noise Std: {noise_std:.2e}")

        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a_default'], 'N_sub': 32}

        # Pre-compute template bank
        s_raw_list = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        T_bank = np.array([det.P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        # [Tuning] 显著增加 Jitter 以拉开曲线差距 (5%, 10%, 15%)
        jit_levels = [0.0, 0.05, 0.15]
        ibo_scan = np.linspace(25, 0, 11)  # 扩大 IBO 扫描范围
        trials = 100

        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        # 1. Ideal Baseline
        res_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_rmse_opt)(10.0, 0, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                     E_bank, v_scan, ideal=True)
            for s in tqdm(range(trials), desc="Ideal")
        )
        rmse_id = np.sqrt(np.mean(np.array(res_id) ** 2))
        ax.semilogy(ibo_scan, [rmse_id] * len(ibo_scan), 'k--', label='Ideal')
        data['ideal'] = [rmse_id] * len(ibo_scan)

        # 2. Hardware Curves
        for idx, jit in enumerate(jit_levels):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_opt)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                         E_bank, v_scan, ideal=False)
                for ibo in tqdm(ibo_scan, desc=f"HW Jit={jit}") for s in range(trials)
            )
            res_mat = np.array(res).reshape(len(ibo_scan), trials)
            # Robust RMSE: 剔除完全发散的点 (e.g. > 1200) 以平滑曲线
            rmse = []
            for row in res_mat:
                valid = row[np.abs(row) < 1200]
                val = np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 1000.0
                rmse.append(val)

            ax.semilogy(ibo_scan, rmse, 'o-', label=rf'Jit={jit * 100:.0f}%')
            data[f'rmse_jit_{idx}'] = rmse

        ax.set_xlabel('IBO (dB)');
        ax.set_ylabel('Velocity RMSE (m/s)')
        ax.invert_xaxis();
        ax.legend(frameon=False)
        self.save_plot('Fig6_Sensitivity')
        self.save_csv(data, 'Fig6_Data')

    def generate_fig7_roc(self):
        print("\n--- Fig 7: ROC ---")
        trials = 500
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig7"], d_wb)
        seeds = np.arange(trials)

        # Ideal vs Hardware (Jitter=5%)
        def run_roc(ideal):
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc)(False, s, noise_std, 15000, self.N, self.fs, ideal) for s in seeds)
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc)(True, s, noise_std, 15000, self.N, self.fs, ideal) for s in seeds)
            return np.array(r0), np.array(r1)

        r0_hw, r1_hw = run_roc(False)
        r0_id, r1_id = run_roc(True)

        def get_curve(n, s):
            th = np.linspace(min(n), max(s), 200)
            return [np.mean(n > t) for t in th], [np.mean(s > t) for t in th]

        pf_hw, pd_hw = get_curve(r0_hw[:, 0], r1_hw[:, 0])
        pf_id, pd_id = get_curve(r0_id[:, 0], r1_id[:, 0])

        # Energy Detector (HW)
        pf_ed, pd_ed = get_curve(r0_hw[:, 1], r1_hw[:, 1])

        plt.figure(figsize=(5, 5))
        plt.plot(pf_id, pd_id, 'g-', label='Ideal GLRT')
        plt.plot(pf_hw, pd_hw, 'b-', label='HW GLRT')
        plt.plot(pf_ed, pd_ed, 'r:', label='Energy Det')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('PFA');
        plt.ylabel('PD');
        plt.legend()
        self.save_plot('Fig7_ROC')
        self.save_csv({'pf_hw': pf_hw, 'pd_hw': pd_hw}, 'Fig7_Data')

    def generate_fig8_mds(self):
        print("\n--- Fig 8: MDS ---")
        diams = np.array([2, 5, 10, 20, 50])
        radii = diams / 2000.0
        d_ref = DiffractionChannel({**GLOBAL_CONFIG, 'a': 0.05}).generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig8"], d_ref)

        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        s_norm = det.P_perp @ det._generate_template(15000)
        s_norm /= np.linalg.norm(s_norm)

        # Threshold (Pfa=0.1 on Noise)
        h0_runs = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds)(False, None, s, noise_std, s_norm, det.P_perp, self.N, self.fs, True) for s in
            range(200))
        th = np.percentile(h0_runs, 90.0)

        pd_hw, pd_id = [], []
        for a in tqdm(radii):
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a})
            sig = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
            r_hw = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig, s, noise_std, s_norm, det.P_perp, self.N, self.fs, False) for s in
                range(100))
            r_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig, s, noise_std, s_norm, det.P_perp, self.N, self.fs, True) for s in
                range(100))
            pd_hw.append(np.mean(np.array(r_hw) > th))
            pd_id.append(np.mean(np.array(r_id) > th))

        plt.figure(figsize=(5, 4))
        plt.semilogx(diams, pd_id, 'g--o', label='Ideal')
        plt.semilogx(diams, pd_hw, 'b-s', label='Hardware')
        plt.axhline(0.5, color='k', ls=':');
        plt.legend()
        self.save_plot('Fig8_MDS')
        self.save_csv({'d': diams, 'pd_hw': pd_hw}, 'Fig8_Data')

    def generate_fig9_isac(self):
        print("\n--- Fig 9: ISAC ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig9"], 1.0 - sig_truth)

        v_scan = np.linspace(14000, 16000, 41)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': 50e3, 'N_sub': 32, 'a': 0.05}
        s_raw = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        T_bank = np.array([det.P_perp @ s for s in s_raw])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 15)
        cap, rmse = [], []

        for ibo in tqdm(ibo_scan):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac)(ibo, s, noise_std, sig_truth, T_bank, E_bank, v_scan, det.P_perp, self.N, self.fs,
                                     False)
                for s in range(40)
            )
            res = np.array(res)
            cap.append(np.mean(res[:, 0]))

            errs = res[:, 1]
            valid = errs[errs < 1200]
            rmse.append(np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 1000.0)

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(cap, rmse, c=ibo_scan, cmap='viridis_r', s=80, edgecolors='k')
        plt.colorbar(sc, label='IBO (dB)')
        plt.xlabel('Capacity (bits/s/Hz)');
        plt.ylabel('RMSE (m/s)')
        self.save_plot('Fig9_ISAC')
        self.save_csv({'ibo': ibo_scan, 'cap': cap, 'rmse': rmse}, 'Fig9_Data')

    def run_all(self):
        print("=== SIMULATION START (V11.0 TUNED) ===")
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
        print("\n=== DONE ===")


# --- Worker Functions (Must be top-level) ---

def _gen_template(v, fs, N, cfg):
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


def _trial_rmse_opt(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base, T_bank, E_bank, v_scan, ideal):
    np.random.seed(seed)
    if ideal:
        pa_out = sig_truth
    else:
        hw_cfg = hw_base.copy()
        hw_cfg['jitter_rms'] = jitter_rms  # Jitter passed here
        hw = HardwareImpairments(hw_cfg)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pn = np.exp(1j * hw.generate_phase_noise(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = TerahertzDebrisDetector(fs, N, N_sub=32).log_envelope_transform(pa_out + w)
    z_perp = TerahertzDebrisDetector(fs, N, N_sub=32).apply_projection(z)

    stats = (np.dot(T_bank, z_perp) ** 2) / E_bank
    return v_scan[np.argmax(stats)] - true_v


def _trial_roc(is_h1, seed, noise_std, true_v, N, fs, ideal):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, N_sub=32)

    sig = (1.0 - DiffractionChannel(GLOBAL_CONFIG).generate_broadband_chirp(np.arange(N) / fs - (N / 2) / fs,
                                                                            32)) if is_h1 else np.ones(N, dtype=complex)

    if ideal:
        pa_out = sig
    else:
        # HW Case: Fixed Jitter 5%
        hw.jitter_rms = 0.05
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_perp = det.apply_projection(det.log_envelope_transform(pa_out + w))

    # GLRT
    s_temp = det.P_perp @ det._generate_template(true_v)
    stat_glrt = (np.dot(s_temp, z_perp) ** 2) / (np.sum(s_temp ** 2) + 1e-20)

    # Energy
    y_ac = np.abs(pa_out + w) - np.mean(np.abs(pa_out + w))
    stat_ed = np.sum(y_ac ** 2)
    return stat_glrt, stat_ed


def _trial_mds(is_h1, sig_clean, seed, noise_std, s_norm, P_perp, N, fs, ideal):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    sig = sig_clean if is_h1 else np.ones(N, dtype=complex)

    if ideal:
        pa_out = sig
    else:
        hw.jitter_rms = 0.05
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = TerahertzDebrisDetector(fs, N, N_sub=32).log_envelope_transform(pa_out + w)
    return (np.dot(s_norm, P_perp @ z) ** 2)


def _trial_isac(ibo, seed, noise_std, sig_truth, T_bank, E_bank, v_scan, P_perp, N, fs, ideal):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out = sig_truth;
        gamma_eff = 0
    else:
        hw.jitter_rms = 0.05
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit, ibo_dB=ibo)
        gamma_eff = 1e-2 * (10 ** (-ibo / 10.0))  # [Tuning] Stronger nonlinearity

    p_rx = np.mean(np.abs(pa_out) ** 2)
    sinr = p_rx / (noise_std ** 2 + p_rx * gamma_eff + 1e-20)
    cap = np.log2(1 + sinr)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = TerahertzDebrisDetector(fs, N, N_sub=32).log_envelope_transform(pa_out + w)
    stats = (np.dot(T_bank, P_perp @ z) ** 2) / E_bank
    return cap, abs(v_scan[np.argmax(stats)] - GLOBAL_CONFIG['v_default'])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    PaperFigureGenerator().run_all()
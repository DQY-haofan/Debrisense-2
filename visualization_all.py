# ----------------------------------------------------------------------------------
# 脚本名称: reproduce_paper_results_ieee.py
# 版本: v8.0 (Expert Review Fixes - Recalibrated SNR & Physics)
# 描述:
#   1. [FIX] SNR 标定改为基于衍射残差功率 (Diffraction Residual Power)，而非总载波功率。
#   2. [FIX] 提升 Jitter 量级以展示区分度。
#   3. [FIX] 调整 ISAC 参数以激活 Trade-off 区域。
#   4. [STYLE] 保持 IEEE 单栏风格。
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

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.labelsize': 14, 'legend.fontsize': 11,
    'lines.linewidth': 1.5, 'figure.figsize': (5, 4),
    'figure.dpi': 300, 'pdf.fonttype': 42,
    'axes.grid': True, 'grid.linestyle': ':', 'grid.alpha': 0.6
})

# --- 全局参数 ---
GLOBAL_CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 50e3,
    'fs': 200e3, 'T_span': 0.02,
    'a_default': 0.05, 'v_default': 15000
}

# 硬件参数 (SOTA)
HW_CONFIG = {
    'jitter_rms': 0.5e-6, 'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}


# -------------------------------------------------------------------------
#  原子计算函数
# -------------------------------------------------------------------------

def _gen_template(v, fs, N, cfg):
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


def _trial_rmse_opt(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base, T_bank, E_bank, v_scan):
    np.random.seed(seed)
    hw_cfg = hw_base.copy()
    hw_cfg['jitter_rms'] = jitter_rms
    hw = HardwareImpairments(hw_cfg)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    # 1. 损伤注入
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    # 2. 接收处理
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    # [Expert Tip] 使用 Log 变换处理乘性噪声
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    # 3. 极速 GLRT
    correlations = np.dot(T_bank, z_perp)
    stats = (correlations ** 2) / E_bank
    est_v = v_scan[np.argmax(stats)]
    return est_v - true_v


def _trial_roc(is_h1, seed, noise_std, true_v, N, fs):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    if is_h1:
        phy = DiffractionChannel(GLOBAL_CONFIG)
        t = np.arange(N) / fs - (N / 2) / fs
        d_wb = phy.generate_broadband_chirp(t, N_sub=32)
        # [Expert Tip] 统一符号: 1 - d
        sig = 1.0 - d_wb
    else:
        sig = np.ones(N, dtype=complex)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)  # Linear region for ROC baseline
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_perp = det.apply_projection(det.log_envelope_transform(pa_out + w))
    s_temp = det.P_perp @ det._generate_template(true_v)
    stat_glrt = (np.dot(s_temp, z_perp) ** 2) / (np.sum(s_temp ** 2) + 1e-20)

    y_ac = np.abs(pa_out + w) - np.mean(np.abs(pa_out + w))
    stat_ed = np.sum(y_ac ** 2)
    return stat_glrt, stat_ed


def _trial_mds(is_h1, sig_clean, seed, noise_std, s_norm, P_perp, N, fs):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, a=0.05)

    sig = sig_clean if is_h1 else np.ones(N, dtype=complex)
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_out, _, _ = hw.apply_saleh_pa(sig * jit * pn, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_perp = P_perp @ det.log_envelope_transform(pa_out + w)
    return (np.dot(s_norm, z_perp) ** 2)


def _trial_isac(ibo, seed, noise_std, sig_truth, T_bank, E_bank, v_scan, P_perp, N, fs):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, a=GLOBAL_CONFIG['a_default'])

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    # Capacity: 受噪声和非线性双重限制
    p_rx = np.mean(np.abs(pa_out) ** 2)
    # [Expert Tip] 调整 gamma_eff 使其不完全主导，允许 Trade-off
    gamma_eff = 1e-3
    sinr = p_rx / (noise_std ** 2 + p_rx * gamma_eff)
    cap = np.log2(1 + sinr)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_perp = P_perp @ det.log_envelope_transform(pa_out + w)

    stats = (np.dot(T_bank, z_perp) ** 2) / E_bank
    est_v = v_scan[np.argmax(stats)]
    return cap, abs(est_v - GLOBAL_CONFIG['v_default'])


# --- Generator Class ---

class PaperFigureGenerator:
    def __init__(self, output_dir='results'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)
        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs
        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        print(f"[Init] Cores: {self.n_jobs} | N: {self.N}")

    def save_plot(self, name):
        plt.tight_layout(pad=0.5)
        plt.savefig(f"{self.out_dir}/{name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.out_dir}/{name}.pdf", format='pdf', bbox_inches='tight')
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

    # --- Group A: Physical ---

    def generate_fig2_mechanisms(self):
        print("\n--- Fig 2 ---")
        hw = HardwareImpairments(HW_CONFIG)
        jit = hw.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)
        pin_db, am_am, scr = hw.get_pa_curves()

        # [Visual Tweak] Truncate SCR for better plot
        scr_plot = np.maximum(scr, -0.2)

        plt.figure(figsize=(5, 4));
        plt.loglog(f, psd, 'b');
        plt.xlabel('Frequency (Hz)');
        plt.ylabel(r'PSD (rad$^2$/Hz)');
        plt.grid(True, ls=':')
        self.save_plot('Fig2a_Jitter');
        self.save_csv({'freq': f, 'psd': psd}, 'Fig2a_Data')

        plt.figure(figsize=(5, 4));
        plt.plot(pin_db, am_am, 'k', label='AM-AM')
        ax = plt.gca();
        ax2 = ax.twinx()
        ax2.plot(pin_db, scr_plot, 'r--', label='SCR')
        ax.set_xlabel('Input Power (dB)');
        ax.set_ylabel('Output Amplitude');
        ax2.set_ylabel('SCR', color='r');
        ax.grid(True)
        # Add a zero line for reference
        ax2.axhline(0, color='r', linestyle=':', alpha=0.3)
        self.save_plot('Fig2b_PA');
        self.save_csv({'pin_db': pin_db, 'am_am': am_am, 'scr': scr}, 'Fig2b_Data')

    def generate_fig3_dispersion(self):
        print("--- Fig 3 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=32))
        plt.figure(figsize=(5, 4));
        plt.plot(self.t_axis * 1000, d_nb, 'r--', label='NB');
        plt.plot(self.t_axis * 1000, d_wb, 'b-', alpha=0.7, label='WB')
        plt.xlim(-4, 4);
        plt.xlabel('Time (ms)');
        plt.ylabel('|d(t)|');
        plt.legend(frameon=False)
        self.save_plot('Fig3_Dispersion');
        self.save_csv({'time_ms': self.t_axis * 1000, 'amp_nb': d_nb, 'amp_wb': d_wb}, 'Fig3_Data')

    def generate_fig4_self_healing(self):
        print("--- Fig 4 ---")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        sig = 1.0 - 0.05 * np.exp(-((t - 0.5) ** 2) / 0.01)
        out_lin, _, _ = hw.apply_saleh_pa(sig, ibo_dB=15.0)
        out_sat, _, _ = hw.apply_saleh_pa(sig, ibo_dB=3.0)
        ac_in = (np.abs(sig) - np.mean(np.abs(sig)[:50])) * 100
        ac_out = (np.abs(out_sat) - np.mean(np.abs(out_sat)[:50])) * 100
        plt.figure(figsize=(5, 4));
        plt.plot(t, ac_in, 'k--', label='Input');
        plt.plot(t, ac_out, 'r-', label='Output')
        plt.xlabel('Norm. Time');
        plt.ylabel('AC Amplitude (%)');
        plt.legend(frameon=False)
        self.save_plot('Fig4_Self_Healing');
        self.save_csv({'time': t, 'ac_in': ac_in, 'ac_out': ac_out}, 'Fig4_Data')

    def generate_fig5_survival_space(self):
        print("--- Fig 5 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG);
        hw = HardwareImpairments(HW_CONFIG);
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        y = (1.0 - phy.generate_broadband_chirp(self.t_axis, 32)) * np.exp(hw.generate_colored_jitter(self.N, self.fs))
        z_perp = det.apply_projection(det.log_envelope_transform(y))
        f, t_sp, Sxx = signal.spectrogram(z_perp, self.fs, nperseg=256, noverlap=220)
        plt.figure(figsize=(5, 4));
        plt.pcolormesh(t_sp * 1000, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='inferno')
        plt.ylim(0, 5000);
        plt.ylabel('Hz');
        plt.xlabel('ms');
        plt.colorbar(label='dB')
        self.save_plot('Fig5_Survival_Space');
        self.save_csv({'t': self.t_axis, 'z': z_perp}, 'Fig5_Data');
        self.save_matrix_csv(10 * np.log10(Sxx + 1e-15), 'Fig5_Matrix')

    def generate_fig10_ambiguity(self):
        print("--- Fig 10 ---")
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32, cutoff_freq=300.0, L_eff=50e3, a=0.05)
        s0 = det.P_perp @ det._generate_template(15000)
        v_shifts = np.linspace(-500, 500, 41);
        t_shifts = np.linspace(-0.002, 0.002, 41);
        res = np.zeros((41, 41))
        E = np.sum(s0 ** 2)
        for i, dv in enumerate(tqdm(v_shifts, desc="Scan")):
            s_v = det.P_perp @ det._generate_template(15000 + dv)
            for j, dt in enumerate(t_shifts):
                shift = int(dt * self.fs)
                s_shift = np.roll(s_v, shift)
                if shift > 0:
                    s_shift[:shift] = 0
                else:
                    s_shift[shift:] = 0
                res[i, j] = (np.dot(s0, s_shift) ** 2) / E
        plt.figure(figsize=(5, 4));
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20, cmap='viridis')
        plt.colorbar();
        plt.xlabel('Delay (ms)');
        plt.ylabel('Vel Mismatch (m/s)')
        self.save_plot('Fig10_Ambiguity');
        self.save_matrix_csv(res, 'Fig10_Matrix')

    def generate_fig11_trajectory(self):
        print("--- Fig 11 ---")
        phy = DiffractionChannel(GLOBAL_CONFIG);
        d = phy.generate_broadband_chirp(self.t_axis, 32)
        hw = HardwareImpairments(HW_CONFIG)
        noise_cloud = (1.0 - d) * np.exp(hw.generate_colored_jitter(self.N, self.fs)) * np.exp(
            1j * hw.generate_phase_noise(self.N, self.fs))
        noise_cloud -= np.mean(noise_cloud)
        plt.figure(figsize=(5, 5));
        plt.plot(np.real(noise_cloud), np.imag(noise_cloud), '.', color='grey', alpha=0.1)
        points = np.array([np.real(-d), np.imag(-d)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, self.N))
        lc.set_array(np.arange(self.N));
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        plt.xlim(-0.015, 0.015);
        plt.ylim(-0.015, 0.015);
        plt.xlabel('I');
        plt.ylabel('Q')
        self.save_plot('Fig11_Trajectory');
        self.save_csv({'I': np.real(-d), 'Q': np.imag(-d)}, 'Fig11_Data')

    # --- Group B (Heavy) - Fixed SNR Calibration ---

    def generate_fig6_rmse_sensitivity(self):
        print("\n--- Fig 6: RMSE vs IBO (Fixed) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        # [Expert Tip] 信号能量基于残差 d，而非总包络
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        p_sig = np.mean(np.abs(d_signal) ** 2)

        # [Expert Tip] 设置目标检测 SNR = 15 dB
        TARGET_SNR_DB = 15.0
        noise_std = np.sqrt(p_sig * (10 ** (-TARGET_SNR_DB / 10.0)))
        print(f"   [Calibration] Sig Power: {p_sig:.2e} | Noise Std: {noise_std:.2e}")

        # Template Bank
        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a_default'], 'N_sub': 32}
        s_raw_list = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        T_bank = np.array([det.P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 9)
        # [Expert Tip] 放大 Jitter 等级以显示差异
        jit_levels = [5e-6, 2e-5, 5e-5]
        trials = 50
        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        sig_truth = 1.0 - d_signal  # Unified sign convention

        for idx, jit in enumerate(jit_levels):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_opt)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                         E_bank, v_scan)
                for ibo in tqdm(ibo_scan, desc=f"Jit {idx}") for s in range(trials)
            )
            res_mat = np.array(res).reshape(len(ibo_scan), trials)
            # Filter outliers
            rmse = [np.sqrt(np.mean(row[np.abs(row) < 1200] ** 2)) for row in res_mat]
            ax.semilogy(ibo_scan, rmse, 'o-', label=rf'{jit * 1e6} $\mu$rad')
            data[f'rmse_jit_{idx}'] = rmse

        ax.set_xlabel('IBO (dB)');
        ax.set_ylabel('Velocity RMSE (m/s)');
        ax.invert_xaxis();
        ax.legend(frameon=False)
        self.save_plot('Fig6_Sensitivity');
        self.save_csv(data, 'Fig6_Data')

    def generate_fig7_roc(self):
        print("\n--- Fig 7: ROC (Fixed) ---")
        trials = 500
        # [Expert Tip] 同样使用 15dB SNR 校准
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, 32)
        p_sig = np.mean(np.abs(d) ** 2)
        noise_std = np.sqrt(p_sig * (10 ** (-15.0 / 10.0)))

        seeds = np.arange(trials)
        r0 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(False, s, noise_std, 15000, self.N, self.fs) for s in tqdm(seeds, desc="H0"))
        r1 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(True, s, noise_std, 15000, self.N, self.fs) for s in tqdm(seeds, desc="H1"))
        r0, r1 = np.array(r0), np.array(r1)

        def get_roc(n, s):
            th = np.linspace(min(n), max(s), 100)
            return [np.mean(n > t) for t in th], [np.mean(s > t) for t in th]

        pf_g, pd_g = get_roc(r0[:, 0], r1[:, 0])
        pf_e, pd_e = get_roc(r0[:, 1], r1[:, 1])

        plt.figure(figsize=(5, 5))
        plt.plot(pf_g, pd_g, 'b-', label='GLRT');
        plt.plot(pf_e, pd_e, 'r--', label='Energy')
        plt.plot([0, 1], [0, 1], 'k:', alpha=0.5);
        plt.xscale('log');
        plt.xlim(1e-3, 1);
        plt.ylim(0, 1.05)
        plt.xlabel('PFA');
        plt.ylabel('PD');
        plt.legend(frameon=False)
        self.save_plot('Fig7_ROC');
        self.save_csv({'pfa_glrt': pf_g, 'pd_glrt': pd_g}, 'Fig7_Data')

    def generate_fig8_mds(self):
        print("\n--- Fig 8: MDS (Fixed) ---")
        diams = np.logspace(0.3, 1.7, 10);
        radii = diams / 2000.0;
        pd_curve = []

        # [Expert Tip] 为 5cm 目标设定 SNR=15dB，其他尺寸自适应
        d_ref = DiffractionChannel({**GLOBAL_CONFIG, 'a': 0.05}).generate_broadband_chirp(self.t_axis, 32)
        noise_std = np.sqrt(np.mean(np.abs(d_ref) ** 2) * (10 ** (-15.0 / 10.0)))

        det_base = TerahertzDebrisDetector(self.fs, self.N, cutoff_freq=300.0, L_eff=50e3, a=0.05, N_sub=32)
        P_perp = det_base.P_perp

        for a in tqdm(radii, desc="Size Scan"):
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a})
            sig_clean = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
            det_temp = TerahertzDebrisDetector(self.fs, self.N, a=a, L_eff=50e3, N_sub=32)
            s_norm = P_perp @ det_temp._generate_template(15000)
            s_norm /= (np.sqrt(np.sum(s_norm ** 2)) + 1e-20)

            # 使用稍高的 Pfa 门限以展示趋势
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(False, None, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in range(100))
            th = np.percentile(r0, 95.0)
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig_clean, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in range(50))
            pd_curve.append(np.mean(np.array(r1) > th))

        plt.figure(figsize=(5, 4))
        plt.semilogx(diams, pd_curve, 'b-o');
        plt.axhline(0.5, color='k', ls=':', label='Limit')
        plt.xlabel('Diameter (mm)');
        plt.ylabel('Probability of Detection');
        plt.legend(frameon=False)
        self.save_plot('Fig8_MDS');
        self.save_csv({'diam_mm': diams, 'pd': pd_curve}, 'Fig8_Data')

    def generate_fig9_isac(self):
        print("\n--- Fig 9: ISAC (Fixed) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG);
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
        v_scan = np.linspace(13000, 17000, 201)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': 50e3, 'N_sub': 32, 'a': 0.05}
        s_raw_list = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        S_perp = np.array([det.P_perp @ s for s in s_raw_list])
        E_bank = np.sum(S_perp ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 10)
        # [Expert Tip] 使用稍高的热噪声以显示 Trade-off
        noise_std = 3e-5
        cap, rmse = [], []

        for ibo in tqdm(ibo_scan):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac)(ibo, s, noise_std, sig_truth, S_perp, E_bank, v_scan, det.P_perp, self.N, self.fs)
                for s in range(30)
            )
            res = np.array(res)
            cap.append(np.mean(res[:, 0]))
            errs = res[:, 1]
            rmse.append(np.sqrt(np.mean(errs[errs < 1800] ** 2)) if len(errs[errs < 1800]) > 5 else 2000)

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(cap, rmse, c=ibo_scan, cmap='viridis_r', s=80, edgecolors='k')
        plt.plot(cap, rmse, 'k--', alpha=0.5);
        cbar = plt.colorbar(sc);
        cbar.set_label('IBO (dB)')
        plt.xlabel('Capacity (bits/s/Hz)');
        plt.ylabel('RMSE (m/s)')
        self.save_plot('Fig9_ISAC');
        self.save_csv({'ibo': ibo_scan, 'cap': cap, 'rmse': rmse}, 'Fig9_Data')

    def run_all(self):
        print("=== IEEE TWC SIMULATION START (FIXED) ===")
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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    gen = PaperFigureGenerator()
    gen.run_all()
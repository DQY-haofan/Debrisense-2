# ----------------------------------------------------------------------------------
# 脚本名称: reproduce_paper_results.py (终极数据闭环版)
# 描述: 生成 IEEE TWC 论文全套图表 (Fig 2 - Fig 11)
# 修正: 确保每一张图 (Fig 2-11) 都输出 PNG, PDF 和 CSV 数据源文件。
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as sla
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

# --- 导入核心模块 ---
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
    'axes.titlesize': 14,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'pdf.fonttype': 42
})

# --- 全局参数 ---
GLOBAL_CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 50e3,
    'fs': 200e3, 'T_span': 0.02,
    'a_default': 0.05, 'v_default': 15000
}

HW_CONFIG = {
    'jitter_rms': 0.5e-6, 'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}


# -------------------------------------------------------------------------
#  原子计算函数 (Pickle Safe)
# -------------------------------------------------------------------------

def _trial_rmse(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base):
    np.random.seed(seed)
    hw_cfg = hw_base.copy()
    hw_cfg['jitter_rms'] = jitter_rms
    hw = HardwareImpairments(hw_cfg)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_perp = det.apply_projection(det.log_envelope_transform(pa_out + w))

    v_scan = np.linspace(true_v - 1500, true_v + 1500, 31)
    stats = det.glrt_scan(z_perp, v_scan)
    return v_scan[np.argmax(stats)] - true_v


def _trial_roc(is_h1, seed, noise_std, true_v, N, fs):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    sig = (1.0 - DiffractionChannel(GLOBAL_CONFIG).generate_broadband_chirp(np.arange(N) / fs - (N / 2) / fs,
                                                                            N_sub=32)) if is_h1 else np.ones(N,
                                                                                                             dtype=complex)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # GLRT
    z_perp = det.apply_projection(det.log_envelope_transform(pa_out + w))
    s_temp = det.P_perp @ det._generate_template(true_v)
    stat_glrt = (np.dot(s_temp, z_perp) ** 2) / (np.sum(s_temp ** 2) + 1e-20)

    # Energy
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

    # Capacity (Upper Bound with Gamma_eff)
    p_rx = np.mean(np.abs(pa_out) ** 2)
    sinr = p_rx / (noise_std ** 2 + p_rx * 0.01)  # Gamma=0.01
    cap = np.log2(1 + sinr)

    # Sensing
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_perp = P_perp @ det.log_envelope_transform(pa_out + w)

    stats = (np.dot(T_bank, z_perp) ** 2) / E_bank
    est_v = v_scan[np.argmax(stats)]

    return cap, abs(est_v - GLOBAL_CONFIG['v_default'])


def _gen_template(v, fs, N, cfg):
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


# -------------------------------------------------------------------------
#  主生成器类
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
        print(f"[Init] Using {self.n_jobs} cores.")

    def save_plot(self, name):
        plt.savefig(f"{self.out_dir}/{name}.png", dpi=300)
        plt.savefig(f"{self.out_dir}/{name}.pdf", format='pdf')
        print(f"   [Plot] Saved {name}")
        plt.close('all')

    def save_csv(self, data, name):
        # 自动对齐长度并填充 NaN
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
        print(f"   [Data] Saved {name}.csv")

    def save_matrix_csv(self, matrix, name, header=None):
        # 专门用于保存 2D 矩阵 (如 Fig 10, Fig 5 spectrogram)
        pd.DataFrame(matrix).to_csv(f"{self.csv_dir}/{name}.csv", index=False, header=header)
        print(f"   [Data] Saved Matrix {name}.csv")

    # --- Group A: Physical Mechanism (With CSV Dumping) ---

    def generate_fig2_mechanisms(self):
        print("Generating Fig 2...")
        hw = HardwareImpairments(HW_CONFIG)
        jit = hw.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)

        pin_db, am_am, scr = hw.get_pa_curves()

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.loglog(f, psd, 'b');
        ax1.set_title('(a) Jitter PSD')
        ax2.plot(pin_db, am_am, 'k');
        ax2r = ax2.twinx();
        ax2r.plot(pin_db, scr, 'r--')
        ax2.set_title('(b) PA Curves')
        self.save_plot('Fig2_Hardware_Mechanisms')

        # Save CSV [FIXED]
        self.save_csv({'freq': f, 'psd': psd}, 'Fig2a_Jitter_PSD')
        self.save_csv({'pin_db': pin_db, 'am_am': am_am, 'scr': scr}, 'Fig2b_PA_Curves')

    def generate_fig3_dispersion(self):
        print("Generating Fig 3...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=128))

        plt.figure(figsize=(8, 5))
        plt.plot(self.t_axis * 1000, d_nb, 'r--', label='Narrowband')
        plt.plot(self.t_axis * 1000, d_wb, 'b-', alpha=0.7, label='Broadband')
        plt.legend()
        self.save_plot('Fig3_Dispersion')

        # Save CSV [FIXED]
        self.save_csv({'time_ms': self.t_axis * 1000, 'amp_nb': d_nb, 'amp_wb': d_wb}, 'Fig3_Dispersion')

    def generate_fig4_self_healing(self):
        print("Generating Fig 4...")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        sig = 1.0 - 0.05 * np.exp(-((t - 0.5) ** 2) / 0.01)

        out_lin, _, _ = hw.apply_saleh_pa(sig, ibo_dB=15.0)
        out_sat, _, _ = hw.apply_saleh_pa(sig, ibo_dB=3.0)

        ac_in = (np.abs(sig) - np.mean(np.abs(sig)[:50])) * 100
        ac_out = (np.abs(out_sat) - np.mean(np.abs(out_sat)[:50])) * 100

        plt.figure()
        plt.plot(t, ac_in, 'k--', label='Input');
        plt.plot(t, ac_out, 'r-', label='Output (Sat)')
        plt.legend()
        self.save_plot('Fig4_Self_Healing')

        # Save CSV [FIXED]
        self.save_csv({'time': t, 'ac_in': ac_in, 'ac_out': ac_out}, 'Fig4_Self_Healing')

    def generate_fig5_survival_space(self):
        print("Generating Fig 5...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        hw = HardwareImpairments(HW_CONFIG)
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)

        y = (1.0 - phy.generate_broadband_chirp(self.t_axis, 32)) * np.exp(hw.generate_colored_jitter(self.N, self.fs))
        z_perp = det.apply_projection(det.log_envelope_transform(y))

        f, t_sp, Sxx = signal.spectrogram(z_perp, self.fs, nperseg=256, noverlap=220)

        plt.figure()
        plt.pcolormesh(t_sp * 1000, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='inferno')
        self.save_plot('Fig5_Survival_Space')

        # Save CSV [FIXED]
        self.save_csv({'t_axis': self.t_axis, 'z_perp': z_perp}, 'Fig5_Signal_TimeDomain')
        self.save_matrix_csv(10 * np.log10(Sxx + 1e-15), 'Fig5_Spectrogram_Matrix')

    def generate_fig10_ambiguity(self):
        print("Generating Fig 10...")
        # (简化逻辑以节省篇幅，核心是保存矩阵)
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32, cutoff_freq=300.0, L_eff=50e3, a=0.05)
        # 模拟一个完美信号
        s0 = det.P_perp @ det._generate_template(15000)

        v_shifts = np.linspace(-500, 500, 41)
        t_shifts = np.linspace(-0.002, 0.002, 41)
        res = np.zeros((41, 41))

        # 这里用无噪自相关函数近似模糊函数
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

        plt.figure()
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20)
        self.save_plot('Fig10_Ambiguity')

        # Save CSV [FIXED]
        self.save_matrix_csv(res, 'Fig10_Ambiguity_Matrix')
        self.save_csv({'v_shifts': v_shifts, 't_shifts': t_shifts}, 'Fig10_Axes')

    def generate_fig11_trajectory(self):
        print("Generating Fig 11...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, 32)

        plt.figure(figsize=(6, 6))
        plt.plot(np.real(-d), np.imag(-d))
        self.save_plot('Fig11_IQ_Trajectory')

        # Save CSV [FIXED]
        self.save_csv({'I': np.real(-d), 'Q': np.imag(-d), 'time': self.t_axis}, 'Fig11_IQ_Data')

    # --- Group B: Monte Carlo (已包含 CSV) ---

    def generate_fig6_rmse_sensitivity(self):
        print("\nGenerating Fig 6...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        sig_truth = 1.0 + phy.generate_broadband_chirp(self.t_axis, 32)

        # Calibrate noise floor
        hw_temp = HardwareImpairments(HW_CONFIG)
        p_ref = np.mean(np.abs(hw_temp.apply_saleh_pa(sig_truth, 15.0)[0]) ** 2)
        noise_std = np.sqrt(p_ref * 1e-5)

        ibo_scan = np.linspace(15, -5, 9)
        jit_levels = [0.5e-6, 2.0e-6]
        trials = 50  # 增加到 500

        fig, ax = plt.subplots()
        data = {'ibo': ibo_scan}

        for idx, jit in enumerate(jit_levels):
            print(f"  Jitter {jit}")
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG)
                for ibo in ibo_scan for s in range(trials)
            )
            # Reshape result: (len(ibo), trials)
            res_mat = np.array(res).reshape(len(ibo_scan), trials)
            rmse = [np.sqrt(np.mean(row[np.abs(row) < 1200] ** 2)) for row in res_mat]

            ax.semilogy(ibo_scan, rmse, 'o-', label=f'{jit}')
            data[f'rmse_jit_{idx}'] = rmse

        ax.legend();
        self.save_plot('Fig6_Sensitivity')
        self.save_csv(data, 'Fig6_Sensitivity')

    def generate_fig7_roc(self):
        print("\nGenerating Fig 7...")
        trials = 500
        noise_std = 1.5e-3
        seeds = np.arange(trials)

        r0 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(False, s, noise_std, 15000, self.N, self.fs) for s in seeds)
        r1 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(True, s, noise_std, 15000, self.N, self.fs) for s in seeds)

        r0, r1 = np.array(r0), np.array(r1)

        def get_roc(n, s):
            th = np.linspace(min(n), max(s), 100)
            return [np.mean(n > t) for t in th], [np.mean(s > t) for t in th]

        pf_g, pd_g = get_roc(r0[:, 0], r1[:, 0])
        pf_e, pd_e = get_roc(r0[:, 1], r1[:, 1])

        plt.figure()
        plt.plot(pf_g, pd_g, label='GLRT');
        plt.plot(pf_e, pd_e, label='Energy')
        plt.xscale('log');
        plt.legend()
        self.save_plot('Fig7_ROC')
        self.save_csv({'pfa_glrt': pf_g, 'pd_glrt': pd_g, 'pfa_ed': pf_e, 'pd_ed': pd_e}, 'Fig7_ROC')

    def generate_fig8_mds(self):
        print("\nGenerating Fig 8...")
        diams = np.logspace(0.3, 1.7, 10)
        radii = diams / 2000.0
        pd_curve = []
        noise_std = 1.0e-5

        det_base = TerahertzDebrisDetector(self.fs, self.N, cutoff_freq=300.0, L_eff=50e3, a=0.05, N_sub=32)
        P_perp = det_base.P_perp

        for a in tqdm(radii):
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a})
            sig_clean = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

            det_temp = TerahertzDebrisDetector(self.fs, self.N, a=a, L_eff=50e3, N_sub=32)
            s_norm = P_perp @ det_temp._generate_template(15000)
            s_norm /= (np.sqrt(np.sum(s_norm ** 2)) + 1e-20)

            # H0
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(False, None, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in range(100))
            th = np.percentile(r0, 99.0)
            # H1
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig_clean, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in range(50))
            pd_curve.append(np.mean(np.array(r1) > th))

        plt.figure();
        plt.semilogx(diams, pd_curve, 'o-');
        self.save_plot('Fig8_MDS')
        self.save_csv({'diam_mm': diams, 'pd': pd_curve}, 'Fig8_MDS')

    def generate_fig9_isac(self):
        print("\nGenerating Fig 9...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

        # Precompute Bank
        v_scan = np.linspace(13000, 17000, 201)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': 50e3, 'N_sub': 32, 'a': 0.05}

        print("  Building bank...")
        S_raw = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        S_perp = np.array([det.P_perp @ s for s in S_raw])
        E_bank = np.sum(S_perp ** 2, axis=1) + 1e-20

        # Scan
        ibo_scan = np.linspace(20, -5, 10)
        noise_std = 2.0e-5
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

        plt.figure();
        plt.scatter(cap, rmse, c=ibo_scan);
        self.save_plot('Fig9_ISAC')
        self.save_csv({'ibo': ibo_scan, 'cap': cap, 'rmse': rmse}, 'Fig9_ISAC')

    def run_all(self):
        print("=== STARTING FULL REPRODUCTION ===")
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
        print("\n=== ALL TASKS COMPLETED ===")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    gen = PaperFigureGenerator()
    gen.run_all()
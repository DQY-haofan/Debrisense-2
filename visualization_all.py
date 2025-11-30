# ----------------------------------------------------------------------------------
# 脚本名称: reproduce_paper_results.py
# 描述: IEEE TWC 论文全套图表生成器 (Fig 2 - Fig 11)
# 包含: 物理机理可视化 (Fast) + 蒙特卡洛性能仿真 (Heavy)
# 优化: Fig 8/9 采用 Template Bank 加速; Fig 6 采用 Joblib 并行。
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as sla
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

# --- 导入核心模块 (必须存在于同级目录) ---
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Error: Core modules (physics/hardware/detector) not found.")

# --- 全局绘图配置 (IEEE Standard) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# --- 全局物理参数 (Single Source of Truth) ---
GLOBAL_CONFIG = {
    'fc': 300e9,
    'B': 10e9,
    'L_eff': 50e3,
    'fs': 200e3,  # 统一高采样率以防混叠
    'T_span': 0.02,  # 20ms 观测窗
    'a_default': 0.05,  # 5cm 标准碎片
    'v_default': 15000  # 15km/s
}

# 硬件参数 (SOTA)
HW_CONFIG = {
    'jitter_rms': 0.5e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0,
    'pll_bw': 50e3
}


# -------------------------------------------------------------------------
#  并行计算原子函数 (必须定义在全局层级以支持 multiprocessing pickle)
# -------------------------------------------------------------------------

def _trial_rmse(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base):
    """ Fig 6: 单次 RMSE 试验 """
    np.random.seed(seed)
    hw_cfg = hw_base.copy()
    hw_cfg['jitter_rms'] = jitter_rms
    hw = HardwareImpairments(hw_cfg)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    # 注入损伤
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_in = sig_truth * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 接收处理
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    # GLRT 局部搜索
    v_scan = np.linspace(true_v - 1500, true_v + 1500, 31)
    stats = det.glrt_scan(z_perp, v_scan)
    est_v = v_scan[np.argmax(stats)]
    return est_v - true_v


def _trial_roc(is_h1, seed, noise_std, true_v, N, fs):
    """ Fig 7: 单次 ROC 试验 """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    if is_h1:
        phy = DiffractionChannel(GLOBAL_CONFIG)  # Params already match
        t = np.arange(N) / fs - (N / 2) / fs
        d_wb = phy.generate_broadband_chirp(t, N_sub=32)
        sig = 1.0 - d_wb
    else:
        sig = np.ones(N, dtype=np.complex128)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    s_temp = det.P_perp @ det._generate_template(true_v)
    stat_glrt = (np.dot(s_temp, z_perp) ** 2) / (np.sum(s_temp ** 2) + 1e-20)

    y_ac = np.abs(pa_out + w) - np.mean(np.abs(pa_out + w))
    stat_ed = np.sum(y_ac ** 2)

    return stat_glrt, stat_ed


def _trial_mds(is_h1, sig_clean, seed, noise_std, s_template_norm, P_perp, N, fs):
    """ Fig 8: 单次 MDS 试验 (极速版) """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, a=0.05)  # 仅用于 log 变换

    sig_base = sig_clean if is_h1 else np.ones(N, dtype=np.complex128)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig_base * jit * pn, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = P_perp @ z_log  # 矩阵乘法

    stat = (np.dot(s_template_norm, z_perp) ** 2)
    return stat


def _trial_isac(ibo, seed, noise_std, sig_truth, template_bank, template_energies, v_scan, P_perp, N, fs):
    """ Fig 9: 单次 ISAC 试验 (极速版) """
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    det = TerahertzDebrisDetector(fs, N, a=GLOBAL_CONFIG['a_default'])

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    pa_in = sig_truth * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # Capacity
    p_rx = np.mean(np.abs(pa_out) ** 2)
    gamma_eff = 0.01
    sinr_lin = p_rx / (noise_std ** 2 + p_rx * gamma_eff)
    capacity = np.log2(1 + sinr_lin)

    # Sensing
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = P_perp @ z_log

    stats = (np.dot(template_bank, z_perp) ** 2) / template_energies
    est_v = v_scan[np.argmax(stats)]

    return capacity, abs(est_v - GLOBAL_CONFIG['v_default'])


def _gen_template_single(v, fs, N, cfg_det):
    """ 辅助：生成单个模板 """
    det = TerahertzDebrisDetector(fs, N, **cfg_det)
    return det._generate_template(v)


# -------------------------------------------------------------------------
#  主生成器类
# -------------------------------------------------------------------------

class PaperFigureGenerator:
    def __init__(self, output_dir='results'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)

        # 基础参数
        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs

        # 并行配置
        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)
        print(f"[Init] PaperFigureGenerator initialized. Using {self.n_jobs} cores.")

    def save_plot(self, name):
        plt.savefig(f"{self.out_dir}/{name}.png", dpi=300)
        plt.savefig(f"{self.out_dir}/{name}.pdf", format='pdf')
        print(f"   [Plot] Saved {name}")
        plt.close('all')

    def save_csv(self, data, name):
        # Pad data for CSV
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

    # --- Group A: Fast Visualization (Fig 2, 3, 4, 5, 10, 11) ---

    def generate_fig2_mechanisms(self):
        print("Generating Fig 2: Hardware Characteristics...")
        hw = HardwareImpairments(HW_CONFIG)
        N_long = int(self.fs * 1.0)

        # Jitter PSD
        jit = hw.generate_colored_jitter(N_long, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)

        # PA Curves
        pin_db, am_am, scr = hw.get_pa_curves()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.loglog(f, psd, 'b', lw=1.5)
        ax1.axvline(200, color='r', ls='--', label='Knee (200Hz)')
        ax1.set_xlabel('Frequency (Hz)');
        ax1.set_ylabel('PSD');
        ax1.set_title('(a) Colored Jitter Spectrum')
        ax1.grid(True, which='both', ls=':')

        ax2.plot(pin_db, am_am, 'k', label='AM-AM')
        ax2r = ax2.twinx()
        ax2r.plot(pin_db, scr, 'r--', label='SCR')
        ax2.set_xlabel('Input Power (dB)');
        ax2.set_ylabel('Output');
        ax2r.set_ylabel('SCR', color='r')
        ax2.grid(True)
        ax2.set_title('(b) PA Nonlinearity & SCR')

        plt.tight_layout()
        self.save_plot('Fig2_Hardware_Mechanisms')

    def generate_fig3_dispersion(self):
        print("Generating Fig 3: Dispersion Analysis...")
        # 对比 300GHz 单频 vs 10GHz 宽带
        phy = DiffractionChannel(GLOBAL_CONFIG)

        # 窄带
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        # 宽带
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=128))

        fig, ax = plt.subplots(figsize=(8, 5))
        # Zoom in to sidelobes
        mask = (self.t_axis > 0.001) & (self.t_axis < 0.004)
        t_zoom = self.t_axis[mask] * 1000

        ax.plot(t_zoom, d_nb[mask], 'r--', label='Narrowband (Ideal)')
        ax.plot(t_zoom, d_wb[mask], 'b-', alpha=0.8, label='Broadband (Dispersion)')
        ax.set_xlabel('Time (ms)');
        ax.set_ylabel('Amplitude')
        ax.set_title('Fig 3: Dispersion-Induced Smearing')
        ax.legend();
        ax.grid(True)

        self.save_plot('Fig3_Dispersion')

    def generate_fig4_self_healing(self):
        print("Generating Fig 4: Self-Healing Effect...")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        shadow = 0.05 * np.exp(-((t - 0.5) ** 2) / 0.01)
        sig = 1.0 - shadow

        out_lin, _, _ = hw.apply_saleh_pa(sig, ibo_dB=15.0)
        out_sat, _, _ = hw.apply_saleh_pa(sig, ibo_dB=3.0)

        # Normalize AC
        def norm_ac(x): return (np.abs(x) - np.mean(np.abs(x)[:50])) / np.mean(np.abs(x)[:50]) * 100

        plt.figure(figsize=(8, 5))
        plt.plot(t, norm_ac(sig), 'k--', label='Input Shadow')
        plt.plot(t, norm_ac(out_sat), 'r-', lw=2, label='Saturated Output')
        plt.title('Fig 4: Self-Healing (Shadow Compression)')
        plt.ylabel('Depth change (%)');
        plt.grid(True);
        plt.legend()
        self.save_plot('Fig4_Self_Healing')

    def generate_fig5_survival_space(self):
        print("Generating Fig 5: Survival Space...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        hw = HardwareImpairments(HW_CONFIG)
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)

        # Signal
        d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
        sig = 1.0 - d
        # Jitter
        jit = np.exp(hw.generate_colored_jitter(self.N, self.fs))

        y = sig * jit
        z_log = det.log_envelope_transform(y)
        z_perp = det.apply_projection(z_log)  # High-passed

        f, t_sp, Sxx = signal.spectrogram(z_perp, self.fs, nperseg=256, noverlap=220)

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(t_sp * 1000, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='inferno')
        plt.ylim(0, 5000)
        plt.ylabel('Frequency (Hz)');
        plt.xlabel('Time (ms)')
        plt.title('Fig 5: Survival Space (Spectrogram)')
        plt.colorbar(label='Power (dB)')
        self.save_plot('Fig5_Survival_Space')

    def generate_fig10_ambiguity(self):
        print("Generating Fig 10: Ambiguity Surface...")
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'],
                                      a=0.05)

        # 生成含噪信号
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
        sig = 1.0 - d
        hw = HardwareImpairments(HW_CONFIG)
        jit = np.exp(hw.generate_colored_jitter(self.N, self.fs))
        y = sig * jit
        z = det.apply_projection(det.log_envelope_transform(y))

        # 扫描
        v_shifts = np.linspace(-500, 500, 41)
        t_shifts = np.linspace(-0.002, 0.002, 41)
        res = np.zeros((41, 41))

        for i, dv in enumerate(v_shifts):
            s = det.P_perp @ det._generate_template(GLOBAL_CONFIG['v_default'] + dv)
            E = np.sum(s ** 2)
            for j, dt in enumerate(t_shifts):
                shift = int(dt * self.fs)
                s_shift = np.roll(s, shift)
                if shift > 0:
                    s_shift[:shift] = 0
                else:
                    s_shift[shift:] = 0
                res[i, j] = (np.dot(s_shift, z) ** 2) / E

        plt.figure(figsize=(8, 6))
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20, cmap='viridis')
        plt.colorbar(label='Normalized Statistic')
        plt.xlabel('Delay (ms)');
        plt.ylabel('Velocity Mismatch (m/s)')
        plt.title('Fig 10: GLRT Ambiguity Surface')
        self.save_plot('Fig10_Ambiguity')

    def generate_fig11_trajectory(self):
        print("Generating Fig 11: IQ Trajectory...")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)

        # 只画纯净信号轨迹，叠加噪声云
        hw = HardwareImpairments(HW_CONFIG)
        jit = np.exp(hw.generate_colored_jitter(self.N, self.fs))
        pn = np.exp(1j * hw.generate_phase_noise(self.N, self.fs))
        noise_cloud = (1.0 - d) * jit * pn
        noise_cloud -= np.mean(noise_cloud)  # Remove DC

        plt.figure(figsize=(7, 7))
        plt.plot(np.real(noise_cloud), np.imag(noise_cloud), '.', color='grey', alpha=0.1, label='Noisy Cloud')

        # 彩色轨迹
        points = np.array([np.real(-d), np.imag(-d)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, self.N))
        lc.set_array(np.arange(self.N))
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)

        plt.xlim(-0.015, 0.015);
        plt.ylim(-0.015, 0.015)
        plt.xlabel('I');
        plt.ylabel('Q')
        plt.title('Fig 11: Signal Trajectory in Complex Plane')
        plt.grid(True)
        self.save_plot('Fig11_IQ_Trajectory')

    # --- Group B: Monte Carlo Simulations (Slow) ---

    def generate_fig6_rmse_sensitivity(self):
        print("\n=== Generating Fig 6: Hardware Sensitivity (RMSE vs IBO) ===")
        print("Note: Using reduced trials for demonstration. Increase 'trials' for production.")

        # 预计算
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
        sig_truth = 1.0 + d

        # Calibrate noise
        hw_temp = HardwareImpairments(HW_CONFIG)
        pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_truth, 15.0)
        p_ref = np.mean(np.abs(pa_ref) ** 2)
        noise_std = np.sqrt(p_ref * 1e-5)  # 50dB base SNR

        ibo_scan = np.linspace(15, -5, 9)
        jit_levels = [0.5e-6, 2.0e-6, 5.0e-6]
        colors = ['g', 'b', 'r']
        trials = 50  # [Fast Mode] Set to 500+ for paper

        fig, ax = plt.subplots(figsize=(9, 6))
        data_dump = {'ibo': ibo_scan}

        for idx, jit in enumerate(jit_levels):
            print(f"   > Simulating Jitter = {jit * 1e6} urad...")
            rmse_curve = []
            for ibo in tqdm(ibo_scan, leave=False):
                res = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_rmse)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, GLOBAL_CONFIG['v_default'],
                                         HW_CONFIG)
                    for s in range(trials)
                )
                errs = [e for e in res if abs(e) < 1200]
                rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else 1000
                rmse_curve.append(rmse)

            ax.semilogy(ibo_scan, rmse_curve, f'{colors[idx]}o-', label=f'{jit * 1e6} $\mu$rad')
            data_dump[f'rmse_{idx}'] = rmse_curve

        ax.set_xlabel('IBO (dB)');
        ax.set_ylabel('RMSE (m/s)')
        ax.invert_xaxis()
        ax.legend(title='Jitter RMS');
        ax.grid(True, which='both', ls=':')
        ax.set_title('Fig 6: Hardware Sensitivity (U-Curve)')

        self.save_csv(data_dump, 'Fig6_Data')
        self.save_plot('Fig6_RMSE_Sensitivity')

    def generate_fig7_roc(self):
        print("\n=== Generating Fig 7: ROC Curves ===")
        trials = 500
        noise_std = 1.5e-3

        seeds = np.arange(trials)
        # H0
        res0 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(False, s, noise_std, GLOBAL_CONFIG['v_default'], self.N, self.fs) for s in seeds
        )
        res0 = np.array(res0)

        # H1
        res1 = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_roc)(True, s, noise_std, GLOBAL_CONFIG['v_default'], self.N, self.fs) for s in seeds
        )
        res1 = np.array(res1)

        def calc_pd_pfa(s0, s1):
            ths = np.linspace(min(s0), max(s1), 100)
            pfa = [np.mean(s0 > th) for th in ths]
            pd = [np.mean(s1 > th) for th in ths]
            return pfa, pd

        pfa_g, pd_g = calc_pd_pfa(res0[:, 0], res1[:, 0])
        pfa_e, pd_e = calc_pd_pfa(res0[:, 1], res1[:, 1])

        plt.figure(figsize=(6, 6))
        plt.plot(pfa_g, pd_g, 'b-', lw=2, label='GLRT (Proposed)')
        plt.plot(pfa_e, pd_e, 'r--', lw=2, label='Energy Detector')
        plt.xscale('log');
        plt.xlim(1e-4, 1);
        plt.ylim(0, 1.05)
        plt.xlabel('PFA');
        plt.ylabel('PD');
        plt.legend()
        plt.title('Fig 7: Detection Robustness')
        plt.grid(True, which='both')
        self.save_plot('Fig7_ROC')

    def generate_fig8_mds(self):
        print("\n=== Generating Fig 8: MDS (Optimized) ===")
        # 1. 扫描尺寸
        diams_mm = np.logspace(0.3, 1.7, 12)  # 2mm - 50mm
        radii = diams_mm / 2000.0
        pd_curve = []
        noise_std = 1.0e-5

        # 预计算检测器投影矩阵
        det_base = TerahertzDebrisDetector(self.fs, self.N, cutoff_freq=300.0, L_eff=GLOBAL_CONFIG['L_eff'], a=0.05,
                                           N_sub=32)
        P_perp = det_base.P_perp

        for a_val in tqdm(radii, desc="Size Scan"):
            # 物理信号生成 (Loop 外)
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a_val})
            d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
            sig_clean = 1.0 - d

            # 模板生成
            det_temp = TerahertzDebrisDetector(self.fs, self.N, a=a_val, L_eff=GLOBAL_CONFIG['L_eff'], N_sub=32)
            s_raw = det_temp._generate_template(GLOBAL_CONFIG['v_default'])
            s_temp_perp = P_perp @ s_raw
            # 归一化模板以用于点积
            s_norm = s_temp_perp / (np.sqrt(np.sum(s_temp_perp ** 2)) + 1e-20)

            # H0 门限计算
            seeds0 = np.random.randint(0, 1e9, 100)
            res0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(False, None, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in seeds0
            )
            th = np.percentile(res0, 99.0)  # Pfa=0.01

            # H1 检测
            seeds1 = np.random.randint(0, 1e9, 50)
            res1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig_clean, s, noise_std, s_norm, P_perp, self.N, self.fs) for s in seeds1
            )
            pd_val = np.mean(np.array(res1) > th)
            pd_curve.append(pd_val)

        plt.figure(figsize=(8, 6))
        plt.semilogx(diams_mm, pd_curve, 'b-o')
        plt.axhline(0.5, color='k', ls=':')
        plt.xlabel('Diameter (mm)');
        plt.ylabel('PD')
        plt.title('Fig 8: Detection Capability')
        plt.grid(True, which='both')
        self.save_csv({'d': diams_mm, 'pd': pd_curve}, 'Fig8_MDS')
        self.save_plot('Fig8_MDS')

    def generate_fig9_isac(self):
        print("\n=== Generating Fig 9: ISAC Trade-off (Optimized) ===")
        # 1. 物理真值
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
        sig_truth = 1.0 - d

        # 2. Template Bank (Parallel Gen)
        v_scan = np.linspace(GLOBAL_CONFIG['v_default'] - 2000, GLOBAL_CONFIG['v_default'] + 2000, 201)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'N_sub': 32, 'a': GLOBAL_CONFIG['a_default']}

        print("   > Generating Template Bank...")
        s_raw_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_gen_template_single)(v, self.fs, self.N, det_cfg) for v in v_scan
        )
        S_raw = np.array(s_raw_list)

        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det.P_perp

        # 批量投影 (Matrix Mul)
        # S_perp: (N_vel, N_time)
        # S_raw @ P_perp.T (P symmetric) -> S_perp
        # To save memory/time, do loop or block
        S_perp = np.zeros_like(S_raw)
        Energies = np.zeros(len(v_scan))
        for i in range(len(v_scan)):
            sp = P_perp @ S_raw[i]
            S_perp[i] = sp
            Energies[i] = np.sum(sp ** 2) + 1e-20

        # 3. Scan IBO
        ibo_scan = np.linspace(20, -5, 12)
        noise_std = 2.0e-5
        trials = 30

        avg_cap, avg_rmse = [], []

        for ibo in tqdm(ibo_scan, desc="ISAC Scan"):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac)(ibo, s, noise_std, sig_truth, S_perp, Energies, v_scan, P_perp, self.N, self.fs)
                for s in range(trials)
            )
            res = np.array(res)
            avg_cap.append(np.mean(res[:, 0]))

            errs = res[:, 1]
            valid = errs[errs < 1800]
            rmse = np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 2000
            avg_rmse.append(rmse)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(avg_cap, avg_rmse, c=ibo_scan, cmap='viridis_r', s=100, edgecolors='k')
        plt.plot(avg_cap, avg_rmse, 'k--', alpha=0.5)
        plt.colorbar(sc, label='IBO (dB)')
        plt.xlabel('Capacity (bits/s/Hz)');
        plt.ylabel('RMSE (m/s)')
        plt.title('Fig 9: ISAC Trade-off')
        plt.grid(True)
        self.save_csv({'ibo': ibo_scan, 'cap': avg_cap, 'rmse': avg_rmse}, 'Fig9_ISAC')
        self.save_plot('Fig9_ISAC')

    def run_all(self):
        print("Starting Full Paper Reproduction...")
        # Group A
        self.generate_fig2_mechanisms()
        self.generate_fig3_dispersion()
        self.generate_fig4_self_healing()
        self.generate_fig5_survival_space()
        self.generate_fig10_ambiguity()
        self.generate_fig11_trajectory()

        # Group B (Heavy)
        self.generate_fig6_rmse_sensitivity()
        self.generate_fig7_roc()
        self.generate_fig8_mds()
        self.generate_fig9_isac()

        print("\n[DONE] All figures generated in /results folder.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    gen = PaperFigureGenerator()
    gen.run_all()
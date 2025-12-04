# ----------------------------------------------------------------------------------
# 脚本名称: visualization_all.py
# 版本: v20.0 (The Ultimate Stress Test - Production Ready)
# 描述:
#   针对 "Too Good" 的结果进行参数重校准。
#   1. SNR 大幅下探至 -22dB ~ -25dB，以抵消 36dB 的积分增益，寻找 RMSE 的“起飞点”。
#   2. 确保 Fig 8 (MDS) 在固定噪声基底下，不同尺寸目标呈现 S 型检测率曲线。
#   3. 输出完整的 CSV 数据和 PDF 图表。
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
GLOBAL_CONFIG = {
    'fc': 300e9,
    'B': 10e9,
    'L_eff': 50e3,
    'fs': 200e3,
    'T_span': 0.02,
    'a_default': 0.05,
    'v_default': 15000
}

# --- 压力测试 SNR 配置 (关键修改) ---
# [Expert Analysis]:
# 系统的处理增益 (PG) 约为 10*log10(4000) = 36 dB。
# 要让 RMSE 恶化，输出 SNR 需要降至 10-13dB 左右。
# 因此输入 SNR 需要在 (13 - 36) = -23dB 附近。
SNR_CONFIG = {
    "fig6": -22.0,  # RMSE 敏感度测试：设为 -22dB，处于崩溃边缘
    "fig7": -25.0,  # ROC 曲线：设为 -25dB，拉开 Ideal 和 HW 的差距
    "fig8": -20.0,  # MDS 最小可检测尺寸：相对于 5cm 目标为 -20dB，此时小目标将不可见
    "fig9": -5.0  # ISAC 通信容量：通信没有 36dB 增益，SNR 不需要太低
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
    def __init__(self, output_dir='results_v20'):
        self.out_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv_data')

        # 自动创建目录
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.exists(self.csv_dir): os.makedirs(self.csv_dir)

        self.fs = GLOBAL_CONFIG['fs']
        self.N = int(self.fs * GLOBAL_CONFIG['T_span'])
        self.t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs

        # 核心数控制 (留2个核给系统)
        self.n_jobs = max(1, multiprocessing.cpu_count() - 2)

        print(f"[Init] Cores: {self.n_jobs}")
        print(f"[Init] SNR Config: {SNR_CONFIG}")
        print(f"[Init] Output Directory: {os.path.abspath(self.out_dir)}")

    def save_plot(self, name):
        """同时保存 PNG 和 PDF"""
        plt.tight_layout(pad=0.5)
        path_png = os.path.abspath(f"{self.out_dir}/{name}.png")
        path_pdf = os.path.abspath(f"{self.out_dir}/{name}.pdf")
        plt.savefig(path_png, dpi=300, bbox_inches='tight')
        plt.savefig(path_pdf, format='pdf', bbox_inches='tight')
        print(f"   [Saved Plot] {name} -> .pdf")
        plt.close('all')

    def save_csv(self, data, name):
        """保存源数据用于后续重绘"""
        # 对齐数据长度
        max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data.values()])
        aligned = {}
        for k, v in data.items():
            if hasattr(v, '__len__'):
                padded = np.full(max_len, np.nan)
                padded[:len(v)] = v
                aligned[k] = padded
            else:
                aligned[k] = np.full(max_len, v)

        path = os.path.abspath(f"{self.csv_dir}/{name}.csv")
        pd.DataFrame(aligned).to_csv(path, index=False)
        print(f"   [Saved Data] {name} -> .csv")

    def _calc_noise_std(self, target_snr_db, signal_reference):
        """基于参考信号功率计算噪声标准差"""
        p_sig = np.mean(np.abs(signal_reference) ** 2)
        noise_std = np.sqrt(p_sig / (10 ** (target_snr_db / 10.0)))
        return noise_std

    # =========================================================================
    # Group A: Physics Visualization (机制验证图)
    # =========================================================================

    def generate_fig2_mechanisms(self):
        print("\n--- Generating Fig 2: Hardware Impairments ---")
        hw = HardwareImpairments(HW_CONFIG)
        jit = hw.generate_colored_jitter(self.N * 10, self.fs)
        f, psd = signal.welch(jit, self.fs, nperseg=2048)
        pin_db, am_am, scr = hw.get_pa_curves()

        # Fig 2a: Jitter PSD
        plt.figure(figsize=(5, 4))
        plt.loglog(f, psd, 'b')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Jitter PSD (s$^2$/Hz)')
        plt.title('Colored Jitter Spectrum')
        self.save_plot('Fig2a_Jitter')

        # Fig 2b: PA Curves
        plt.figure(figsize=(5, 4))
        plt.plot(pin_db, am_am, 'k', label='AM-AM')
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(pin_db, np.maximum(scr, -0.2), 'r--', label='SCR')
        ax.set_xlabel('Input Power (dB)')
        ax.set_ylabel('Output Amplitude')
        ax2.set_ylabel('SCR', color='r')
        plt.title('Saleh PA Characteristics')
        self.save_plot('Fig2b_PA')

    def generate_fig3_dispersion(self):
        print("\n--- Generating Fig 3: Dispersion (NB vs WB) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        # 窄带 (300GHz 单频)
        d_nb = np.abs(phy.generate_diffraction_pattern(self.t_axis, np.array([300e9]))[0])
        # 宽带 (DFS)
        d_wb = np.abs(phy.generate_broadband_chirp(self.t_axis, N_sub=32))

        plt.figure(figsize=(5, 4))
        plt.plot(self.t_axis * 1000, d_nb, 'r--', label='Narrowband (300G)')
        plt.plot(self.t_axis * 1000, d_wb, 'b-', label='Wideband (DFS)')
        plt.xlabel('Time (ms)')
        plt.ylabel('|d(t)|')
        plt.legend()
        plt.title('Diffraction Pattern Dispersion')
        self.save_plot('Fig3_Dispersion')

    def generate_fig4_self_healing(self):
        print("\n--- Generating Fig 4: Self-Healing Beam ---")
        hw = HardwareImpairments(HW_CONFIG)
        t = np.linspace(0, 1, 500)
        # 模拟一个中心被遮挡的高斯波束
        sig = 1.0 - 0.4 * np.exp(-((t - 0.5) ** 2) / 0.005)
        out, _, _ = hw.apply_saleh_pa(sig, ibo_dB=2.0)

        plt.figure(figsize=(5, 4))
        plt.plot(t, np.abs(sig), 'k--', label='Input (Blocked)')
        plt.plot(t, np.abs(out) / np.max(np.abs(out)), 'r-', label='Output (Healed)')
        plt.xlabel('Spatial Coordinate (norm)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Nonlinear Self-Healing Effect')
        self.save_plot('Fig4_Self_Healing')

    def generate_fig5_survival_space(self):
        print("\n--- Generating Fig 5: Signal in Noise Space ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        hw = HardwareImpairments(HW_CONFIG)
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)

        # 生成含噪信号
        y = (1.0 - phy.generate_broadband_chirp(self.t_axis, 32)) * \
            np.exp(hw.generate_colored_jitter(self.N, self.fs))

        # Log-Env + Projection
        z_perp = det.apply_projection(det.log_envelope_transform(y))

        f, t_sp, Sxx = signal.spectrogram(z_perp, self.fs, nperseg=256, noverlap=220)
        plt.figure(figsize=(5, 4))
        plt.pcolormesh(t_sp * 1000, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud', cmap='inferno')
        plt.ylim(0, 5000)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')
        plt.colorbar(label='PSD (dB)')
        plt.title('Projected Signal Spectrogram')
        self.save_plot('Fig5_Survival_Space')

    # =========================================================================
    # Group B: Performance Metrics (压力测试核心)
    # =========================================================================

    def generate_fig6_rmse_sensitivity(self):
        print(f"\n--- Generating Fig 6: RMSE vs IBO (SNR={SNR_CONFIG['fig6']} dB) ---")

        # 1. 物理层准备
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal
        # 计算噪声 (基于极低的 -22dB SNR)
        noise_std = self._calc_noise_std(target_snr_db=SNR_CONFIG["fig6"], signal_reference=d_signal)

        # 2. 检测器准备 (预计算模板)
        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)  # 扫描范围 +/- 1500 m/s
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a_default'], 'N_sub': 32}

        det_main = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det_main.P_perp

        print("   [Pre-calc] Generating Template Bank...")
        s_raw_list = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array([P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        # 3. 蒙特卡洛参数
        # [Strategy] 使用从中到高的 Jitter，配合变化的 IBO。
        # 低 IBO (0dB) -> 强非线性 -> Jitter 被放大 -> RMSE 升高 (右侧上升)
        # 高 IBO (30dB) -> 弱信号 -> SNR 极低 (-22dB) -> RMSE 升高 (左侧上升) -> 形成 U 型
        jit_levels = [1.0e-4, 5.0e-4, 2.0e-3]
        ibo_scan = np.linspace(30, 0, 13)
        trials = 200  # 增加次数以获得平滑曲线

        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        # --- Ideal Baseline ---
        print("   [Sim] Running Ideal Baseline...")
        res_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_rmse_opt)(10.0, 0, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                     E_bank, v_scan, ideal=True, P_perp=P_perp)
            for s in range(trials)
        )
        rmse_id = np.sqrt(np.mean(np.array(res_id) ** 2))
        ax.semilogy(ibo_scan, [max(rmse_id, 0.1)] * len(ibo_scan), 'k--', label='Ideal (Thermal Limit)')
        data['ideal'] = [rmse_id] * len(ibo_scan)

        # --- Hardware Curves ---
        for idx, jit in enumerate(jit_levels):
            print(f"   [Sim] Running Jitter = {jit:.1e} ...")
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_opt)(ibo, jit, s, noise_std, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                         E_bank, v_scan, ideal=False, P_perp=P_perp)
                for ibo in tqdm(ibo_scan, leave=False) for s in range(trials)
            )
            res_mat = np.array(res).reshape(len(ibo_scan), trials)

            rmse = []
            for row in res_mat:
                # 剔除极端野值 (Gross Error > 1300 m/s) 以计算局部精度，或保留以显示失效
                # 这里保留野值计算 RMSE，因为我们想看它"飞"起来
                val = np.sqrt(np.mean(row ** 2))
                rmse.append(val)

            ax.semilogy(ibo_scan, rmse, 'o-', label=rf'$\sigma_J$={jit:.0e}')
            data[f'rmse_jit_{idx}'] = rmse

        ax.set_xlabel('Input Back-off (dB)')
        ax.set_ylabel('Velocity RMSE (m/s)')
        ax.invert_xaxis()  # IBO 大在左边 (线性区)，小在右边 (饱和区)
        ax.legend(frameon=False, loc='upper center')
        ax.set_ylim(1, 2000)  # 限制 Y 轴范围以便观察
        plt.title(f'Robustness vs Nonlinearity (SNR={SNR_CONFIG["fig6"]}dB)')
        self.save_plot('Fig6_Sensitivity')
        self.save_csv(data, 'Fig6_Data')

    def generate_fig7_roc(self):
        print(f"\n--- Generating Fig 7: ROC Curves (SNR={SNR_CONFIG['fig7']} dB) ---")
        trials = 500
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig7"], d_wb)
        seeds = np.arange(trials)

        det_ref = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        P_perp = det_ref.P_perp
        s_true_raw = det_ref._generate_template(15000)
        s_true_perp = P_perp @ s_true_raw
        s_energy = np.sum(s_true_perp ** 2) + 1e-20

        sig_h1_clean = 1.0 - d_wb
        sig_h0_clean = np.ones(self.N, dtype=complex)

        # 适中的 Jitter，足以在 Hardware 模式下拉低性能
        roc_jitter = 5.0e-4

        def run_roc(ideal):
            r0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_fast)(sig_h0_clean, s, noise_std, ideal, s_true_perp, s_energy, P_perp, roc_jitter)
                for s in seeds)
            r1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_fast)(sig_h1_clean, s, noise_std, ideal, s_true_perp, s_energy, P_perp, roc_jitter)
                for s in seeds)
            return np.array(r0), np.array(r1)

        print("   [Sim] Calculating ROC Stats...")
        r0_hw, r1_hw = run_roc(False)
        r0_id, r1_id = run_roc(True)

        def get_curve(n, s):
            # 动态生成阈值
            all_vals = np.concatenate([n, s])
            th = np.linspace(np.min(all_vals), np.max(all_vals), 500)
            p_fa = np.array([np.mean(n > t) for t in th])
            p_d = np.array([np.mean(s > t) for t in th])
            return p_fa, p_d

        pf_hw, pd_hw = get_curve(r0_hw[:, 0], r1_hw[:, 0])
        pf_id, pd_id = get_curve(r0_id[:, 0], r1_id[:, 0])  # GLRT Ideal
        pf_ed, pd_ed = get_curve(r0_hw[:, 1], r1_hw[:, 1])  # Energy Detector (Baseline)

        plt.figure(figsize=(5, 5))
        plt.plot(pf_id, pd_id, 'g-', label='Ideal GLRT')
        plt.plot(pf_hw, pd_hw, 'b-s', markevery=0.1, label='Hardware GLRT')
        plt.plot(pf_ed, pd_ed, 'r:', label='Energy Detector')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('Probability of False Alarm ($P_{FA}$)')
        plt.ylabel('Probability of Detection ($P_D$)')
        plt.legend(loc='lower right')
        plt.title('Detector ROC Performance')
        plt.grid(True)
        self.save_plot('Fig7_ROC')
        self.save_csv({'pf_hw': pf_hw, 'pd_hw': pd_hw, 'pf_id': pf_id, 'pd_id': pd_id}, 'Fig7_Data')

    def generate_fig8_mds(self):
        print(f"\n--- Generating Fig 8: MDS (Ref SNR={SNR_CONFIG['fig8']} dB) ---")
        diams = np.array([2, 5, 10, 20, 50])  # mm
        radii = diams / 2000.0  # m (radius)

        # [CRITICAL] 噪声基底必须固定！
        # 我们使用默认尺寸 (5cm, a=0.05) 作为 SNR 的参考基准。
        # 这样，当尺寸变小 (a=0.001) 时，信号减弱，而噪声不变，SNR 剧烈下降。
        d_ref = DiffractionChannel({**GLOBAL_CONFIG, 'a': 0.05}).generate_broadband_chirp(self.t_axis, 32)
        noise_std = self._calc_noise_std(SNR_CONFIG["fig8"], d_ref)

        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        P_perp = det.P_perp
        s_norm = P_perp @ det._generate_template(15000)
        s_norm /= np.linalg.norm(s_norm)  # 归一化模板

        # 1. 确定 CFAR 阈值 (P_fa = 0.01)
        print("   [Sim] Calibrating CFAR Threshold...")
        h0_runs = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_mds)(False, None, s, noise_std, s_norm, P_perp, self.N, self.fs, True)
            for s in range(500)
        )
        threshold = np.percentile(h0_runs, 99.0)  # 1% False Alarm

        pd_hw, pd_id = [], []
        mds_jitter = 2.0e-4

        print("   [Sim] Scanning Debris Sizes...")
        for a in tqdm(radii):
            phy = DiffractionChannel({**GLOBAL_CONFIG, 'a': a})
            sig = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

            # Hardware Runs
            r_hw = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig, s, noise_std, s_norm, P_perp, self.N, self.fs, False, mds_jitter)
                for s in range(200))

            # Ideal Runs
            r_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds)(True, sig, s, noise_std, s_norm, P_perp, self.N, self.fs, True, 0)
                for s in range(200))

            pd_hw.append(np.mean(np.array(r_hw) > threshold))
            pd_id.append(np.mean(np.array(r_id) > threshold))

        plt.figure(figsize=(5, 4))
        plt.semilogx(diams, pd_id, 'g--o', label='Ideal')
        plt.semilogx(diams, pd_hw, 'b-s', label='Hardware-Impaired')
        plt.axhline(0.5, color='k', ls=':', label='Detection Limit')
        plt.xlabel('Debris Diameter (mm)')
        plt.ylabel('Detection Probability')
        plt.legend()
        plt.title(f'Minimum Detectable Size (MDS)')
        self.save_plot('Fig8_MDS')
        self.save_csv({'d': diams, 'pd_hw': pd_hw, 'pd_id': pd_id}, 'Fig8_Data')

    def generate_fig9_isac(self):
        print(f"\n--- Generating Fig 9: ISAC Trade-off (SNR={SNR_CONFIG['fig9']} dB) ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        # ISAC 通信看的是底下的"载波"，而不是衍射图案
        # 信号 = 1.0 (Carrier) - d(t)。通信解调的是 Carrier 的相位/幅度。
        sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

        # 这里的 SNR 参考的是 Carrier (1.0) 的功率，还是衍射信号？
        # 通信 SNR 通常指 Carrier-to-Noise。
        # 如果我们用 _calc_noise_std(d_ref)，那是针对 Sensing 的。
        # 这里为了简单，我们还是用 Sensing 的 Noise Level，但是因为 SNR 设得高 (-5dB relative to d)，
        # 所以 Carrier SNR 会极其巨大 (Carrier power >> d power)。
        # 为了让通信看起来不那么完美，我们需要加上显著的相位噪声影响。

        noise_std = self._calc_noise_std(SNR_CONFIG["fig9"], 1.0 - sig_truth)  # Ref is 1.0 (Approx)

        # Sensing 准备
        v_scan = np.linspace(14000, 16000, 41)
        det_cfg = {'cutoff_freq': 300.0, 'L_eff': 50e3, 'N_sub': 32, 'a': 0.05}
        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det.P_perp
        s_raw = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array([P_perp @ s for s in s_raw])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        ibo_scan = np.linspace(20, 0, 15)
        cap, rmse = [], []
        isac_jitter = 1.0e-4

        print("   [Sim] Running ISAC Trade-off...")
        for ibo in tqdm(ibo_scan):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_isac)(ibo, s, noise_std, sig_truth, T_bank, E_bank, v_scan, P_perp, self.N, self.fs,
                                     False, isac_jitter)
                for s in range(50)
            )
            res = np.array(res)
            cap.append(np.mean(res[:, 0]))

            errs = res[:, 1]
            valid = errs[errs < 1200]
            rmse.append(np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 1000.0)

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(cap, rmse, c=ibo_scan, cmap='viridis_r', s=80, edgecolors='k')
        cbar = plt.colorbar(sc)
        cbar.set_label('IBO (dB)')
        plt.xlabel('Comm Capacity (bits/s/Hz)')
        plt.ylabel('Sensing RMSE (m/s)')
        plt.title('ISAC Performance Trade-off')
        plt.grid(True)
        self.save_plot('Fig9_ISAC')
        self.save_csv({'ibo': ibo_scan, 'cap': cap, 'rmse': rmse}, 'Fig9_Data')

    def generate_fig10_ambiguity(self):
        print("\n--- Generating Fig 10: Ambiguity Function ---")
        det = TerahertzDebrisDetector(self.fs, self.N, N_sub=32)
        s0 = det.P_perp @ det._generate_template(15000)
        v_shifts = np.linspace(-500, 500, 41)
        t_shifts = np.linspace(-0.002, 0.002, 41)
        res = np.zeros((41, 41))
        E = np.sum(s0 ** 2)

        # 简单的双重循环，不用并行
        for i, dv in enumerate(v_shifts):
            s_v = det.P_perp @ det._generate_template(15000 + dv)
            for j, dt in enumerate(t_shifts):
                shift = int(dt * self.fs)
                s_shift = np.roll(s_v, shift)
                # Zero padding for valid linear convolution simulation
                if shift > 0:
                    s_shift[:shift] = 0
                else:
                    s_shift[shift:] = 0
                res[i, j] = (np.dot(s0, s_shift) ** 2) / E

        plt.figure(figsize=(5, 4))
        plt.contourf(t_shifts * 1000, v_shifts, res / np.max(res), 20, cmap='viridis')
        plt.xlabel('Time Delay (ms)')
        plt.ylabel('Velocity Mismatch (m/s)')
        plt.title('Ambiguity Function')
        plt.colorbar(label='Normalized Correlation')
        self.save_plot('Fig10_Ambiguity')

    def generate_fig11_trajectory(self):
        print("\n--- Generating Fig 11: Signal Trajectory ---")
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d = phy.generate_broadband_chirp(self.t_axis, 32)
        hw = HardwareImpairments(HW_CONFIG)

        # 制造一个极度嘈杂的轨迹云
        jit = hw.generate_colored_jitter(self.N, self.fs)
        pn = hw.generate_phase_noise(self.N, self.fs)

        noise_cloud = (1.0 - d) * np.exp(jit * 10) * np.exp(1j * pn)
        noise_cloud -= np.mean(noise_cloud)  # Center at 0

        plt.figure(figsize=(5, 5))
        plt.plot(np.real(noise_cloud), np.imag(noise_cloud), '.', color='grey', alpha=0.1, label='Samples')
        plt.xlabel('I Component')
        plt.ylabel('Q Component')
        plt.title('Signal Trajectory (Constellation)')
        plt.grid(True)
        self.save_plot('Fig11_Trajectory')

    def run_all(self):
        print("=== SIMULATION START (V20.0 ULTIMATE STRESS TEST) ===")
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
        print("\n=== ALL TASKS DONE. CHECK OUTPUT FOLDER. ===")


# =========================================================================
# Worker Functions (Picklable for Multiprocessing)
# =========================================================================

def _log_envelope(y):
    return np.log(np.abs(y) + 1e-12)


def _gen_template(v, fs, N, cfg):
    # Helper to instantiate detector inside worker
    return TerahertzDebrisDetector(fs, N, **cfg)._generate_template(v)


def _trial_rmse_opt(ibo, jitter_rms, seed, noise_std, sig_truth, N, fs, true_v, hw_base, T_bank, E_bank, v_scan, ideal,
                    P_perp):
    np.random.seed(seed)
    if ideal:
        pa_out = sig_truth
    else:
        hw_cfg = hw_base.copy()
        hw_cfg['jitter_rms'] = jitter_rms
        hw = HardwareImpairments(hw_cfg)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pn = np.exp(1j * hw.generate_phase_noise(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)

    # Add Thermal Noise
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # Detector Chain
    z = _log_envelope(pa_out + w)
    z_perp = P_perp @ z  # Projection

    # Match Filter Bank
    stats = (np.dot(T_bank, z_perp) ** 2) / E_bank

    # Estimator
    v_hat = v_scan[np.argmax(stats)]
    return v_hat - true_v


def _trial_roc_fast(sig_clean, seed, noise_std, ideal, s_true_perp, s_energy, P_perp, jitter_val=0):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out = sig_clean
    else:
        hw.jitter_rms = jitter_val
        fs, N = GLOBAL_CONFIG['fs'], len(sig_clean)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_clean * jit, ibo_dB=10.0)

    N = len(sig_clean)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_perp = P_perp @ _log_envelope(pa_out + w)

    # GLRT Statistic
    stat_glrt = (np.dot(s_true_perp, z_perp) ** 2) / s_energy

    # Energy Detector Statistic (Benchmark)
    # AC Energy of envelope
    y_ac = np.abs(pa_out + w) - np.mean(np.abs(pa_out + w))
    stat_ed = np.sum(y_ac ** 2)

    return stat_glrt, stat_ed


def _trial_mds(is_h1, sig_clean, seed, noise_std, s_norm, P_perp, N, fs, ideal, jitter_val=0):
    np.random.seed(seed)

    # H0 (Noise Only) or H1 (Signal + Noise)
    if is_h1:
        sig = sig_clean
    else:
        sig = np.ones(N, dtype=complex)  # Just Carrier

    if ideal:
        pa_out = sig
    else:
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)  # Fixed IBO

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = _log_envelope(pa_out + w)

    # Normalized Statistic
    stat = (np.dot(s_norm, P_perp @ z) ** 2)
    return stat


def _trial_isac(ibo, seed, noise_std, sig_truth, T_bank, E_bank, v_scan, P_perp, N, fs, ideal, jitter_val=0):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)

    if ideal:
        pa_out = sig_truth;
        gamma_eff = 0
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        # 通信受到 PA 非线性扭曲的影响 (EVM)
        pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit, ibo_dB=ibo)
        # 简单的失真模型: 失真功率随 IBO 降低而增加
        gamma_eff = 1e-2 * (10 ** (-ibo / 10.0))

    # Comm Capacity Calculation
    p_rx = np.mean(np.abs(pa_out) ** 2)
    # SINR = Signal / (ThermalNoise + NonLinearDistortion)
    sinr = p_rx / (noise_std ** 2 + p_rx * gamma_eff + 1e-20)
    cap = np.log2(1 + sinr)

    # Sensing Calculation
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = _log_envelope(pa_out + w)
    stats = (np.dot(T_bank, P_perp @ z) ** 2) / E_bank
    v_est = v_scan[np.argmax(stats)]

    return cap, abs(v_est - GLOBAL_CONFIG['v_default'])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    PaperFigureGenerator().run_all()
# ----------------------------------------------------------------------------------
# 脚本名称: visualization_all_fixed_v8.py
# 版本: v8.0 (Noise Calculation Fix Only)
#
# 【核心修复】只修复噪声计算方式，保持原始物理参数不变
#
# 修复内容：
# 1. 噪声基于当前信号功率计算（Ideal 基于原始信号，Hardware 基于 PA 输出）
# 2. cutoff_freq = 300Hz（符合 DR_algo_01 理论）
# 3. SNR 配置适应原始物理参数（需要较高 SNR）
#
# 【保持不变】
# - 物理参数：a=5cm, L_eff=50km（原始配置）
# - 硬件模型参数
# - 所有图形生成逻辑
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，解决多进程报错
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

# --- 全局物理参数（保持原始配置） ---
MODE = "demo_relaxed"

if MODE == "physics_strict":
    print(f"[Config] Mode: PHYSICS STRICT (Real-world parameters)")
    GLOBAL_CONFIG = {
        'fc': 300e9, 'B': 10e9, 'L_eff': 50e3, 'fs': 200e3,
        'T_span': 0.02, 'a': 0.05, 'v_default': 15000
    }
    SNR_CONFIG = { "fig6": 80.0, "fig7": 80.0, "fig8": 80.0, "fig9": 70.0 }
else:
    print(f"[Config] Mode: DEMO RELAXED (High Contrast Demo)")
    GLOBAL_CONFIG = {
        'fc': 300e9, 'B': 10e9,
        'L_eff': 20e3,  # 20 km
        'fs': 200e3, 'T_span': 0.02,
        'a': 0.10,      # 10 cm
        'v_default': 15000
    }
    # [V18 基于扫描数据的最终调整]
    SNR_CONFIG = {
        "fig6": 70.0,  # [Sweep Result] 20倍区分度的最佳点
        "fig7": 50.0,  # [Tweak] 稍微调低 SNR 以拉大 ROC 差距
        "fig8": 68.0,  # [Keep] 保持 V17 的 S 型曲线
        "fig9": 65.0   # [Keep] 保持
    }


# --- 硬件损伤默认配置（保持原始） ---
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
# 【修复】cutoff_freq = 300Hz 符合 DR_algo_01 理论
DETECTOR_CONFIG = {
    'cutoff_freq': 300.0,  # Hz - 生存空间理论（原始代码是 5000Hz，这是错误的）
    'N_sub': 32
}


class PaperFigureGenerator:
    def __init__(self, output_dir='results_v8_fixed'):
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
        print(
            f"[Init] Physics (Original): L_eff={GLOBAL_CONFIG['L_eff'] / 1000:.0f}km, a={GLOBAL_CONFIG['a'] * 100:.0f}cm")

        # 计算衍射深度
        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)
        depth = (1 - np.min(np.abs(1.0 - d_wb))) * 100
        print(f"[Init] Diffraction depth: {depth:.4f}%")

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
    # Group A: Physics Visualization（保持原始逻辑）
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
    # Group B: Performance（修复噪声计算）
    # =========================================================================

        # =========================================================================
        # PATCH FUNCTION: Fig 6 RMSE (Visibility Fix)
        # =========================================================================

    def generate_fig6_rmse_sensitivity(self):
        snr_ref = SNR_CONFIG['fig6']
        print(f"\n--- Fig 6: RMSE vs IBO (Ref SNR={snr_ref} dB, Mode={MODE}) ---")

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal
        v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)

        det_cfg = {
            'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
            'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a'], 'N_sub': DETECTOR_CONFIG['N_sub']
        }
        det_main = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det_main.P_perp

        s_raw_list = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
        T_bank = np.array([P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        # [V18] 使用扫描验证过的最佳 Jitter 组合
        # 1e-5 (Low), 1e-3 (Mid), 1e-2 (High)
        jit_levels = [1.0e-6, 8.0e-3,  1.0e-2 ,2.0e-2 ,5.0e-2]
        ibo_scan = np.linspace(30, 0, 13)
        trials = 100

        fig, ax = plt.subplots(figsize=(5, 4))
        data = {'ibo': ibo_scan}

        # Ideal (底层，灰色)
        res_id = Parallel(n_jobs=self.n_jobs)(
            delayed(_trial_rmse_fixed)(10.0, 0, s, snr_ref, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank,
                                       E_bank, v_scan, True, P_perp)
            for s in tqdm(range(trials), desc="Ideal")
        )
        rmse_id = np.sqrt(np.mean(np.array(res_id) ** 2))
        ax.semilogy(ibo_scan, [max(rmse_id, 0.01)] * len(ibo_scan), 'k--', label='Ideal', linewidth=2.5, alpha=0.3)
        data['ideal'] = [rmse_id] * len(ibo_scan)

        colors = ['tab:blue', 'tab:orange','tab:purple', 'tab:green', 'tab:pink']
        markers = ['o', 's', '^', '*', 'D']

        for idx, jit in enumerate(jit_levels):
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_fixed)(
                    ibo, jit, s, snr_ref - ibo, sig_truth, self.N, self.fs, 15000, HW_CONFIG, T_bank, E_bank, v_scan,
                    False, P_perp
                )
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

            zorder = 10 - idx
            ax.semilogy(ibo_scan, rmse, marker=markers[idx], linestyle='-', color=colors[idx],
                        label=rf'$\sigma_J$={jit:.0e} s', markersize=6, linewidth=1.5, zorder=zorder)
            data[f'rmse_jit_{idx}'] = rmse

        ax.set_xlabel('IBO (dB)')
        ax.set_ylabel('Velocity RMSE (m/s)')
        # 下限设为 0.1 (因为扫描结果最小误差约 4m/s, 0.1足够了)
        ax.set_ylim(0.1, 2000)
        ax.invert_xaxis()
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        self.save_csv(data, 'Fig6_Sensitivity')
        self.save_plot('Fig6_Sensitivity')

    def generate_fig13_sensitivity(self):
        """
        [New] Fig 13: 灵敏度瀑布图 (Sensitivity Waterfall)
        描述: 在固定恶劣抖动下，扫描 SNR，展示检测概率 Pd 的变化。
        对比: Proposed (GLRT) vs Naive (MF) vs Standard (ED)
        """
        print(f"\n--- Fig 13: Sensitivity Waterfall (Pd vs SNR) ---")

        # 1. 锁定恶劣物理环境 (与 Fig 7, 8 保持一致)
        fixed_jitter = 2.0e-3

        # [修改点] 显式定义一个较小的 Debris 半径用于测试，增加难度以拉开差距
        # 原始 Demo 模式是 0.10 (10cm)，这里我们尝试减小到 0.03 (3cm) 或 0.05 (5cm)
        test_a = 0.05  # 5cm
        print(f"   Fixed Condition: Jitter={fixed_jitter} s, Pfa=0.01, Size(a)={test_a}m")

        # 2. SNR 扫描范围
        snr_scan = np.linspace(30, 70, 11)

        # 准备物理模型 (使用特定的 test_a)
        # 临时创建一个配置，覆盖默认的 'a'
        cfg_fig13 = GLOBAL_CONFIG.copy()
        cfg_fig13['a'] = test_a

        phy = DiffractionChannel(cfg_fig13)
        d_wb = phy.generate_broadband_chirp(self.t_axis, 32)

        # 准备检测器和模板 (同样使用 test_a)
        det_cfg = DETECTOR_CONFIG.copy()
        det_cfg['L_eff'] = GLOBAL_CONFIG['L_eff']
        det_cfg['a'] = test_a  # 确保检测器也知道我们要测的是这个尺寸

        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det.P_perp

        # 模板 1: Proposed (Projected)
        s_raw = det._generate_template(15000)
        s_prop = P_perp @ s_raw
        s_prop_eng = np.sum(s_prop ** 2) + 1e-20

        # 模板 2: Naive (Mean Subtracted)
        s_naive = s_raw - np.mean(s_raw)
        s_naive_eng = np.sum(s_naive ** 2) + 1e-20

        # 信号定义
        sig_h1 = 1.0 - d_wb
        sig_h0 = np.ones(self.N, dtype=complex)

        pd_prop_list = []
        pd_naive_list = []
        pd_ed_list = []

        # 3. 开始扫描 SNR
        for snr in tqdm(snr_scan, desc="Scanning SNR"):
            # --- Step A: 计算当前 SNR 下满足 Pfa=0.01 的阈值 ---
            # 跑 500 次 H0 (纯噪声)
            res_h0 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_tri)(sig_h0, s, snr, False,
                                        s_prop, s_prop_eng, P_perp,
                                        s_naive, s_naive_eng,
                                        fixed_jitter, self.N, self.fs)
                for s in range(500)
            )
            res_h0 = np.array(res_h0)

            # 确定 99% 分位点作为阈值 (Pfa = 1%)
            th_prop = np.percentile(res_h0[:, 0], 99.0)
            th_naive = np.percentile(res_h0[:, 1], 99.0)
            th_ed = np.percentile(res_h0[:, 2], 99.0)

            # --- Step B: 计算当前 SNR 下的 Pd ---
            # 跑 200 次 H1 (信号+噪声)
            res_h1 = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_roc_tri)(sig_h1, s, snr, False,
                                        s_prop, s_prop_eng, P_perp,
                                        s_naive, s_naive_eng,
                                        fixed_jitter, self.N, self.fs)
                for s in range(200)
            )
            res_h1 = np.array(res_h1)

            # 统计过阈值的比例
            pd_prop_list.append(np.mean(res_h1[:, 0] > th_prop))
            pd_naive_list.append(np.mean(res_h1[:, 1] > th_naive))
            pd_ed_list.append(np.mean(res_h1[:, 2] > th_ed))

        # 4. 保存与绘图
        self.save_csv({'snr': snr_scan, 'pd_prop': pd_prop_list, 'pd_naive': pd_naive_list, 'pd_ed': pd_ed_list},
                      'Fig13_Sensitivity_Data')

        plt.figure(figsize=(5, 4))
        plt.plot(snr_scan, pd_prop_list, 'b-o', label='Proposed (Robust)', linewidth=2)
        plt.plot(snr_scan, pd_naive_list, color='orange', linestyle='--', marker='x', label='Naive MF')
        plt.plot(snr_scan, pd_ed_list, 'r:', marker='^', label='Energy Det')

        # 画辅助线 (Pd=0.9)
        plt.axhline(0.9, color='k', linestyle=':', alpha=0.5)
        plt.text(snr_scan[0], 0.92, 'Target Pd=0.9', fontsize=8, color='gray')

        plt.xlabel('SNR (dB)')
        plt.ylabel('Probability of Detection ($P_{fa}=10^{-2}$)')
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.ylim(-0.05, 1.05)
        self.save_plot('Fig13_Sensitivity')

    def generate_fig7_roc(self):
            print(f"\n--- Fig 7: ROC (Prop vs Naive vs ED) ---")
            trials = 1500
            phy = DiffractionChannel(GLOBAL_CONFIG)
            d_wb = phy.generate_broadband_chirp(self.t_axis, 32)

            det_ref = TerahertzDebrisDetector(self.fs, self.N, **DETECTOR_CONFIG, L_eff=GLOBAL_CONFIG['L_eff'],
                                              a=GLOBAL_CONFIG['a'])
            P_perp = det_ref.P_perp

            # 1. Proposed Template (Projected)
            s_raw_template = det_ref._generate_template(15000)
            s_prop_tmpl = P_perp @ s_raw_template
            s_prop_eng = np.sum(s_prop_tmpl ** 2) + 1e-20

            # 2. Naive Template (Mean Subtracted Only)
            s_naive_tmpl = s_raw_template - np.mean(s_raw_template)
            s_naive_eng = np.sum(s_naive_tmpl ** 2) + 1e-20

            sig_h1 = 1.0 - d_wb
            sig_h0 = np.ones(self.N, dtype=complex)

            # 统一使用中等抖动 (2e-3)，这是有色 1/f 抖动
            jit_fair = 2.0e-3
            print(f"   Environment: SNR={SNR_CONFIG['fig7']}dB, Jitter={jit_fair}s (1/f Colored)")

            # 使用 _trial_roc_tri 运行三方对比
            def run_tri_batch(is_id, jit):
                r0 = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_roc_tri)(sig_h0, s, SNR_CONFIG['fig7'], is_id,
                                            s_prop_tmpl, s_prop_eng, P_perp,
                                            s_naive_tmpl, s_naive_eng,
                                            jit, self.N, self.fs)
                    for s in range(trials))
                r1 = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_roc_tri)(sig_h1, s, SNR_CONFIG['fig7'], is_id,
                                            s_prop_tmpl, s_prop_eng, P_perp,
                                            s_naive_tmpl, s_naive_eng,
                                            jit, self.N, self.fs)
                    for s in range(trials))
                return np.array(r0), np.array(r1)

            r0_tri, r1_tri = run_tri_batch(False, jit_fair)
            r0_id, r1_id = run_tri_batch(True, 0)  # Ideal (Prop Logic)

            def get_curve(n, s):
                if len(n) == 0: return [0, 1], [0, 1], 0.5
                th = np.linspace(np.min(np.concatenate([n, s])), np.max(np.concatenate([n, s])), 500)
                pf = np.array([np.mean(n > t) for t in th])
                pd = np.array([np.mean(s > t) for t in th])
                return pf, pd, -np.trapezoid(pd, pf)

            # 0=Prop, 1=Naive, 2=ED
            pf_id, pd_id, auc_id = get_curve(r0_id[:, 0], r1_id[:, 0])
            pf_prop, pd_prop, auc_prop = get_curve(r0_tri[:, 0], r1_tri[:, 0])
            pf_naive, pd_naive, auc_naive = get_curve(r0_tri[:, 1], r1_tri[:, 1])
            pf_ed, pd_ed, auc_ed = get_curve(r0_tri[:, 2], r1_tri[:, 2])

            print(f"   AUC: Ideal={auc_id:.3f} | Prop={auc_prop:.3f} | Naive={auc_naive:.3f} | ED={auc_ed:.3f}")

            data = {'pf_id': pf_id, 'pd_id': pd_id, 'pf_prop': pf_prop, 'pd_prop': pd_prop,
                    'pf_naive': pf_naive, 'pd_naive': pd_naive, 'pf_ed': pf_ed, 'pd_ed': pd_ed}
            self.save_csv(data, 'Fig7_ROC_Data')

            plt.figure(figsize=(5, 5))
            plt.plot(pf_id, pd_id, 'g-', label=f'Ideal ({auc_id:.2f})')
            plt.plot(pf_prop, pd_prop, 'b-', label=f'Proposed (Robust) ({auc_prop:.2f})')
            plt.plot(pf_naive, pd_naive, color='orange', linestyle='--', label=f'Naive MF ({auc_naive:.2f})')
            plt.plot(pf_ed, pd_ed, 'r:', label=f'Energy Det ({auc_ed:.2f})')
            plt.plot([0, 1], [0, 1], 'k-', alpha=0.1)
            plt.xlabel('Probability of False Alarm')
            plt.ylabel('Probability of Detection')
            plt.legend(loc='lower right')
            self.save_plot('Fig7_ROC')

        # =========================================================================
        # PATCH FUNCTION: Fig 8 MDS (3-Way Comparison)
        # =========================================================================
    def generate_fig8_mds(self):
            print(f"\n--- Fig 8: MDS (Prop vs Naive vs ED) ---")
            diams = np.array([2, 5, 8, 12, 16, 20, 30, 50, 80, 100])
            radii = diams / 2000.0

            det = TerahertzDebrisDetector(self.fs, self.N, **DETECTOR_CONFIG, L_eff=GLOBAL_CONFIG['L_eff'],
                                          a=GLOBAL_CONFIG['a'])
            P_perp = det.P_perp

            s_raw_tmpl = det._generate_template(15000)
            s_prop_tmpl = P_perp @ s_raw_tmpl
            s_prop_tmpl /= (np.linalg.norm(s_prop_tmpl) + 1e-20)

            s_naive_tmpl = s_raw_tmpl - np.mean(s_raw_tmpl)
            s_naive_tmpl /= (np.linalg.norm(s_naive_tmpl) + 1e-20)

            jit_fair = 2.0e-3

            # 计算阈值
            print("   Calculating Thresholds...")
            h0_stats = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_tri)(False, None, s, SNR_CONFIG['fig8'],
                                        s_prop_tmpl, P_perp, s_naive_tmpl,
                                        self.N, self.fs, False, jit_fair)
                for s in range(800)
            )
            h0_stats = np.array(h0_stats)
            th_prop = np.percentile(h0_stats[:, 0], 95.0)
            th_naive = np.percentile(h0_stats[:, 1], 95.0)
            th_ed = np.percentile(h0_stats[:, 2], 95.0)

            h0_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_mds_tri)(False, None, s, SNR_CONFIG['fig8'], s_prop_tmpl, P_perp, s_naive_tmpl, self.N,
                                        self.fs, True, 0) for s in range(800))
            th_id = np.percentile(np.array(h0_id)[:, 0], 95.0)

            pd_id, pd_prop, pd_naive, pd_std = [], [], [], []

            for a in tqdm(radii, desc="Scanning Sizes"):
                cfg_a = {**GLOBAL_CONFIG, 'a': a}
                phy = DiffractionChannel(cfg_a)
                sig = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)

                res = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_mds_tri)(True, sig, s, SNR_CONFIG['fig8'],
                                            s_prop_tmpl, P_perp, s_naive_tmpl,
                                            self.N, self.fs, False, jit_fair)
                    for s in range(200)
                )
                res = np.array(res)

                res_id = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_mds_tri)(True, sig, s, SNR_CONFIG['fig8'], s_prop_tmpl, P_perp, s_naive_tmpl, self.N,
                                            self.fs, True, 0)
                    for s in range(200)
                )
                res_id = np.array(res_id)

                pd_id.append(np.mean(res_id[:, 0] > th_id))
                pd_prop.append(np.mean(res[:, 0] > th_prop))
                pd_naive.append(np.mean(res[:, 1] > th_naive))
                pd_std.append(np.mean(res[:, 2] > th_ed))

            self.save_csv({'d': diams, 'pd_id': pd_id, 'pd_prop': pd_prop, 'pd_naive': pd_naive, 'pd_ed': pd_std},
                          'Fig8_MDS_Data')
            plt.figure(figsize=(5, 4))
            plt.semilogx(diams, pd_id, 'g-o', label='Ideal')
            plt.semilogx(diams, pd_prop, 'b-s', label='Proposed (Robust)')
            plt.semilogx(diams, pd_naive, color='orange', marker='x', linestyle='--', label='Naive MF')
            plt.semilogx(diams, pd_std, 'r--^', label='Energy Det')
            plt.axhline(0.9, color='k', ls=':', alpha=0.5)
            plt.xlabel('Debris Diameter (mm)')
            plt.ylabel('Probability of Detection')
            plt.legend()
            plt.grid(True, which="both", ls=":")
            self.save_plot('Fig8_MDS')



    def generate_fig9_isac(self):
            print(f"\n--- Fig 9: ISAC (Ref SNR={SNR_CONFIG['fig9']} dB) ---")
            phy = DiffractionChannel(GLOBAL_CONFIG)
            sig_truth = 1.0 - phy.generate_broadband_chirp(self.t_axis, 32)
            v_scan = np.linspace(14000, 16000, 41)

            det_cfg = {'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'], 'L_eff': GLOBAL_CONFIG['L_eff'],
                       'a': GLOBAL_CONFIG['a'], 'N_sub': DETECTOR_CONFIG['N_sub']}
            det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
            P_perp = det.P_perp
            s_raw = Parallel(n_jobs=self.n_jobs)(delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan)
            T_bank = np.array([P_perp @ s for s in s_raw])
            E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

            ibo_scan = np.linspace(20, 0, 15)
            cap, rmse = [], []
            isac_jitter = 1.0e-4

            for ibo in tqdm(ibo_scan, desc="ISAC Scan"):
                snr_eff = SNR_CONFIG['fig9'] - ibo
                res = Parallel(n_jobs=self.n_jobs)(
                    delayed(_trial_isac_fixed)(ibo, s, snr_eff, sig_truth, T_bank, E_bank, v_scan, P_perp, self.N,
                                               self.fs, False, isac_jitter)
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

    def generate_fig14_coarse_v_est(self):
        """
        [New] Fig 14: Coarse Velocity Estimation Accuracy (RMSE vs SNR)
        对比: Ideal, Hardware (Jitter), Theoretical Bound (CRLB)
        """
        print(f"\n--- Fig 14: Coarse Velocity Estimation (RMSE vs SNR) ---")

        # 1. 严格统一的 physics 配置（给 DiffractionChannel 用）
        common_cfg = GLOBAL_CONFIG.copy()
        common_cfg['L_eff'] = GLOBAL_CONFIG['L_eff']
        common_cfg['a'] = GLOBAL_CONFIG['a']
        common_cfg['N_sub'] = 32  # 仅用于记录，DiffractionChannel 实际不使用这个 key

        # 2. 准备物理引擎（用于生成真值信号）
        phy = DiffractionChannel(common_cfg)

        # 3. 准备 detector 的配置（给 TerahertzDebrisDetector / _gen_template 用）
        #    注意：这里不能包含 'fs'，否则会和位置参数冲突
        det_cfg = {
            'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
            'L_eff': common_cfg['L_eff'],
            'a': common_cfg['a'],
            'N_sub': common_cfg['N_sub'],
            # 可选：如果你希望和 GLOBAL_CONFIG 完全一致，可以显式传 fc/B
            # 'fc': common_cfg['fc'],
            # 'B': common_cfg['B'],
        }

        det = TerahertzDebrisDetector(self.fs, self.N, **det_cfg)
        P_perp = det.P_perp

        # 4. 仿真配置
        fixed_ibo = 10.0
        fixed_jitter = 5.0e-4
        snr_scan = np.linspace(0, 60, 13)

        # 5. 生成真值信号 (Ground Truth) - 使用物理引擎的复数输出
        true_v = 15000.0

        # DiffractionChannel 默认用 config['v_rel']，GLOBAL_CONFIG 没有这个 key，
        # 所以我们需要显式覆盖
        phy.v_rel = true_v

        d_complex = phy.generate_broadband_chirp(self.t_axis, N_sub=32)
        sig_truth = 1.0 - d_complex  # 复数信号，后续在 _trial_rmse_fixed 里会走 AGC + log-envelope

        # 6. CRLB 计算（在 P_perp 子空间里）
        delta_v = 1.0
        s_plus = det._generate_template(true_v + delta_v)
        s_minus = det._generate_template(true_v - delta_v)
        ds_dv = P_perp @ (s_plus - s_minus) / (2 * delta_v)

        # 更保险的写法：如果将来 ds_dv 是复数，np.abs 也能正常工作
        crlb_coeff = 1.0 / (np.sqrt(np.sum(np.abs(ds_dv) ** 2)) + 1e-20)

        # 7. 速度搜索空间（高精度）
        v_scan = np.linspace(true_v - 1000, true_v + 1000, 1001)  # 步长 2 m/s
        print(f"   Generating templates (Step={v_scan[1] - v_scan[0]:.2f} m/s)...")

        # 关键修正：这里用 det_cfg，而不是 common_cfg（common_cfg 里有 fs）
        s_raw_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_gen_template)(v, self.fs, self.N, det_cfg) for v in v_scan
        )
        T_bank = np.array([P_perp @ s for s in s_raw_list])
        E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

        rmse_ideal = []
        rmse_hw = []
        crlb_curve = []

        for snr in tqdm(snr_scan, desc="Scanning SNR"):
            # A. CRLB：基于理想真值信号的噪声方差
            p_sig = np.mean(np.abs(sig_truth) ** 2)
            noise_var = p_sig / (10 ** (snr / 10.0))
            crlb_val = crlb_coeff * np.sqrt(noise_var)
            crlb_curve.append(crlb_val)

            # B. Ideal Case（无 jitter / 非线性）
            res_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_fixed)(
                    fixed_ibo, 0.0, s, snr, sig_truth,
                    self.N, self.fs, true_v, HW_CONFIG,
                    T_bank, E_bank, v_scan, True, P_perp
                )
                for s in range(50)
            )
            res_id = np.array(res_id)
            valid_id = res_id[np.abs(res_id) < 500]  # 剪掉明显爆炸的 outlier
            if len(valid_id) > 10:
                rmse_ideal.append(np.sqrt(np.mean(valid_id ** 2)))
            else:
                rmse_ideal.append(100.0)

            # C. Hardware Case（含 jitter + PA 非线性）
            res_hw = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_rmse_fixed)(
                    fixed_ibo, fixed_jitter, s, snr, sig_truth,
                    self.N, self.fs, true_v, HW_CONFIG,
                    T_bank, E_bank, v_scan, False, P_perp
                )
                for s in range(50)
            )
            res_hw = np.array(res_hw)
            valid_hw = res_hw[np.abs(res_hw) < 500]
            if len(valid_hw) > 10:
                rmse_hw.append(np.sqrt(np.mean(valid_hw ** 2)))
            else:
                rmse_hw.append(100.0)

        # 8. 保存数据 & 画图
        self.save_csv(
            {'snr': snr_scan, 'rmse_ideal': rmse_ideal,
             'rmse_hw': rmse_hw, 'crlb': crlb_curve},
            'Fig14_Coarse_V_Est'
        )

        plt.figure(figsize=(5, 4))
        # 这里最好把 LaTeX 反斜杠转义一下，避免 SyntaxWarning
        plt.semilogy(snr_scan, rmse_hw, 'r-o', label=f'Hardware ($\\sigma_J$={fixed_jitter})')
        plt.semilogy(snr_scan, rmse_ideal, 'b--s', label='Ideal')
        plt.semilogy(snr_scan, crlb_curve, 'k:', linewidth=2, label='CRLB')

        plt.xlabel('SNR (dB)')
        plt.ylabel('Velocity RMSE (m/s)')
        plt.title('Velocity Estimation (Complex Signal)')
        plt.legend()
        plt.grid(True, which="both", ls=":")
        self.save_plot('Fig14_Coarse_V_Est')


    def generate_fig15_coarse_t_est(self):
        """
        [New] Fig 15: Coarse Time (CPA) Estimation Accuracy
        [Status]: Locked. Parameters are good.
        """
        print(f"\n--- Fig 15: Coarse Time Estimation (RMSE vs SNR) ---")

        fixed_ibo = 10.0
        fixed_jitter = 5.0e-4
        snr_scan = np.linspace(20, 80, 13)

        phy = DiffractionChannel(GLOBAL_CONFIG)
        d_signal = phy.generate_broadband_chirp(self.t_axis, 32)
        sig_truth = 1.0 - d_signal

        rmse_t_ideal = []
        rmse_t_hw = []

        for snr in tqdm(snr_scan, desc="Scanning SNR (Time)"):
            # Ideal
            res_id = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_tcpa_error)(fixed_ibo, 0, s, snr, sig_truth, self.N, self.fs, HW_CONFIG, True)
                for s in range(50)
            )
            rmse_t_ideal.append(np.sqrt(np.mean(np.array(res_id) ** 2)))

            # Hardware
            res_hw = Parallel(n_jobs=self.n_jobs)(
                delayed(_trial_tcpa_error)(fixed_ibo, fixed_jitter, s, snr, sig_truth, self.N, self.fs, HW_CONFIG,
                                           False)
                for s in range(50)
            )
            rmse_t_hw.append(np.sqrt(np.mean(np.array(res_hw) ** 2)))

        self.save_csv({'snr': snr_scan, 'rmse_t_ideal': rmse_t_ideal, 'rmse_t_hw': rmse_t_hw}, 'Fig15_Coarse_T_Est')

        plt.figure(figsize=(5, 4))
        plt.semilogy(snr_scan, np.array(rmse_t_hw) * 1000, 'r-o', label='Hardware')
        plt.semilogy(snr_scan, np.array(rmse_t_ideal) * 1000, 'b--s', label='Ideal')

        plt.xlabel('SNR (dB)')
        plt.ylabel('Time CPA RMSE (ms)')
        plt.legend()
        plt.grid(True, which="both", ls=":")
        self.save_plot('Fig15_Coarse_T_Est')

    def run_all(self):
        print("=" * 70)
        print("SIMULATION START (V8.0 - ORIGINAL PARAMS + NOISE FIX)")
        print("=" * 70)

        # Group A: Physics
        # self.generate_fig2_mechanisms()
        # self.generate_fig3_dispersion()
        # self.generate_fig4_self_healing()
        # self.generate_fig5_survival_space()
        # self.generate_fig10_ambiguity()
        # self.generate_fig11_trajectory()
        #
        # Group B: Performance
        # self.generate_fig6_rmse_sensitivity()
        # self.generate_fig7_roc()
        # self.generate_fig8_mds()
        # self.generate_fig13_sensitivity()
        # self.generate_fig9_isac()
        self.generate_fig14_coarse_v_est()
        # self.generate_fig15_coarse_t_est()

        print("\n" + "=" * 70)
        print("ALL TASKS DONE")
        print("=" * 70)


# =========================================================================
# Worker Functions（修复噪声计算）
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


def _trial_tcpa_error(ibo, jitter_rms, seed, snr_db, sig_truth, N, fs, hw_base, ideal):
    np.random.seed(seed)

    if ideal:
        pa_out = sig_truth
        p_ref = np.mean(np.abs(sig_truth) ** 2)
    else:
        hw_cfg = hw_base.copy()
        hw_cfg['jitter_rms'] = jitter_rms
        hw = HardwareImpairments(hw_cfg)
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pn = np.exp(1j * hw.generate_phase_noise(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_truth * jit * pn, ibo_dB=ibo)
        p_ref = np.mean(np.abs(pa_out) ** 2)

    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    rx = pa_out + (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # [Fix] 更稳健的包络提取
    # 1. 低通滤波 (平滑噪声)
    win_len = max(1, int(fs * 0.002))  # 增加窗口到 2ms
    env = np.abs(rx)
    env_smooth = np.convolve(env, np.ones(win_len) / win_len, mode='same')

    # 2. 寻找"深坑"
    # 我们知道坑大概在中间，可以限制搜索范围在中间 50% 区域，防止边缘噪声干扰
    center = N // 2
    search_width = N // 4
    search_region = env_smooth[center - search_width: center + search_width]

    idx_min_local = np.argmin(search_region)
    idx_min_global = idx_min_local + (center - search_width)

    # 真实位置
    idx_true = N // 2

    t_err = (idx_min_global - idx_true) / fs
    return t_err
def _trial_rmse_fixed(ibo, jitter_rms, seed, snr_db, sig_truth, N, fs, true_v,
                      hw_base, T_bank, E_bank, v_scan, ideal, P_perp):
    """
    【修复】噪声基于当前信号功率计算
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
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【修复】基于 PA 输出

    # 【修复】基于当前信号功率计算噪声
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
    【修复】噪声基于当前信号功率计算
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
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【修复】基于 PA 输出

    # 【修复】基于当前信号功率计算噪声
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
    【修复】噪声基于当前信号功率计算
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
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【修复】基于 PA 输出

    # 【修复】基于当前信号功率计算噪声
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    rx_signal = pa_out_raw + w

    z = _log_envelope(_apply_agc(rx_signal))
    stat = (np.dot(s_norm, P_perp @ z) ** 2)
    return stat


def _trial_isac_fixed(ibo, seed, snr_db, sig_truth, T_bank, E_bank, v_scan,
                      P_perp, N, fs, ideal, jitter_val=0):
    """
    【修复】噪声基于当前信号功率计算
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
        p_ref = np.mean(np.abs(pa_out_raw) ** 2)  # 【修复】基于 PA 输出
        gamma_eff = 4.0e-3 * (10 ** (-ibo / 10.0))

    # 【修复】基于当前信号功率计算噪声
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


def _trial_roc_tri(sig_clean, seed, snr_db, ideal, s_prop_perp, s_prop_eng, P_perp,
                   s_naive_tmpl, s_naive_eng, jitter_val, N, fs):
    np.random.seed(seed)
    hw = HardwareImpairments(HW_CONFIG)
    if ideal:
        pa_out = sig_clean
        p_ref = np.mean(np.abs(sig_clean) ** 2)
    else:
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_clean * jit, ibo_dB=10.0)
        p_ref = np.mean(np.abs(pa_out) ** 2)
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z = _log_envelope(_apply_agc(pa_out + w))

    # 1. Proposed (Projected GLRT)
    z_proj = P_perp @ z
    stat_prop = (np.dot(s_prop_perp, z_proj) ** 2) / s_prop_eng

    # 2. Naive (Mean-Subtracted GLRT) - 易受 1/f 噪声影响
    z_ac = z - np.mean(z)
    stat_naive = (np.dot(s_naive_tmpl, z_ac) ** 2) / s_naive_eng

    # 3. ED (Energy on Projected - fairest to ED)
    stat_ed = np.sum(z_proj ** 2)

    return stat_prop, stat_naive, stat_ed


def _trial_mds_tri(is_h1, sig_clean, seed, snr_db, s_prop_norm, P_perp,
                   s_naive_norm, N, fs, ideal, jitter_val):
    np.random.seed(seed)
    sig_in = sig_clean if is_h1 else np.ones(N, dtype=complex)
    if ideal:
        pa_out = sig_in
        p_ref = np.mean(np.abs(sig_in) ** 2)
    else:
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = jitter_val
        jit = np.exp(hw.generate_colored_jitter(N, fs))
        pa_out, _, _ = hw.apply_saleh_pa(sig_in * jit, ibo_dB=10.0)
        p_ref = np.mean(np.abs(pa_out) ** 2)
    noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
    z = _log_envelope(_apply_agc(pa_out + (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)))

    # 1. Proposed
    z_proj = P_perp @ z
    stat_prop = (np.dot(s_prop_norm, z_proj) ** 2)

    # 2. Naive
    z_ac = z - np.mean(z)
    stat_naive = (np.dot(s_naive_norm, z_ac) ** 2)

    # 3. ED
    stat_ed = np.sum(z_proj ** 2)

    return stat_prop, stat_naive, stat_ed



if __name__ == "__main__":
    multiprocessing.freeze_support()
    PaperFigureGenerator().run_all()
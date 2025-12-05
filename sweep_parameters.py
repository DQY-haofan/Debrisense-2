# ----------------------------------------------------------------------------------
# 脚本名称: sweep_fig6_only.py
# 功能: 专门扫描 Fig 6 的最佳 SNR 和 Jitter 配置
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

# 导入核心模块
try:
    from visualization_all import GLOBAL_CONFIG, HW_CONFIG, DETECTOR_CONFIG, _trial_rmse_fixed, _gen_template
    from physics_engine import DiffractionChannel
    from detector import TerahertzDebrisDetector
    from hardware_model import HardwareImpairments
except ImportError:
    # 尝试兼容 patch 后的导入
    from visualization_patch_final import GLOBAL_CONFIG, DETECTOR_CONFIG
    from visualization_all import HW_CONFIG, _trial_rmse_fixed, _gen_template
    from physics_engine import DiffractionChannel
    from detector import TerahertzDebrisDetector
    from hardware_model import HardwareImpairments

# 强制使用 Demo 模式配置
GLOBAL_CONFIG['L_eff'] = 20e3
GLOBAL_CONFIG['a'] = 0.10
N_CORES = max(1, multiprocessing.cpu_count() - 2)
FS = GLOBAL_CONFIG['fs']
N_SAMPLES = int(FS * GLOBAL_CONFIG['T_span'])
T_AXIS = np.arange(N_SAMPLES) / FS - (N_SAMPLES / 2) / FS

print(f"--- FIG 6 SWEEP TOOL ---")
print(f"Physics: L={GLOBAL_CONFIG['L_eff'] / 1000}km, a={GLOBAL_CONFIG['a'] * 100}cm")


def run_sweep():
    # 准备基础数据
    phy = DiffractionChannel(GLOBAL_CONFIG)
    d_signal = phy.generate_broadband_chirp(T_AXIS, 32)
    sig_truth = 1.0 - d_signal

    v_scan = np.linspace(15000 - 1500, 15000 + 1500, 31)
    det_cfg = {
        'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
        'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a'], 'N_sub': DETECTOR_CONFIG['N_sub']
    }
    det = TerahertzDebrisDetector(FS, N_SAMPLES, **det_cfg)
    P_perp = det.P_perp

    s_raw_list = Parallel(n_jobs=N_CORES)(delayed(_gen_template)(v, FS, N_SAMPLES, det_cfg) for v in v_scan)
    T_bank = np.array([P_perp @ s for s in s_raw_list])
    E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

    # === 扫描配置 ===
    # 我们测试 3 种 SNR 环境
    snr_candidates = [70.0, 80.0, 90.0, 100.0]

    # 我们测试 1 组"巨大差异"的 Jitter 组合
    # Low=1e-5 (Proposed), Mid=1e-3, High=1e-2 (Standard Failure)
    jit_levels = [1.0e-5, 1.0e-3, 1.0e-2]

    # 只测一个典型的 IBO 点 (比如 15dB，处于线性区，热噪声主导)
    # 和一个非线性点 (0dB)
    ibo_test_points = [15.0, 0.0]
    trials = 40  # 速测

    print(f"\n{'SNR':<6} | {'IBO':<5} | {'RMSE(Low)':<10} | {'RMSE(High)':<10} | {'Ratio(H/L)':<10} | {'Verdict'}")
    print("-" * 70)

    for snr in snr_candidates:
        for ibo in ibo_test_points:
            # 计算有效 SNR
            snr_eff = snr - ibo

            # 跑 Low Jitter
            res_low = Parallel(n_jobs=N_CORES)(
                delayed(_trial_rmse_fixed)(ibo, jit_levels[0], s, snr_eff, sig_truth, N_SAMPLES, FS, 15000, HW_CONFIG,
                                           T_bank, E_bank, v_scan, False, P_perp)
                for s in range(trials)
            )
            rmse_low = np.sqrt(np.mean(np.array(res_low) ** 2))

            # 跑 High Jitter
            res_high = Parallel(n_jobs=N_CORES)(
                delayed(_trial_rmse_fixed)(ibo, jit_levels[-1], s, snr_eff, sig_truth, N_SAMPLES, FS, 15000, HW_CONFIG,
                                           T_bank, E_bank, v_scan, False, P_perp)
                for s in range(trials)
            )
            rmse_high = np.sqrt(np.mean(np.array(res_high) ** 2))

            # 评价分离度
            ratio = rmse_high / (rmse_low + 1e-6)
            verdict = "BAD"
            if ratio > 1.5: verdict = "OK"
            if ratio > 5.0: verdict = "GOOD"
            if ratio > 10.0: verdict = "EXCELLENT"

            print(f"{snr:<6.1f} | {ibo:<5.1f} | {rmse_low:<10.2f} | {rmse_high:<10.2f} | {ratio:<10.2f} | {verdict}")


if __name__ == "__main__":
    run_sweep()
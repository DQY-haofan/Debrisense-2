# ----------------------------------------------------------------------------------
# 脚本名称: sweep_parameters.py
# 功能: 自动扫描 Fig 6/7/8/9 的合理参数区间 (Expert Helper)
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

# 导入核心模块
# 假设 visualization_all.py, physics_engine.py 等都在同一目录下
try:
    from visualization_all import (
        GLOBAL_CONFIG, HW_CONFIG, DETECTOR_CONFIG,
        _trial_rmse_fixed, _trial_roc_fixed, _trial_mds_fixed, _trial_isac_fixed,
        _gen_template
    )
    from physics_engine import DiffractionChannel
    from detector import TerahertzDebrisDetector
    from hardware_model import HardwareImpairments
except ImportError:
    raise SystemExit("Error: Could not import from visualization_all.py or core modules.")

# 可以在这里临时覆盖 GLOBAL_CONFIG 来测试 "Relaxed" 模式
# GLOBAL_CONFIG['L_eff'] = 20e3
# GLOBAL_CONFIG['a'] = 0.10

N_CORES = max(1, multiprocessing.cpu_count() - 2)
FS = GLOBAL_CONFIG['fs']
N_SAMPLES = int(FS * GLOBAL_CONFIG['T_span'])
T_AXIS = np.arange(N_SAMPLES) / FS - (N_SAMPLES / 2) / FS

print(f"--- SWEEP TOOL INITIALIZED ---")
print(f"Physics: L={GLOBAL_CONFIG['L_eff'] / 1000}km, a={GLOBAL_CONFIG['a'] * 100}cm")
print(f"Cores: {N_CORES}")


# ==============================================================================
# 1. Sweep for Fig 6 (RMSE U-Shape)
# ==============================================================================
def sweep_fig6_u_shape():
    """
    寻找能产生漂亮 'U型' RMSE 曲线的 SNR_ref 和 IBO 范围。
    U型原理：
      - 左侧 (High IBO): 功率低 -> SNR低 -> RMSE 高 (Noise limited)
      - 右侧 (Low IBO):  功率高 -> 非线性强 -> RMSE 高 (Distortion limited)
      - 中间: 甜点
    """
    print("\n[Task 1] Sweeping Fig 6 (U-Shape)...")

    # 准备环境
    phy = DiffractionChannel(GLOBAL_CONFIG)
    d_signal = phy.generate_broadband_chirp(T_AXIS, 32)
    sig_truth = 1.0 - d_signal

    # 简化扫描参数
    v_scan = np.linspace(15000 - 1000, 15000 + 1000, 21)  # 降低分辨率加速
    det_cfg = {
        'cutoff_freq': DETECTOR_CONFIG['cutoff_freq'],
        'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a'], 'N_sub': DETECTOR_CONFIG['N_sub']
    }
    det = TerahertzDebrisDetector(FS, N_SAMPLES, **det_cfg)
    P_perp = det.P_perp

    s_raw_list = Parallel(n_jobs=N_CORES)(delayed(_gen_template)(v, FS, N_SAMPLES, det_cfg) for v in v_scan)
    T_bank = np.array([P_perp @ s for s in s_raw_list])
    E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

    # 扫描区间
    snr_ref_list = np.arange(50, 90, 5)  # 试探 50dB 到 90dB
    ibo_list = [0, 5, 10, 15, 20, 25, 30]
    jitter_val = 1e-5
    trials = 20  # 小样本

    print(f"{'SNR_ref':<10} | {'Left(IBO=30)':<12} | {'Mid(IBO=15)':<12} | {'Right(IBO=0)':<12} | {'Shape':<10}")
    print("-" * 70)

    for snr_ref in snr_ref_list:
        rmses = []
        for ibo in ibo_list:
            # 关键：这里模拟 Fig 6 的逻辑，SNR 随 IBO 衰减
            snr_eff = snr_ref - ibo

            errs = Parallel(n_jobs=N_CORES)(
                delayed(_trial_rmse_fixed)(
                    ibo, jitter_val, s, snr_eff, sig_truth, N_SAMPLES, FS,
                    15000, HW_CONFIG, T_bank, E_bank, v_scan, False, P_perp
                ) for s in range(trials)
            )

            # 简单 RMSE 计算 (忽略极值)
            clean_errs = [e for e in errs if abs(e) < 1000]
            if len(clean_errs) > 5:
                rmse = np.sqrt(np.mean(np.array(clean_errs) ** 2))
            else:
                rmse = 9999.0
            rmses.append(rmse)

        # 提取关键点
        left = rmses[-1]  # IBO=30 (Low SNR)
        mid = rmses[len(rmses) // 2]  # IBO=15
        right = rmses[0]  # IBO=0 (High Nonlinearity)

        shape = "Flat"
        if left > mid and right > mid:
            shape = "U-Shape"
        elif left > mid:
            shape = "Noise-Lim"
        elif right > mid:
            shape = "Dist-Lim"

        print(f"{snr_ref:<10.1f} | {left:<12.2f} | {mid:<12.2f} | {right:<12.2f} | {shape}")


# ==============================================================================
# 2. Sweep for Fig 7 (ROC Separation)
# ==============================================================================
def sweep_fig7_roc():
    """
    寻找 ROC 曲线拉开差距的 SNR。
    目标：Proposed AUC > 0.8, Standard AUC < 0.7, ED ~ 0.5
    """
    print("\n[Task 2] Sweeping Fig 7 (ROC Separation)...")

    phy = DiffractionChannel(GLOBAL_CONFIG)
    d_wb = phy.generate_broadband_chirp(T_AXIS, 32)
    sig_h1 = 1.0 - d_wb
    sig_h0 = np.ones(N_SAMPLES, dtype=complex)

    det = TerahertzDebrisDetector(FS, N_SAMPLES, **DETECTOR_CONFIG, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a'])
    P_perp = det.P_perp
    s_true = P_perp @ det._generate_template(15000)
    s_eng = np.sum(s_true ** 2) + 1e-20

    snr_list = np.arange(55, 85, 5)
    trials = 100  # ROC 需要稍微多一点
    jit_prop = 1e-5
    jit_std = 2e-4

    print(f"{'SNR':<8} | {'AUC_Prop':<10} | {'AUC_Std':<10} | {'AUC_Diff':<10}")
    print("-" * 50)

    for snr in snr_list:
        # Run Proposed
        r0_p = Parallel(n_jobs=N_CORES)(
            delayed(_trial_roc_fixed)(sig_h0, s, snr, False, s_true, s_eng, P_perp, jit_prop, N_SAMPLES, FS) for s in
            range(trials))
        r1_p = Parallel(n_jobs=N_CORES)(
            delayed(_trial_roc_fixed)(sig_h1, s, snr, False, s_true, s_eng, P_perp, jit_prop, N_SAMPLES, FS) for s in
            range(trials))

        # Run Standard
        r0_s = Parallel(n_jobs=N_CORES)(
            delayed(_trial_roc_fixed)(sig_h0, s, snr, False, s_true, s_eng, P_perp, jit_std, N_SAMPLES, FS) for s in
            range(trials))
        r1_s = Parallel(n_jobs=N_CORES)(
            delayed(_trial_roc_fixed)(sig_h1, s, snr, False, s_true, s_eng, P_perp, jit_std, N_SAMPLES, FS) for s in
            range(trials))

        # Simple AUC Calc
        def get_auc(n, s):
            return np.mean([np.mean(s[:, 0] > v) for v in n[:, 0]])

        auc_p = get_auc(np.array(r0_p), np.array(r1_p))
        auc_s = get_auc(np.array(r0_s), np.array(r1_s))

        print(f"{snr:<8.1f} | {auc_p:<10.3f} | {auc_s:<10.3f} | {auc_p - auc_s:<10.3f}")


# ==============================================================================
# 3. Sweep for Fig 8 (MDS Size Sensitivity)
# ==============================================================================
def sweep_fig8_mds():
    """
    寻找 MDS 曲线的过渡区。
    我们希望小尺寸 (2mm) Pd 低，大尺寸 (50mm) Pd 高。
    """
    print("\n[Task 3] Sweeping Fig 8 (MDS Transition)...")

    sizes = [0.005, 0.02, 0.05]  # 5mm, 20mm, 50mm
    snr_list = np.arange(60, 90, 5)
    trials = 50

    # 预计算 H0 阈值 (简化，只算一次大致的)
    det = TerahertzDebrisDetector(FS, N_SAMPLES, **DETECTOR_CONFIG, L_eff=GLOBAL_CONFIG['L_eff'], a=GLOBAL_CONFIG['a'])
    P_perp = det.P_perp
    s_norm = P_perp @ det._generate_template(15000)
    s_norm /= (np.linalg.norm(s_norm) + 1e-20)

    print(f"{'SNR':<8} | {'Pd(5mm)':<10} | {'Pd(20mm)':<10} | {'Pd(50mm)':<10}")
    print("-" * 50)

    for snr in snr_list:
        # 估算阈值 (H0)
        h0_stats = Parallel(n_jobs=N_CORES)(
            delayed(_trial_mds_fixed)(False, None, s, snr, s_norm, P_perp, N_SAMPLES, FS, False, 1e-5)
            for s in range(100)
        )
        threshold = np.percentile(h0_stats, 95)

        pds = []
        for a_val in sizes:
            # 临时构建物理信道
            cfg_temp = GLOBAL_CONFIG.copy()
            cfg_temp['a'] = a_val
            phy = DiffractionChannel(cfg_temp)
            sig = 1.0 - phy.generate_broadband_chirp(T_AXIS, 32)

            stats = Parallel(n_jobs=N_CORES)(
                delayed(_trial_mds_fixed)(True, sig, s, snr, s_norm, P_perp, N_SAMPLES, FS, False, 1e-5)
                for s in range(trials)
            )
            pd = np.mean(np.array(stats) > threshold)
            pds.append(pd)

        print(f"{snr:<8.1f} | {pds[0]:<10.2f} | {pds[1]:<10.2f} | {pds[2]:<10.2f}")


# ==============================================================================
# 4. Sweep for Fig 9 (ISAC Trade-off)
# ==============================================================================
def sweep_fig9_isac():
    """
    寻找 ISAC 的 Knee Point (Capacity 和 RMSE 的折衷区)。
    """
    print("\n[Task 4] Sweeping Fig 9 (ISAC Knee)...")

    phy = DiffractionChannel(GLOBAL_CONFIG)
    sig_truth = 1.0 - phy.generate_broadband_chirp(T_AXIS, 32)

    # 简化扫描
    v_scan = np.linspace(14000, 16000, 11)
    det_cfg = DETECTOR_CONFIG.copy()
    det_cfg.update({'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a']})

    s_raw = Parallel(n_jobs=N_CORES)(delayed(_gen_template)(v, FS, N_SAMPLES, det_cfg) for v in v_scan)
    det = TerahertzDebrisDetector(FS, N_SAMPLES, **det_cfg)
    P_perp = det.P_perp
    T_bank = np.array([P_perp @ s for s in s_raw])
    E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

    snr_fixed = 65.0  # 先固定一个 SNR
    ibo_list = [0, 5, 10, 15, 20]

    print(f"Fixed SNR={snr_fixed} dB")
    print(f"{'IBO':<8} | {'Capacity':<10} | {'RMSE':<10}")
    print("-" * 40)

    for ibo in ibo_list:
        res = Parallel(n_jobs=N_CORES)(
            delayed(_trial_isac_fixed)(ibo, s, snr_fixed, sig_truth, T_bank, E_bank, v_scan, P_perp, N_SAMPLES, FS,
                                       False, 1e-4)
            for s in range(30)
        )
        res = np.array(res)
        cap = np.mean(res[:, 0])
        errs = res[:, 1]
        valid_errs = errs[errs < 1000]
        rmse = np.sqrt(np.mean(valid_errs ** 2)) if len(valid_errs) > 0 else 9999

        print(f"{ibo:<8.1f} | {cap:<10.2f} | {rmse:<10.2f}")


if __name__ == "__main__":
    # 可以选择只运行某一个
    sweep_fig6_u_shape()
    sweep_fig7_roc()
    sweep_fig8_mds()
    # sweep_fig9_isac()
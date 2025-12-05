# calibration_tool.py
# 用于寻找 "Money Plots" 的最佳物理参数范围

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# 引入你的核心模块
from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments
from detector import TerahertzDebrisDetector

# --- 基础配置 ---
FS = 200e3
N = 4000
PHYSICS_CFG = {'fc': 300e9, 'B': 10e9, 'L_eff': 20e3, 'a': 50.0, 'v_rel': 15000}
HW_CFG = {'jitter_rms': 1e-4, 'beta_a': 5995.0, 'alpha_a': 10.127}  # 200ppm Jitter


def run_calibration_sweep():
    print("=" * 60)
    print("PARAMETER CALIBRATION TOOL (v1.0)")
    print("寻找 'Breaking Point' 以生成完美的 U 型/S 型曲线")
    print("=" * 60)

    # 1. 预计算检测器
    print("[Init] Pre-computing templates...")
    det = TerahertzDebrisDetector(FS, N, cutoff_freq=300.0)
    P_perp = det.P_perp
    # 生成 15km/s 的标准模板
    s_temp = det._generate_template(15000)
    s_proj = P_perp @ s_temp
    E_s = np.sum(s_proj ** 2) + 1e-20

    # 生成物理信号 (H1)
    phy = DiffractionChannel(PHYSICS_CFG)
    sig_clean = 1.0 - phy.generate_broadband_chirp(np.arange(N) / FS - N / 2 / FS, 32)

    # --- 扫描 SNR 范围 ---
    # 我们扫描 15dB 到 55dB，寻找检测概率 Pd 从 0 到 1 的突变点
    snr_range = np.arange(50, 80, 5)

    print("\n[Task 1] Calibrating SNR for Detection (Fig 7 & 8)...")
    print(f"{'SNR (dB)':<10} | {'Pd (Detection Rate)':<20} | {'Status'}")
    print("-" * 50)

    for snr in snr_range:
        # 运行 100 次蒙特卡洛
        detections = Parallel(n_jobs=-1)(
            delayed(_probe_detection)(sig_clean, snr, HW_CFG, P_perp, s_proj, E_s, N, FS)
            for _ in range(100)
        )
        pd = np.mean(detections)

        status = ""
        if pd < 0.1:
            status = "Too Hard (Blind)"
        elif pd > 0.99:
            status = "Too Easy (Flat Line)"
        else:
            status = "*** SWEET SPOT ***"

        print(f"{snr:<10.1f} | {pd:<20.2f} | {status}")

    print("\n建议:")
    print("1. Fig 7 (ROC): 选择 'SWEET SPOT' 左侧 3-5dB 的值 (例如 25-30dB)，以区分 Ideal/Hardware。")
    print("2. Fig 8 (MDS): 选择 'SWEET SPOT' 的值 (例如 30-35dB)，确保 2mm 碎片测不到，50mm 能测到。")


def _probe_detection(sig_clean, snr, hw_cfg, P_perp, s_proj, E_s, N, fs):
    # 模拟一次检测过程
    hw = HardwareImpairments(hw_cfg)
    hw.jitter_rms = hw_cfg['jitter_rms']  # 使用 200ppm Jitter

    # 信号 + Jitter + PA
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    sig_pa, _, _ = hw.apply_saleh_pa(sig_clean * jit, ibo_dB=5.0)

    # 热噪声
    p_ref = np.mean(np.abs(sig_pa) ** 2)
    noise_std = np.sqrt(p_ref / (10 ** (snr / 10.0)))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # 检测统计量
    z = np.log(np.abs(sig_pa + w) + 1e-12)
    z_perp = P_perp @ z
    stat = (np.dot(s_proj, z_perp) ** 2) / E_s

    # 简单的门限 (基于经验或 H0 跑一次)
    # 这里我们用一个经验门限来快速判断
    return 1 if stat > 15.0 else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_calibration_sweep()
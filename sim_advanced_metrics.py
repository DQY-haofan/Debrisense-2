# ----------------------------------------------------------------------------------
# 脚本名称: sim_advanced_metrics.py (优化版)
# 描述: 生成 Fig 8 (MDS) 和 Fig 9 (ISAC Trade-off)
# 优化: 物理信号预计算 + 模板库并行生成
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

# 导入模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Missing modules. Ensure physics_engine.py, hardware_model.py, detector.py are present.")

# 绘图配置
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42})

# --- 全局配置 ---
L_eff = 50e3
fs = 200e3
T_span = 0.02
N = int(fs * T_span)
t_axis = np.arange(N) / fs - (N / 2) / fs
true_v = 15000.0

# 硬件参数 (SOTA)
config_hw = {
    'jitter_rms': 0.5e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0,
    'pll_bw': 50e3
}

# 检测器参数 (N_sub=32 对齐物理引擎)
config_det_base = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'N_sub': 32}

# 并行核心数
num_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cores - 2)


def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)


def save_csv(data, filename):
    folder = 'results/csv_data'
    ensure_dir(folder)
    pd.DataFrame(data).to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


# =========================================================================
# Task 1: Fig 8 - 探测能力边界 (MDS)
# 优化策略: 物理信号在循环外生成一次，循环内只加噪声
# =========================================================================

def run_noise_injection_trial(is_h1, sig_clean, seed, noise_std):
    """
    仅负责注入噪声和检测，不进行物理计算
    """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    # 检测器实例化开销极小
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)  # a这里不影响log变换和投影

    # 1. 选择信号
    if is_h1:
        sig_base = sig_clean
    else:
        sig_base = np.ones(N, dtype=np.complex128)

    # 2. 注入硬件损伤 (快速 FFT)
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    # 3. PA & 热噪声
    pa_in = sig_base * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 4. 检测统计量
    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    # 获取对应尺寸的模板 (注意: 这里需要稍微小心，如果 sig_clean 对应的 a 变了，
    # 理论上检测器应该用对应的模板。为了简化，我们假设检测器完全匹配目标尺寸)
    # 在主循环中我们会传入匹配的模板向量 s_template_perp
    return z_perp


def run_fig8_mds():
    print("\n=== Running Task 1: Fig 8 (MDS Scan) [Optimized] ===")

    noise_std = 1.0e-5
    diameters_mm = np.logspace(0.3, 1.7, 10)  # 减少点数以加速预览 (10个点足够画曲线)
    radii_m = diameters_mm / 2000.0
    pd_curve = []

    # 预实例化检测器以获取投影矩阵 (P_perp 与 a 无关)
    det_base = TerahertzDebrisDetector(fs, N, **config_det_base)
    P_perp = det_base.P_perp

    for a_val in tqdm(radii_m, desc="Size Scan"):
        # [优化点 1]：在循环外生成纯净物理信号 (耗时操作)
        phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': true_v})
        d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
        sig_clean = 1.0 - d_wb

        # [优化点 2]: 生成匹配模板 (用于 GLRT)
        # 检测器需要重新实例化以更新内部参数 a (虽然 _generate_template 会用到)
        det_temp = TerahertzDebrisDetector(fs, N, a=a_val, **config_det_base)
        s_raw = det_temp._generate_template(true_v)
        s_template = P_perp @ s_raw
        energy = np.sum(s_template ** 2) + 1e-20

        # A. H0 统计量 (无目标) - 并行计算
        # 注意: H0 不需要 sig_clean，传 None 或 ones
        seeds_h0 = np.random.randint(0, 1e9, 100)  # 减少次数: 200 -> 100
        results_h0 = Parallel(n_jobs=n_jobs)(
            delayed(run_noise_injection_trial)(False, None, s, noise_std)
            for s in seeds_h0
        )
        # 计算统计量
        stats_h0 = [(np.dot(s_template, z_p) ** 2) / energy for z_p in results_h0]
        threshold = np.percentile(stats_h0, 99.0)

        # B. H1 统计量 (有目标)
        seeds_h1 = np.random.randint(0, 1e9, 50)  # 减少次数: 100 -> 50
        results_h1 = Parallel(n_jobs=n_jobs)(
            delayed(run_noise_injection_trial)(True, sig_clean, s, noise_std)
            for s in seeds_h1
        )
        stats_h1 = [(np.dot(s_template, z_p) ** 2) / energy for z_p in results_h1]

        pd_val = np.mean(np.array(stats_h1) > threshold)
        pd_curve.append(pd_val)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogx(diameters_mm, pd_curve, 'b-o', linewidth=2, label='300 GHz FSR')
    plt.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Debris Diameter (mm)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title('Fig 8: Detection Capability (MDS)')
    plt.grid(True, which='both')
    plt.savefig('results/Fig8_MDS.png')
    save_csv({'d_mm': diameters_mm, 'pd': pd_curve}, 'Fig8_MDS')


# =========================================================================
# Task 2: Fig 9 - ISAC Trade-off
# 优化策略: 并行生成模板库 + 矩阵化 GLRT
# =========================================================================

def generate_single_template(v, det_config):
    """ 辅助函数: 单个模板生成 (用于并行化) """
    det = TerahertzDebrisDetector(fs, N, a=0.05, **det_config)
    s_raw = det._generate_template(v)
    return s_raw


def precompute_template_bank_parallel(v_scan):
    print(f"   [Optim] Parallel generating {len(v_scan)} templates...")

    # [优化点 3]: 并行生成原始模板
    # 注意: 这里只生成 s_raw，投影在主线程做 (矩阵乘法很快)
    s_raw_list = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_template)(v, config_det_base)
        for v in v_scan
    )

    # 转换为矩阵
    S_raw = np.array(s_raw_list)  # (N_vel, N)

    # 获取投影矩阵
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)
    P_perp = det.P_perp

    # 批量投影: S_perp = S_raw @ P_perp.T (因为 P是对称的)
    # 但由于 N~4000, 矩阵乘法 (N_vel, N) @ (N, N) 较慢
    # 更快的方法是: S_perp = S_raw - (S_raw @ H) @ H.T
    # 这里直接用检测器自带的逻辑循环投影也行，或者利用 apply_projection 的矩阵化

    print("   [Optim] Projecting templates...")
    S_perp = np.zeros_like(S_raw)
    energies = np.zeros(len(v_scan))

    for i in range(len(v_scan)):
        s_p = P_perp @ S_raw[i]
        S_perp[i] = s_p
        energies[i] = np.sum(s_p ** 2) + 1e-20

    return S_perp, energies, P_perp


def run_isac_trial_fast(ibo, seed, noise_std, sig_truth, template_bank, template_energies, v_scan, P_perp):
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)  # 仅用于log变换

    # 1. 损伤
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    # 2. PA
    pa_in = sig_truth * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # Comm Capacity (Upper Bound)
    p_rx = np.mean(np.abs(pa_out) ** 2)
    gamma_eff = 0.01  # SOTA hardware factor
    sinr_lin = p_rx / (noise_std ** 2 + p_rx * gamma_eff)
    capacity = np.log2(1 + sinr_lin)

    # Sensing RMSE
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = P_perp @ z_log

    # Matrix GLRT
    correlations = np.dot(template_bank, z_perp)
    stats = (correlations ** 2) / template_energies
    est_v = v_scan[np.argmax(stats)]

    return capacity, abs(est_v - true_v)


def run_fig9_isac_optimized():
    print("\n=== Running Task 2: Fig 9 (ISAC Trade-off) [Optimized] ===")

    # 1. 物理真值 (只算一次)
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v})
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_truth = 1.0 - d_wb

    # 2. 模板库
    v_scan = np.linspace(true_v - 2000, true_v + 2000, 201)  # 减少密度: 401 -> 201
    template_bank, energies, P_perp = precompute_template_bank_parallel(v_scan)

    # 3. 扫描 IBO
    noise_std = 2.0e-5
    ibo_points = np.linspace(20, -5, 10)  # 减少点数: 15 -> 10
    trials = 30  # 减少次数: 50 -> 30

    avg_cap = []
    avg_rmse = []

    for ibo in tqdm(ibo_points, desc="IBO Scan"):
        res = Parallel(n_jobs=n_jobs)(
            delayed(run_isac_trial_fast)(
                ibo, s, noise_std, sig_truth, template_bank, energies, v_scan, P_perp
            ) for s in range(trials)
        )
        res = np.array(res)
        avg_cap.append(np.mean(res[:, 0]))

        errs = res[:, 1]
        valid = errs[errs < 1800]  # 去除野值
        rmse = np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 2000.0
        avg_rmse.append(rmse)

    # 绘图
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(avg_cap, avg_rmse, c=ibo_points, cmap='viridis_r', s=100, edgecolors='k')
    plt.plot(avg_cap, avg_rmse, 'k--', alpha=0.5)
    plt.colorbar(sc, label='IBO (dB)')
    plt.xlabel('Capacity (bits/s/Hz)')
    plt.ylabel('RMSE (m/s)')
    plt.title('Fig 9: ISAC Trade-off')
    plt.grid(True)
    plt.savefig('results/Fig9_ISAC.png')
    save_csv({'ibo': ibo_points, 'cap': avg_cap, 'rmse': avg_rmse}, 'Fig9_ISAC')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    ensure_dir('results')

    run_fig8_mds()
    run_fig9_isac_optimized()

    print("\n[Done] All simulations completed successfully.")
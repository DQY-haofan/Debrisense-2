# ----------------------------------------------------------------------------------
# 脚本名称: main_sim_montecarlo.py (核心蒙特卡洛仿真 - 最终修复版 v4.2)
#
# 描述:
#   本脚本是生成 IEEE TWC 论文核心图表 "Fig 6: RMSE vs SNR (The Error Floor)" 的主程序。
#   修复了 toeplitz 函数的引用错误，并**增加了多进程死锁的鲁棒性**。
#
# 核心功能与升级点:
#   1. [鲁棒性增强]: 尝试设置 multiprocessing start method 为 'spawn'，防止 Colab 环境死锁。
#   2. [物理真值]: 使用 10GHz 宽带 DFS 信号作为 Ground Truth。
#   3. [多级扫描]: 扫描不同 Jitter 强度，绘制 Error Floor 趋势。
#   4. [CPU并行]: 使用 joblib 进行稳定高效的 CPU 多核并行加速 (增加 verbose 输出)。
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla  # <--- 修复: 显式导入 scipy.linalg
from tqdm import tqdm
import os
import csv
import multiprocessing
from joblib import Parallel, delayed
import time

# 导入自定义模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    print(
        "Error: Dependent modules not found. Please ensure physics_engine.py, hardware_model.py, detector.py are in the same folder.")
    raise SystemExit

# 设置 Matplotlib 绘图标准
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'lines.linewidth': 2.0,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def calc_capacity(snr_linear):
    """ 计算香农容量 C = log2(1 + SNR) """
    return np.log2(1 + snr_linear)


def calc_bcrlb(snr_linear, jitter_cov_inv, h_sensitivity, scr=1.0):
    """
    计算包含 SCR (自愈效应) 修正的贝叶斯克拉美-罗下界 (BCRLB)
    BCRLB = sqrt( 1 / (J_jitter + J_thermal) )
    """
    # jitter_cov_inv 和 h_sensitivity 已经是 NumPy 数组
    h_eff = h_sensitivity * scr

    # 1. 抖动信息量 (NumPy 线性代数)
    j_jitter = np.dot(h_eff.T, np.dot(jitter_cov_inv, h_eff))

    # 2. 热噪声信息量
    j_thermal = snr_linear * np.sum(h_sensitivity ** 2) * (scr ** 2)

    return np.sqrt(1.0 / (j_jitter + J_thermal))


# --- 单次蒙特卡洛试验函数 ---
def run_trial(ibo, jitter_rms, seed, noise_std, sig_broadband_truth, N, fs, true_v, hw_base_config, det_config):
    """
    执行单次仿真闭环：
    宽带真值 -> 有色抖动 -> PA非线性 -> 热噪声 -> 对数检测器 -> GLRT估计
    """
    np.random.seed(seed)

    hw_config = hw_base_config.copy()
    hw_config['jitter_rms'] = jitter_rms

    hw_local = HardwareImpairments(hw_config)
    det_local = TerahertzDebrisDetector(fs, N, **det_config)

    # 1. 生成有色 Jitter
    jitter = hw_local.generate_colored_jitter(N, fs)
    a_jitter = np.exp(jitter)

    # 2. PA 非线性与自愈效应
    pa_in = sig_broadband_truth * a_jitter
    pa_out, _, _ = hw_local.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 3. 添加热噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 4. 检测器处理
    z_log = det_local.log_envelope_transform(y_rx)
    z_perp = det_local.apply_projection(z_log)

    # 5. GLRT 速度搜索
    v_scan = np.linspace(true_v - 1500, true_v + 1500, 61)
    stats = det_local.glrt_scan(z_perp, v_scan)

    est_v = v_scan[np.argmax(stats)]

    return est_v - true_v


def main():
    # --- 修复: 尝试设置 multiprocessing start method ---
    if multiprocessing.current_process().name == 'MainProcess' and multiprocessing.get_start_method(
            allow_none=True) is None:
        try:
            # 'spawn' is more stable in Colab/Jupyter environment
            multiprocessing.set_start_method('spawn', force=True)
            print("[INFO] Multiprocessing start method set to 'spawn' for stability.")
        except RuntimeError:
            pass

    print("=== IEEE TWC Simulation: Fig 6 (RMSE vs SNR with Jitter Tiers) ===")
    print("Initializing High-Fidelity Physics Simulation...")
    ensure_dir('results')

    # --- CPU 核心数检测 ---
    num_cores = multiprocessing.cpu_count()
    n_jobs = max(1, num_cores - 2)
    print(f"[INFO] Detected {num_cores} cores. Using {n_jobs} cores for CPU parallel processing.")

    # --- 1. 系统参数配置 ---
    L_eff = 50e3
    fs = 20e3
    T_span = 0.05
    N = int(fs * T_span)
    t_axis = np.linspace(-T_span / 2, T_span / 2, N)
    true_v = 15000.0

    # 模块配置
    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v}
    config_hw_base = {'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
    config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'a': 0.05}

    # --- 2. 预计算：生成宽带物理真值 ---
    print("Pre-computing Broadband Physics Signal (10GHz DFS)...")
    phy = DiffractionChannel(config_phy)
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=64)
    sig_broadband_truth = 1.0 + d_wb

    # 预计算 BCRLB 所需的灵敏度向量 h (NumPy)
    det_temp = TerahertzDebrisDetector(fs, N, **config_det)
    s_v = det_temp._generate_template(true_v)
    s_v_plus = det_temp._generate_template(true_v + 1.0)
    h_vec = (s_v_plus - s_v) / 1.0  # 梯度

    # 噪声底校准
    hw_temp = HardwareImpairments(config_hw_base)
    pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_broadband_truth, 15.0)
    p_ref = np.mean(np.abs(pa_ref) ** 2)
    noise_std = np.sqrt(p_ref * 1e-5)
    print(f"Calibrated Noise Floor (Std): {noise_std:.2e}")

    # --- 3. 仿真扫描配置 ---
    ibo_scan = np.linspace(20, -5, 15)
    jitter_levels = [1e-6, 3e-6, 10e-6]
    trials_per_point = 500

    results = {}

    plt.figure(figsize=(10, 7))
    colors = ['g', 'b', 'r']

    # --- 4. 开始循环扫描 ---
    for idx, j_rms in enumerate(jitter_levels):
        print(f"\n--- Simulating Jitter Level: {j_rms * 1e6:.1f} urad ---")

        # 预计算该 Jitter 等级下的协方差矩阵 (NumPy)
        hw_config_j = config_hw_base.copy()
        hw_config_j['jitter_rms'] = j_rms
        hw_j = HardwareImpairments(hw_config_j)

        long_jit = hw_j.generate_colored_jitter(N * 200, fs)
        r_xx = np.correlate(long_jit, long_jit, mode='full')[len(long_jit) - 1: len(long_jit) - 1 + N] / len(long_jit)

        # 修正后的协方差逆矩阵计算 (使用 sla.toeplitz 和 np.linalg.inv)
        C_inv = np.linalg.inv(sla.toeplitz(r_xx) + np.eye(N) * 1e-12)

        rmse_sim_list = []
        bcrlb_theo_list = []
        cap_curve = []

        # IBO 扫描
        for ibo in tqdm(ibo_scan, desc=f"Scanning Power (IBO)"):
            # --- A. 理论值计算 ---
            _, scr_tr, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            scr = np.mean(scr_tr)

            pa_out, _, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            p_sig = np.var(pa_out)
            snr_lin = p_sig / (noise_std ** 2)

            p_tot = np.mean(np.abs(pa_out) ** 2)
            comm_snr = p_tot / (noise_std ** 2)
            cap = calc_capacity(comm_snr)
            cap_curve.append(cap)

            bcrlb = calc_bcrlb(snr_lin, C_inv, h_vec, scr=scr)
            bcrlb_theo_list.append(bcrlb)

            # --- B. 并行蒙特卡洛仿真 ---
            seeds = np.random.randint(0, 1e9, trials_per_point)

            # 增加 verbose=10 输出，帮助诊断 joblib 状态
            errs = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_trial)(
                    ibo, j_rms, s, noise_std, sig_broadband_truth, N, fs, true_v, config_hw_base, config_det
                ) for s in seeds
            )

            # 鲁棒统计 RMSE
            valid_errs = [e for e in errs if abs(e) < 1200]
            if len(valid_errs) > 10:
                rmse = np.sqrt(np.mean(np.array(valid_errs) ** 2))
            else:
                rmse = 1000.0

            rmse_sim_list.append(rmse)

        # 绘制该 Jitter 等级的曲线
        label = f"{j_rms * 1e6:.0f} $\mu$rad"
        plt.semilogy(ibo_scan, rmse_sim_list, f'{colors[idx]}o-', label=f'Sim: Jitter={label}', markersize=5)
        plt.semilogy(ibo_scan, bcrlb_theo_list, f'{colors[idx]}--', linewidth=1.5, alpha=0.6)

        # 存储结果 (为 CSV 准备)
        results[f"rmse_sim_{label}"] = rmse_sim_list
        results[f"bcrlb_{label}"] = bcrlb_theo_list
        results[f"capacity"] = cap_curve

    # --- 5. 绘图修饰 (双轴图) ---
    ax1 = plt.gca()
    ax1.set_xlabel('Input Back-Off (dB) [High Power $\leftarrow$]')
    ax1.set_ylabel('Ranging RMSE (m/s)', color='k')
    ax1.invert_xaxis()
    ax1.grid(True, which='both', linestyle=':')

    # 右轴：通信容量 (Capacity)
    ax2 = ax1.twinx()
    line_cap, = ax2.plot(ibo_scan, cap_curve, 'k-s', alpha=0.3, linewidth=8, label='Comm Capacity')
    ax2.set_ylabel('Spectral Efficiency (bits/s/Hz)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # 图例合并
    lines_rmse = [line for line in ax1.lines]
    lines_cap = [line_cap]

    legend_elements = lines_rmse + lines_cap
    legend_labels = [l.get_label() for l in legend_elements]
    ax1.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True,
               shadow=True, title="Legend")

    plt.title('Fig 6: ISAC Performance Trade-off & Error Floor Verification')
    plt.tight_layout()

    # --- 6. 保存结果 ---
    plot_path = 'results/Fig6_RMSE_vs_SNR_MultiJitter'
    plt.savefig(f'{plot_path}.png', dpi=300)
    plt.savefig(f'{plot_path}.pdf', format='pdf')

    # 保存数值数据 (CSV)
    results['ibo'] = ibo_scan
    csv_path = 'results/Fig6_data.csv'
    keys = sorted(results.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[results[k] for k in keys]))
    print(f"\n[Success] Fig 6 Generated. Data saved to {csv_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
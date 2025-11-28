# ----------------------------------------------------------------------------------
# 脚本名称: main_sim_montecarlo.py (核心蒙特卡洛仿真 - 升级版 v3.0)
#
# 描述:
#   本脚本是生成 IEEE TWC 论文核心图表 "Fig 6: RMSE vs SNR (The Error Floor)" 的主程序。
#   它执行了高保真的闭环仿真，验证在硬件受限 (Hardware-Limited) 机制下的感知极限。
#
# 核心功能与升级点:
#   1. [物理真值]: 强制接入 physics_engine，使用 10GHz 宽带 DFS 信号作为 Ground Truth，
#      包含真实的色散模糊效应 (Dispersion Smearing)，而非窄带近似。
#   2. [多级扫描]: 自动扫描不同的 Jitter 强度 (1urad, 3urad, 10urad)，直观展示
#      Error Floor 随平台稳定性恶化而上移的现象。
#   3. [ISAC 权衡]: 计算通信容量 (Capacity) 并绘制双轴图，展示 PA 饱和对
#      通信(有利)与感知(有害)的矛盾影响。
#   4. [并行加速]: 使用 joblib 实现多核并行计算。
#
# 输出:
#   - results/Fig6_RMSE_vs_SNR_MultiJitter.png (.pdf)
#   - results/Fig6_data.csv (用于 TikZ/Matlab 复绘)
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
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

# 设置 IEEE 论文绘图标准
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
    # 考虑 SCR 导致的灵敏度压缩: H_eff = SCR * H_linear
    h_eff = h_sensitivity * scr

    # 1. 抖动信息量 (Jitter Information) - 这是一个常数基底
    j_jitter = np.dot(h_eff.T, np.dot(jitter_cov_inv, h_eff))

    # 2. 热噪声信息量 (Thermal Information) - 随功率线性增加
    j_thermal = snr_linear * np.sum(h_sensitivity ** 2) * (scr ** 2)

    return np.sqrt(1.0 / (j_jitter + j_thermal))


# --- 单次蒙特卡洛试验函数 (Broadband Physics Aware) ---
def run_trial(ibo, jitter_rms, seed, noise_std, sig_broadband_truth, N, fs, true_v, hw_base_config, det_config):
    """
    执行单次仿真闭环：
    宽带真值 -> 有色抖动 -> PA非线性 -> 热噪声 -> 对数检测器 -> GLRT估计
    """
    np.random.seed(seed)

    # 动态更新 Jitter 配置
    hw_config = hw_base_config.copy()
    hw_config['jitter_rms'] = jitter_rms

    # 实例化局部对象
    hw_local = HardwareImpairments(hw_config)
    det_local = TerahertzDebrisDetector(fs, N, **det_config)

    # 1. 生成有色 Jitter (Colored Jitter)
    jitter = hw_local.generate_colored_jitter(N, fs)
    a_jitter = np.exp(jitter)

    # 2. PA 非线性与自愈效应
    # 输入 = 宽带真值信号 * 抖动幅度
    # 注意: sig_broadband_truth 已经是 (1 + d(t)) 的形式
    pa_in = sig_broadband_truth * a_jitter
    pa_out, _, _ = hw_local.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 3. 添加热噪声
    # noise_std 是固定的底噪，不随 IBO 变化 (模拟接收机固定噪声系数)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 4. 检测器处理 (Log-Envelope -> Projection)
    z_log = det_local.log_envelope_transform(y_rx)
    z_perp = det_local.apply_projection(z_log)

    # 5. GLRT 速度搜索
    # 搜索范围 +/- 1500 m/s
    v_scan = np.linspace(true_v - 1500, true_v + 1500, 61)
    stats = det_local.glrt_scan(z_perp, v_scan)

    # 峰值估计
    est_v = v_scan[np.argmax(stats)]

    return est_v - true_v


def main():
    print("=== IEEE TWC Simulation: Fig 6 (RMSE vs SNR with Jitter Tiers) ===")
    print("Initializing High-Fidelity Physics Simulation...")
    ensure_dir('results')

    # 并行核心数配置
    num_cores = multiprocessing.cpu_count()
    n_jobs = max(1, num_cores - 2)
    print(f"Using {n_jobs} cores for parallel processing.")

    # --- 1. 系统参数配置 ---
    L_eff = 50e3  # 50km (为了让衍射条纹在 50ms 内更清晰)
    fs = 20e3  # 20kHz 采样率
    T_span = 0.05  # 50ms 观测窗口
    N = int(fs * T_span)
    t_axis = np.linspace(-T_span / 2, T_span / 2, N)
    true_v = 15000.0  # 目标速度

    # 模块配置
    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v}
    config_hw_base = {'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
    config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'a': 0.05}

    # --- 2. 预计算：生成宽带物理真值 (Broadband Truth) ---
    print("Pre-computing Broadband Physics Signal (10GHz DFS)...")
    phy = DiffractionChannel(config_phy)
    # 关键点：使用 physics_engine 生成宽带信号
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=64)
    sig_broadband_truth = 1.0 + d_wb

    # 预计算 BCRLB 所需的灵敏度向量 h (基于窄带近似是合理的，因为理论界通常基于理想模型)
    det_temp = TerahertzDebrisDetector(fs, N, **config_det)
    s_v = det_temp._generate_template(true_v)
    s_v_plus = det_temp._generate_template(true_v + 1.0)
    h_vec = (s_v_plus - s_v) / 1.0  # 差分近似导数

    # 噪声底校准：设定为线性区信号功率的 -50dB
    hw_temp = HardwareImpairments(config_hw_base)
    pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_broadband_truth, 15.0)  # 15dB IBO (Linear)
    p_ref = np.mean(np.abs(pa_ref) ** 2)
    noise_std = np.sqrt(p_ref * 1e-5)
    print(f"Calibrated Noise Floor (Std): {noise_std:.2e}")

    # --- 3. 仿真扫描配置 ---
    # IBO 扫描范围：20dB (线性/低功率) -> -5dB (饱和/高功率)
    ibo_scan = np.linspace(20, -5, 15)

    # [升级] 多级 Jitter 扫描
    jitter_levels = [1e-6, 3e-6, 10e-6]  # 1urad, 3urad, 10urad
    trials_per_point = 500  # 每个点的蒙特卡洛次数

    results = {}  # 用于存储数据以便保存 CSV

    # 初始化绘图
    plt.figure(figsize=(10, 7))
    colors = ['g', 'b', 'r']  # 对应三个 Jitter 等级

    # --- 4. 开始循环扫描 ---
    for idx, j_rms in enumerate(jitter_levels):
        print(f"\n--- Simulating Jitter Level: {j_rms * 1e6:.1f} urad ---")

        # 预计算该 Jitter 等级下的协方差矩阵 (用于 BCRLB)
        hw_config_j = config_hw_base.copy()
        hw_config_j['jitter_rms'] = j_rms
        hw_j = HardwareImpairments(hw_config_j)

        # 生成长序列以估计自相关
        long_jit = hw_j.generate_colored_jitter(N * 200, fs)
        r_xx = np.correlate(long_jit, long_jit, mode='full')[len(long_jit) - 1: len(long_jit) - 1 + N] / len(long_jit)
        # 协方差求逆 (加微小正则项防止奇异)
        C_inv = la.inv(la.toeplitz(r_xx) + np.eye(N) * 1e-12)

        rmse_sim_list = []
        bcrlb_theo_list = []
        cap_curve = []

        # IBO 扫描
        for ibo in tqdm(ibo_scan, desc=f"Scanning Power (IBO)"):
            # --- A. 理论值计算 ---
            # 计算 SCR (灵敏度压缩比)
            _, scr_tr, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            scr = np.mean(scr_tr)

            # 计算输出信噪比
            pa_out, _, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            p_sig = np.var(pa_out)  # AC 功率 (用于感知)
            snr_lin = p_sig / (noise_std ** 2)

            # 计算通信容量 (使用总功率)
            p_tot = np.mean(np.abs(pa_out) ** 2)
            comm_snr = p_tot / (noise_std ** 2)
            cap = calc_capacity(comm_snr)
            cap_curve.append(cap)

            # 计算 BCRLB (含 SCR 修正)
            bcrlb = calc_bcrlb(snr_lin, C_inv, h_vec, scr=scr)
            bcrlb_theo_list.append(bcrlb)

            # --- B. 并行蒙特卡洛仿真 ---
            seeds = np.random.randint(0, 1e9, trials_per_point)

            # 并行执行
            errs = Parallel(n_jobs=n_jobs)(
                delayed(run_trial)(
                    ibo, j_rms, s, noise_std, sig_broadband_truth, N, fs, true_v, config_hw_base, config_det
                ) for s in seeds
            )

            # 鲁棒统计 RMSE (剔除异常值)
            valid_errs = [e for e in errs if abs(e) < 1200]
            if len(valid_errs) > 10:
                rmse = np.sqrt(np.mean(np.array(valid_errs) ** 2))
            else:
                rmse = 1000.0  # 视作检测失败

            rmse_sim_list.append(rmse)

        # 存储结果
        label = f"{j_rms * 1e6:.0f} $\mu$rad"
        results[f"rmse_sim_{label}"] = rmse_sim_list
        results[f"bcrlb_{label}"] = bcrlb_theo_list

        # 绘制该 Jitter 等级的曲线
        # 实线圆点: 仿真值
        plt.semilogy(ibo_scan, rmse_sim_list, f'{colors[idx]}o-', label=f'Sim: Jitter={label}', markersize=5)
        # 虚线: 理论 BCRLB
        plt.semilogy(ibo_scan, bcrlb_theo_list, f'{colors[idx]}--', linewidth=1.5, alpha=0.6)

    # --- 5. 绘图修饰 (双轴图) ---
    ax1 = plt.gca()
    ax1.set_xlabel('Input Back-Off (dB) [High Power $\leftarrow$]')
    ax1.set_ylabel('Ranging RMSE (m/s)', color='k')
    ax1.invert_xaxis()  # 左侧为高功率(低IBO)，符合直觉
    ax1.grid(True, which='both', linestyle=':')

    # 添加区域标注
    ylim = ax1.get_ylim()
    ax1.text(18, ylim[0] * 1.5, "Linear Region\n(Thermal Limited)", color='green', fontsize=10, ha='center')
    ax1.text(0, ylim[0] * 1.5, "Saturation Region\n(Hardware Limited)", color='red', fontsize=10, ha='center')

    # 左轴图例 (RMSE)
    legend1 = ax1.legend(loc='lower left', title="Sensing Performance")

    # 右轴：通信容量 (Capacity)
    # 使用最后一次循环的 Capacity 数据 (Capacity 对 Jitter 不敏感，主要受 Power 影响)
    ax2 = ax1.twinx()
    line_cap, = ax2.plot(ibo_scan, cap_curve, 'k-s', alpha=0.3, linewidth=8, label='Comm Capacity')
    ax2.set_ylabel('Spectral Efficiency (bits/s/Hz)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # 右轴图例
    ax2.legend([line_cap], ['Capacity'], loc='upper right')

    plt.title('Fig 6: ISAC Performance Trade-off & Error Floor Verification')
    plt.tight_layout()

    # 保存图像
    plot_path = 'results/Fig6_RMSE_vs_SNR_MultiJitter'
    plt.savefig(f'{plot_path}.png', dpi=300)
    plt.savefig(f'{plot_path}.pdf', format='pdf')
    print(f"\n[Plot Saved] {plot_path}.png/.pdf")

    # --- 6. 保存数值数据 (CSV) ---
    results['ibo'] = ibo_scan
    results['capacity'] = cap_curve

    csv_path = 'results/Fig6_data.csv'
    keys = sorted(results.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*[results[k] for k in keys]))
    print(f"[Data Saved] {csv_path}")


if __name__ == "__main__":
    # Windows 下必须保护
    multiprocessing.freeze_support()
    main()
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Missing modules")

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42})


def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)


def save_csv(data_dict, filename, folder='results/csv_data'):
    if not os.path.exists(folder): os.makedirs(folder)
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


# --- 公共配置 ---
L_eff = 50e3
fs = 20e3
T_span = 0.05
N = int(fs * T_span)
t_axis = np.linspace(-T_span / 2, T_span / 2, N)
true_v = 15000.0

config_hw = {'jitter_rms': 2.0e-6, 'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
# 检测器使用固定的参考尺寸 a=0.05m 进行归一化
config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'a': 0.05}

num_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cores - 2)


# ==========================================
# Task 1: Fig 8 - 最小可探测尺寸 (MDS)
# ==========================================

def get_h0_stat(seed, noise_std):
    """ H0 (无目标) 仿真: 获取噪声背景下的检测统计量以确定门限 """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, **config_det)

    # 纯硬件噪声: Jitter + Phase Noise
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    sig = np.ones(N, dtype=np.complex128)  # 仅有载波

    pa_in = sig * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)  # 线性区工作
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # 检测流程
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    # 用标准模板匹配
    s_temp = det.P_perp @ det._generate_template(true_v)
    denom = np.sum(s_temp ** 2)
    if denom < 1e-15: return 0.0

    stat = (np.dot(s_temp, z_perp) ** 2) / denom
    return stat


def get_h1_stat(a_val, seed, noise_std):
    """ H1 (有目标) 仿真: 注入不同尺寸的碎片信号 """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, **config_det)

    # 物理引擎: 设置当前扫描的半径 a_val
    phy_conf = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': true_v}
    phy = DiffractionChannel(phy_conf)

    # 生成宽带衍射信号
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig = 1.0 + d_wb

    # 硬件损伤
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_in = sig * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    s_temp = det.P_perp @ det._generate_template(true_v)
    denom = np.sum(s_temp ** 2)
    stat = (np.dot(s_temp, z_perp) ** 2) / denom
    return stat


def run_fig8_mds():
    print("\n--- Running Fig 8: Minimum Detectable Size (MDS) ---")

    # 1. 确定门限 (Pfa = 1e-3, 实际上 TWC 需要 1e-6, 为速度做近似)
    print("Step 1: Establishing CFAR Threshold (Pfa=1e-3)...")
    noise_std = 1.0e-3  # 高信噪比环境
    h0_trials = 5000
    seeds_h0 = np.random.randint(0, 1e9, h0_trials)

    stats_h0 = Parallel(n_jobs=n_jobs)(
        delayed(get_h0_stat)(s, noise_std) for s in seeds_h0
    )
    threshold = np.percentile(stats_h0, 99.9)
    print(f"   Threshold: {threshold:.2f}")

    # 2. 扫描尺寸 (1mm 到 100mm)
    diameters_mm = np.logspace(0, 2, 15)  # 对数分布
    radii_m = diameters_mm / 2000.0  # 转换为半径 (米)

    pd_curve = []
    trials_h1 = 200

    print("Step 2: Sweeping Debris Size...")
    for a_val in tqdm(radii_m):
        seeds_h1 = np.random.randint(0, 1e9, trials_h1)
        stats_h1 = Parallel(n_jobs=n_jobs)(
            delayed(get_h1_stat)(a_val, s, noise_std) for s in seeds_h1
        )
        # 统计过门限的比例
        pd_val = np.mean(np.array(stats_h1) > threshold)
        pd_curve.append(pd_val)

    # 3. 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogx(diameters_mm, pd_curve, 'b-o', linewidth=2, label='300 GHz FSR System')

    # 标注 MDS 点 (Pd > 0.5)
    pd_array = np.array(pd_curve)
    cross_indices = np.where(pd_array > 0.5)[0]
    if len(cross_indices) > 0:
        mds_d = diameters_mm[cross_indices[0]]
        plt.axvline(mds_d, color='r', linestyle='--', label=f'MDS $\\approx$ {mds_d:.1f} mm')
        plt.axhline(0.5, color='k', linestyle=':', alpha=0.5)

    plt.xlabel('Debris Diameter (mm)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title(f'Fig 8: Detection Capability (Pfa=1e-3)')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    plt.savefig('results/Fig8_Detection_Capability.png', dpi=300)
    plt.savefig('results/Fig8_Detection_Capability.pdf', format='pdf')

    # 保存数据
    save_csv({'diameter_mm': diameters_mm, 'pd': pd_curve}, 'Fig8_MDS_Data')


# ==========================================
# Task 2: Fig 9 - 通感一体化权衡 (ISAC Trade-off)
# ==========================================

def run_isac_trial_single(ibo, seed, noise_std, sig_truth):
    """ 单次试验: 返回 (Capacity, RMSE) """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, **config_det)

    # 1. 硬件损伤注入
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    # 2. PA 放大 (关键步骤: IBO 决定了线性和功率)
    pa_in = sig_truth * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 3. 指标 A: 通信容量
    # Capacity = log2(1 + SNR_out)
    # 实际上应计算 SINR，这里简化为输出信噪比
    p_rx = np.mean(np.abs(pa_out) ** 2)
    snr_lin = p_rx / (noise_std ** 2)
    capacity = np.log2(1 + snr_lin)

    # 4. 指标 B: 感知 RMSE
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    v_scan = np.linspace(true_v - 1500, true_v + 1500, 41)

    # 局部 GLRT 扫描
    stats = []
    for v in v_scan:
        s_raw = det._generate_template(v)
        s_perp = det.P_perp @ s_raw
        denom = np.sum(s_perp ** 2) + 1e-15
        stats.append((np.dot(s_perp, z_perp) ** 2) / denom)

    est_v = v_scan[np.argmax(stats)]
    err_v = abs(est_v - true_v)

    return capacity, err_v


def run_fig9_isac():
    print("\n--- Running Fig 9: ISAC Trade-off Analysis ---")

    # 预计算物理信号
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v})
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_truth = 1.0 + d_wb

    # 噪声定标 (以 40dB SNR 为基准)
    hw_temp = HardwareImpairments(config_hw)
    pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_truth, 15.0)
    p_ref = np.mean(np.abs(pa_ref) ** 2)
    noise_std = np.sqrt(p_ref * 1e-4)

    # 扫描 IBO: 从线性区 (20dB) 到 深度饱和区 (-5dB)
    ibo_points = np.linspace(20, -5, 15)
    trials = 50

    avg_cap = []
    avg_rmse = []

    for ibo in tqdm(ibo_points, desc="Scanning IBO"):
        seeds = np.random.randint(0, 1e9, trials)
        res = Parallel(n_jobs=n_jobs)(
            delayed(run_isac_trial_single)(ibo, s, noise_std, sig_truth) for s in seeds
        )
        res = np.array(res)

        avg_cap.append(np.mean(res[:, 0]))
        avg_rmse.append(np.sqrt(np.mean(res[:, 1] ** 2)))

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 7))

    # 散点图，颜色代表 IBO
    sc = ax.scatter(avg_cap, avg_rmse, c=ibo_points, cmap='viridis_r', s=100, zorder=5, edgecolors='k')
    ax.plot(avg_cap, avg_rmse, 'k--', alpha=0.5, zorder=1)

    cbar = plt.colorbar(sc)
    cbar.set_label('Input Back-Off (dB) [High Power $\leftarrow$]')

    ax.set_xlabel('Comm Capacity (bits/s/Hz)')
    ax.set_ylabel('Sensing RMSE (m/s)')
    ax.set_title('Fig 9: ISAC Trade-off (Pareto Frontier)')
    ax.grid(True, linestyle=':')

    # 标注区域
    ax.annotate('Linear Region\n(High Accuracy, Low Cap)',
                xy=(avg_cap[0], avg_rmse[0]), xytext=(avg_cap[0], avg_rmse[0] + 100),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    ax.annotate('Saturation Region\n(Self-Healing Error)',
                xy=(avg_cap[-1], avg_rmse[-1]), xytext=(avg_cap[-1], avg_rmse[-1] + 100),
                arrowprops=dict(facecolor='red', shrink=0.05), ha='center')

    plt.tight_layout()
    plt.savefig('results/Fig9_ISAC_Tradeoff.png', dpi=300)
    plt.savefig('results/Fig9_ISAC_Tradeoff.pdf', format='pdf')

    # 保存数据
    save_csv({'ibo': ibo_points, 'capacity': avg_cap, 'rmse': avg_rmse}, 'Fig9_ISAC_Data')


if __name__ == "__main__":
    ensure_dir('results/csv_data')
    multiprocessing.freeze_support()

    # 依次执行
    run_fig8_mds()
    run_fig9_isac()

    print("\n[Done] Advanced metrics generated successfully.")
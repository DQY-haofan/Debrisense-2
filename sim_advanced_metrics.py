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
    raise SystemExit("Missing modules")

# 绘图配置 (移除不需要的 note)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42})


def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)


def save_csv(data_dict, filename, folder='results/csv_data'):
    """ 标准化 CSV 保存函数 """
    if not os.path.exists(folder): os.makedirs(folder)
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv to {folder}")


# --- 公共配置 (参数微调以确保可见性) ---
L_eff = 50e3
fs = 20e3
T_span = 0.1  # [TWEAK] 增加到 100ms 以获得更多处理增益
N = int(fs * T_span)
t_axis = np.linspace(-T_span / 2, T_span / 2, N)
true_v = 15000.0

# 硬件配置 [TWEAK: 降低噪声以验证逻辑]
# jitter_rms: 2e-6 -> 0.5e-6 (SOTA)
# L_1MHz: -85 -> -95 (Better Oscillator)
config_hw = {
    'jitter_rms': 0.5e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0
}

# 检测器配置
config_det_base = {'cutoff_freq': 300.0, 'L_eff': L_eff}

num_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cores - 2)


# =========================================================================
# Task 1: Fig 8 - 探测能力边界 / 最小可探测尺寸 (MDS)
# =========================================================================

def get_h0_stat(seed, noise_std):
    """ H0 (无目标) 仿真 """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))
    sig = np.ones(N, dtype=np.complex128)

    pa_in = sig * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    s_temp = det.P_perp @ det._generate_template(true_v)
    denom = np.sum(s_temp ** 2)
    if denom < 1e-15: return 0.0

    stat = (np.dot(s_temp, z_perp) ** 2) / denom
    return stat


def get_h1_stat(a_val, seed, noise_std):
    """ H1 (有目标) 仿真 """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)

    # [FIX] 检测器使用匹配尺寸
    det = TerahertzDebrisDetector(fs, N, a=a_val, **config_det_base)

    # 物理引擎: 真实尺寸
    phy_conf = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': true_v}
    phy = DiffractionChannel(phy_conf)

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
    print("\n=== Running Task 1: Fig 8 (Minimum Detectable Size) ===")

    # [TWEAK] 极低热噪声，专注于 Jitter/PhaseNoise 限制
    noise_std = 1.0e-4

    print("Step 1: Establishing CFAR Threshold (Pfa=1e-3)...")
    h0_trials = 2000
    seeds_h0 = np.random.randint(0, 1e9, h0_trials)

    stats_h0 = Parallel(n_jobs=n_jobs)(
        delayed(get_h0_stat)(s, noise_std) for s in seeds_h0
    )
    threshold = np.percentile(stats_h0, 99.9)
    print(f"   Threshold: {threshold:.2f}")

    # 扫描尺寸 (1mm 到 100mm)
    diameters_mm = np.logspace(0, 2, 20)
    radii_m = diameters_mm / 2000.0

    pd_curve = []
    trials_h1 = 100

    print("Step 2: Sweeping Debris Size...")
    for a_val in tqdm(radii_m):
        seeds_h1 = np.random.randint(0, 1e9, trials_h1)
        stats_h1 = Parallel(n_jobs=n_jobs)(
            delayed(get_h1_stat)(a_val, s, noise_std) for s in seeds_h1
        )
        pd_val = np.mean(np.array(stats_h1) > threshold)
        pd_curve.append(pd_val)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.semilogx(diameters_mm, pd_curve, 'b-o', linewidth=2, label='300 GHz FSR System')

    # 标注 MDS
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
    plt.tight_layout()

    plt.savefig('results/Fig8_Detection_Capability.png', dpi=300)
    plt.savefig('results/Fig8_Detection_Capability.pdf', format='pdf')

    save_csv({'diameter_mm': diameters_mm, 'pd': pd_curve}, 'Fig8_MDS_Data')


# =========================================================================
# Task 2: Fig 9 - 通感一体化权衡 (ISAC Trade-off)
# =========================================================================

def run_isac_trial_single(ibo, seed, noise_std, sig_truth):
    """ 单次试验 """
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_in = sig_truth * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # A. Capacity
    p_rx = np.mean(np.abs(pa_out) ** 2)
    snr_lin = p_rx / (noise_std ** 2)
    capacity = np.log2(1 + snr_lin)

    # B. Sensing RMSE
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    # [TWEAK] 更密集的扫描以捕获峰值
    v_scan = np.linspace(true_v - 1500, true_v + 1500, 101)

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
    print("\n=== Running Task 2: Fig 9 (ISAC Trade-off) ===")

    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v})
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_truth = 1.0 + d_wb

    # 噪声定标 (50dB Base SNR)
    hw_temp = HardwareImpairments(config_hw)
    pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_truth, 15.0)
    p_ref = np.mean(np.abs(pa_ref) ** 2)
    noise_std = np.sqrt(p_ref * 1e-5)

    # 扫描 IBO
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
        # 鲁棒平均 (剔除完全失败点)
        valid_errs = [e for e in res[:, 1] if e < 1400]
        if len(valid_errs) > 5:
            rmse = np.sqrt(np.mean(np.array(valid_errs) ** 2))
        else:
            rmse = 1500.0
        avg_rmse.append(rmse)

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(avg_cap, avg_rmse, c=ibo_points, cmap='viridis_r', s=100, zorder=5, edgecolors='k')
    ax.plot(avg_cap, avg_rmse, 'k--', alpha=0.5, zorder=1)

    cbar = plt.colorbar(sc)
    cbar.set_label('Input Back-Off (dB) [High Power $\leftarrow$]')

    ax.set_xlabel('Comm Capacity (bits/s/Hz)')
    ax.set_ylabel('Sensing RMSE (m/s)')
    ax.set_title('Fig 9: ISAC Trade-off (Pareto Frontier)')
    ax.grid(True, linestyle=':')

    # [CLEAN] 移除文字标注以保持图片干净，信息保留在 CSV 中
    # ax.annotate... (Removed per instruction)

    plt.tight_layout()
    plt.savefig('results/Fig9_ISAC_Tradeoff.png', dpi=300)
    plt.savefig('results/Fig9_ISAC_Tradeoff.pdf', format='pdf')

    save_csv({'ibo': ibo_points, 'capacity': avg_cap, 'rmse': avg_rmse}, 'Fig9_ISAC_Data')


if __name__ == "__main__":
    ensure_dir('results')
    ensure_dir('results/csv_data')
    multiprocessing.freeze_support()

    run_fig8_mds()
    run_fig9_isac()

    print("\n[Done] All advanced metrics generated. Check 'results/csv_data/' for data.")
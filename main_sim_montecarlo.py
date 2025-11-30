import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from tqdm import tqdm
import os
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd

# 导入自定义模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    print("Error: Dependent modules not found.")
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


def save_csv(data_dict, filename, folder='results/csv_data'):
    """ 统一 CSV 保存接口: 强制保存到 results/csv_data """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 填充 NaN 以对齐长度 (因为不同曲线长度可能一致，但为了鲁棒性)
    max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data_dict.values()])
    aligned_data = {}
    for k, v in data_dict.items():
        if hasattr(v, '__len__'):
            padded = np.full(max_len, np.nan)
            padded[:len(v)] = v
            aligned_data[k] = padded
        else:
            aligned_data[k] = np.full(max_len, v)

    df = pd.DataFrame(aligned_data)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv to {folder}")


def calc_bcrlb(snr_linear, jitter_cov_inv, h_sensitivity, scr=1.0):
    """ 计算包含 SCR (自愈效应) 修正的 BCRLB """
    h_eff = h_sensitivity * scr
    j_jitter = np.dot(h_eff.T, np.dot(jitter_cov_inv, h_eff))
    j_thermal = snr_linear * np.sum(h_sensitivity ** 2) * (scr ** 2)
    return np.sqrt(1.0 / (j_jitter + j_thermal))


# --- 单次蒙特卡洛试验函数 ---
def run_trial(ibo, jitter_rms, seed, noise_std, sig_broadband_truth, N, fs, true_v, hw_base_config, det_config):
    """
    执行单次仿真闭环：
    宽带真值 -> 有色抖动 + 相位噪声 -> PA非线性 -> 热噪声 -> 对数检测器 -> GLRT估计
    """
    np.random.seed(seed)

    hw_config = hw_base_config.copy()
    hw_config['jitter_rms'] = jitter_rms

    hw_local = HardwareImpairments(hw_config)
    det_local = TerahertzDebrisDetector(fs, N, **det_config)

    # 1. 生成有色 Jitter (Amplitude)
    jitter = hw_local.generate_colored_jitter(N, fs)
    a_jitter = np.exp(jitter)

    # 2. 生成相位噪声 (Phase)
    theta_pn = hw_local.generate_phase_noise(N, fs)
    phase_noise = np.exp(1j * theta_pn)

    # 3. 物理互动: 信号受到幅度与相位的双重调制
    pa_in = sig_broadband_truth * a_jitter * phase_noise

    # 4. PA 非线性
    pa_out, _, _ = hw_local.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 5. 添加热噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 6. 检测器处理
    z_log = det_local.log_envelope_transform(y_rx)
    z_perp = det_local.apply_projection(z_log)

    # 7. GLRT 速度搜索
    v_scan = np.linspace(true_v - 1500, true_v + 1500, 31)
    stats = det_local.glrt_scan(z_perp, v_scan)

    est_v = v_scan[np.argmax(stats)]

    return est_v - true_v


def main():
    if multiprocessing.current_process().name == 'MainProcess':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    print("=== IEEE TWC Simulation: Fig 6 (Hardware Sensitivity) ===")
    ensure_dir('results')
    ensure_dir('results/csv_data')

    num_cores = multiprocessing.cpu_count()
    n_jobs = max(1, num_cores - 2)
    print(f"[INFO] Using {n_jobs} cores for parallel processing.")

    # --- 1. 系统参数 ---
    L_eff = 50e3
    fs = 20e3
    T_span = 0.05
    N = int(fs * T_span)
    t_axis = np.linspace(-T_span / 2, T_span / 2, N)
    true_v = 15000.0

    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v}
    config_hw_base = {'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
    config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'a': 0.05}

    # --- 2. 预计算 ---
    print("Pre-computing Broadband Physics Signal...")
    phy = DiffractionChannel(config_phy)
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=64)
    sig_broadband_truth = 1.0 + d_wb

    det_temp = TerahertzDebrisDetector(fs, N, **config_det)
    s_v = det_temp._generate_template(true_v)
    s_v_plus = det_temp._generate_template(true_v + 1.0)
    h_vec = (s_v_plus - s_v) / 1.0

    # 噪声底校准
    hw_temp = HardwareImpairments(config_hw_base)
    pa_ref, _, _ = hw_temp.apply_saleh_pa(sig_broadband_truth, 15.0)
    p_ref = np.mean(np.abs(pa_ref) ** 2)
    noise_std = np.sqrt(p_ref * 1e-5)  # 50dB Base SNR
    print(f"Calibrated Noise Floor (Std): {noise_std:.2e}")

    # --- 3. 仿真扫描 ---
    ibo_scan = np.linspace(15, -5, 11)
    # Jitter 曲线族 (Family of Curves)
    jitter_levels = [0.5e-6, 2.0e-6, 5.0e-6]
    jitter_labels = ['SOTA (0.5urad)', 'Typical (2.0urad)', 'Poor (5.0urad)']
    colors = ['g', 'b', 'r']

    trials_per_point = 200
    results_dict = {'ibo': ibo_scan}

    fig, ax1 = plt.subplots(figsize=(10, 7))

    for idx, (j_rms, label) in enumerate(zip(jitter_levels, jitter_labels)):
        print(f"\n--- Simulating Curve: {label} ---")

        # 计算理论协方差 (近似)
        hw_config_j = config_hw_base.copy()
        hw_config_j['jitter_rms'] = j_rms
        hw_j = HardwareImpairments(hw_config_j)
        long_jit = hw_j.generate_colored_jitter(N * 200, fs)
        r_xx = np.correlate(long_jit, long_jit, mode='full')[len(long_jit) - 1: len(long_jit) - 1 + N] / len(long_jit)
        C_inv = np.linalg.inv(sla.toeplitz(r_xx) + np.eye(N) * 1e-12)

        rmse_sim = []
        bcrlb_theo = []

        for ibo in tqdm(ibo_scan):
            # A. 理论值
            _, scr_tr, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            scr = np.mean(scr_tr)

            pa_out, _, _ = hw_j.apply_saleh_pa(sig_broadband_truth, ibo)
            p_sig = np.var(pa_out)
            snr_lin = p_sig / (noise_std ** 2)

            bcrlb = calc_bcrlb(snr_lin, C_inv, h_vec, scr=scr)
            bcrlb_theo.append(bcrlb)

            # B. 仿真
            seeds = np.random.randint(0, 1e9, trials_per_point)
            errs = Parallel(n_jobs=n_jobs)(
                delayed(run_trial)(ibo, j_rms, s, noise_std, sig_broadband_truth, N, fs, true_v, config_hw_base,
                                   config_det)
                for s in seeds
            )

            valid_errs = [e for e in errs if abs(e) < 1200]
            if len(valid_errs) > 10:
                rmse = np.sqrt(np.mean(np.array(valid_errs) ** 2))
            else:
                rmse = 1000.0
            rmse_sim.append(rmse)

        # 绘图
        ax1.semilogy(ibo_scan, rmse_sim, f'{colors[idx]}o-', label=f'Sim: {label}')
        ax1.semilogy(ibo_scan, bcrlb_theo, f'{colors[idx]}--', alpha=0.5)

        # 存储数据
        results_dict[f'rmse_sim_{idx}'] = rmse_sim
        results_dict[f'bcrlb_{idx}'] = bcrlb_theo

    ax1.set_xlabel(r'Input Back-Off (dB) [$\leftarrow$ High Power]')
    ax1.set_ylabel('Ranging RMSE (m/s)')
    ax1.invert_xaxis()
    ax1.grid(True, which='both', linestyle=':')
    ax1.set_ylim(50, 2000)
    ax1.legend(loc='best', title="Platform Stability")

    plt.title('Fig 6: Hardware Sensitivity Analysis')
    plt.tight_layout()
    plt.savefig('results/Fig6_Hardware_Sensitivity.png', dpi=300)
    plt.savefig('results/Fig6_Hardware_Sensitivity.pdf', format='pdf')

    save_csv(results_dict, 'Fig6_Sensitivity_Data')
    print("\n[Success] Fig 6 generated. Data saved to results/csv_data/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
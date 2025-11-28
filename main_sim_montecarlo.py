import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm
import os
import csv
import multiprocessing
from joblib import Parallel, delayed
import time

# 导入模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    print("Error: Dependent modules not found.")
    raise SystemExit

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'lines.linewidth': 1.5, 'pdf.fonttype': 42})


def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def calc_capacity(snr_linear):
    """ 计算香农容量 (bits/s/Hz) """
    return np.log2(1 + snr_linear)


def calc_bcrlb(snr_linear, jitter_cov_inv, h_sensitivity, scr=1.0):
    """ 计算考虑 SCR 的 BCRLB """
    h_eff = h_sensitivity * scr
    j_jitter = np.dot(h_eff.T, np.dot(jitter_cov_inv, h_eff))
    j_thermal = snr_linear * np.sum(h_sensitivity ** 2) * (scr ** 2)
    return np.sqrt(1.0 / (j_jitter + j_thermal))


# --- 修改：单次试验现在接收宽带信号作为输入 ---
def run_single_trial_broadband(ibo, seed, noise_std, sig_broadband_truth, N, fs, true_v, hw_config, det_config):
    np.random.seed(seed)

    # 本地实例化轻量级对象
    hw_local = HardwareImpairments(hw_config)
    # 检测器依然使用内部的"窄带近似"模板，这正是我们要验证的：窄带检测器能否搞定宽带信号
    det_local = TerahertzDebrisDetector(fs, N, **det_config)

    # 1. 生成有色 Jitter
    jitter = hw_local.generate_colored_jitter(N, fs)
    a_jitter = np.exp(jitter)

    # 2. PA 非线性 (宽带信号 * Jitter)
    # sig_broadband_truth 已经是 1 + d(t) 的形式
    pa_in = sig_broadband_truth * a_jitter
    pa_out, _, _ = hw_local.apply_saleh_pa(pa_in, ibo_dB=ibo)

    # 3. 热噪声
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 4. 检测流程
    z_log = det_local.log_envelope_transform(y_rx)
    z_perp = det_local.apply_projection(z_log)

    # 5. GLRT 搜索
    v_scan = np.linspace(true_v - 2000, true_v + 2000, 41)
    stats = det_local.glrt_scan(z_perp, v_scan)
    est_v = v_scan[np.argmax(stats)]

    return est_v - true_v


def main():
    print("Initializing High-Fidelity Physics Simulation...")
    ensure_dir('results')

    num_cores = multiprocessing.cpu_count()
    n_jobs = max(1, num_cores - 2)

    # --- 1. 物理参数配置 ---
    # 注意：L_eff 使用 500km 以符合 DR 报告，如果跑不动再降到 50km
    L_eff_sim = 500e3
    fs = 20e3
    T_span = 0.05  # 50ms 足够捕捉 500km 处的慢速条纹
    N = int(fs * T_span)
    t_axis = np.linspace(-T_span / 2, T_span / 2, N)
    true_v = 15000.0

    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff_sim, 'a': 0.05, 'v_rel': true_v}
    config_hw = {'jitter_rms': 2.0e-6, 'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
    config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff_sim, 'a': 0.05}

    # --- 2. [关键修正] 生成真值宽带信号 ---
    print(f"Generating Broadband Truth Signal (10GHz DFS, L={L_eff_sim / 1000}km)...")
    phy_engine = DiffractionChannel(config_phy)
    # d_broadband 是复数衍射项，必须加上直流载波 1.0
    d_broadband = phy_engine.generate_broadband_chirp(t_axis, N_sub=64)
    sig_broadband_truth = 1.0 + d_broadband

    # 为了计算理论界，我们需要计算窄带灵敏度向量 h
    # 因为 BCRLB 是基于检测器模型的理论极限
    det_temp = TerahertzDebrisDetector(fs, N, **config_det)
    s_v = det_temp._generate_template(true_v)
    s_v_plus = det_temp._generate_template(true_v + 1.0)
    h_vec = (s_v_plus - s_v) / 1.0  # 梯度

    # Jitter 协方差准备
    hw_temp = HardwareImpairments(config_hw)
    long_jitter = hw_temp.generate_colored_jitter(N * 200, fs)
    r_xx = np.correlate(long_jitter, long_jitter, mode='full')[len(long_jitter) - 1:len(long_jitter) - 1 + N] / len(
        long_jitter)
    C_inv = la.inv(la.toeplitz(r_xx) + np.eye(N) * 1e-10)

    # --- 3. 扫描配置 ---
    # IBO 范围：从线性区 (15dB) 到 深度饱和 (-5dB)
    ibo_range = np.linspace(15, -5, 15)
    mc_trials = 500  # 500次足以画出趋势

    # 噪声设置：为了凸显 PA 和 Jitter 效应，热噪声设得低一点
    noise_std = 1e-3

    results = {'ibo': [], 'rmse': [], 'crb': [], 'capacity': []}

    print(f"Starting Scan: {len(ibo_range)} points, {mc_trials} trials/point")

    for ibo in tqdm(ibo_range):
        # A. 理论值计算
        _, scr_trace, _ = hw_temp.apply_saleh_pa(sig_broadband_truth, ibo)
        avg_scr = np.mean(scr_trace)

        # 计算输出信噪比 (用于 BCRLB 和 Capacity)
        # 信号功率 (AC) vs 热噪声功率
        pa_out_ref, _, _ = hw_temp.apply_saleh_pa(sig_broadband_truth, ibo)
        p_sig = np.var(pa_out_ref)  # AC power
        snr_lin = p_sig / (noise_std ** 2)

        # 计算通信容量 (简单香农公式，假设带宽归一化或关注频谱效率)
        # 这里用总功率信噪比 (DC+AC) / Noise，因为通信是用全功率的
        p_total = np.mean(np.abs(pa_out_ref) ** 2)
        comm_snr = p_total / (noise_std ** 2)
        cap = calc_capacity(comm_snr)

        crb = calc_bcrlb(snr_lin, C_inv, h_vec, scr=avg_scr)

        # B. 蒙特卡洛
        seeds = np.random.randint(0, 1e9, mc_trials)
        errors = Parallel(n_jobs=n_jobs)(
            delayed(run_single_trial_broadband)(
                ibo, s, noise_std, sig_broadband_truth, N, fs, true_v, config_hw, config_det
            ) for s in seeds
        )

        # 鲁棒 RMSE (剔除异常值)
        valid_errs = [e for e in errors if abs(e) < 1000]
        rmse = np.sqrt(np.mean(np.array(valid_errs) ** 2)) if valid_errs else 1000.0

        results['ibo'].append(ibo)
        results['rmse'].append(rmse)
        results['crb'].append(crb)
        results['capacity'].append(cap)

    # --- 4. 双轴绘图 (Sensing vs Comms Trade-off) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：RMSE (感知)
    l1, = ax1.semilogy(results['ibo'], results['rmse'], 'b-o', label='Sensing RMSE (Sim)')
    l2, = ax1.semilogy(results['ibo'], results['crb'], 'b--', label='Sensing BCRLB (Theory)')
    ax1.set_xlabel('Input Back-Off (dB) [High Power ->]')
    ax1.set_ylabel('Ranging RMSE (m/s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.invert_xaxis()  # 左边是高 IBO (低功率)，右边是低 IBO (高功率)

    # 右轴：Capacity (通信)
    ax2 = ax1.twinx()
    l3, = ax2.plot(results['ibo'], results['capacity'], 'r-s', label='Comm Capacity (bits/s/Hz)')
    ax2.set_ylabel('Spectral Efficiency', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 标注 ISAC 权衡区
    plt.title('ISAC Trade-off: PA Saturation vs Performance')
    ax1.grid(True, which='both', linestyle=':')

    # 合并图例
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')

    plt.tight_layout()
    plt.savefig('results/ISAC_Tradeoff_MoneyPlot.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
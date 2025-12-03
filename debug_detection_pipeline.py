import numpy as np
import matplotlib.pyplot as plt
from physics_engine import DiffractionChannel
from detector import TerahertzDebrisDetector
from hardware_model import HardwareImpairments

# --- 简单配置 ---
GLOBAL_CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 50e3,
    'fs': 200e3, 'T_span': 0.02,
    'a_default': 0.05, 'v_default': 15000,
    # 调试用的 SNR，确保足够高以验证算法本身
    'debug_snr_db': 25.0
}

HW_CONFIG = {
    'jitter_rms': 2.0e-6, 'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 10e3
}


def _calc_noise_std(signal_reference, target_snr_db):
    """
    核心修复：基于前向散射残差(d)的能量来计算噪声水平。
    """
    sigma2_sig = np.mean(np.abs(signal_reference) ** 2)
    noise_std = np.sqrt(sigma2_sig / (10 ** (target_snr_db / 10.0)))
    return noise_std


def step0_physics_baseline():
    print("\n=== Step 0: Physics & Energy Check ===")
    fs = GLOBAL_CONFIG['fs']
    N = int(GLOBAL_CONFIG['T_span'] * fs)
    t_axis = np.arange(N) / fs - (N / 2) / fs

    # 1. 生成物理信号 d(t)
    phy = DiffractionChannel({
        'fc': GLOBAL_CONFIG['fc'], 'B': GLOBAL_CONFIG['B'],
        'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a_default'],
        'v_rel': GLOBAL_CONFIG['v_default']
    })
    d = phy.generate_broadband_chirp(t_axis, N_sub=32)

    # 2. 能量统计
    power_d = np.mean(np.abs(d) ** 2)
    max_d = np.max(np.abs(d))
    print(f"[Physics] Mean Power of d(t): {power_d:.3e}")
    print(f"[Physics] Max Amplitude of d(t): {max_d:.3e}")

    # 3. 检查投影矩阵是否"误杀"了信号
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, N_sub=32)

    # 信号转换到对数域近似 (假设线性区 ln(1-d) ~ -d)
    # 这里直接用 -d 来测试线性投影的损耗
    d_perp = det.apply_projection(d)
    power_d_perp = np.mean(np.abs(d_perp) ** 2)
    ratio = power_d_perp / power_d
    print(f"[Detector] Energy preservation after P_perp (300Hz cut): {ratio * 100:.2f}%")

    if ratio < 0.1:
        print("!! WARNING !! P_perp is killing >90% of the signal energy. Check cutoff_freq vs Chirp rate.")
    else:
        print("[Detector] Projection looks healthy.")

    return d, t_axis


def step1_ideal_awgn_glrt(d, t_axis):
    print("\n=== Step 1: Ideal AWGN + GLRT Validation ===")
    fs = GLOBAL_CONFIG['fs']
    N = len(t_axis)

    # 设定目标 SNR (相对于 d)
    target_snr = GLOBAL_CONFIG['debug_snr_db']
    noise_std = _calc_noise_std(d, target_snr)
    print(f"[Setup] Target SNR: {target_snr} dB")
    print(f"[Setup] Calculated Noise Std: {noise_std:.3e}")

    # 生成接收信号 y = (1 - d) + n
    # 注意：GLRT 是在 log 域工作的，所以我们输入物理信号
    sig_truth = 1.0 - d
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_received = sig_truth + noise

    # Detector Pipeline
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0,
                                  L_eff=GLOBAL_CONFIG['L_eff'],
                                  a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)

    # 1. Log Envelope
    z = det.log_envelope_transform(y_received)

    # 2. Projection
    z_perp = det.apply_projection(z)

    # 3. GLRT Scan
    v_true = GLOBAL_CONFIG['v_default']
    # 扫描范围：真实值附近 +/- 1000 m/s
    v_scan = np.linspace(v_true - 1000, v_true + 1000, 51)

    glrt_stats = det.glrt_scan(z_perp, v_scan)

    # 结果分析
    idx_peak = np.argmax(glrt_stats)
    v_hat = v_scan[idx_peak]
    peak_val = glrt_stats[idx_peak]

    # 次大峰值 (排除主峰附近的旁瓣)
    # 简单找一个远离主峰的点
    mask = np.abs(v_scan - v_hat) > 200
    if np.any(mask):
        second_peak = np.max(glrt_stats[mask])
    else:
        second_peak = 0.0

    contrast = peak_val / (second_peak + 1e-20)

    print(f"[Result] True v: {v_true}, Est v: {v_hat}")
    print(f"[Result] RMSE: {abs(v_hat - v_true):.2f} m/s")
    print(f"[Result] Peak Value: {peak_val:.2f}, Contrast (vs side): {contrast:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(v_scan, glrt_stats, 'b-o')
    plt.axvline(v_true, color='r', linestyle='--', label='True v')
    plt.title(f"Ideal GLRT (SNR={target_snr}dB)")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Test Statistic")
    plt.grid(True)
    plt.legend()
    # plt.show() # Agent环境注释掉

    if abs(v_hat - v_true) < 100 and contrast > 2.0:
        print(">>> SUCCESS: Ideal GLRT is working correctly! <<<")
        return True
    else:
        print(">>> FAILURE: Ideal GLRT failed. Stop here and fix Detector/SNR. <<<")
        return False


def step2_hardware_impairments(d, t_axis):
    print("\n=== Step 2: Hardware Impairments Stress Test ===")
    fs = GLOBAL_CONFIG['fs']
    N = len(t_axis)
    # [FIX] 必须传入 L_eff 和 a 等物理参数，否则使用错误的默认值(500km)会导致匹配失败
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0,
                                  L_eff=GLOBAL_CONFIG['L_eff'],
                                  a=GLOBAL_CONFIG['a_default'],
                                  N_sub=32)
    hw = HardwareImpairments(HW_CONFIG)

    sig_truth = 1.0 - d
    target_snr = GLOBAL_CONFIG['debug_snr_db']
    noise_std = _calc_noise_std(d, target_snr)  # 保持噪声水平不变，考察硬件影响

    scenarios = {
        "Ideal": (False, False, False),
        "Jitter Only": (True, False, False),
        "Jitter + PA": (True, True, False),
        "Full Hardware": (True, True, True)
    }

    v_true = GLOBAL_CONFIG['v_default']
    v_scan = np.linspace(v_true - 1000, v_true + 1000, 21)  # 粗扫

    print(f"{'Scenario':<15} | {'RMSE (m/s)':<10} | {'Peak Val':<10}")
    print("-" * 40)

    for name, (use_jit, use_pa, use_pn) in scenarios.items():
        # 构建受损信号
        signal_chain = sig_truth

        # 1. Jitter (Multiplicative)
        if use_jit:
            jit = np.exp(hw.generate_colored_jitter(N, fs))
            signal_chain = signal_chain * jit

        # 2. Phase Noise
        if use_pn:
            pn = np.exp(1j * hw.generate_phase_noise(N, fs))
            signal_chain = signal_chain * pn

        # 3. PA (Nonlinear)
        if use_pa:
            # 使用较小的 IBO 模拟一定程度的非线性
            signal_chain, _, _ = hw.apply_saleh_pa(signal_chain, ibo_dB=10.0)

        # 4. AWGN
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
        y_final = signal_chain + noise

        # Detection
        z = det.log_envelope_transform(y_final)
        z_perp = det.apply_projection(z)
        stats = det.glrt_scan(z_perp, v_scan)

        v_hat = v_scan[np.argmax(stats)]
        rmse = abs(v_hat - v_true)
        peak = np.max(stats)

        print(f"{name:<15} | {rmse:<10.1f} | {peak:<10.2f}")


if __name__ == "__main__":
    d_base, t_base = step0_physics_baseline()
    success = step1_ideal_awgn_glrt(d_base, t_base)
    if success:
        step2_hardware_impairments(d_base, t_base)
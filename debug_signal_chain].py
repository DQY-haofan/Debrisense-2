import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal
import os

# 导入模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Missing modules")

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 150})


def debug_run():
    print("=== Deep Diagnostic Run: Signal Chain Inspection ===")
    if not os.path.exists('results/debug'): os.makedirs('results/debug')

    # 1. 场景配置 (典型 LNT 场景)
    L_eff = 50e3
    fs = 20e3
    T_span = 0.05
    N = int(fs * T_span)
    t_axis = np.linspace(-T_span / 2, T_span / 2, N)
    true_v = 15000.0

    # 目标参数: 2cm 直径 (a=0.01m) - 这应该是可检测的
    target_a = 0.01

    # 硬件配置 (保持与 sim_advanced_metrics 一致)
    config_hw = {'jitter_rms': 2.0e-6, 'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127,
                 'L_1MHz': -90.0, 'L_floor': -120.0}  # 稍微降低噪声以排查问题

    # 物理引擎
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': target_a, 'v_rel': true_v})

    # 检测器 (关键：这里我们先用匹配的 a 进行诊断，确认算法本身是否工作)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=L_eff, a=target_a)

    # --- Step 1: 纯物理信号 ---
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_clean = 1.0 + d_wb
    print(f"Signal Depth (Max): {np.max(np.abs(d_wb)):.2e}")

    # --- Step 2: 硬件损伤注入 ---
    hw = HardwareImpairments(config_hw)

    # Jitter
    jit_raw = hw.generate_colored_jitter(N, fs)
    jit = np.exp(jit_raw)
    print(f"Jitter RMS (log): {np.std(jit_raw):.2e} rad")

    # Phase Noise
    theta_pn = hw.generate_phase_noise(N, fs)
    pn = np.exp(1j * theta_pn)
    print(f"Phase Noise RMS: {np.std(theta_pn):.2e} rad")

    # 合成输入
    pa_in = sig_clean * jit * pn

    # PA (Linear Region IBO=15dB to isolate detection logic)
    pa_out, scr, _ = hw.apply_saleh_pa(pa_in, ibo_dB=15.0)
    print(f"PA SCR (Linearity): {np.mean(scr):.4f}")

    # Thermal Noise (High SNR condition for debug)
    # Calibrate signal power
    p_sig = np.mean(np.abs(pa_out) ** 2)
    desired_snr_db = 40
    noise_std = np.sqrt(p_sig * 10 ** (-desired_snr_db / 10))
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    y_rx = pa_out + w

    # --- Step 3: 检测器处理 ---
    # A. Log Transform
    z_log = det.log_envelope_transform(y_rx)

    # B. Projection
    z_perp = det.apply_projection(z_log)

    # C. Template Matching
    s_temp = det.P_perp @ det._generate_template(true_v)

    # D. GLRT Scan
    v_scan = np.linspace(true_v - 2000, true_v + 2000, 100)
    glrt_profile = det.glrt_scan(z_perp, v_scan)

    # --- Visualization ---
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 1. Time Domain: Raw vs Physics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_axis * 1000, np.abs(y_rx), 'k-', alpha=0.3, label='Received Envelope (Noisy)')
    # Overlay scaled signal for comparison
    scale = np.mean(np.abs(y_rx))
    ax1.plot(t_axis * 1000, scale * (1.0 - np.real(d_wb)), 'r--', label='Physics Truth (Inverted)')
    ax1.set_title(f'Time Domain (Input) - Target a={target_a * 1000:.1f}mm')
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    # 2. Log Domain & Projection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_axis * 1000, z_log - np.mean(z_log), 'gray', alpha=0.5, label='Log (Centered)')
    ax2.plot(t_axis * 1000, z_perp, 'b-', linewidth=1.5, label='Projected (Signal Space)')
    ax2.plot(t_axis * 1000, s_temp * (np.max(z_perp) / np.max(s_temp)), 'g--', label='Template (Scaled)')
    ax2.set_title('Log-Projection Domain')
    ax2.legend()

    # 3. Frequency Domain (PSD)
    ax3 = fig.add_subplot(gs[1, :])
    f, p_log = scipy.signal.periodogram(z_log, fs)
    f, p_perp = scipy.signal.periodogram(z_perp, fs)
    f, p_temp = scipy.signal.periodogram(s_temp, fs)

    ax3.semilogy(f, p_log, 'gray', alpha=0.3, label='Log Noise Floor')
    ax3.semilogy(f, p_perp, 'b-', alpha=0.8, label='Projected Signal')
    ax3.semilogy(f, p_temp * (np.max(p_perp) / np.max(p_temp)), 'g--', label='Target Template Spectrum')
    ax3.set_xlim(0, 5000)
    ax3.axvspan(0, 300, color='red', alpha=0.1, label='Blind Zone')
    ax3.set_title('Spectral Analysis')
    ax3.legend()

    # 4. GLRT Profile
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(v_scan, glrt_profile, 'b-o')
    ax4.axvline(true_v, color='r', linestyle='--', label='True Velocity')
    ax4.set_title('GLRT Velocity Scan')
    ax4.set_xlabel('Velocity (m/s)')
    ax4.set_ylabel('Test Statistic')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('results/debug/Diagnostic_Report.png')
    print("Saved results/debug/Diagnostic_Report.png")

    # 简单分析
    peak_v = v_scan[np.argmax(glrt_profile)]
    print(f"\nDiagnostic Result:")
    print(f"  > True Velocity: {true_v}")
    print(f"  > Est Velocity:  {peak_v}")
    print(f"  > Peak Value:    {np.max(glrt_profile):.4f}")

    if np.max(glrt_profile) < 1.0:
        print("  [CRITICAL] Peak is extremely low. Signal is buried or template is mismatched.")
    elif abs(peak_v - true_v) > 500:
        print("  [WARNING] Peak found but velocity error is large.")
    else:
        print("  [PASS] Signal detected successfully.")


if __name__ == "__main__":
    debug_run()
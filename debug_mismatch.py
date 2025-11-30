import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal
import os
import pandas as pd

try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Missing modules.")

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 150})


def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)


def save_csv(data_dict, filename, folder='results/csv_data'):
    ensure_dir(folder)
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
    print(f"   [Data] Saved {filename}.csv")


def run_diagnostic():
    print("=== System Diagnostic & Verification (DFS Template Fix) ===")

    # 1. 场景配置
    L_eff = 50e3
    fs = 200e3
    T_span = 0.02
    N = int(fs * T_span)
    t_axis = np.arange(N) / fs - (N / 2) / fs
    true_v = 15000.0
    target_a = 0.05
    cutoff_freq_debug = 300.0

    config_hw = {
        'jitter_rms': 0.5e-6,
        'f_knee': 200.0,
        'beta_a': 5995.0, 'alpha_a': 10.127,
        'L_1MHz': -100.0, 'L_floor': -130.0, 'pll_bw': 50e3
    }
    noise_std = 1.0e-5

    # 2. 物理层生成 (宽带 DFS)
    # Physics 使用 N_sub=32
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': target_a, 'v_rel': true_v})
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_clean = 1.0 - d_wb

    # 3. 检测器内部检查 (模板验证)
    # Detector 现在也内置了 DFS，默认 N_sub=16
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=cutoff_freq_debug, L_eff=L_eff, a=target_a, B=10e9)

    s_raw = det._generate_template(true_v)
    s_perp = det.P_perp @ s_raw

    # 4. 波形对比 (物理信号的阴影部分 vs 模板)
    phy_signal = -np.real(d_wb)  # Log近似

    corr_coef = np.corrcoef(phy_signal, s_raw)[0, 1]
    print(f"\n[Waveform Check]")
    print(f"  Correlation: {corr_coef:.4f}")

    if corr_coef > 0.95:
        print("  [PASS] Perfect Match! DFS alignment successful.")
    else:
        print("  [FAIL] Still mismatched. Check DFS params.")

    # 5. 完整链路
    hw = HardwareImpairments(config_hw)
    jit_mod = np.exp(hw.generate_colored_jitter(N, fs))
    pn_mod = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_in = sig_clean * jit_mod * pn_mod
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    v_scan = np.linspace(true_v - 2000, true_v + 2000, 101)
    stats = det.glrt_scan(z_perp, v_scan)
    peak_val = np.max(stats)

    print(f"\n[Detection Result]")
    print(f"  Peak Stat: {peak_val:.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(t_axis * 1000, phy_signal, 'k-', label='Physics (DFS-32)', linewidth=2, alpha=0.5)
    ax1.plot(t_axis * 1000, s_raw, 'r--', label='Detector (DFS-16)', linewidth=1.5)
    ax1.set_title(f'Broadband Waveform Match (Corr={corr_coef:.4f})')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(v_scan, stats, 'b-o')
    ax2.axvline(true_v, color='r', linestyle='--')
    ax2.set_title(f'GLRT Scan (Peak={peak_val:.2f})')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('results/debug/Diagnostic_Fix5.png')

    save_csv({'t': t_axis, 'phy': phy_signal, 'det': s_raw, 'v': v_scan, 'stat': stats}, 'debug_fix5_data',
             'results/csv_data')


if __name__ == "__main__":
    run_diagnostic()
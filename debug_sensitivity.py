import numpy as np
import matplotlib.pyplot as plt
from physics_engine import DiffractionChannel
from detector import TerahertzDebrisDetector
from hardware_model import HardwareImpairments

# --- 全局配置 ---
GLOBAL_CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 50e3,
    'fs': 200e3, 'T_span': 0.02,
    'a_default': 0.05, 'v_default': 15000
}

# [FIX] 补全缺失的硬件配置
HW_CONFIG = {
    'jitter_rms': 1.0e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}


def _calc_noise_std(signal_ref, target_snr_db):
    p_sig = np.mean(np.abs(signal_ref) ** 2)
    noise_std = np.sqrt(p_sig / (10 ** (target_snr_db / 10.0)))
    return noise_std


def _log_envelope(y):
    return np.log(np.abs(y) + 1e-12)


def run_sensitivity_probe():
    print("=== SENSITIVITY PROBE START (FIXED) ===")

    # 1. 准备物理信号
    phy = DiffractionChannel(GLOBAL_CONFIG)
    N = int(GLOBAL_CONFIG['fs'] * GLOBAL_CONFIG['T_span'])
    t_axis = np.arange(N) / GLOBAL_CONFIG['fs'] - (N / 2) / GLOBAL_CONFIG['fs']
    d_truth = phy.generate_broadband_chirp(t_axis, 32)
    sig_truth = 1.0 - d_truth

    # [FIX] 显式传入物理参数 L_eff 和 a，防止使用默认值导致失配
    det = TerahertzDebrisDetector(GLOBAL_CONFIG['fs'], N, N_sub=32,
                                  L_eff=GLOBAL_CONFIG['L_eff'],
                                  a=GLOBAL_CONFIG['a_default'])
    P_perp = det.P_perp

    # 生成 GLRT 模板库 (只在真实速度附近扫，为了快)
    v_true = GLOBAL_CONFIG['v_default']
    v_scan = np.linspace(v_true - 500, v_true + 500, 21)

    # 预计算模板能量
    # 注意：这里的 s_raw_list 使用上面修正过的 det 生成
    s_raw_list = [det._generate_template(v) for v in v_scan]
    T_bank = np.array([P_perp @ s for s in s_raw_list])
    E_bank = np.sum(T_bank ** 2, axis=1) + 1e-20

    print(f"\n[Info] Signal |d| max: {np.max(np.abs(d_truth)):.2e}")
    print(f"[Info] Signal |d| mean power: {np.mean(np.abs(d_truth) ** 2):.2e}")

    # --- Probe 1: SNR Limit (Ideal Case) ---
    print("\n--- Probe 1: Finding SNR Limit (Ideal) ---")
    snr_levels = [-10, 0, 5, 10, 15, 20, 25, 30]  # Extended range

    # 记录一个可靠的 SNR 用于 Probe 2
    safe_snr = 30.0

    for snr in snr_levels:
        noise_std = _calc_noise_std(d_truth, snr)

        # Trial (跑 5 次取平均 RMSE，避免单次偶然性)
        rmses = []
        for _ in range(5):
            w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
            y = sig_truth + w
            z = _log_envelope(y)
            z_perp = P_perp @ z

            stats = (np.dot(T_bank, z_perp) ** 2) / E_bank
            v_hat = v_scan[np.argmax(stats)]
            rmses.append(abs(v_hat - v_true))

        avg_rmse = np.mean(rmses)
        peak = np.max(stats)

        status = "PASS" if avg_rmse < 100 else "FAIL"
        print(f"SNR={snr:3d} dB | NoiseStd={noise_std:.1e} | Peak={peak:.1e} | RMSE={avg_rmse:5.1f} | {status}")

        if status == "PASS" and safe_snr > snr:
            safe_snr = snr + 5.0  # 留 5dB 余量给 Probe 2

    print(f"\n[Info] Selected Safe SNR for Probe 2: {safe_snr} dB")

    # --- Probe 2: Jitter Limit (High SNR Case) ---
    print(f"\n--- Probe 2: Finding Jitter Limit (SNR={safe_snr}dB) ---")
    noise_std_fixed = _calc_noise_std(d_truth, safe_snr)

    # Jitter 扫描范围：1e-7 到 1e-4
    # 注意：d 的量级是 1e-5。如果 Jitter > 1e-5，基本就很难了。
    jit_levels = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

    hw = HardwareImpairments(HW_CONFIG)

    for jit_rms in jit_levels:
        hw_cfg = HW_CONFIG.copy()
        hw_cfg['jitter_rms'] = jit_rms
        hw_local = HardwareImpairments(hw_cfg)

        # 跑 5 次取平均
        rmses = []
        peaks = []
        for _ in range(5):
            jitter_vec = np.exp(hw_local.generate_colored_jitter(N, GLOBAL_CONFIG['fs']))
            y = sig_truth * jitter_vec
            w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std_fixed / np.sqrt(2)
            z = _log_envelope(y + w)
            z_perp = P_perp @ z

            stats = (np.dot(T_bank, z_perp) ** 2) / E_bank
            v_hat = v_scan[np.argmax(stats)]
            rmses.append(abs(v_hat - v_true))
            peaks.append(np.max(stats))

        avg_rmse = np.mean(rmses)
        avg_peak = np.mean(peaks)

        # 判定: Jitter 幅度相对于 d 的比例
        ratio = jit_rms / np.max(np.abs(d_truth))
        status = "PASS" if avg_rmse < 100 else "FAIL"

        print(f"Jit={jit_rms:.1e} | Jit/Sig Ratio={ratio:.2f} | Peak={avg_peak:.1e} | RMSE={avg_rmse:5.1f} | {status}")


if __name__ == "__main__":
    run_sensitivity_probe()
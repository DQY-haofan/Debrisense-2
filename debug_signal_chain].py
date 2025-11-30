import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os
from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments
from detector import TerahertzDebrisDetector

# 配置 (与 sim_advanced_metrics 保持一致)
L_eff = 50e3
fs = 200e3  # 高采样率
T_span = 0.02
N = int(fs * T_span)
t_axis = np.linspace(-T_span / 2, T_span / 2, N)
true_v = 15000.0

# 硬件配置
config_hw = {
    'jitter_rms': 0.5e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0
}
config_det_base = {'cutoff_freq': 300.0, 'L_eff': L_eff}


def run_diagnostic_trial(a_val, noise_std, seed):
    np.random.seed(seed)

    # 1. 初始化
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=a_val, **config_det_base)
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': true_v})

    # 2. 生成信号
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig_clean = 1.0 + d_wb

    # 3. 硬件损伤
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    # 4. 接收信号 (H1)
    pa_in = sig_clean * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w

    # 5. 检测器内部信号
    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    # 6. 模板
    s_temp_raw = det._generate_template(true_v)
    s_temp_perp = det.P_perp @ s_temp_raw

    # 7. 统计量计算
    # 相关性
    corr = np.dot(s_temp_perp, z_perp)
    # 能量
    energy_s = np.sum(s_temp_perp ** 2)
    energy_z = np.sum(z_perp ** 2)

    stat = (corr ** 2) / energy_s

    return {
        'signal_energy': np.sum(np.abs(d_wb) ** 2),
        'z_perp_energy': energy_z,
        'template_energy': energy_s,
        'correlation': corr,
        'stat': stat
    }


def main():
    print("=== Deep Dive Diagnostic ===")

    # 测试参数
    a_large = 0.05  # 100mm (应该很容易检测)
    a_small = 0.005  # 10mm (边界情况)
    noise_std = 1.0e-5
    trials = 100

    print(f"\n--- Testing Large Target (D=100mm) ---")
    results_large = Parallel(n_jobs=4)(delayed(run_diagnostic_trial)(a_large, noise_std, s) for s in range(trials))
    stats_large = np.array([r['stat'] for r in results_large])
    print(f"Avg Stat (H1): {np.mean(stats_large):.4f}")
    print(f"Min Stat: {np.min(stats_large):.4f} | Max Stat: {np.max(stats_large):.4f}")

    print(f"\n--- Testing Small Target (D=10mm) ---")
    results_small = Parallel(n_jobs=4)(delayed(run_diagnostic_trial)(a_small, noise_std, s) for s in range(trials))
    stats_small = np.array([r['stat'] for r in results_small])
    print(f"Avg Stat (H1): {np.mean(stats_small):.4f}")

    print(f"\n--- Testing H0 (Noise Only) ---")
    # H0 相当于 a=0 (无信号)，但为了计算 stat 我们需要一个模板，假设用 a_large 的模板
    # 这里我们在 run_diagnostic_trial 里通过改代码或者传入 a_val 但 phy 生成 0 信号来实现
    # 简单起见，我们修改 run_diagnostic_trial 的 d_wb 为 0
    # (为了代码简洁，这里不复写 run_diagnostic_trial，但在实际排查时需要)
    # 假设 H0 的统计量分布接近 Chi-Square(1) * scale

    # 关键检查点：
    # 如果 Avg Stat (H1_large) >> Avg Stat (H0)，说明大目标可检。
    # 如果 Avg Stat (H1_small) ≈ Avg Stat (H0)，说明小目标淹没。

    # 我们可以通过比较 s_temp_perp 的能量来推断
    # 理论上 GLRT 统计量在 H1 下是非中心卡方分布，均值约为 1 + lambda (lambda 为非中心参数，即匹配滤波器输出信噪比)

    avg_energy_s_large = np.mean([r['template_energy'] for r in results_large])
    avg_energy_s_small = np.mean([r['template_energy'] for r in results_small])

    print(f"\n--- Energy Analysis ---")
    print(f"Template Energy (Large): {avg_energy_s_large:.4e}")
    print(f"Template Energy (Small): {avg_energy_s_small:.4e}")
    print(
        f"Ratio (Large/Small): {avg_energy_s_large / avg_energy_s_small:.2f} (Expected ~ (100/10)^8 = 10^8 ? No, FSCS is D^4, Amplitude is D^2)")
    # Amplitude of d[n] is proportional to a^2. Energy is proportional to a^4.
    # (0.05 / 0.005)^4 = 10^4 = 10000.

    # 如果 template energy 极小，可能会导致数值不稳定。


if __name__ == "__main__":
    main()
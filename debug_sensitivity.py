import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# 尝试导入核心模块，如果路径不对请确保这些文件在同一目录下
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
except ImportError:
    raise SystemExit("Error: Core modules (physics_engine, hardware_model) not found.")

# --- 1. 配置：使用“最艰难”的设定 ---
CONFIG = {
    'fc': 300e9, 'B': 10e9, 'fs': 200e3,
    'T_span': 0.02, 'L_eff': 50e3, 'a': 0.05, 'v_default': 15000
}
HW_CFG = {
    'jitter_rms': 1.0e-4,  # 100ppm Jitter
    'f_knee': 200.0,
    'beta_a': 5995.0, 'alpha_a': 10.127,
    'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}


def run_diagnosis():
    print("Running Physics Diagnosis...")

    # --- 2. 生成信号 (Signal) ---
    phy = DiffractionChannel({**CONFIG, 'a': 0.01})  # Radius 10mm = Diam 20mm
    d_signal = phy.generate_broadband_chirp(
        np.arange(int(CONFIG['fs'] * CONFIG['T_span'])) / CONFIG['fs'] - 0.01,
        N_sub=32
    )

    # [FIX] 取绝对值 (Envelope)，将复数信号转换为实数信号
    # 物理意义：我们对比的是"幅度"上的信号强度 vs "幅度"上的噪声强度
    sig_ac = np.abs(d_signal)
    # 去除直流分量 (DC)，只看波动
    sig_ac = sig_ac - np.mean(sig_ac)

    # --- 3. 生成噪声 (Interference) ---
    hw = HardwareImpairments(HW_CFG)
    N = len(d_signal)
    jitter = hw.generate_colored_jitter(N, CONFIG['fs'])

    # --- 4. 计算功率谱密度 (PSD) ---
    # 现在两个输入都是实数，Welch 会返回相同长度的频率向量 (513点)
    f, Pxx_sig = signal.welch(sig_ac, CONFIG['fs'], nperseg=1024)
    f_jit, Pxx_jit = signal.welch(jitter, CONFIG['fs'], nperseg=1024)

    # 归一化对比 (dB/Hz)
    Pxx_sig_db = 10 * np.log10(Pxx_sig + 1e-20)
    Pxx_jit_db = 10 * np.log10(Pxx_jit + 1e-20)

    # --- 5. 可视化判决 ---
    plt.figure(figsize=(8, 6))

    # 画图：确保 x 和 y 维度一致
    plt.semilogx(f, Pxx_sig_db, 'g-', linewidth=2, label='Debris Signal Envelope (20mm)')
    plt.semilogx(f_jit, Pxx_jit_db, 'r--', linewidth=2, label='Hardware Jitter (1e-4)')

    # 标记关键频率
    plt.axvline(300, color='k', linestyle=':', label='Current Cutoff (300Hz)')
    plt.axvline(1000, color='b', linestyle='--', label='Proposed Cutoff (1kHz)')

    plt.title("Spectral Diagnosis: Signal Envelope vs. Jitter Noise")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid(True, which='both', linestyle='-', alpha=0.5)

    # 保存图片
    save_path = 'diagnosis_spectrum.png'
    plt.savefig(save_path, dpi=300)
    print(f"Diagnosis plot saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    run_diagnosis()
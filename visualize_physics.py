import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal
import os

# 导入核心模块
try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    print("Error: Dependent modules not found. Ensure physics_engine.py, hardware_model.py, detector.py are present.")
    raise SystemExit

# 设置绘图风格 (符合 IEEE TWC 标准)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def viz_dispersion_analysis(save_dir='results'):
    """
    Fig A: 宽带色散效应 (The Dispersion Evidence)
    对比 Narrowband (300GHz) 与 Wideband (10GHz DFS) 的时域波形
    """
    print("Generating Fig A: Dispersion Analysis...")

    # 1. 物理参数 (L_eff=50km, a=5cm)
    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': 50e3, 'a': 0.05, 'v_rel': 15000}
    phy = DiffractionChannel(config_phy)

    fs = 100e3  # 高采样率以展示细节
    T_span = 0.01  # 10ms
    t = np.linspace(-T_span / 2, T_span / 2, int(T_span * fs))

    # 2. 生成信号
    d_nb_matrix = phy.generate_diffraction_pattern(t, np.array([config_phy['fc']]))
    d_nb = np.abs(d_nb_matrix[0, :])

    d_wb = np.abs(phy.generate_broadband_chirp(t, N_sub=128))

    # 3. 绘图
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1])

    # 子图 1: 全局视图
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t * 1000, d_nb, 'r--', linewidth=1.5, label='Narrowband (300 GHz)')
    ax1.plot(t * 1000, d_wb, 'b-', linewidth=1.5, alpha=0.7, label='Broadband (10 GHz BW)')
    ax1.set_ylabel('|d(t)|')
    ax1.set_title('(a) Diffraction Pattern: Main Lobe Alignment')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':')
    ax1.set_xlim(-4, 4)

    # 子图 2: 局部放大 (Side-lobes Smearing)
    ax2 = fig.add_subplot(gs[1])
    # 聚焦于 1ms - 3ms (旁瓣区域)
    mask = (t > 0.001) & (t < 0.003)
    t_zoom = t[mask] * 1000
    ax2.plot(t_zoom, d_nb[mask], 'r--', linewidth=2.0, label='Narrowband')
    ax2.plot(t_zoom, d_wb[mask], 'b-', linewidth=2.0, alpha=0.8, label='Broadband')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('|d(t)| (Zoomed)')
    ax2.set_title('(b) Dispersion Smearing on High-Order Sidelobes')

    # 添加标注箭头
    peak_idx = np.argmax(d_nb[mask])
    peak_t = t_zoom[peak_idx]
    peak_val = d_nb[mask][peak_idx]
    ax2.annotate('Amplitude Attenuation\n(Smearing)',
                 xy=(peak_t, peak_val), xytext=(peak_t + 0.5, peak_val + 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'Fig_Dispersion_Analysis')
    plt.savefig(f'{save_path}.png', dpi=300)
    plt.savefig(f'{save_path}.pdf', format='pdf')
    print(f"Saved {save_path}")


def viz_survival_space(save_dir='results'):
    """
    Fig B: 频谱生存空间证明 (The Survival Space Proof)
    展示 Jitter 淹没原始信号，以及投影后 Chirp 信号显现
    """
    print("Generating Fig B: Survival Space Spectrogram...")

    # 1. 参数
    fs = 20e3
    N = int(fs * 0.05)  # 50ms 长窗口以展示时频特征
    t = np.linspace(0, 0.05, N)

    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=50e3, a=0.05)
    hw = HardwareImpairments({'jitter_rms': 5.0e-6, 'f_knee': 200.0})  # 强 Jitter

    # 2. 构造信号
    # 目标信号 (Chirp)
    true_v = 15000.0
    # 手动生成更长时间的 Chirp 用于展示
    # s(t) = -Re{d(t)}
    # 为了 Spectrogram 清晰，我们增强一下信号幅度
    d_chirp = det._generate_template(true_v) * 5.0
    sig_linear = 1.0 + d_chirp

    # Jitter
    jitter = hw.generate_colored_jitter(N, fs)
    a_jitter = np.exp(jitter)

    # 合成接收信号 (Log Domain)
    # y = A * (1-d) => z = ln(A) + ln(1-d) approx J + S
    z_raw = np.log(a_jitter * sig_linear + 1e-12)

    # 投影后信号
    z_perp = det.apply_projection(z_raw)

    # 3. 绘图 (Spectrogram)
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2])

    # 子图 1: 原始信号 PSD
    ax1 = fig.add_subplot(gs[0])
    f, Pxx_raw = scipy.signal.periodogram(z_raw, fs)
    ax1.semilogy(f, Pxx_raw, 'k', alpha=0.6, label='Raw Signal (Jitter Dominated)')
    ax1.axvspan(0, 300, color='red', alpha=0.2, label='Blind Zone (DC-300Hz)')
    ax1.set_ylabel('PSD (dB/Hz)')
    ax1.set_title('(a) Raw Signal Spectrum: Masked by Colored Jitter')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 5000)
    ax1.grid(True, linestyle=':')

    # 子图 2: 投影后信号 PSD
    ax2 = fig.add_subplot(gs[1])
    f, Pxx_proj = scipy.signal.periodogram(z_perp, fs)
    ax2.semilogy(f, Pxx_proj, 'b', alpha=0.8, label='Projected Signal')
    ax2.axvspan(300, 5000, color='green', alpha=0.1, label='Survival Space')
    ax2.set_ylabel('PSD (dB/Hz)')
    ax2.set_title('(b) Projected Spectrum: Jitter Removed, Signal Preserved')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 5000)
    ax2.grid(True, linestyle=':')

    # 子图 3: 时频图对比
    ax3 = fig.add_subplot(gs[2])
    # 使用 spectrogram
    f_spec, t_spec, Sxx = scipy.signal.spectrogram(z_perp, fs, nperseg=256, noverlap=128)
    # Log scale
    Sxx_log = 10 * np.log10(Sxx + 1e-12)

    im = ax3.pcolormesh(t_spec * 1000, f_spec, Sxx_log, shading='gouraud', cmap='inferno')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('(c) Spectrogram after Projection: The Chirp Signature')
    ax3.set_ylim(0, 5000)

    # 标注 Chirp
    ax3.annotate('Target Chirp\n(Slope ~ v^2)',
                 xy=(25, 2000), xytext=(35, 3000),
                 color='white',
                 arrowprops=dict(facecolor='white', shrink=0.05))

    # 标注 Blind Zone
    ax3.axhspan(0, 300, color='white', alpha=0.2, hatch='//')
    ax3.text(5, 100, 'Blind Zone (Null Space)', color='white', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.15)
    cbar.set_label('Power Spectral Density (dB)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'Fig_Survival_Space')
    plt.savefig(f'{save_path}.png', dpi=300)
    plt.savefig(f'{save_path}.pdf', format='pdf')
    print(f"Saved {save_path}")


def viz_self_healing(save_dir='results'):
    """
    Fig C: 灵敏度压缩时域演示 (The SCR Time-Domain Demo)
    对比线性区与饱和区的阴影深度
    """
    print("Generating Fig C: Self-Healing Effect...")

    # 1. 硬件模型
    config_hw = {'beta_a': 5995.0, 'alpha_a': 10.127}
    hw = HardwareImpairments(config_hw)

    # 2. 构造输入信号 (Ideal Shadow)
    t = np.linspace(0, 1, 1000)
    # 高斯阴影，深度 5%
    shadow = 0.05 * np.exp(-((t - 0.5) ** 2) / (0.05 ** 2))
    sig_in_norm = 1.0 - shadow

    # 3. 通过 PA
    # Case 1: Linear (IBO = 15dB)
    out_lin, _, v_lin = hw.apply_saleh_pa(sig_in_norm, ibo_dB=15.0)
    # Case 2: Saturation (IBO = -5dB)
    out_sat, _, v_sat = hw.apply_saleh_pa(sig_in_norm, ibo_dB=-5.0)

    # 4. 归一化以便对比形状 (AC Coupling view)
    # 我们不仅归一化幅度，还要去除直流，只看 AC 部分的对比
    def get_ac_shape(sig):
        mag = np.abs(sig)
        dc = np.mean(mag[:100])  # 取边缘作为基准
        return (mag - dc) / dc  # 相对变化率 (Shadow Depth)

    ac_in = get_ac_shape(sig_in_norm)
    ac_out_lin = get_ac_shape(out_lin)
    ac_out_sat = get_ac_shape(out_sat)

    # 5. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(t, ac_in * 100, 'k--', linewidth=1.5, label='Input: Ideal Shadow (5% Depth)')
    ax.plot(t, ac_out_lin * 100, 'g-', linewidth=2.0, label='Output: Linear PA (IBO=15dB)')
    ax.plot(t, ac_out_sat * 100, 'r-', linewidth=3.0, label='Output: Saturated PA (IBO=-5dB)')

    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Relative Amplitude Change (%)')
    ax.set_title('Self-Healing Effect: Shadow Erasure in Saturation')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':')

    # 添加标注
    depth_in = np.min(ac_in * 100)
    depth_sat = np.min(ac_out_sat * 100)

    ax.annotate('Shadow Preserved',
                xy=(0.5, depth_in), xytext=(0.6, depth_in - 1),
                arrowprops=dict(facecolor='green', shrink=0.05))

    ax.annotate('Shadow Erased (SCR < 0.2)',
                xy=(0.5, depth_sat), xytext=(0.3, depth_sat - 1),
                arrowprops=dict(facecolor='red', shrink=0.05))

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'Fig_Self_Healing')
    plt.savefig(f'{save_path}.png', dpi=300)
    plt.savefig(f'{save_path}.pdf', format='pdf')
    print(f"Saved {save_path}")


if __name__ == "__main__":
    ensure_dir('results')
    viz_dispersion_analysis()
    viz_survival_space()
    viz_self_healing()
    print("\nAll visualization plots generated successfully in 'results/' folder.")
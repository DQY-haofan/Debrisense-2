import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal
import os
import pandas as pd

# 导入模块
from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments
from detector import TerahertzDebrisDetector

# IEEE 绘图标准
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def save_csv(data_dict, filename, folder='results/csv_data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Handle varying lengths by finding max length and padding with NaN
    max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data_dict.values()])

    # Create dict with uniform length
    uniform_data = {}
    for k, v in data_dict.items():
        if hasattr(v, '__len__'):
            if len(v) < max_len:
                # Pad with nan
                padded = np.full(max_len, np.nan)
                padded[:len(v)] = v
                uniform_data[k] = padded
            else:
                uniform_data[k] = v
        else:
            # Broadcast scalar
            uniform_data[k] = np.full(max_len, v)

    df = pd.DataFrame(uniform_data)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


def save_plot(filename, folder='results'):
    if not os.path.exists(folder): os.makedirs(folder)
    plt.savefig(f"{folder}/{filename}.png", dpi=300)
    plt.savefig(f"{folder}/{filename}.pdf", format='pdf')
    print(f"   [Plot] Saved {filename}.png and .pdf")


def fig2_hardware_characteristics():
    print("Generating Fig 2: Hardware Characteristics...")
    hw = HardwareImpairments({})

    # 1. Jitter PSD
    fs = 10e3
    N = 10000
    jitter = hw.generate_colored_jitter(N, fs)
    f, psd = scipy.signal.welch(jitter, fs, nperseg=1024)

    # 2. PA Curves
    pin_db, am_am, scr = hw.get_pa_curves()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Jitter PSD
    ax1.loglog(f, psd, 'b-', linewidth=1.5)
    ax1.axvline(200, color='r', linestyle='--', label='Knee (200Hz)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (rad²/Hz)')
    ax1.set_title('(a) Colored Jitter Spectrum')
    ax1.legend()
    ax1.grid(True, which='both', linestyle=':')

    # Right: PA & SCR
    ax2.plot(pin_db, am_am, 'k-', label='AM-AM Curve')
    ax2.set_xlabel('Input Power (dB relative to Saturation)')
    ax2.set_ylabel('Normalized Output Amplitude')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True)

    ax2r = ax2.twinx()
    ax2r.plot(pin_db, scr, 'r--', label='SCR (Sensitivity)')
    ax2r.set_ylabel('Sensitivity Compression Ratio (SCR)', color='r')
    ax2r.tick_params(axis='y', labelcolor='r')
    ax2r.set_ylim(-0.5, 1.1)

    # 标注工作区
    ax2.axvspan(-30, -5, color='green', alpha=0.1)  # Linear
    ax2.text(-20, 0.5, 'Linear Region', color='green')
    ax2.axvspan(-5, 5, color='red', alpha=0.1)  # Saturation
    ax2.text(0, 0.5, 'Saturation', color='red')

    plt.tight_layout()
    save_plot('Fig2_Hardware_Characteristics')
    plt.close('all') # <--- **FIX: 显式关闭图形对象**

    # Export CSV (Splitting into two files for clarity as lengths differ)
    save_csv({'freq': f, 'psd': psd}, 'Fig2a_Jitter_PSD')
    save_csv({'pin_db': pin_db, 'am_am': am_am, 'scr': scr}, 'Fig2b_PA_Curves')


def fig3_dispersion():
    print("Generating Fig 3: Diffraction Dispersion...")
    phy = DiffractionChannel({'L_eff': 50e3, 'a': 0.05, 'B': 10e9})
    t = np.linspace(-0.005, 0.005, 1000)

    # 窄带 (300GHz) - 修复后的函数应该能正确输出振荡
    d_nb = np.abs(phy.generate_diffraction_pattern(t, np.array([300e9]))[0, :])
    # 宽带 (10GHz BW)
    d_wb = np.abs(phy.generate_broadband_chirp(t, N_sub=128))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t * 1000, d_nb, 'r--', label='Narrowband (300GHz)')
    ax1.plot(t * 1000, d_wb, 'b-', alpha=0.7, label='Broadband (10GHz)')
    ax1.set_title('(a) Main Lobe Alignment')
    ax1.set_ylabel('|d(t)|')
    ax1.legend()
    ax1.grid(True)

    # Zoom in
    mask = (t > 0.001) & (t < 0.003)
    t_mask = t[mask]
    d_nb_mask = d_nb[mask]
    d_wb_mask = d_wb[mask]

    ax2.plot(t_mask * 1000, d_nb_mask, 'r--', linewidth=2)
    ax2.plot(t_mask * 1000, d_wb_mask, 'b-', linewidth=2, alpha=0.8)
    ax2.set_title('(b) Side-lobe Smearing (Dispersion Evidence)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('|d(t)| (Zoomed)')
    ax2.grid(True)

    plt.tight_layout()
    save_plot('Fig3_Dispersion_Analysis')
    plt.close('all') # <--- **FIX: 显式关闭图形对象**

    save_csv({'time_ms': t * 1000, 'amp_nb': d_nb, 'amp_wb': d_wb}, 'Fig3_Dispersion_Analysis')


def fig4_self_healing():
    print("Generating Fig 4: Self-Healing Effect...")
    hw = HardwareImpairments({'beta_a': 5995.0, 'alpha_a': 10.127})
    t = np.linspace(0, 1, 1000)
    shadow = 0.05 * np.exp(-((t - 0.5) ** 2) / (0.05 ** 2))
    sig_in = 1.0 - shadow

    # Linear Case
    out_lin, _, _ = hw.apply_saleh_pa(sig_in, ibo_dB=15.0)

    # Saturated Case (Adjusted IBO to 3dB to avoid inversion artifact)
    out_sat, _, _ = hw.apply_saleh_pa(sig_in, ibo_dB=3.0)

    # Normalize for AC comparison
    def get_ac(s): return (np.abs(s) - np.mean(np.abs(s)[:100])) / np.mean(np.abs(s)[:100]) * 100

    ac_in = get_ac(sig_in)
    ac_sat = get_ac(out_sat)

    plt.figure(figsize=(8, 6))
    plt.plot(t, ac_in, 'k--', linewidth=1.5, label='Input Shadow')
    plt.plot(t, ac_sat, 'r-', linewidth=3.0, label='Saturated Output (IBO=3dB)')
    plt.title('Self-Healing: Shadow Depth Compression')
    plt.ylabel('Relative Amplitude Change (%)')
    plt.xlabel('Time (Normalized)')
    plt.legend()
    plt.grid(True)

    save_plot('Fig4_Self_Healing_Effect')
    plt.close('all') # <--- **FIX: 显式关闭图形对象**
    save_csv({'time': t, 'depth_in': ac_in, 'depth_sat': ac_sat}, 'Fig4_Self_Healing_Effect')


def fig5_survival_space():
    print("Generating Fig 5: Survival Space Spectrogram...")
    fs = 20e3
    N = int(fs * 0.1)  # 100ms
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=50e3, a=0.05)
    hw = HardwareImpairments({'jitter_rms': 5.0e-6, 'f_knee': 200.0})

    # Signal + Jitter
    d_chirp = det._generate_template(15000.0) * 10.0  # Boost signal for visibility
    sig_lin = 1.0 + d_chirp
    jitter = np.exp(hw.generate_colored_jitter(N, fs))

    z_raw = np.log(jitter * sig_lin + 1e-12)
    z_perp = det.apply_projection(z_raw)

    # Normalize for visibility
    z_perp_norm = z_perp / np.std(z_perp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Raw PSD
    f, p_raw = scipy.signal.periodogram(z_raw, fs)
    f, p_perp = scipy.signal.periodogram(z_perp, fs)

    ax1.semilogy(f, p_raw, 'r', alpha=0.5, label='Raw (Jitter)')
    ax1.semilogy(f, p_perp, 'b', alpha=0.8, label='Projected (Signal)')
    ax1.axvspan(300, 5000, color='green', alpha=0.1, label='Survival Space')
    ax1.legend()
    ax1.set_title('(a) Spectral Separation')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD')
    ax1.grid(True)

    # Spectrogram
    f_sp, t_sp, Sxx = scipy.signal.spectrogram(z_perp_norm, fs, nperseg=256, noverlap=200)

    # Manual vmin/vmax for contrast
    vmin = 10 * np.log10(np.max(Sxx)) - 50  # 50dB dynamic range
    vmax = 10 * np.log10(np.max(Sxx))

    im = ax2.pcolormesh(t_sp * 1000, f_sp, 10 * np.log10(Sxx + 1e-12),
                        shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
    ax2.set_ylim(0, 5000)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('(b) Spectrogram: Chirp Revealed in Survival Space')
    plt.colorbar(im, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    save_plot('Fig5_Survival_Space')
    plt.close('all') # <--- **FIX: 显式关闭图形对象**

    save_csv({'freq': f, 'psd_raw': p_raw, 'psd_proj': p_perp}, 'Fig5a_PSD')
    # Spectrogram data is 2D, might be too big for simple CSV, saving flattened axes
    save_csv({'time': t_sp, 'freq_bins': f_sp}, 'Fig5b_Spectrogram_Axes')


if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    # fig3_dispersion 函数虽然包含在 visualize_mechanisms.py 中，但物理特性可视化
    # 应该主要由 visualize_physics.py 完成，因此在主运行块中保留 fig2, fig4, fig5。
    fig2_hardware_characteristics()
    fig4_self_healing()
    fig5_survival_space()
    print("\n[Done] All mechanism figures generated.")
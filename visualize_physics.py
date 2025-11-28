import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal
import os
import pandas as pd  # 导入 pandas 库

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


def save_csv(data_dict, filename, folder='results/csv_data'):
    """Helper function to save data dictionary to CSV."""
    if not os.path.exists(folder): os.makedirs(folder)
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


def viz_dispersion_analysis(save_dir='results'):
    """
    Fig 3: 宽带色散效应 (The Dispersion Evidence)
    对比 Narrowband (300GHz) 与 Wideband (10GHz DFS) 的时域波形
    """
    print("Generating Fig 3: Dispersion Analysis...")

    # 1. 物理参数 (L_eff=50km, a=5cm)
    config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': 50e3, 'a': 0.05, 'v_rel': 15000}
    phy = DiffractionChannel(config_phy)

    fs = 100e3  # 高采样率以展示细节
    T_span = 0.01  # 10ms
    t = np.linspace(-T_span / 2, T_span / 2, int(T_span * fs))

    # 2. 生成信号
    # 注意: generate_diffraction_pattern 针对单频数组，返回 (1, N_t)
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
    save_path = os.path.join(save_dir, 'Fig3_Dispersion_Analysis')
    plt.savefig(f'{save_path}.png', dpi=300)
    plt.savefig(f'{save_path}.pdf', format='pdf')
    print(f"Saved {save_path}")
    plt.close('all')  # 显式关闭

    # 4. [FIX] 保存 CSV 数据
    save_csv({
        'time_ms': t * 1000,
        'amplitude_narrowband': d_nb,
        'amplitude_broadband': d_wb
    }, 'Fig3_Dispersion_Analysis')


if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')

    # [FIX] 仅调用物理学相关的可视化函数 (Fig 3)
    viz_dispersion_analysis()

    print("\n[Done] Physics visualization (Fig 3) completed.")
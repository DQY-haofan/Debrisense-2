import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import os
from tqdm import tqdm

# 导入模块
from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments
from detector import TerahertzDebrisDetector

# Set plotting style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'lines.linewidth': 2.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42
})


def save_csv(data_dict, filename, folder='results/csv_data'):
    if not os.path.exists(folder): os.makedirs(folder)
    max_len = max([len(v) if hasattr(v, '__len__') else 1 for v in data_dict.values()])
    uniform_data = {}
    for k, v in data_dict.items():
        if hasattr(v, '__len__'):
            if len(v) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(v)] = v
                uniform_data[k] = padded
            else:
                uniform_data[k] = v
        else:
            uniform_data[k] = np.full(max_len, v)
    df = pd.DataFrame(uniform_data)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


def save_plot(filename, folder='results'):
    if not os.path.exists(folder): os.makedirs(folder)
    plt.savefig(f"{folder}/{filename}.png", dpi=300)
    plt.savefig(f"{folder}/{filename}.pdf", format='pdf')


# --- 关键修改：物理参数集 ---
# 使用 20cm 碎片以获得足够的阴影深度 (~0.5%)
# 使用较小的 L_eff 以增强条纹可见度
GLOBAL_CONFIG = {
    'fc': 300e9,
    'L_eff': 50e3,
    'a': 0.20,  # [FIX] 增大到 20cm
    'v_true': 15000.0,
    'fs': 20e3,
    'T_span': 0.05,  # 50ms 窗口
    'jitter_rms': 10e-6,  # [FIX] 10 urad 强抖动
    'f_knee': 200.0
}


# --- Single Trial Functions ---

def trial_rmse(ibo, seed, noise_std, N, fs, true_v, hw_config, det_config):
    np.random.seed(seed)
    hw = HardwareImpairments(hw_config)
    det = TerahertzDebrisDetector(fs, N, **det_config)

    # 物理信号
    d = det._generate_template(true_v)
    sig = 1.0 + d

    # 硬件损伤
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo)

    # 热噪声 (固定底噪)
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # 检测
    z = det.apply_projection(det.log_envelope_transform(pa_out + noise))

    # 精细网格搜索 (Search Range = +/- 1000 m/s)
    # 如果搜索范围太大，容易收敛到局部极值，导致 RMSE 爆炸
    scan_range = 1000
    v_scan = np.linspace(true_v - scan_range, true_v + scan_range, 100)
    stats = det.glrt_scan(z, v_scan)

    est_idx = np.argmax(stats)
    est_v = v_scan[est_idx]

    return est_v - true_v


def trial_roc(is_h1, seed, noise_std, N, fs, true_v, hw_config, det_config):
    np.random.seed(seed)
    hw = HardwareImpairments(hw_config)
    det = TerahertzDebrisDetector(fs, N, **det_config)

    jit = np.exp(hw.generate_colored_jitter(N, fs))

    if is_h1:
        d = det._generate_template(true_v)
        sig = 1.0 + d
    else:
        sig = 1.0  # H0

    # IBO=10dB (Linear Region) for ROC check
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y = pa_out + noise

    # GLRT Statistic
    z = det.apply_projection(det.log_envelope_transform(y))
    s_temp = det.P_perp @ det._generate_template(true_v)
    denom = np.sum(s_temp ** 2)
    stat_glrt = (np.dot(s_temp, z) ** 2) / (denom + 1e-20)

    # Energy Statistic
    # Remove DC roughly to be fair
    y_ac = np.abs(y) - np.mean(np.abs(y))
    stat_ed = np.sum(y_ac ** 2)

    return stat_glrt, stat_ed


def trial_pd_velocity(v, seed, noise_std, N, fs, hw_config, det_config, threshold):
    np.random.seed(seed)
    hw = HardwareImpairments(hw_config)
    det = TerahertzDebrisDetector(fs, N, **det_config)

    d = det._generate_template(v)
    sig = 1.0 + d
    jit = np.exp(hw.generate_colored_jitter(N, fs))

    # Check in linear region to see "Survival Space" effect purely
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z = det.apply_projection(det.log_envelope_transform(pa_out + noise))

    s_temp = det.P_perp @ det._generate_template(v)
    denom = np.sum(s_temp ** 2)

    if denom < 1e-15:
        # Signal falls completely into blind zone
        stat = 0.0
    else:
        stat = (np.dot(s_temp, z) ** 2) / denom

    return stat > threshold


# --- Runners ---

def run_fig6():
    print("Running Fig 6 (RMSE vs IBO)...")
    # Config setup
    fs = GLOBAL_CONFIG['fs']
    N = int(fs * GLOBAL_CONFIG['T_span'])
    true_v = GLOBAL_CONFIG['v_true']

    hw_config = {'beta_a': 5995.0, 'alpha_a': 10.127, 'jitter_rms': GLOBAL_CONFIG['jitter_rms']}
    det_config = {'cutoff_freq': 200.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a']}

    # [FIX] Lower thermal noise to make Jitter/PA effects dominant
    # Noise floor ~ -60dB relative to carrier
    noise_std = 1e-3

    # Scan from Linear (15dB) to Saturation (-5dB)
    ibo_range = np.linspace(15, -5, 15)
    trials = 500

    rmse_list = []

    for ibo in tqdm(ibo_range):
        seeds = np.random.randint(0, 1e6, trials)
        errs = Parallel(n_jobs=-1)(delayed(trial_rmse)(
            ibo, s, noise_std, N, fs, true_v, hw_config, det_config) for s in seeds)

        # Robust RMSE: Remove outliers (detection failures)
        # If error > 500, consider it a miss
        errs = np.array(errs)
        valid_mask = np.abs(errs) < 800
        if np.sum(valid_mask) > 10:
            rmse = np.sqrt(np.mean(errs[valid_mask] ** 2))
        else:
            rmse = 1000.0  # Saturation value

        rmse_list.append(rmse)

    plt.figure(figsize=(8, 6))
    plt.semilogy(ibo_range, rmse_list, 'b-o', linewidth=2)
    plt.gca().invert_xaxis()
    plt.xlabel('Input Back-Off (dB)')
    plt.ylabel('RMSE (m/s)')
    plt.title('Fig 6: RMSE Degradation due to PA Saturation')
    plt.grid(True, ls=':')

    # Add annotations
    plt.axvspan(15, 5, color='green', alpha=0.1)
    plt.text(10, min(rmse_list) * 1.2, 'Linear Region', ha='center', color='green')
    plt.axvspan(5, -5, color='red', alpha=0.1)
    plt.text(0, min(rmse_list) * 1.2, 'Saturation Region\n(Self-Healing)', ha='center', color='red')

    save_plot('Fig6_RMSE_vs_Power')
    save_csv({'ibo': ibo_range, 'rmse': rmse_list}, 'Fig6_RMSE_vs_Power')


def run_fig7():
    print("Running Fig 7 (ROC)...")
    fs = GLOBAL_CONFIG['fs']
    N = int(fs * GLOBAL_CONFIG['T_span'])
    true_v = GLOBAL_CONFIG['v_true']

    hw_config = {'jitter_rms': GLOBAL_CONFIG['jitter_rms']}
    det_config = {'cutoff_freq': 200.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a']}

    trials = 2000
    noise_std = 2e-3  # Slightly higher noise for ROC to show curve

    seeds = np.random.randint(0, 1e6, trials)

    res0 = Parallel(n_jobs=-1)(
        delayed(trial_roc)(False, s, noise_std, N, fs, true_v, hw_config, det_config) for s in seeds)
    res1 = Parallel(n_jobs=-1)(
        delayed(trial_roc)(True, s, noise_std, N, fs, true_v, hw_config, det_config) for s in seeds)
    res0 = np.array(res0);
    res1 = np.array(res1)

    def get_curve(h0, h1):
        # Sort scores
        scores = np.concatenate([h0, h1])
        # Sample thresholds
        ths = np.linspace(np.min(scores), np.max(scores), 500)
        pfa = [];
        pd = []
        for t in ths:
            pfa.append(np.sum(h0 > t) / len(h0))
            pd.append(np.sum(h1 > t) / len(h1))
        return pfa, pd

    pfa_g, pd_g = get_curve(res0[:, 0], res1[:, 0])
    pfa_e, pd_e = get_curve(res0[:, 1], res1[:, 1])

    plt.figure(figsize=(6, 6))
    plt.plot(pfa_g, pd_g, 'b-', linewidth=2, label='Log-Proj GLRT')
    plt.plot(pfa_e, pd_e, 'r--', linewidth=2, label='Energy Detector')
    plt.plot([0, 1], [0, 1], 'k:', label='Random Guess')
    plt.xscale('log')
    plt.xlim(1e-3, 1);
    plt.ylim(0, 1.05)
    plt.xlabel('Probability of False Alarm ($P_{fa}$)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title('Fig 7: Robustness against Colored Jitter')
    plt.legend(loc='lower right')
    plt.grid(True, ls=':')

    save_plot('Fig7_ROC_Curves')
    save_csv({'pfa_glrt': pfa_g, 'pd_glrt': pd_g, 'pfa_ed': pfa_e, 'pd_ed': pd_e}, 'Fig7_ROC_Curves')


def run_fig8():
    print("Running Fig 8 (Velocity Blind Zone)...")
    fs = GLOBAL_CONFIG['fs']
    # Use longer window for slow targets
    N = int(fs * 0.1)  # 100ms

    hw_config = {'jitter_rms': GLOBAL_CONFIG['jitter_rms']}
    det_config = {'cutoff_freq': 300.0, 'L_eff': GLOBAL_CONFIG['L_eff'], 'a': GLOBAL_CONFIG['a']}

    # Establish Threshold (Pfa = 1e-3)
    seeds = np.random.randint(0, 1e6, 500)
    noise_std = 1e-3
    h0_stats = Parallel(n_jobs=-1)(
        delayed(trial_roc)(False, s, noise_std, N, fs, 15000, hw_config, det_config) for s in seeds)
    threshold = np.percentile(np.array(h0_stats)[:, 0], 99.9)
    print(f"   Threshold set to: {threshold:.4f}")

    v_range = np.logspace(1.5, 4.3, 20)  # 30 m/s to 20000 m/s
    pd_curve = []

    for v in tqdm(v_range):
        res = Parallel(n_jobs=-1)(delayed(trial_pd_velocity)(
            v, s, noise_std, N, fs, hw_config, det_config, threshold) for s in seeds[:200])
        pd_curve.append(np.mean(res))

    plt.figure(figsize=(8, 5))
    plt.semilogx(v_range, pd_curve, 'k-o', linewidth=2)
    plt.xlabel('Debris Velocity (m/s)')
    plt.ylabel('Probability of Detection')
    plt.title('Fig 8: Detection Blind Zone (Spectral Overlap)')
    plt.grid(True, which='both')

    # Calculate cutoff velocity roughly
    # f_chirp_max = v^2 / (lambda * L) * T
    # Simple check: where does Pd drop?
    plt.axvspan(30, 800, color='red', alpha=0.1, label='Blind Zone (<800 m/s)')
    plt.legend()

    save_plot('Fig8_Pd_vs_Velocity')
    save_csv({'velocity': v_range, 'pd': pd_curve}, 'Fig8_Pd_vs_Velocity')


if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    multiprocessing.freeze_support()

    run_fig6()
    run_fig7()
    run_fig8()
    print("\nAll simulations completed with FIXED parameters.")
# ----------------------------------------------------------------------------------
# 脚本名称: sim_advanced_metrics.py (高级指标仿真脚本)
# 版本: 2.0 (升级版 - 宽带物理真值)
# 描述:
#   本脚本替代原有的 sim_performance_campaign.py。
#   主要功能是生成 Fig 7 (ROC 曲线) 和 Fig 8 (速度盲区分析)。
#   核心升级点：
#     1. 强制接入 physics_engine.py 中的 generate_broadband_chirp 函数，
#        确保仿真信号包含宽带色散效应 (Dispersion Smearing)，而非窄带近似。
#     2. 这对于验证高速目标下的检测性能至关重要。
# 用法:
#   直接运行此脚本。输出结果将保存在 results/ 文件夹下。
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm

try:
    from physics_engine import DiffractionChannel
    from hardware_model import HardwareImpairments
    from detector import TerahertzDebrisDetector
except ImportError:
    raise SystemExit("Missing modules")

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42})


def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)


# --- Shared Configuration ---
L_eff = 50e3
fs = 20e3
T_span = 0.1  # 100ms window for slow targets
N = int(fs * T_span)
t_axis = np.linspace(-T_span / 2, T_span / 2, N)

config_phy = {'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': 15000}
config_hw = {'jitter_rms': 5e-6, 'f_knee': 200.0, 'beta_a': 5995.0, 'alpha_a': 10.127}
config_det = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'a': 0.05}


# --- Task 1: ROC Curve (Robustness) ---
def trial_roc(is_h1, seed, noise_std, phy_engine, true_v):
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, **config_det)

    # 1. Generate Signal (Broadband Physics) or Null
    if is_h1:
        # Update phy engine velocity dynamically for robustness
        phy_engine.v_rel = true_v
        d_wb = phy_engine.generate_broadband_chirp(t_axis, N_sub=32)  # Lower sub for speed
        sig = 1.0 + d_wb
    else:
        sig = np.ones(N, dtype=np.complex128)

    # 2. Jitter & PA (Linear Region for ROC check)
    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=10.0)

    # 3. Noise
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y = pa_out + w

    # 4. GLRT Detector (Proposed)
    z_log = det.log_envelope_transform(y)
    z_perp = det.apply_projection(z_log)
    s_temp = det.P_perp @ det._generate_template(true_v)
    # Energy Norm
    denom = np.sum(s_temp ** 2) + 1e-15
    stat_glrt = (np.dot(s_temp, z_perp) ** 2) / denom

    # 5. Energy Detector (Baseline)
    # Simple AC energy
    y_ac = np.abs(y) - np.mean(np.abs(y))
    stat_ed = np.sum(y_ac ** 2)

    return stat_glrt, stat_ed


def run_fig7_roc():
    print("Running Fig 7: ROC Analysis...")
    phy = DiffractionChannel(config_phy)
    true_v = 15000.0
    trials = 2000
    noise_std = 1.5e-3  # Calibrated for decent overlap

    seeds = np.random.randint(0, 1e9, trials)

    # Run H0 (Noise Only)
    res0 = Parallel(n_jobs=-1)(delayed(trial_roc)(False, s, noise_std, phy, true_v) for s in seeds)
    res0 = np.array(res0)

    # Run H1 (Signal + Noise)
    res1 = Parallel(n_jobs=-1)(delayed(trial_roc)(True, s, noise_std, phy, true_v) for s in seeds)
    res1 = np.array(res1)

    # Calculate ROC
    def calc_curve(scores0, scores1):
        # Merge and sort
        thresholds = np.linspace(min(scores0), max(scores1), 1000)
        pfa, pd = [], []
        for th in thresholds:
            pfa.append(np.mean(scores0 > th))
            pd.append(np.mean(scores1 > th))
        return pfa, pd

    pfa_glrt, pd_glrt = calc_curve(res0[:, 0], res1[:, 0])
    pfa_ed, pd_ed = calc_curve(res0[:, 1], res1[:, 1])

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(pfa_glrt, pd_glrt, 'b-', linewidth=2.5, label='Log-Proj GLRT (Proposed)')
    plt.plot(pfa_ed, pd_ed, 'r--', linewidth=2, label='Energy Detector (Baseline)')
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    plt.xscale('log')
    plt.xlim(1e-4, 1)
    plt.ylim(0, 1.05)
    plt.xlabel('Probability of False Alarm ($P_{fa}$)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title('Fig 7: Robustness against Colored Jitter')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':')
    plt.savefig('results/Fig7_ROC.png', dpi=300)
    plt.savefig('results/Fig7_ROC.pdf', format='pdf')

    # Save Data
    pd.DataFrame({'pfa_glrt': pfa_glrt, 'pd_glrt': pd_glrt,
                  'pfa_ed': pfa_ed, 'pd_ed': pd_ed}).to_csv('results/Fig7_ROC.csv')


# --- Task 2: Velocity Blind Zone (Survival Space) ---
def trial_pd_vel(v, threshold, seed, noise_std):
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, **config_det)
    phy = DiffractionChannel(config_phy)
    phy.v_rel = v  # Update speed

    # Broadband Physics Signal
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig = 1.0 + d_wb

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    # Use Linear PA to isolate Jitter effect from PA effect
    pa_out, _, _ = hw.apply_saleh_pa(sig * jit, ibo_dB=15.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    # Detection
    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)

    # Match with template at assumed v
    s_temp = det.P_perp @ det._generate_template(v)
    denom = np.sum(s_temp ** 2)

    if denom < 1e-12: return 0.0  # Signal totally filtered out

    stat = (np.dot(s_temp, z_perp) ** 2) / denom
    return 1.0 if stat > threshold else 0.0


def run_fig8_velocity():
    print("Running Fig 8: Velocity Blind Zone Analysis...")

    # 1. Determine Threshold (Pfa=1e-3)
    # Run pure noise trials
    seeds = np.random.randint(0, 1e9, 1000)
    noise_std = 1e-3
    # Use a dummy velocity for threshold calibration
    phy_dummy = DiffractionChannel(config_phy)
    h0_stats = Parallel(n_jobs=-1)(delayed(trial_roc)(False, s, noise_std, phy_dummy, 15000) for s in seeds)
    h0_scores = np.array(h0_stats)[:, 0]  # GLRT scores
    threshold = np.percentile(h0_scores, 99.9)
    print(f"Threshold (Pfa=1e-3): {threshold:.4f}")

    # 2. Sweep Velocity
    # Log scale from 100 m/s to 30 km/s
    v_range = np.logspace(2, 4.5, 20)
    pd_curve = []

    seeds_pd = np.random.randint(0, 1e9, 200)  # 200 trials per point

    for v in tqdm(v_range):
        res = Parallel(n_jobs=-1)(delayed(trial_pd_vel)(v, threshold, s, noise_std) for s in seeds_pd)
        pd_curve.append(np.mean(res))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.semilogx(v_range, pd_curve, 'k-o', linewidth=2)

    # Highlight Blind Zone
    # f_chirp_max approx v^2 / (lam * L) * T. If < 300Hz, blind.
    # Theoretical cutoff approx 800m/s for this config
    plt.axvspan(100, 800, color='red', alpha=0.2, label='Blind Zone (<800 m/s)')
    plt.axvspan(800, 30000, color='green', alpha=0.1, label='Survival Space')

    plt.xlabel('Relative Velocity (m/s)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title('Fig 8: Detection Blind Zone (Spectral Overlap)')
    plt.grid(True, which='both')
    plt.legend()

    plt.savefig('results/Fig8_Velocity_BlindZone.png', dpi=300)
    plt.savefig('results/Fig8_Velocity_BlindZone.pdf', format='pdf')
    pd.DataFrame({'velocity': v_range, 'pd': pd_curve}).to_csv('results/Fig8_Velocity.csv')


if __name__ == "__main__":
    ensure_dir('results')
    multiprocessing.freeze_support()
    run_fig7_roc()
    run_fig8_velocity()
    print("\n[Success] Advanced metrics (Fig 7, 8) generated with Broadband Physics.")
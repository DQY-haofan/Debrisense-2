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


def save_csv(data_dict, filename, folder='results/csv_data'):
    if not os.path.exists(folder): os.makedirs(folder)
    df = pd.DataFrame(data_dict)
    df.to_csv(f"{folder}/{filename}.csv", index=False)
    print(f"   [Data] Saved {filename}.csv")


# --- Config ---
L_eff = 50e3
fs = 200e3
T_span = 0.02
N = int(fs * T_span)
t_axis = np.linspace(-T_span / 2, T_span / 2, N)
true_v = 15000.0

config_hw = {
    'jitter_rms': 0.5e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0
}
config_det_base = {'cutoff_freq': 300.0, 'L_eff': L_eff, 'N_sub': 32}  # [FIX] Sync N_sub

num_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cores - 2)


# --- Kernel Functions ---

def run_trial_stat(is_h1, a_val, seed, noise_std):
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=a_val, **config_det_base)

    if is_h1:
        phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': true_v})
        # [FIX] Physics uses N_sub=32
        d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
        sig = 1.0 - d_wb
    else:
        sig = np.ones(N, dtype=np.complex128)

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_in = sig * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=10.0)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)

    z_log = det.log_envelope_transform(pa_out + w)
    z_perp = det.apply_projection(z_log)
    s_temp = det.P_perp @ det._generate_template(true_v)

    energy = np.sum(s_temp ** 2)
    if energy < 1e-20: return 0.0

    stat = (np.dot(s_temp, z_perp) ** 2) / energy
    return stat


def run_fig8_mds():
    print("\n=== Fig 8: MDS (Matched Filter V4) ===")

    # 极低噪声以展示物理极限
    noise_std = 1.0e-6

    diameters_mm = np.logspace(0.3, 1.7, 15)
    radii_m = diameters_mm / 2000.0

    pd_curve = []

    for a_val in tqdm(radii_m):
        seeds_h0 = np.random.randint(0, 1e9, 200)
        stats_h0 = Parallel(n_jobs=n_jobs)(
            delayed(run_trial_stat)(False, a_val, s, noise_std) for s in seeds_h0
        )
        threshold = np.percentile(stats_h0, 99.0)

        seeds_h1 = np.random.randint(0, 1e9, 100)
        stats_h1 = Parallel(n_jobs=n_jobs)(
            delayed(run_trial_stat)(True, a_val, s, noise_std) for s in seeds_h1
        )

        pd_val = np.mean(np.array(stats_h1) > threshold)
        pd_curve.append(pd_val)

    plt.figure(figsize=(8, 6))
    plt.semilogx(diameters_mm, pd_curve, 'b-o', linewidth=2)
    plt.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Debris Diameter (mm)')
    plt.ylabel('Probability of Detection ($P_d$)')
    plt.title(f'Fig 8: Detection Capability (Correlation=1.0)')
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig('results/Fig8_Detection_Capability.png', dpi=300)
    plt.savefig('results/Fig8_Detection_Capability.pdf', format='pdf')
    save_csv({'diameter_mm': diameters_mm, 'pd': pd_curve}, 'Fig8_MDS_Data')


def run_isac_trial(ibo, seed, noise_std):
    np.random.seed(seed)
    hw = HardwareImpairments(config_hw)
    det = TerahertzDebrisDetector(fs, N, a=0.05, **config_det_base)
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': 0.05, 'v_rel': true_v})

    # [FIX] N_sub=32 match
    d_wb = phy.generate_broadband_chirp(t_axis, N_sub=32)
    sig = 1.0 - d_wb

    jit = np.exp(hw.generate_colored_jitter(N, fs))
    pn = np.exp(1j * hw.generate_phase_noise(N, fs))

    pa_in = sig * jit * pn
    pa_out, _, _ = hw.apply_saleh_pa(pa_in, ibo_dB=ibo)

    p_rx = np.mean(np.abs(pa_out) ** 2)
    snr_lin = p_rx / (noise_std ** 2)
    capacity = np.log2(1 + snr_lin)

    w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
    y_rx = pa_out + w
    z_log = det.log_envelope_transform(y_rx)
    z_perp = det.apply_projection(z_log)

    v_scan = np.linspace(true_v - 1500, true_v + 1500, 101)
    stats = []
    for v in v_scan:
        s_raw = det._generate_template(v)
        s_perp = det.P_perp @ s_raw
        denom = np.sum(s_perp ** 2) + 1e-20
        stats.append((np.dot(s_perp, z_perp) ** 2) / denom)

    est_v = v_scan[np.argmax(stats)]
    err_v = abs(est_v - true_v)

    return capacity, err_v


def run_fig9_isac():
    print("\n=== Fig 9: ISAC Trade-off ===")
    noise_std = 1.0e-5
    ibo_points = np.linspace(20, -5, 15)

    avg_cap, avg_rmse = [], []

    for ibo in tqdm(ibo_points):
        res = Parallel(n_jobs=n_jobs)(
            delayed(run_isac_trial)(ibo, s, noise_std) for s in range(50)
        )
        res = np.array(res)
        avg_cap.append(np.mean(res[:, 0]))
        errs = res[:, 1]
        valid = errs[errs < 1400]
        rmse = np.sqrt(np.mean(valid ** 2)) if len(valid) > 5 else 1500.0
        avg_rmse.append(rmse)

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(avg_cap, avg_rmse, c=ibo_points, cmap='viridis_r', s=100, edgecolors='k')
    plt.colorbar(sc, label='Input Back-Off (dB)')
    plt.xlabel('Capacity (bits/s/Hz)')
    plt.ylabel('RMSE (m/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/Fig9_ISAC_Tradeoff.png')
    plt.savefig('results/Fig9_ISAC_Tradeoff.pdf')
    save_csv({'ibo': ibo_points, 'capacity': avg_cap, 'rmse': avg_rmse}, 'Fig9_ISAC_Data')


if __name__ == "__main__":
    ensure_dir('results/csv_data')
    multiprocessing.freeze_support()
    run_fig8_mds()
    run_fig9_isac()
    print("\n[Done] All advanced metrics generated.")
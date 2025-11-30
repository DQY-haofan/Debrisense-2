import numpy as np
import matplotlib.pyplot as plt
from physics_engine import DiffractionChannel
from detector import TerahertzDebrisDetector


def debug_mismatch():
    print("=== Template vs Signal Mismatch Check ===")

    # Config
    fs = 200e3
    N = 4000  # 20ms
    t = np.linspace(-0.01, 0.01, N)
    L_eff = 50e3
    a_val = 0.05  # 5cm radius
    v_true = 15000.0

    # 1. Physics Signal (The Truth)
    phy = DiffractionChannel({'fc': 300e9, 'B': 10e9, 'L_eff': L_eff, 'a': a_val, 'v_rel': v_true})
    d_broadband = phy.generate_broadband_chirp(t, N_sub=32)

    # Signal is the shadow (Real part drop)
    # y = 1 - d. We are detecting 'd'.
    sig_physics = np.real(d_broadband)

    # 2. Detector Template (The Guess)
    det = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0, L_eff=L_eff, a=a_val)
    s_template = det._generate_template(v_true)  # This is -Re(d) usually

    # 3. Projection Effect
    sig_proj = det.apply_projection(sig_physics)
    temp_proj = det.apply_projection(s_template)

    # Normalize for comparison
    sig_norm = sig_proj / np.max(np.abs(sig_proj))
    temp_norm = temp_proj / np.max(np.abs(temp_proj))

    # Check Correlation
    corr = np.dot(sig_norm, temp_norm) / np.sqrt(np.sum(sig_norm ** 2) * np.sum(temp_norm ** 2))
    print(f"Correlation Coefficient: {corr:.4f}")

    if abs(corr) < 0.8:
        print("[FAIL] Severe Mismatch! Check Frequency/Phase definitions.")
    else:
        print("[PASS] Template matches Physics.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t * 1000, sig_norm, 'b-', label='Physics Signal (Projected)', linewidth=2, alpha=0.7)
    plt.plot(t * 1000, temp_norm, 'r--', label='Detector Template (Projected)', linewidth=1.5)
    plt.title(f"Waveform Alignment Check (Corr={corr:.4f})")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/debug/Mismatch_Check.png')
    print("Saved results/debug/Mismatch_Check.png")


if __name__ == "__main__":
    import os

    if not os.path.exists('results/debug'): os.makedirs('results/debug')
    debug_mismatch()
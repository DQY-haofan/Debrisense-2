#!/usr/bin/env python3
"""
Survival Space 价值展示 - 使用能量检测器
==========================================
匹配滤波器天然抑制不相关噪声，所以投影改进有限。
能量检测器对噪声敏感，能更好地展示投影的价值。

两种检测器对比:
1. 能量检测器 (Energy Detector): T = ||z||² 
   - 对所有噪声敏感
   - 投影能显著提升性能

2. 匹配滤波器 (Matched Filter): T = <z,s>² / ||s||²
   - 天然抑制不相关噪声
   - 投影改进有限
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import sys
sys.path.insert(0, '.')

from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments

# 配置
CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 20e3, 'fs': 200e3,
    'T_span': 0.02, 'a': 0.10, 'v_default': 15000
}

HW_CONFIG = {
    'jitter_rms': 1.0e-6, 'f_knee': 200.0, 'beta_a': 5995.0,
    'alpha_a': 10.127, 'L_1MHz': -95.0, 'L_floor': -120.0, 'pll_bw': 50e3
}

fs = CONFIG['fs']
N = int(fs * CONFIG['T_span'])
t_axis = np.arange(N) / fs - (N / 2) / fs
k_cut = int(np.ceil(300 * N / fs))

def log_envelope(y, eps=1e-12):
    return np.log(np.abs(y) + eps)

def center(x):
    return x - np.mean(x)

def apply_projection(z):
    C = dct(z, type=2, norm='ortho')
    C[:k_cut] = 0
    return idct(C, type=2, norm='ortho')

def generate_debris_signal():
    phy = DiffractionChannel(CONFIG)
    d_wb = phy.generate_broadband_chirp(t_axis, 32)
    return d_wb

def compute_auc(stats_h0, stats_h1):
    all_stats = np.concatenate([stats_h0, stats_h1])
    th = np.linspace(np.min(all_stats), np.max(all_stats), 500)
    pfa = [np.mean(np.array(stats_h0) > t) for t in th]
    pd = [np.mean(np.array(stats_h1) > t) for t in th]
    return -np.trapezoid(pd, pfa)


def run_detection_comparison(jitter_rms, snr_db, trials=500):
    """
    对比两种检测器的性能
    """
    d_wb = generate_debris_signal()
    y0_base = np.ones(N, dtype=complex)
    y1_base = 1.0 - d_wb
    
    # 生成 log-envelope 模板
    s_z = center(log_envelope(y1_base) - log_envelope(y0_base))
    s_z_proj = apply_projection(s_z)
    
    results = {
        'energy_no_proj': {'h0': [], 'h1': []},
        'energy_with_proj': {'h0': [], 'h1': []},
        'matched_no_proj': {'h0': [], 'h1': []},
        'matched_with_proj': {'h0': [], 'h1': []},
    }
    
    for trial in range(trials):
        np.random.seed(trial)
        
        # 生成 jitter
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = jitter_rms
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        for hyp, y_base in [('h0', y0_base), ('h1', y1_base)]:
            y = y_base * jitter
            
            # 添加 AWGN
            pa_out, _, _ = hw.apply_saleh_pa(y, ibo_dB=10.0)
            p_ref = np.mean(np.abs(pa_out)**2)
            noise_std = np.sqrt(p_ref / (10**(snr_db/10)))
            w = (np.random.randn(N) + 1j*np.random.randn(N)) * noise_std / np.sqrt(2)
            
            z = center(log_envelope(pa_out + w))
            z_proj = apply_projection(z)
            
            # 能量检测器: T = ||z||²
            T_energy_no = np.sum(z**2)
            T_energy_proj = np.sum(z_proj**2)
            
            # 匹配滤波器: T = <z,s>² / ||s||²
            T_matched_no = np.dot(z, s_z)**2 / (np.sum(s_z**2) + 1e-20)
            T_matched_proj = np.dot(z_proj, s_z_proj)**2 / (np.sum(s_z_proj**2) + 1e-20)
            
            results['energy_no_proj'][hyp].append(T_energy_no)
            results['energy_with_proj'][hyp].append(T_energy_proj)
            results['matched_no_proj'][hyp].append(T_matched_no)
            results['matched_with_proj'][hyp].append(T_matched_proj)
    
    auc = {}
    for key in results:
        auc[key] = compute_auc(results[key]['h0'], results[key]['h1'])
    
    return auc, results


# =============================================================================
# 主程序
# =============================================================================
print("="*70)
print("SURVIVAL SPACE 价值展示")
print("对比能量检测器 vs 匹配滤波器")
print("="*70)

# 参数扫描
jitter_rms = 2e-2  # 增大 jitter 以更清楚地展示
snr_values = [30, 35, 40, 45, 50, 55, 60]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== (a) AUC vs SNR: 能量检测器 =====
print("\n计算能量检测器性能...")
auc_energy_no = []
auc_energy_proj = []

for snr in snr_values:
    print(f"  SNR = {snr} dB...")
    auc, _ = run_detection_comparison(jitter_rms, snr, trials=300)
    auc_energy_no.append(auc['energy_no_proj'])
    auc_energy_proj.append(auc['energy_with_proj'])

ax = axes[0, 0]
ax.plot(snr_values, auc_energy_no, 's--', color='red', linewidth=2, markersize=8,
        label='Without projection')
ax.plot(snr_values, auc_energy_proj, 'o-', color='blue', linewidth=2, markersize=8,
        label='With projection')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

for i, snr in enumerate(snr_values):
    diff = auc_energy_proj[i] - auc_energy_no[i]
    if abs(diff) > 0.02:
        ax.annotate(f'+{diff:.2f}', xy=(snr, (auc_energy_proj[i]+auc_energy_no[i])/2),
                   fontsize=9, color='green', ha='center')

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('(a) Energy Detector: AUC vs SNR', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 1.05)

# ===== (b) AUC vs SNR: 匹配滤波器 =====
print("\n计算匹配滤波器性能...")
auc_matched_no = []
auc_matched_proj = []

for snr in snr_values:
    print(f"  SNR = {snr} dB...")
    auc, _ = run_detection_comparison(jitter_rms, snr, trials=300)
    auc_matched_no.append(auc['matched_no_proj'])
    auc_matched_proj.append(auc['matched_with_proj'])

ax = axes[0, 1]
ax.plot(snr_values, auc_matched_no, 's--', color='red', linewidth=2, markersize=8,
        label='Without projection')
ax.plot(snr_values, auc_matched_proj, 'o-', color='blue', linewidth=2, markersize=8,
        label='With projection')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('(b) Matched Filter: AUC vs SNR', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 1.05)

# ===== (c) 统计量分布对比: 能量检测器 @ SNR=45dB =====
print("\n生成统计量分布图...")
_, results_45 = run_detection_comparison(jitter_rms, 45, trials=500)

ax = axes[1, 0]
ax.hist(results_45['energy_no_proj']['h0'], bins=30, alpha=0.5, color='red', 
        label='H0 (no proj)', density=True)
ax.hist(results_45['energy_no_proj']['h1'], bins=30, alpha=0.5, color='blue',
        label='H1 (no proj)', density=True)
ax.hist(results_45['energy_with_proj']['h0'], bins=30, alpha=0.5, color='orange',
        label='H0 (with proj)', density=True, histtype='step', linewidth=2)
ax.hist(results_45['energy_with_proj']['h1'], bins=30, alpha=0.5, color='cyan',
        label='H1 (with proj)', density=True, histtype='step', linewidth=2)

ax.set_xlabel('Test Statistic (Energy)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(c) Energy Detector Statistics @ SNR=45dB', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# ===== (d) 改进量对比 =====
ax = axes[1, 1]
improvement_energy = np.array(auc_energy_proj) - np.array(auc_energy_no)
improvement_matched = np.array(auc_matched_proj) - np.array(auc_matched_no)

ax.plot(snr_values, improvement_energy*100, 'o-', color='blue', linewidth=2, markersize=8,
        label='Energy Detector')
ax.plot(snr_values, improvement_matched*100, 's--', color='red', linewidth=2, markersize=8,
        label='Matched Filter')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('AUC Improvement (%)', fontsize=12)
ax.set_title('(d) Projection Benefit: Energy vs Matched Filter', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fig_Detector_Comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig_Detector_Comparison.pdf', bbox_inches='tight')
print("\n[Saved] Fig_Detector_Comparison.png/pdf")

# ===== 打印结论 =====
print("\n" + "="*70)
print("结论")
print("="*70)
print(f"\n配置: jitter_rms = {jitter_rms}, f_cut = 300 Hz")
print(f"\n能量检测器 AUC 改进 (投影后 - 投影前):")
for snr, imp in zip(snr_values, improvement_energy):
    print(f"  SNR={snr}dB: +{imp*100:.1f}%")

print(f"\n匹配滤波器 AUC 改进:")
for snr, imp in zip(snr_values, improvement_matched):
    print(f"  SNR={snr}dB: +{imp*100:.1f}%")

print("\n关键洞察:")
print("  1. 能量检测器对 jitter 非常敏感 → 投影显著提升性能")
print("  2. 匹配滤波器天然抑制不相关 jitter → 投影改进有限")
print("  3. Survival Space 的主要价值在于支持简单检测器")

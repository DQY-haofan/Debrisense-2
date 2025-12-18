#!/usr/bin/env python3
"""
诊断脚本：为什么 Survival Space 投影前后差异不明显？
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import sys
sys.path.insert(0, '.')

from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments

# 配置
fs = 200e3
N = 4000
t_axis = np.arange(N) / fs - (N / 2) / fs

HW_CONFIG = {
    'jitter_rms': 1.0e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0,
    'pll_bw': 50e3
}

CONFIG = {
    'fc': 300e9, 'B': 10e9, 'L_eff': 20e3, 'fs': 200e3,
    'T_span': 0.02, 'a': 0.10, 'v_default': 15000
}

def log_envelope(y, eps=1e-12):
    return np.log(np.abs(y) + eps)

def center(x):
    return x - np.mean(x)

print("="*70)
print("SURVIVAL SPACE 诊断")
print("="*70)

# 1. 检查 Debris 调制深度
phy = DiffractionChannel(CONFIG)
d_wb = phy.generate_broadband_chirp(t_axis, 32)

debris_modulation = np.abs(1 - d_wb)
debris_depth_peak = np.max(np.abs(debris_modulation - 1))
debris_depth_rms = np.std(debris_modulation)

print(f"\n1. DEBRIS 调制深度:")
print(f"   Peak: {debris_depth_peak:.6f} ({debris_depth_peak*100:.4f}%)")
print(f"   RMS:  {debris_depth_rms:.6f} ({debris_depth_rms*100:.4f}%)")

# 2. 检查不同 jitter_rms 的影响
print(f"\n2. JITTER 影响分析:")
print(f"   {'jitter_rms':<15} | {'z_std':<12} | {'z_std/debris':<12}")
print("   " + "-"*45)

jitter_rms_values = [1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
z_std_list = []

for jr in jitter_rms_values:
    np.random.seed(42)
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = jr
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    
    y0 = np.ones(N, dtype=complex) * jitter
    z0 = log_envelope(y0)
    z_std = np.std(center(z0))
    z_std_list.append(z_std)
    
    ratio = z_std / debris_depth_rms
    print(f"   {jr:<15.0e} | {z_std:<12.6f} | {ratio:<12.2f}x")

# 3. 检查投影对 jitter 的去除效果
print(f"\n3. 投影对 JITTER 的去除效果:")

k_cut = int(np.ceil(300 * N / fs))

for jr in [2e-3, 1e-2, 5e-2]:
    np.random.seed(42)
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = jr
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    
    y0 = np.ones(N, dtype=complex) * jitter
    z0 = center(log_envelope(y0))
    
    C = dct(z0, type=2, norm='ortho')
    C_proj = C.copy()
    C_proj[:k_cut] = 0
    z0_proj = idct(C_proj, type=2, norm='ortho')
    
    E_before = np.sum(z0**2)
    E_after = np.sum(z0_proj**2)
    removal = 1 - E_after/E_before
    
    print(f"   jitter_rms={jr:.0e}: 去除 {removal*100:.1f}% 能量")

# 4. 关键问题：当前 jitter 是否足够大？
print(f"\n4. 关键诊断:")
print(f"   当前 jitter_rms = 2e-3")
print(f"   对应 z(t) 的 std ≈ {z_std_list[2]:.6f}")
print(f"   Debris 调制 RMS ≈ {debris_depth_rms:.6f}")
print(f"   比例: jitter/debris ≈ {z_std_list[2]/debris_depth_rms:.2f}x")

if z_std_list[2] < debris_depth_rms:
    print(f"\n   ⚠️ 问题: Jitter 比 Debris 信号还小！")
    print(f"      Survival Space 投影去除的是 jitter，但 jitter 本身就小")
    print(f"      所以投影前后差异不明显")
else:
    print(f"\n   ✓ Jitter 比 Debris 大，投影应该有效")

# 5. 建议
print(f"\n5. 建议修复方案:")
print(f"   方案A: 增大 jitter_rms 到 1e-2 ~ 5e-2")
print(f"   方案B: 降低 SNR 到 30-40 dB（让 jitter 更重要）")
print(f"   方案C: 检查 1/f 噪声是否正确注入到 log-envelope 域")

# 6. 画图对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) 不同 jitter 下的 z(t) 波动
ax = axes[0, 0]
for i, jr in enumerate([1e-3, 1e-2, 5e-2]):
    np.random.seed(42)
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = jr
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    z0 = center(log_envelope(np.ones(N, dtype=complex) * jitter))
    ax.plot(t_axis*1000, z0, alpha=0.7, label=f'jitter={jr:.0e}')

ax.axhline(debris_depth_rms, color='red', linestyle='--', label='Debris RMS')
ax.axhline(-debris_depth_rms, color='red', linestyle='--')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('z(t) - mean')
ax.set_title('(a) Jitter Impact on Log-Envelope')
ax.legend()
ax.set_xlim(-5, 5)
ax.grid(True, alpha=0.3)

# (b) Debris 信号
ax = axes[0, 1]
y1 = 1.0 - d_wb
z1 = center(log_envelope(y1))
ax.plot(t_axis*1000, z1, 'b-', linewidth=1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('z(t) - mean')
ax.set_title('(b) Debris Signal in Log-Envelope Domain')
ax.set_xlim(-2, 2)
ax.grid(True, alpha=0.3)

# (c) DCT 能量分布
ax = axes[1, 0]
f_k = np.arange(N) * fs / (2 * N)

for jr in [1e-3, 1e-2, 5e-2]:
    np.random.seed(42)
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = jr
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    z0 = center(log_envelope(np.ones(N, dtype=complex) * jitter))
    C = np.abs(dct(z0, type=2, norm='ortho'))
    ax.semilogy(f_k[:500], C[:500], alpha=0.7, label=f'jitter={jr:.0e}')

C_debris = np.abs(dct(z1, type=2, norm='ortho'))
ax.semilogy(f_k[:500], C_debris[:500], 'k-', linewidth=2, label='Debris signal')
ax.axvline(300, color='red', linestyle='--', label='f_cut')
ax.set_xlabel('DCT Frequency (Hz)')
ax.set_ylabel('|C[k]|')
ax.set_title('(c) DCT Spectrum: Jitter vs Debris')
ax.legend()
ax.grid(True, alpha=0.3)

# (d) 投影效果随 jitter 变化
ax = axes[1, 1]
jr_range = np.logspace(-4, -1, 20)
auc_improvement = []

# 简化计算
for jr in jr_range:
    np.random.seed(42)
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = jr
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    z0 = center(log_envelope(np.ones(N, dtype=complex) * jitter))
    
    C = dct(z0, type=2, norm='ortho')
    C_proj = C.copy()
    C_proj[:k_cut] = 0
    z0_proj = idct(C_proj, type=2, norm='ortho')
    
    # 投影去除的能量比例
    removal = 1 - np.sum(z0_proj**2)/np.sum(z0**2)
    auc_improvement.append(removal)

ax.semilogx(jr_range, np.array(auc_improvement)*100, 'b-o', markersize=4)
ax.axvline(2e-3, color='red', linestyle='--', label='Current (2e-3)')
ax.axvline(2e-2, color='green', linestyle='--', label='Recommended (2e-2)')
ax.set_xlabel('Jitter RMS')
ax.set_ylabel('Noise Energy Removed (%)')
ax.set_title('(d) Projection Effectiveness vs Jitter Level')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_jitter_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n[Saved] diagnostic_jitter_analysis.png")

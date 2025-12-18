#!/usr/bin/env python3
"""
Survival Space Validation - FINAL VERSION
==========================================
解决专家指出的所有核心问题：

A) 模板定义对齐：同时实现 deterministic 和 expected 模板，并验证一致性
B) Summary数字一致：固定 ρ_noise 为实际计算值
C) DCT频率映射：明确定义 f_k = k * fs / (2N)
D) AUC性能曲线：添加 AUC vs SNR 对照图

Author: THz-ISL Paper Team
Date: 2024 (Final Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
import os
import sys
import hashlib
import yaml
from datetime import datetime

sys.path.insert(0, '.')

from physics_engine import DiffractionChannel
from hardware_model import HardwareImpairments

# =============================================================================
# 配置
# =============================================================================
CONFIG = {
    # 物理参数
    'fc': 300e9,
    'B': 10e9,
    'L_eff': 20e3,
    'fs': 200e3,
    'T_span': 0.02,
    'a': 0.10,
    'v_default': 15000,
    
    # 噪声/硬件参数
    'f_knee': 200.0,
    'jitter_rms': 2.0e-3,
    'ibo_dB': 10.0,
    
    # 数值参数
    'eps_log': 1e-12,
    'delta_stat': 1e-20,
    
    # Monte Carlo 参数
    'M_expected': 200,      # 期望模板的MC次数
    'trials_auc': 500,      # AUC计算的MC次数
    'master_seed': 42,
}

HW_CONFIG = {
    'jitter_rms': 1.0e-6,
    'f_knee': 200.0,
    'beta_a': 5995.0,
    'alpha_a': 10.127,
    'L_1MHz': -95.0,
    'L_floor': -120.0,
    'pll_bw': 50e3
}

# 物理常数
c = 3e8
wavelength = c / CONFIG['fc']

# 采样参数
fs = CONFIG['fs']
N = int(fs * CONFIG['T_span'])
t_axis = np.arange(N) / fs - (N / 2) / fs

# 输出目录
OUTPUT_DIR = "survival_space_FINAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 核心函数
# =============================================================================
def log_envelope(y, eps=None):
    """Log-envelope 变换: z[n] = log(|y[n]| + ε)"""
    if eps is None:
        eps = CONFIG['eps_log']
    return np.log(np.abs(y) + eps)


def center(x):
    """去均值"""
    return x - np.mean(x)


def dct_frequency_mapping(k, N, fs):
    """
    DCT 频率映射 (C) 的明确定义
    
    f_k = k * fs / (2N)
    
    这是 DCT-II 基函数的等效频率映射，用于可视化和 cutoff 选择。
    """
    return k * fs / (2 * N)


def build_projection_operator(f_cut, N, fs):
    """
    构建 survival-space 投影算子 P_perp
    
    P_perp = I - H H^T
    
    其中 H 包含 DCT 低频基向量 (f < f_cut)
    """
    k_max = int(np.ceil(f_cut * 2 * N / fs))
    return k_max


def apply_projection(z, k_max):
    """应用 DCT 投影"""
    C = dct(z, type=2, norm='ortho')
    C_proj = C.copy()
    C_proj[:k_max] = 0
    return idct(C_proj, type=2, norm='ortho'), C, C_proj


def fresnel_radius(wavelength, L_eff):
    return np.sqrt(wavelength * L_eff)


def signal_bandwidth(v_rel, r_F):
    return v_rel / (2 * r_F)


# =============================================================================
# [A] 模板生成 - 同时实现 deterministic 和 expected
# =============================================================================
def generate_debris_signal(v_rel):
    """生成干净的 debris 调制信号 d_wb"""
    config = {**CONFIG, 'v_default': v_rel}
    phy = DiffractionChannel(config)
    d_wb = phy.generate_broadband_chirp(t_axis, 32)
    return d_wb


def build_sz_deterministic(v_rel):
    """
    [A-1] 确定性模板 (nominal, noise-free, jitter-free)
    
    s_z^det = center(z1_clean - z0_clean)
    
    其中:
    - y0_clean = 1 (H0: no debris)
    - y1_clean = 1 - d_wb (H1: debris present)
    """
    d_wb = generate_debris_signal(v_rel)
    
    y0_clean = np.ones(N, dtype=complex)
    y1_clean = 1.0 - d_wb
    
    z0_clean = log_envelope(y0_clean)
    z1_clean = log_envelope(y1_clean)
    
    s_z_det = center(z1_clean - z0_clean)
    
    return s_z_det


def build_sz_expected(v_rel, snr_db, M=None, seed=None):
    """
    [A-2] 期望模板 (Monte-Carlo average with hardware/noise)
    
    s_z^exp = center( E[z_H1 - z_H0] )
    
    用 M 次 Monte-Carlo 估计期望
    """
    if M is None:
        M = CONFIG['M_expected']
    if seed is None:
        seed = CONFIG['master_seed']
    
    d_wb = generate_debris_signal(v_rel)
    
    y0_base = np.ones(N, dtype=complex)
    y1_base = 1.0 - d_wb
    
    z_diff_sum = np.zeros(N)
    
    for m in range(M):
        # 派生子种子
        sub_seed = int(hashlib.md5(f"{seed}_{m}".encode()).hexdigest()[:8], 16) % (2**31)
        np.random.seed(sub_seed)
        
        # 生成 jitter
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = CONFIG['jitter_rms']
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        # H0 with jitter + noise
        y0_m = y0_base * jitter
        pa_out_0, _, _ = hw.apply_saleh_pa(y0_m, ibo_dB=CONFIG['ibo_dB'])
        p_ref = np.mean(np.abs(pa_out_0) ** 2)
        noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
        w0 = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
        z0_m = log_envelope(pa_out_0 + w0)
        
        # H1 with jitter + noise (same jitter realization)
        np.random.seed(sub_seed)  # 重置以获得相同jitter
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = CONFIG['jitter_rms']
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        y1_m = y1_base * jitter
        pa_out_1, _, _ = hw.apply_saleh_pa(y1_m, ibo_dB=CONFIG['ibo_dB'])
        w1 = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
        z1_m = log_envelope(pa_out_1 + w1)
        
        z_diff_sum += (z1_m - z0_m)
    
    s_z_exp = center(z_diff_sum / M)
    
    return s_z_exp


# =============================================================================
# 能量指标计算
# =============================================================================
def compute_eta(s_z, f_cut):
    """计算能量保留率 η_z = ||P_perp s_z||^2 / ||s_z||^2"""
    k_max = build_projection_operator(f_cut, N, fs)
    s_z_proj, _, _ = apply_projection(s_z, k_max)
    
    E_raw = np.sum(s_z ** 2)
    E_proj = np.sum(s_z_proj ** 2)
    
    return E_proj / E_raw if E_raw > 1e-20 else 0.0


def compute_rho_noise(f_cut, n_trials=100, seed=None):
    """
    计算噪声能量去除比例 ρ_noise
    
    定义：用 H0 的 z[n] 去均值作为 "噪声"
    ρ_noise = 1 - ||P_perp n_center||^2 / ||n_center||^2
    """
    if seed is None:
        seed = CONFIG['master_seed']
    
    k_max = build_projection_operator(f_cut, N, fs)
    
    y0_base = np.ones(N, dtype=complex)
    
    E_total = 0
    E_proj = 0
    
    for trial in range(n_trials):
        sub_seed = int(hashlib.md5(f"{seed}_noise_{trial}".encode()).hexdigest()[:8], 16) % (2**31)
        np.random.seed(sub_seed)
        
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = CONFIG['jitter_rms']
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        y0_m = y0_base * jitter
        pa_out, _, _ = hw.apply_saleh_pa(y0_m, ibo_dB=CONFIG['ibo_dB'])
        z0 = log_envelope(pa_out)
        
        n_center = center(z0)
        n_proj, _, _ = apply_projection(n_center, k_max)
        
        E_total += np.sum(n_center ** 2)
        E_proj += np.sum(n_proj ** 2)
    
    rho = 1 - E_proj / E_total if E_total > 1e-20 else 0.0
    return rho


# =============================================================================
# [D] 检测统计量与 AUC 计算
# =============================================================================
def compute_test_statistic(z, s_z, use_projection, f_cut):
    """
    计算归一化匹配统计量
    
    T = <z_used, s_used>^2 / (||s_used||^2 + δ)
    """
    k_max = build_projection_operator(f_cut, N, fs)
    
    if use_projection:
        z_used, _, _ = apply_projection(center(z), k_max)
        s_used, _, _ = apply_projection(s_z, k_max)
    else:
        z_used = center(z)
        s_used = s_z
    
    T = (np.dot(z_used, s_used) ** 2) / (np.sum(s_used ** 2) + CONFIG['delta_stat'])
    return T


def compute_auc(snr_db, f_cut, use_projection, template_mode='det', trials=None, seed=None):
    """
    计算 AUC
    
    Args:
        snr_db: SNR in dB
        f_cut: cutoff frequency
        use_projection: 是否使用投影
        template_mode: 'det' (deterministic) or 'exp' (expected)
        trials: MC trials 数量
        seed: 随机种子
    """
    if trials is None:
        trials = CONFIG['trials_auc']
    if seed is None:
        seed = CONFIG['master_seed']
    
    v_rel = CONFIG['v_default']
    
    # 生成模板
    if template_mode == 'det':
        s_z = build_sz_deterministic(v_rel)
    else:
        s_z = build_sz_expected(v_rel, snr_db, M=100, seed=seed)
    
    d_wb = generate_debris_signal(v_rel)
    y0_base = np.ones(N, dtype=complex)
    y1_base = 1.0 - d_wb
    
    stats_h0 = []
    stats_h1 = []
    
    for trial in range(trials):
        sub_seed = int(hashlib.md5(f"{seed}_auc_{snr_db}_{f_cut}_{trial}".encode()).hexdigest()[:8], 16) % (2**31)
        np.random.seed(sub_seed)
        
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = CONFIG['jitter_rms']
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        # H0
        y0_m = y0_base * jitter
        pa_out_0, _, _ = hw.apply_saleh_pa(y0_m, ibo_dB=CONFIG['ibo_dB'])
        p_ref = np.mean(np.abs(pa_out_0) ** 2)
        noise_std = np.sqrt(p_ref / (10 ** (snr_db / 10.0)))
        w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
        z0 = log_envelope(pa_out_0 + w)
        T0 = compute_test_statistic(z0, s_z, use_projection, f_cut)
        stats_h0.append(T0)
        
        # H1 (same jitter)
        np.random.seed(sub_seed)
        hw = HardwareImpairments(HW_CONFIG)
        hw.jitter_rms = CONFIG['jitter_rms']
        jitter = np.exp(hw.generate_colored_jitter(N, fs))
        
        y1_m = y1_base * jitter
        pa_out_1, _, _ = hw.apply_saleh_pa(y1_m, ibo_dB=CONFIG['ibo_dB'])
        w = (np.random.randn(N) + 1j * np.random.randn(N)) * noise_std / np.sqrt(2)
        z1 = log_envelope(pa_out_1 + w)
        T1 = compute_test_statistic(z1, s_z, use_projection, f_cut)
        stats_h1.append(T1)
    
    # 计算 AUC
    all_stats = np.concatenate([stats_h0, stats_h1])
    thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 500)
    pfa = np.array([np.mean(np.array(stats_h0) > t) for t in thresholds])
    pd = np.array([np.mean(np.array(stats_h1) > t) for t in thresholds])
    auc = -np.trapezoid(pd, pfa)
    
    return auc, stats_h0, stats_h1


# =============================================================================
# 图生成函数
# =============================================================================
def save_figure(fig, name, subdir=None):
    """保存图像"""
    if subdir:
        path = os.path.join(OUTPUT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
    else:
        path = OUTPUT_DIR
    
    fig.savefig(os.path.join(path, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(path, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] {name}.png/pdf")


def generate_fig_eta_comparison():
    """
    [Task-2] η_z 曲线对比：deterministic vs expected
    """
    print("\n" + "="*60)
    print("Generating: η_z Comparison (det vs exp)")
    print("="*60)
    
    f_cut_values = np.array([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500])
    v_values = [10000, 15000, 20000]
    
    r_F = fresnel_radius(wavelength, CONFIG['L_eff'])
    f_max = signal_bandwidth(15000, r_F)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：deterministic
    for v in v_values:
        s_z_det = build_sz_deterministic(v)
        eta_list = [compute_eta(s_z_det, fc) for fc in f_cut_values]
        ax1.plot(f_cut_values, eta_list, 'o-', label=f'$v_{{rel}}$={v/1000:.0f} km/s', linewidth=2)
    
    ax1.axvline(300, color='red', linestyle='--', linewidth=2, label='$f_{cut}$=300 Hz')
    ax1.axvline(200, color='gray', linestyle=':', linewidth=1.5, label='$f_{knee}$=200 Hz')
    ax1.axvline(f_max/3, color='orange', linestyle=':', linewidth=1.5, label=f'$f_{{max}}/3$={f_max/3:.0f} Hz')
    ax1.axhline(0.99, color='green', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Cutoff Frequency $f_{cut}$ (Hz)', fontsize=12)
    ax1.set_ylabel('$\\eta_z^{det}$', fontsize=12)
    ax1.set_title('(a) Deterministic Template', fontsize=12)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.90, 1.01)
    ax1.set_xlim(0, 1600)
    
    # 右图：expected (只用一个速度，因为计算量大)
    print("   Computing expected template η (this takes time)...")
    v = 15000
    s_z_exp = build_sz_expected(v, snr_db=50, M=100)
    eta_exp = [compute_eta(s_z_exp, fc) for fc in f_cut_values]
    
    s_z_det = build_sz_deterministic(v)
    eta_det = [compute_eta(s_z_det, fc) for fc in f_cut_values]
    
    ax2.plot(f_cut_values, eta_det, 's-', color='blue', label='Deterministic $s_z^{det}$', linewidth=2)
    ax2.plot(f_cut_values, eta_exp, 'o--', color='red', label='Expected $s_z^{exp}$', linewidth=2)
    
    ax2.axvline(300, color='green', linestyle='--', linewidth=2, label='$f_{cut}$=300 Hz')
    ax2.axhline(0.99, color='green', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Cutoff Frequency $f_{cut}$ (Hz)', fontsize=12)
    ax2.set_ylabel('$\\eta_z$', fontsize=12)
    ax2.set_title('(b) Det vs Exp Comparison ($v_{rel}$=15 km/s)', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.90, 1.01)
    ax2.set_xlim(0, 1600)
    
    plt.tight_layout()
    save_figure(fig, 'Fig_Eta_Comparison')
    
    # 打印关键数值
    idx_300 = np.where(f_cut_values == 300)[0][0]
    print(f"\n   η at f_cut=300 Hz:")
    print(f"      Deterministic: {eta_det[idx_300]:.4f}")
    print(f"      Expected:      {eta_exp[idx_300]:.4f}")


def generate_fig_dct_spectrum():
    """
    [Task-3] DCT 谱对比：det vs exp
    """
    print("\n" + "="*60)
    print("Generating: DCT Spectrum Comparison")
    print("="*60)
    
    v_rel = 15000
    r_F = fresnel_radius(wavelength, CONFIG['L_eff'])
    f_max = signal_bandwidth(v_rel, r_F)
    
    s_z_det = build_sz_deterministic(v_rel)
    s_z_exp = build_sz_expected(v_rel, snr_db=50, M=100)
    
    # DCT
    C_det = dct(s_z_det, type=2, norm='ortho')
    C_exp = dct(s_z_exp, type=2, norm='ortho')
    
    # 频率映射 [C]: f_k = k * fs / (2N)
    f_k = np.arange(N) * fs / (2 * N)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(f_k[:800], np.abs(C_det[:800]) + 1e-15, 'b-', linewidth=1.5, 
                label='$|C_z^{det}[k]|$', alpha=0.8)
    ax.semilogy(f_k[:800], np.abs(C_exp[:800]) + 1e-15, 'r--', linewidth=1.5, 
                label='$|C_z^{exp}[k]|$', alpha=0.8)
    
    ax.axvline(CONFIG['f_knee'], color='gray', linestyle=':', linewidth=2, label='$f_{knee}$=200 Hz')
    ax.axvline(300, color='green', linestyle='--', linewidth=2, label='$f_{cut}$=300 Hz')
    ax.axvline(f_max, color='purple', linestyle='-.', linewidth=1.5, label=f'$f_{{max}}$={f_max:.0f} Hz')
    
    ax.axvspan(0, CONFIG['f_knee'], alpha=0.15, color='red', label='1/f region')
    ax.axvspan(CONFIG['f_knee'], 300, alpha=0.1, color='orange')
    ax.axvspan(300, f_max, alpha=0.08, color='green', label='Signal band')
    
    ax.set_xlabel('DCT Frequency $f_k = k \\cdot f_s / (2N)$ (Hz)', fontsize=12)
    ax.set_ylabel('DCT Coefficient Magnitude', fontsize=12)
    ax.set_title('DCT Spectrum: Deterministic vs Expected Template', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4000)
    ax.set_ylim(1e-10, 1e-2)
    
    plt.tight_layout()
    save_figure(fig, 'Fig_DCT_Spectrum')


def generate_fig_rho_noise():
    """
    [Task-4] ρ_noise 曲线，固定 baseline 数值
    """
    print("\n" + "="*60)
    print("Generating: ρ_noise Curve")
    print("="*60)
    
    f_cut_values = np.array([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])
    
    print("   Computing ρ_noise (this takes time)...")
    rho_list = []
    for fc in f_cut_values:
        rho = compute_rho_noise(fc, n_trials=100)
        rho_list.append(rho)
        print(f"      f_cut={fc} Hz: ρ_noise = {rho:.4f} ({rho*100:.1f}%)")
    
    # 获取 baseline 值
    idx_300 = np.where(f_cut_values == 300)[0][0]
    rho_baseline = rho_list[idx_300]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(f_cut_values, np.array(rho_list) * 100, 'ro-', linewidth=2, markersize=8)
    ax.axvline(300, color='green', linestyle='--', linewidth=2)
    ax.axhline(rho_baseline * 100, color='blue', linestyle=':', alpha=0.5)
    
    ax.annotate(f'{rho_baseline*100:.1f}%', xy=(300, rho_baseline*100), 
                xytext=(400, rho_baseline*100 - 10),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlabel('Cutoff Frequency $f_{cut}$ (Hz)', fontsize=12)
    ax.set_ylabel('Noise Energy Removed $\\rho_{noise}$ (%)', fontsize=12)
    ax.set_title('1/f Noise Rejection Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    save_figure(fig, 'Fig_Rho_Noise')
    
    print(f"\n   [B] BASELINE VALUE: ρ_noise = {rho_baseline*100:.1f}% at f_cut=300 Hz")
    
    return rho_baseline


def generate_fig_auc_vs_snr():
    """
    [Task-5] AUC vs SNR: 投影前 vs 投影后对照
    """
    print("\n" + "="*60)
    print("Generating: AUC vs SNR (with/without projection)")
    print("="*60)
    
    snr_values = [30, 40, 45, 50, 55, 60]
    f_cut = 300
    
    auc_no_proj = []
    auc_with_proj = []
    
    print("   Computing AUC curves (this takes significant time)...")
    for snr in snr_values:
        print(f"      SNR = {snr} dB...")
        
        auc_np, _, _ = compute_auc(snr, f_cut, use_projection=False, trials=300)
        auc_wp, _, _ = compute_auc(snr, f_cut, use_projection=True, trials=300)
        
        auc_no_proj.append(auc_np)
        auc_with_proj.append(auc_wp)
        
        print(f"         No projection: AUC = {auc_np:.3f}")
        print(f"         With projection: AUC = {auc_wp:.3f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(snr_values, auc_no_proj, 's--', color='red', linewidth=2, markersize=8,
            label='Without $P_\\perp$ projection')
    ax.plot(snr_values, auc_with_proj, 'o-', color='blue', linewidth=2, markersize=8,
            label='With $P_\\perp$ projection')
    
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random guess')
    ax.axhline(1.0, color='green', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Detection Performance: AUC vs SNR ($f_{{cut}}$={f_cut} Hz)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(28, 62)
    ax.set_ylim(0.4, 1.05)
    
    # 添加改进量标注
    for i, snr in enumerate(snr_values):
        improvement = auc_with_proj[i] - auc_no_proj[i]
        if improvement > 0.01:
            ax.annotate(f'+{improvement:.2f}', 
                       xy=(snr, (auc_with_proj[i] + auc_no_proj[i])/2),
                       fontsize=9, color='green', ha='center')
    
    plt.tight_layout()
    save_figure(fig, 'Fig_AUC_vs_SNR')
    
    return snr_values, auc_no_proj, auc_with_proj


def generate_fig_concept_final(rho_baseline):
    """
    生成最终概念图，使用固定的 ρ_noise baseline 值
    """
    print("\n" + "="*60)
    print("Generating: Final Concept Figure")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    v_rel = 15000
    r_F = fresnel_radius(wavelength, CONFIG['L_eff'])
    f_max = signal_bandwidth(v_rel, r_F)
    
    # 生成信号
    d_wb = generate_debris_signal(v_rel)
    y0_base = np.ones(N, dtype=complex)
    y1_base = 1.0 - d_wb
    
    np.random.seed(CONFIG['master_seed'])
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = CONFIG['jitter_rms']
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    
    pa_out_1, _, _ = hw.apply_saleh_pa(y1_base * jitter, ibo_dB=CONFIG['ibo_dB'])
    z_h1 = log_envelope(pa_out_1)
    
    np.random.seed(CONFIG['master_seed'])
    hw = HardwareImpairments(HW_CONFIG)
    hw.jitter_rms = CONFIG['jitter_rms']
    jitter = np.exp(hw.generate_colored_jitter(N, fs))
    pa_out_0, _, _ = hw.apply_saleh_pa(y0_base * jitter, ibo_dB=CONFIG['ibo_dB'])
    z_h0 = log_envelope(pa_out_0)
    
    k_max = build_projection_operator(300, N, fs)
    f_k = np.arange(N) * fs / (2 * N)
    
    # (a) 时域
    ax = axes[0, 0]
    ax.plot(t_axis * 1000, center(z_h0), 'r-', alpha=0.7, linewidth=0.8, label='H0')
    ax.plot(t_axis * 1000, center(z_h1), 'b-', alpha=0.7, linewidth=0.8, label='H1')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('$z(t) - \\bar{z}$', fontsize=11)
    ax.set_title('(a) Log-Envelope $z(t) = \\ln|r(t)|$', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    
    # (b) DCT before
    C_h1 = dct(center(z_h1), type=2, norm='ortho')
    C_h0 = dct(center(z_h0), type=2, norm='ortho')
    
    ax = axes[0, 1]
    ax.semilogy(f_k[:500], np.abs(C_h0[:500]) + 1e-12, 'r-', alpha=0.7, linewidth=1, label='H0')
    ax.semilogy(f_k[:500], np.abs(C_h1[:500]) + 1e-12, 'b-', alpha=0.7, linewidth=1, label='H1')
    ax.axvline(300, color='green', linestyle='--', linewidth=2, label='$f_{cut}$')
    ax.axvline(200, color='orange', linestyle=':', linewidth=1.5, label='$f_{knee}$')
    ax.axvspan(0, 300, alpha=0.15, color='red')
    ax.set_xlabel('$f_k = k \\cdot f_s/(2N)$ (Hz)', fontsize=11)
    ax.set_ylabel('$|C_z[k]|$', fontsize=11)
    ax.set_title('(b) DCT Before $P_\\perp$', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2500)
    
    # (c) DCT after
    C_h1_proj = C_h1.copy()
    C_h1_proj[:k_max] = 0
    C_h0_proj = C_h0.copy()
    C_h0_proj[:k_max] = 0
    
    ax = axes[0, 2]
    ax.semilogy(f_k[:500], np.abs(C_h0_proj[:500]) + 1e-12, 'r-', alpha=0.7, linewidth=1, label='H0')
    ax.semilogy(f_k[:500], np.abs(C_h1_proj[:500]) + 1e-12, 'b-', alpha=0.7, linewidth=1, label='H1')
    ax.axvline(300, color='green', linestyle='--', linewidth=2)
    ax.axvspan(0, 300, alpha=0.15, color='red', label='Nulled')
    ax.axvspan(300, 2500, alpha=0.08, color='green', label='Preserved')
    ax.set_xlabel('$f_k$ (Hz)', fontsize=11)
    ax.set_ylabel('$|C_z[k]|$', fontsize=11)
    ax.set_title('(c) DCT After $P_\\perp$', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2500)
    
    # (d) 时域 after
    z_h1_proj = idct(C_h1_proj, type=2, norm='ortho')
    z_h0_proj = idct(C_h0_proj, type=2, norm='ortho')
    
    ax = axes[1, 0]
    ax.plot(t_axis * 1000, z_h0_proj, 'r-', alpha=0.7, linewidth=0.8, label='H0')
    ax.plot(t_axis * 1000, z_h1_proj, 'b-', alpha=0.7, linewidth=0.8, label='H1')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('$P_\\perp z$', fontsize=11)
    ax.set_title('(d) After Projection', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    
    # (e) Detection
    auc, stats_h0, stats_h1 = compute_auc(50, 300, use_projection=True, trials=300)
    
    ax = axes[1, 1]
    ax.hist(stats_h0, bins=30, alpha=0.6, color='red', label='H0', density=True)
    ax.hist(stats_h1, bins=30, alpha=0.6, color='blue', label='H1', density=True)
    ax.set_xlabel('Test Statistic $T$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'(e) Detection (AUC={auc:.3f}, SNR=50dB)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (f) Summary [B] 使用固定的 baseline 值
    ax = axes[1, 2]
    ax.axis('off')
    
    # [B] 修正：使用实际计算的 ρ_noise 值
    rho_pct = rho_baseline * 100
    
    summary_text = f"""SURVIVAL SPACE DESIGN (FINAL)

Physical Parameters:
  - Fresnel radius: r_F = {r_F:.2f} m
  - Crossing time: T_cross = {2*r_F/v_rel*1000:.2f} ms
  - Signal bandwidth: f_max = {f_max:.0f} Hz

Projection Operator P_perp:
  - Nulls DCT coefficients for f < f_cut
  - Frequency mapping: f_k = k*fs/(2N)
  - Preserved subspace: [f_cut, fs/2]

Design Choice: f_cut = 300 Hz
  - Condition: f_knee < f_cut << f_max
  - f_cut/f_knee = 1.5 (noise margin)
  - f_cut/f_max = 0.18 (signal margin)

Key Metrics (f_cut = 300 Hz):
  - Signal retention: eta_z > 99%
  - Noise removed: rho = {rho_pct:.1f}%
  - Detection AUC = {auc:.3f} (SNR=50dB)

Working Range:
  f_knee < f_max/3 = {f_max/3:.0f} Hz"""
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('(f) Design Summary', fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, 'Fig_Concept_FINAL')
    
    return auc


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("="*70)
    print("SURVIVAL SPACE VALIDATION - FINAL VERSION")
    print("="*70)
    print("\nAddressing expert concerns:")
    print("  [A] Template definition alignment (det vs exp)")
    print("  [B] Summary numerical consistency")
    print("  [C] DCT frequency mapping: f_k = k*fs/(2N)")
    print("  [D] AUC performance curves")
    
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"Configuration: fs={fs/1e3}kHz, N={N}, f_knee={CONFIG['f_knee']}Hz")
    
    r_F = fresnel_radius(wavelength, CONFIG['L_eff'])
    f_max = signal_bandwidth(CONFIG['v_default'], r_F)
    print(f"Physical: r_F={r_F:.2f}m, f_max={f_max:.0f}Hz, f_max/3={f_max/3:.0f}Hz")
    
    # 生成所有图
    generate_fig_eta_comparison()
    generate_fig_dct_spectrum()
    rho_baseline = generate_fig_rho_noise()
    snr_vals, auc_np, auc_wp = generate_fig_auc_vs_snr()
    auc_final = generate_fig_concept_final(rho_baseline)
    
    # 保存配置
    config_snapshot = {
        'config': CONFIG,
        'hw_config': HW_CONFIG,
        'results': {
            'rho_noise_baseline': float(rho_baseline),
            'auc_at_50dB': float(auc_final),
            'f_max': float(f_max),
            'f_knee_limit': float(f_max/3),
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"  ρ_noise at f_cut=300 Hz: {rho_baseline*100:.1f}%")
    print(f"  AUC at SNR=50dB: {auc_final:.3f}")
    print(f"  Working range: f_knee < {f_max/3:.0f} Hz")
    print("="*70)


if __name__ == "__main__":
    main()

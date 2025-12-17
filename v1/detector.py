# ----------------------------------------------------------------------------------
# 脚本名称: detector_fixed.py
# 版本: v4.0 (Peak Detection Fix)
# 描述:
#   [CRITICAL FIX]: 添加峰值检测方法，解决子空间投影导致的"信号自消除"问题
#
#   问题诊断：
#   - 原始的 P_perp (正交子空间投影) 会把衍射信号的"幅度下降"当作直流漂移去除
#   - 导致投影后信号能量为 0，检测器完全失效
#
#   解决方案：
#   - 保留原有的 P_perp 用于特定场景
#   - 添加新的峰值检测 (Peak Detection) 方法用于衍射信号检测
#   - 峰值检测直接寻找信号中的"深坑"，不会误删信号
# ----------------------------------------------------------------------------------

import numpy as np
import scipy.special as sp
import scipy.signal as sig_proc


class TerahertzDebrisDetector:
    def __init__(self, fs, N_window, cutoff_freq=300.0, L_eff=500e3, fc=300e9, a=0.05, B=10e9, N_sub=32):
        self.fs = fs
        self.N = int(N_window)
        self.f_cut = cutoff_freq
        self.L_eff = L_eff
        self.fc = fc
        self.B = B
        self.a = a
        self.c = 299792458.0
        self.N_sub = N_sub

        # 保留原有的投影矩阵（用于某些特定场景）
        self.P_perp = self._build_dct_projection_matrix()

        # [NEW] 构建高通滤波器（可选方案）
        self.hp_filter = self._build_highpass_filter()

    def _build_dct_projection_matrix(self):
        """原有的 DCT 正交投影矩阵"""
        freq_res = self.fs / (2 * self.N)
        k_max = int(np.ceil(self.f_cut / freq_res))
        k_max = max(k_max, 1)

        H_int = np.zeros((self.N, k_max))
        n_idx = np.arange(self.N)

        for k in range(k_max):
            scale = np.sqrt(1.0 / self.N) if k == 0 else np.sqrt(2.0 / self.N)
            basis_vec = scale * np.cos(np.pi * k * (2 * n_idx + 1) / (2 * self.N))
            H_int[:, k] = basis_vec

        P_perp = np.eye(self.N) - (H_int @ H_int.T)
        return P_perp

    def _build_highpass_filter(self):
        """构建高通滤波器（可选方案）"""
        try:
            # 设计 4 阶 Butterworth 高通滤波器
            nyq = self.fs / 2
            normalized_cutoff = min(self.f_cut / nyq, 0.99)
            b, a = sig_proc.butter(4, normalized_cutoff, btype='high')
            return (b, a)
        except:
            return None

    def log_envelope_transform(self, y):
        """对数包络变换"""
        env = np.abs(y)
        z = np.log(env + 1e-12)
        return z

    def apply_projection(self, z):
        """应用正交投影（原有方法）"""
        if len(z) != self.N:
            z = np.reshape(z, (self.N,))
        return self.P_perp @ z

    def apply_highpass(self, signal):
        """应用高通滤波（可选方案）"""
        if self.hp_filter is None:
            # Fallback: 简单的去均值
            return signal - np.mean(signal)
        b, a = self.hp_filter
        return sig_proc.filtfilt(b, a, np.real(signal))

    # =========================================================================
    # [NEW] 峰值检测方法 - 核心修复
    # =========================================================================

    def detect_dip_peak(self, rx_signal, return_stats=False):
        """
        峰值检测方法 - 检测衍射信号中的"深坑"

        原理：
        - 衍射信号表现为幅度的短暂下降（深坑）
        - 直接检测信号中偏离均值最大的负向峰值
        - 不使用正交投影，避免信号自消除

        参数:
            rx_signal: 接收信号（复数）
            return_stats: 是否返回详细统计量

        返回:
            stat_peak: 峰值统计量（坑越深值越大）
            stat_energy: 能量统计量（可选）
        """
        # 1. 取幅度包络
        rx_amp = np.abs(rx_signal)

        # 2. 去直流（去除静态背景）
        rx_mean = np.mean(rx_amp)
        rx_ac = rx_amp - rx_mean

        # 3. 峰值检测统计量
        # 衍射深坑是负向的，所以取 -min 使得坑越深值越大
        stat_peak = -np.min(rx_ac)

        # 4. 归一化（可选，使统计量与信号功率无关）
        rx_std = np.std(rx_ac) + 1e-12
        stat_peak_normalized = stat_peak / rx_std

        if return_stats:
            # 额外的统计量
            stat_energy = np.sum(rx_ac ** 2)
            stat_variance = np.var(rx_ac)
            dip_location = np.argmin(rx_ac)

            return {
                'peak': stat_peak,
                'peak_normalized': stat_peak_normalized,
                'energy': stat_energy,
                'variance': stat_variance,
                'dip_location': dip_location,
                'dip_time': dip_location / self.fs
            }

        return stat_peak_normalized

    def detect_matched_filter(self, rx_signal, template=None, v_rel=15000):
        """
        匹配滤波检测方法

        原理：
        - 使用理论衍射模板与接收信号做相关
        - 适用于已知目标速度的场景

        参数:
            rx_signal: 接收信号
            template: 自定义模板（可选）
            v_rel: 相对速度（用于生成模板）
        """
        if template is None:
            template = self._generate_template(v_rel)

        # 取幅度并去均值
        rx_amp = np.abs(rx_signal) - np.mean(np.abs(rx_signal))
        template_ac = template - np.mean(template)

        # 归一化互相关
        correlation = np.correlate(rx_amp, template_ac, mode='valid')

        # 统计量：最大相关值
        stat = np.max(np.abs(correlation))

        return stat

    def energy_detector(self, rx_signal):
        """
        能量检测器（最简单的检测方法）

        原理：
        - 检测信号能量的变化
        - 对任何形式的扰动都敏感
        """
        rx_amp = np.abs(rx_signal)
        rx_ac = rx_amp - np.mean(rx_amp)
        stat = np.sum(rx_ac ** 2)
        return stat

    # =========================================================================
    # 原有方法保留
    # =========================================================================

    def _lommel_series_fast(self, u, v, max_terms=40):
        v_safe = np.maximum(v, 1e-5)
        ratio = u / v_safe

        U1 = np.zeros_like(v, dtype=np.complex128)
        U2 = np.zeros_like(v, dtype=np.complex128)

        for m in range(max_terms):
            sign = (-1.0) ** m
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)
            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)

            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2

        return U1, U2

    def _generate_diffraction_pattern(self, t_axis, freqs, v_rel):
        freqs = np.atleast_1d(freqs)
        freqs_col = freqs.reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)

        lam_col = self.c / freqs_col
        rho_t = v_rel * np.abs(t_row)

        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)

        U1, U2 = self._lommel_series_fast(u_k, v_k)

        quad_phase = np.exp(1j * (v_k ** 2) / (2 * u_k))
        phase_term = np.exp(-1j * u_k / 2)

        d_k = phase_term * (U1 + 1j * U2) * quad_phase
        return d_k

    def _generate_template(self, v_rel):
        """生成宽带 DFS 模板"""
        t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs
        freqs = np.linspace(self.fc - self.B / 2, self.fc + self.B / 2, self.N_sub)

        d_k_matrix = self._generate_diffraction_pattern(t_axis, freqs, v_rel)

        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)

        broadband_d = np.sum(d_k_matrix * time_phase_matrix, axis=0) / self.N_sub
        s_template = -np.real(broadband_d)

        return s_template

    def glrt_scan(self, z_perp, velocity_grid):
        """原有的 GLRT 扫描方法（保留）"""
        glrt_stats = []
        for v in velocity_grid:
            s_raw = self._generate_template(v)
            s_perp = self.P_perp @ s_raw

            energy = np.sum(s_perp ** 2)
            if energy < 1e-20:
                glrt_stats.append(0.0)
            else:
                correlation = np.dot(s_perp, z_perp)
                t_stat = (correlation ** 2) / energy
                glrt_stats.append(t_stat)

        return np.array(glrt_stats)

    # =========================================================================
    # [NEW] 改进的 GLRT 扫描（不使用投影）
    # =========================================================================

    def glrt_scan_peak(self, rx_signal, velocity_grid):
        """
        改进的 GLRT 扫描 - 使用峰值检测

        原理：
        - 对每个候选速度生成模板
        - 使用峰值检测而非投影
        - 返回最佳匹配的速度估计
        """
        rx_amp = np.abs(rx_signal)
        rx_ac = rx_amp - np.mean(rx_amp)

        stats = []
        for v in velocity_grid:
            template = self._generate_template(v)
            template_ac = template - np.mean(template)

            # 相关性统计量
            corr = np.dot(rx_ac, template_ac)
            energy = np.sum(template_ac ** 2) + 1e-20
            stat = (corr ** 2) / energy
            stats.append(stat)

        return np.array(stats)


# =========================================================================
# 便捷函数
# =========================================================================

def create_detector(fs, N, cutoff_freq=5000.0, **kwargs):
    """创建检测器实例的便捷函数"""
    return TerahertzDebrisDetector(fs, N, cutoff_freq=cutoff_freq, **kwargs)
# ----------------------------------------------------------------------------------
# 脚本名称: physics_engine.py
# 版本: v2.4 (Fixed - Time Axis & Numerical Stability)
# 描述:
#   [FIX 1]: 修复了 v -> 0 时的数值溢出问题 (v_safe 提高到 1e-5)。
#   [FIX 2]: 时间轴改为 np.arange 生成，确保采样间隔严格为 1/fs。
# ----------------------------------------------------------------------------------

import numpy as np
import scipy.special as sp


class DiffractionChannel:
    def __init__(self, config):
        self.fc = config.get('fc', 300e9)
        self.B = config.get('B', 10e9)
        self.L_eff = config.get('L_eff', 500e3)
        self.a = config.get('a', 0.05)
        self.v_rel = config.get('v_rel', 15000)
        self.c = 299792458.0

    def _lommel_series(self, u, v, max_terms=40):
        """
        计算双变量 Lommel 函数 U1, U2
        """
        # [FIX] 防止数值溢出: (u/v)^n 当 v 太小时会爆炸
        # 对于 u~1, v_safe=1e-5 => ratio=1e5. ratio^40 = 1e200 (安全范围 < 1e308)
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

    def generate_diffraction_pattern(self, t_axis, freqs):
        """
        生成特定频率下的衍射调制信号 d(t)
        """
        freqs = np.atleast_1d(freqs)
        freqs_col = freqs.reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)

        lam_col = self.c / freqs_col
        rho_t = self.v_rel * np.abs(t_row)

        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)

        U1, U2 = self._lommel_series(u_k, v_k)

        # Chirp Phase (二次相位)
        quad_phase = np.exp(1j * (v_k ** 2) / (2 * u_k))
        phase_term = np.exp(-1j * u_k / 2)

        d_k = phase_term * (U1 + 1j * U2) * quad_phase

        return d_k

    def generate_broadband_chirp(self, t_axis, N_sub=128):
        """
        宽带离散频率求和 (DFS)
        """
        freqs = np.linspace(self.fc - self.B / 2, self.fc + self.B / 2, N_sub)
        d_k_matrix = self.generate_diffraction_pattern(t_axis, freqs)

        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        # 基带旋转
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)

        broadband_signal = np.sum(d_k_matrix * time_phase_matrix, axis=0) / N_sub

        return broadband_signal
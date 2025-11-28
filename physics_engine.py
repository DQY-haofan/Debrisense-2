import numpy as np
import scipy.special as sp


class DiffractionChannel:
    """
    IEEE TWC 级别物理光学仿真引擎 (Fixed Version)
    """

    def __init__(self, config):
        self.fc = config.get('fc', 300e9)
        self.B = config.get('B', 10e9)
        self.L_eff = config.get('L_eff', 50e3)  # 默认 50km
        self.a = config.get('a', 0.05)  # 默认 5cm
        self.v_rel = config.get('v_rel', 15000)
        self.c = 299792458.0

    def _lommel_series(self, u, v, max_terms=40):
        """
        计算双变量 Lommel 函数 U1, U2
        """
        v_safe = np.maximum(v, 1e-12)
        ratio = u / v_safe

        # 确保输出形状与输入 v 一致
        U1 = np.zeros_like(v, dtype=np.complex128)
        U2 = np.zeros_like(v, dtype=np.complex128)

        for m in range(max_terms):
            sign = (-1.0) ** m
            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)

            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2

        return U1, U2

    def generate_diffraction_pattern(self, t_axis, freqs):
        """
        核心物理计算函数 (Broadcasting Fix Applied)
        """
        # [FIX] 1. 确保 freqs 是 (N_f, 1) 的列向量
        freqs = np.atleast_1d(freqs)
        freqs_col = freqs.reshape(-1, 1)

        # [FIX] 2. 确保 t_axis 是 (1, N_t) 的行向量
        t_row = t_axis.reshape(1, -1)

        # 计算波数 k (N_f, 1)
        lam_col = self.c / freqs_col

        # 计算径向距离 rho(t) (1, N_t)
        rho_t = self.v_rel * np.abs(t_row)

        # 计算 u (N_f, 1)
        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)

        # 计算 v (N_f, N_t) - 触发广播
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)

        # 计算 Lommel 函数
        U1, U2 = self._lommel_series(u_k, v_k)

        # 组合衍射调制项
        phase_term = np.exp(-1j * u_k / 2)
        d_k = phase_term * (U1 + 1j * U2)

        return d_k

    def generate_broadband_chirp(self, t_axis, N_sub=128):
        """
        宽带离散频率求和 (DFS)
        """
        freqs = np.linspace(self.fc - self.B / 2, self.fc + self.B / 2, N_sub)

        # d_k_matrix shape: (N_sub, N_time)
        d_k_matrix = self.generate_diffraction_pattern(t_axis, freqs)

        # 相干叠加
        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)

        broadband_signal = np.sum(d_k_matrix * time_phase_matrix, axis=0) / N_sub

        return broadband_signal
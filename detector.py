# ----------------------------------------------------------------------------------
# 脚本名称: detector.py
# 版本: v3.1 (Fixed - N_sub Alignment)
# 描述:
#   [FIX]: 将 DFS 子载波数 N_sub 参数化 (默认 32)。
#   确保与 Physics Engine 的频率网格 100% 重合，从而实现 Correlation = 1.0。
# ----------------------------------------------------------------------------------

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.signal


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
        # [NEW] 显式保存子载波数量，默认为 32 以匹配物理引擎
        self.N_sub = N_sub
        self.P_perp = self._build_dct_projection_matrix()

    def _build_dct_projection_matrix(self):
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

    def log_envelope_transform(self, y):
        env = np.abs(y)
        z = np.log(env + 1e-12)
        return z

    def apply_projection(self, z):
        if len(z) != self.N:
            z = np.reshape(z, (self.N,))
        return self.P_perp @ z

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
        """
        生成宽带 DFS 模板
        """
        t_axis = np.arange(self.N) / self.fs - (self.N / 2) / self.fs

        # [FIX] 使用 self.N_sub (32)，而不是硬编码的 16
        freqs = np.linspace(self.fc - self.B / 2, self.fc + self.B / 2, self.N_sub)

        d_k_matrix = self._generate_diffraction_pattern(t_axis, freqs, v_rel)

        delta_f = (freqs - self.fc).reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)

        broadband_d = np.sum(d_k_matrix * time_phase_matrix, axis=0) / self.N_sub

        s_template = -np.real(broadband_d)

        return s_template

    def glrt_scan(self, z_perp, velocity_grid):
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
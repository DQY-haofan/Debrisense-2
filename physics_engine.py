# ----------------------------------------------------------------------------------
# 脚本名称: physics_engine.py
# 版本: v3.0 (Numba Ready & Vectorized)
# ----------------------------------------------------------------------------------

import numpy as np
import scipy.special as sp

# 尝试导入 numba 进行 JIT 加速
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    # 定义一个空的装饰器作为 fallback
    def jit(nopython=True):
        def decorator(func):
            return func

        return decorator


class DiffractionChannel:
    def __init__(self, config):
        self.fc = config.get('fc', 300e9)
        self.B = config.get('B', 10e9)
        self.L_eff = config.get('L_eff', 500e3)
        self.a = config.get('a', 0.05)
        self.v_rel = config.get('v_rel', 15000)
        self.c = 299792458.0

    def generate_broadband_chirp(self, t_axis, N_sub=32):
        """
        宽带离散频率求和 (DFS) - 物理层核心
        """
        # 1. 频率网格
        freqs = np.linspace(self.fc - self.B / 2, self.fc + self.B / 2, N_sub)

        # 2. 空间参数矩阵化 (Vectorization)
        # freqs: (N_sub, 1)
        # t_axis: (1, N_t)
        freqs_col = freqs.reshape(-1, 1)
        t_row = t_axis.reshape(1, -1)

        lam_col = self.c / freqs_col
        rho_t = self.v_rel * np.abs(t_row)

        # Lommel 变量 u, v
        # u 是常数(对时间)，但随频率变化
        u_k = (2 * np.pi * self.a ** 2) / (lam_col * self.L_eff)
        # v 随时间和频率变化
        v_k = (2 * np.pi * self.a * rho_t) / (lam_col * self.L_eff)

        # 3. 计算 Lommel 函数 (耗时核心)
        if HAS_NUMBA:
            # 展平数组以喂给 Numba (如果需要极致优化，这里可以进一步展开)
            # 但目前的 Numpy 向量化已经足够快，这里保持兼容性
            U1, U2 = self._lommel_series_numpy(u_k, v_k)
        else:
            U1, U2 = self._lommel_series_numpy(u_k, v_k)

        # 4. 合成衍射场
        # 二次相位项 (Chirp 本质)
        quad_phase = np.exp(1j * (v_k ** 2) / (2 * u_k))
        phase_term = np.exp(-1j * u_k / 2)

        d_k_matrix = phase_term * (U1 + 1j * U2) * quad_phase

        # 5. 基带旋转与叠加
        delta_f = (freqs - self.fc).reshape(-1, 1)
        time_phase_matrix = np.exp(1j * 2 * np.pi * delta_f * t_row)

        # 相干累加
        broadband_signal = np.sum(d_k_matrix * time_phase_matrix, axis=0) / N_sub

        return broadband_signal

    def _lommel_series_numpy(self, u, v, max_terms=40):
        """
        高度向量化的 Numpy 实现，无需显式循环处理数组元素
        """
        v_safe = np.maximum(v, 1e-5)
        ratio = u / v_safe

        U1 = np.zeros_like(v, dtype=np.complex128)
        U2 = np.zeros_like(v, dtype=np.complex128)

        # 级数求和循环 (无法避免，但只有 40 次，开销很小)
        for m in range(max_terms):
            sign = (-1.0) ** m
            # scipy.special.jv 支持数组输入
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)

            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)

            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2

        return U1, U2
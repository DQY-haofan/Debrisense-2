import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal

# 设置绘图风格 (符合 IEEE 论文标准)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 150
})


class TerahertzDebrisDetector:
    """
    IEEE TWC 级别鲁棒检测器实现
    核心架构：对数包络变换 -> DCT 子空间投影 -> GLRT 匹配滤波器组
    Ref: DR Algo 03
    """

    def __init__(self, fs, N_window, cutoff_freq=300.0, L_eff=500e3, fc=300e9, a=0.05):
        """
        :param fs: 采样率 (Hz)
        :param N_window: 处理窗口长度 (samples), e.g., 对应 3ms-5ms
        :param cutoff_freq: 抖动截止频率 (Hz), 用于构建干扰子空间
        :param L_eff: 有效基线长度 (m)
        :param fc: 载波频率 (Hz)
        :param a: 碎片半径 (m) 用于模板生成
        """
        self.fs = fs
        self.N = int(N_window)
        self.f_cut = cutoff_freq
        self.L_eff = L_eff
        self.lam = 3e8 / fc
        self.a = a

        # --- 1. 预计算投影矩阵 P_perp (DR Algo 03 Section 4.2) ---
        self.P_perp = self._build_dct_projection_matrix()

    def _build_dct_projection_matrix(self):
        """
        构建基于 DCT 的正交投影矩阵，用于剔除低频有色抖动
        P_perp = I - H_int * H_int^T
        """
        freq_res = self.fs / (2 * self.N)
        k_max = int(np.ceil(self.f_cut / freq_res))

        # 强制至少剔除直流 (k=0)
        k_max = max(k_max, 1)

        # 构建干扰子空间矩阵 H_int (N x k_max)
        H_int = np.zeros((self.N, k_max))
        n_idx = np.arange(self.N)

        for k in range(k_max):
            # 归一化系数
            scale = np.sqrt(1.0 / self.N) if k == 0 else np.sqrt(2.0 / self.N)
            # DCT-II Basis
            basis_vec = scale * np.cos(np.pi * k * (2 * n_idx + 1) / (2 * self.N))
            H_int[:, k] = basis_vec

        # 计算正交投影矩阵 P_perp = I - H H^T (利用 DCT 基的正交归一性)
        P_perp = np.eye(self.N) - (H_int @ H_int.T)

        return P_perp

    def log_envelope_transform(self, y):
        """
        对数包络变换：将乘性噪声线性化
        z = ln(|y| + eps)
        Ref: DR Algo 03 Section 4.1
        """
        env = np.abs(y)
        # 添加极小值防止 log(0)
        z = np.log(env + 1e-12)
        return z

    def apply_projection(self, z):
        """
        应用子空间投影
        z_perp = P_perp @ z
        """
        if len(z) != self.N:
            # Try to reshape or truncate if possible, but strict checks are safer for research code
            # For robustness in MC loops sometimes shapes can be (N,1)
            z = np.reshape(z, (self.N,))

        return self.P_perp @ z

    def _generate_template(self, v_rel):
        """
        生成理论衍射模板 s[n] = -Re{d[n]}
        基于 Lommel 函数 (物理光学模型)
        """
        t_axis = np.linspace(-self.N / (2 * self.fs), self.N / (2 * self.fs), self.N)

        # 计算 Lommel 参数
        # u: 静态参数 (Scalar)
        u = (2 * np.pi * self.a ** 2) / (self.lam * self.L_eff)

        # v: 动态参数 v(t) = (2*pi*a / lam*L) * rho(t) (Vector)
        rho_t = v_rel * np.abs(t_axis)
        v_dynamic = (2 * np.pi * self.a * rho_t) / (self.lam * self.L_eff)
        v_safe = np.maximum(v_dynamic, 1e-12)

        # FIX: 初始化 U1, U2 必须与 v_safe (Time dimension) 形状一致
        # u 是 scalar, v_safe 是 (N,)
        # zeros_like(u) 会导致 shape ()，无法存储数组结果
        U1 = np.zeros_like(v_safe, dtype=np.complex128)
        U2 = np.zeros_like(v_safe, dtype=np.complex128)

        ratio = u / v_safe

        for m in range(20):
            sign = (-1.0) ** m
            pow_1 = np.power(ratio, 1 + 2 * m)
            pow_2 = np.power(ratio, 2 + 2 * m)
            bessel_1 = sp.jv(1 + 2 * m, v_safe)
            bessel_2 = sp.jv(2 + 2 * m, v_safe)
            U1 += sign * pow_1 * bessel_1
            U2 += sign * pow_2 * bessel_2

        # d[n] = exp(-ju/2) * (U1 + jU2)
        d_val = np.exp(-1j * u / 2) * (U1 + 1j * U2)

        # 在对数域中，信号表现为 -Re{d[n]} (因为 ln(1-d) approx -d)
        s_template = -np.real(d_val)

        return s_template

    def glrt_scan(self, z_perp, velocity_grid):
        """
        广义似然比检验 (GLRT) 扫描
        Ref: DR Algo 03 Section 5.2
        T(z) = max_v ( |s_perp^T z_perp|^2 / ||s_perp||^2 )
        """
        glrt_stats = []

        for v in velocity_grid:
            # 1. 生成理论模板 s(v)
            s_raw = self._generate_template(v)

            # 2. **CRITICAL**: 模板投影 (Pre-whitening the template)
            s_perp = self.P_perp @ s_raw

            # 3. 能量归一化 (Energy Normalization)
            energy = np.sum(s_perp ** 2)

            if energy < 1e-12:
                glrt_stats.append(0.0)
                continue

            # 4. 计算相关性
            correlation = np.dot(s_perp, z_perp)

            # 5. 计算 GLRT 统计量
            t_stat = (correlation ** 2) / energy
            glrt_stats.append(t_stat)

        return np.array(glrt_stats)


if __name__ == "__main__":
    # 1. 仿真参数配置
    fs = 20e3  # 20 kHz 采样率
    T_span = 0.005  # 5ms 窗口
    N = int(fs * T_span)
    t = np.linspace(-T_span / 2, T_span / 2, N)

    detector = TerahertzDebrisDetector(fs, N, cutoff_freq=300.0)

    # 2. 构造合成信号
    true_v = 15000.0
    s_target_raw = detector._generate_template(true_v)
    linear_mod = 1.0 + s_target_raw

    # B. 有色 Jitter
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    psd_shape = (1.0 / (np.maximum(freqs, 1e-6) / 200.0) ** 4)
    psd_shape[freqs > 200] *= 0.001
    jitter_spec = np.sqrt(psd_shape) * np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
    jitter_noise = np.fft.irfft(jitter_spec, n=N)
    jitter_noise = jitter_noise / np.std(jitter_noise) * 0.05
    A_jitter = np.exp(jitter_noise)

    # C. 合成接收信号
    y_linear = A_jitter * linear_mod

    # 3. 运行检测器流程
    z_log = detector.log_envelope_transform(y_linear)
    z_perp = detector.apply_projection(z_log)

    v_grid = np.linspace(5000, 25000, 100)
    t_stats = detector.glrt_scan(z_perp, v_grid)

    # --- 4. 可视化 ---
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    f_axis, z_log_spec = scipy.signal.periodogram(z_log, fs)
    ax1.semilogy(f_axis, z_log_spec, 'k', alpha=0.7)
    ax1.set_title('Raw Signal Spectrum (Log-Domain)')
    ax1.axvspan(0, 300, color='red', alpha=0.2, label='Blind Zone (Jitter)')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1])
    f_axis, z_perp_spec = scipy.signal.periodogram(z_perp, fs)
    ax2.semilogy(f_axis, z_perp_spec, 'b', alpha=0.9)
    ax2.set_title('Projected Signal Spectrum (Survival Space Extracted)')
    ax2.axvspan(300, 5000, color='green', alpha=0.1, label='Survival Space')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(v_grid / 1000.0, t_stats, 'r.-', linewidth=1.5)
    ax3.axvline(x=true_v / 1000.0, color='g', linestyle='--', label=f'True Velocity ({true_v / 1000} km/s)')
    ax3.set_title('GLRT Statistic vs Velocity Hypothesis')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('detector_verification.png')
    print("Verification plot generated: detector_verification.png")
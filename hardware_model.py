import numpy as np


class HardwareImpairments:
    """
    IEEE TWC 级别硬件损伤仿真引擎 (v2.1 Final)
    包含：
    1. 有色幅度抖动 (Colored Jitter) - 机械微振动
    2. 相位噪声 (Phase Noise) - 振荡器 Leeson 模型
    3. PA 非线性 (Saleh Model) - 射频非线性与自愈效应
    """

    def __init__(self, config):
        # PA 参数 (InGaAs mHEMT)
        self.alpha_a = config.get('alpha_a', 10.127)
        self.beta_a = config.get('beta_a', 5995.0)
        self.alpha_phi = config.get('alpha_phi', 4.0033)
        self.beta_phi = config.get('beta_phi', 9.1040)

        # Jitter 参数
        self.jitter_rms = config.get('jitter_rms', 2.0e-6)
        self.f_knee = config.get('f_knee', 200.0)

        # Phase Noise 参数 (300 GHz carrier)
        self.L_1MHz = config.get('L_1MHz', -85.0)  # dBc/Hz at 1MHz offset
        self.L_floor = config.get('L_floor', -115.0)  # dBc/Hz noise floor
        self.f_corner = config.get('f_corner', 100e3)  # 100 kHz flicker corner

    def generate_colored_jitter(self, N_samples, fs):
        """
        生成有色 Jitter (幅度调制噪声)
        PSD: DC-200Hz 平台, >200Hz 快速滚降 (-80dB/dec)
        """
        freqs = np.fft.rfftfreq(N_samples, d=1 / fs)
        safe_freqs = np.maximum(freqs, 1e-6)

        # PSD 形状: 低频漂移 + 高频滚降
        psd_shape = (1.0 / safe_freqs ** 0.5) * (1.0 / (1.0 + (safe_freqs / self.f_knee) ** 8))
        psd_shape[0] = psd_shape[1]  # 修复 DC

        amplitude = np.sqrt(psd_shape)
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        spectrum = amplitude * random_phase

        jitter_raw = np.fft.irfft(spectrum, n=N_samples)

        # RMS 归一化
        current_rms = np.std(jitter_raw)
        if current_rms == 0: current_rms = 1.0
        jitter_scaled = jitter_raw * (self.jitter_rms / current_rms)

        return jitter_scaled

    def generate_phase_noise(self, N_samples, fs):
        """
        生成符合 Leeson 模型的相位噪声 (Phase Noise)
        依据 DR-Simulation-01 Section 4.2
        包含:
        - 1/f^3 (Flicker FM): 近载波区
        - 1/f^2 (White FM): 锁相环带宽内
        - 1/f^0 (White Phase): 远端热噪声底

        Returns:
            theta_pn (np.ndarray): 相位噪声时域序列 (弧度)
        """
        freqs = np.fft.rfftfreq(N_samples, d=1 / fs)
        safe_freqs = np.maximum(freqs, 1.0)  # 避免 DC 奇点

        # 线性化 dB 值
        S_floor = 10 ** (self.L_floor / 10.0)
        S_1MHz = 10 ** (self.L_1MHz / 10.0)

        # Leeson 模型计算
        # 1. White FM (1/f^2) 区域
        S_white_fm = S_1MHz * (1e6 / safe_freqs) ** 2

        # 2. Flicker FM (1/f^3) 区域
        S_flicker = S_white_fm * (self.f_corner / safe_freqs)

        # 总单边 PSD
        psd_pn = S_white_fm + S_floor + np.where(safe_freqs < self.f_corner, S_flicker, 0)

        # IFFT 合成
        # 功率归一化: amplitude = sqrt(PSD * fs * N / 2)
        amplitude = np.sqrt(psd_pn * fs * N_samples / 2)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))

        # 强制 DC 相位为 0 (载波本身是参考)
        amplitude[0] = 0.0

        theta_pn = np.fft.irfft(amplitude * phase, n=N_samples)

        return theta_pn

    def apply_saleh_pa(self, input_signal, ibo_dB):
        """ 应用 Saleh PA 模型 (带物理电压定标) """
        # V_sat 计算: beta * V^2 = 1 => V_sat = 1/sqrt(beta)
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)

        # 根据 IBO 计算工作点峰值电压
        # 假设输入信号归一化幅度为 1.0 对应 0dB IBO
        v_op_peak = v_sat_phys * (10 ** (-ibo_dB / 20.0))

        r_norm = np.abs(input_signal)
        r_phys = r_norm * v_op_peak
        phase_in = np.angle(input_signal)

        # AM-AM
        denominator_am = 1.0 + self.beta_a * (r_phys ** 2)
        a_out_phys = (self.alpha_a * r_phys) / denominator_am

        # AM-PM
        denominator_pm = 1.0 + self.beta_phi * (r_phys ** 2)
        phi_distortion = (self.alpha_phi * (r_phys ** 2)) / denominator_pm

        output_phys = a_out_phys * np.exp(1j * (phase_in + phi_distortion))

        # SCR (Sensitivity Compression Ratio) 计算 (用于理论曲线修正)
        beta_r2 = self.beta_a * (r_phys ** 2)
        numerator = 1.0 - beta_r2
        denominator = 1.0 + beta_r2
        # 避免除零
        scr = numerator / (denominator + 1e-12)

        return output_phys, scr, v_op_peak

    def get_pa_curves(self):
        """ 生成 PA 特性曲线数据 (用于绘图) """
        r_norm = np.linspace(0, 2.0, 500)
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)
        r_phys = r_norm * v_sat_phys

        denom = 1.0 + self.beta_a * (r_phys ** 2)
        a_out = (self.alpha_a * r_phys) / denom
        a_out_norm = a_out / (self.alpha_a * v_sat_phys)

        beta_r2 = self.beta_a * (r_phys ** 2)
        scr = (1.0 - beta_r2) / (1.0 + beta_r2)

        pin_db = 20 * np.log10(r_norm + 1e-9)

        return pin_db, a_out_norm, scr
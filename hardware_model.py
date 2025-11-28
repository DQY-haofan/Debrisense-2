import numpy as np


class HardwareImpairments:
    """
    IEEE TWC 级别硬件损伤仿真引擎 (Enhanced)
    """

    def __init__(self, config):
        self.alpha_a = config.get('alpha_a', 10.127)
        self.beta_a = config.get('beta_a', 5995.0)
        self.alpha_phi = config.get('alpha_phi', 4.0033)
        self.beta_phi = config.get('beta_phi', 9.1040)

        self.jitter_rms = config.get('jitter_rms', 2.0e-6)
        self.f_knee = config.get('f_knee', 200.0)

    def generate_colored_jitter(self, N_samples, fs):
        """ 生成有色 Jitter (IFFT 方法) """
        freqs = np.fft.rfftfreq(N_samples, d=1 / fs)
        safe_freqs = np.maximum(freqs, 1e-6)

        # PSD: DC-200Hz 平台, >200Hz 滚降
        psd_shape = (1.0 / safe_freqs ** 0.5) * (1.0 / (1.0 + (safe_freqs / self.f_knee) ** 8))
        psd_shape[0] = psd_shape[1]

        amplitude = np.sqrt(psd_shape)
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        spectrum = amplitude * random_phase

        jitter_raw = np.fft.irfft(spectrum, n=N_samples)

        # 归一化
        current_rms = np.std(jitter_raw)
        if current_rms == 0: current_rms = 1.0
        jitter_scaled = jitter_raw * (self.jitter_rms / current_rms)

        return jitter_scaled

    def apply_saleh_pa(self, input_signal, ibo_dB):
        """ 应用 Saleh PA 模型 (带电压定标) """
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)
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

        # SCR 计算
        beta_r2 = self.beta_a * (r_phys ** 2)
        numerator = 1.0 - beta_r2
        denominator = 1.0 + beta_r2
        scr = numerator / denominator  # 注意：过饱和时可能为负

        return output_phys, scr, v_op_peak

    def get_pa_curves(self):
        """ 生成 PA 特性曲线数据 (用于 Fig 2) """
        r_norm = np.linspace(0, 2.0, 500)  # 归一化输入幅度 0 -> 2 (0dB IBO at 1.0)
        # 假设 0dB IBO 对应 r_norm=1.0
        v_sat_phys = 1.0 / np.sqrt(self.beta_a)
        r_phys = r_norm * v_sat_phys  # 映射到物理电压

        # AM-AM
        denom = 1.0 + self.beta_a * (r_phys ** 2)
        a_out = (self.alpha_a * r_phys) / denom

        # 归一化输出以便绘图 (Normalized Output Amplitude)
        # 理想线性输出: alpha * r_phys
        a_out_norm = a_out / (self.alpha_a * v_sat_phys)

        # SCR
        beta_r2 = self.beta_a * (r_phys ** 2)
        scr = (1.0 - beta_r2) / (1.0 + beta_r2)

        # 输入功率 (相对于 P_sat 的 dB)
        pin_db = 20 * np.log10(r_norm + 1e-9)

        return pin_db, a_out_norm, scr
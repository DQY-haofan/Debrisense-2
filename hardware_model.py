import numpy as np


class HardwareImpairments:
    """
    IEEE TWC 级别硬件损伤仿真引擎 (v2.2 - PLL Fixed)
    """

    def __init__(self, config):
        # PA 参数
        self.alpha_a = config.get('alpha_a', 10.127)
        self.beta_a = config.get('beta_a', 5995.0)
        self.alpha_phi = config.get('alpha_phi', 4.0033)
        self.beta_phi = config.get('beta_phi', 9.1040)

        # Jitter 参数
        self.jitter_rms = config.get('jitter_rms', 2.0e-6)
        self.f_knee = config.get('f_knee', 200.0)

        # Phase Noise 参数
        self.L_1MHz = config.get('L_1MHz', -95.0)
        self.L_floor = config.get('L_floor', -120.0)
        self.f_corner = config.get('f_corner', 100e3)
        # [NEW] PLL 环路带宽：低于此频率的相位漂移被跟踪消除
        self.pll_bw = config.get('pll_bw', 10e3)  # 10 kHz

    def generate_colored_jitter(self, N_samples, fs):
        """ 生成有色 Jitter (幅度调制) """
        freqs = np.fft.rfftfreq(N_samples, d=1 / fs)
        safe_freqs = np.maximum(freqs, 1e-6)

        psd_shape = (1.0 / safe_freqs ** 0.5) * (1.0 / (1.0 + (safe_freqs / self.f_knee) ** 8))
        psd_shape[0] = 0.0  # 去除直流

        amplitude = np.sqrt(psd_shape)
        # RMS 归一化前先生成
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        spectrum = amplitude * random_phase
        jitter_raw = np.fft.irfft(spectrum, n=N_samples)

        current_rms = np.std(jitter_raw)
        if current_rms == 0: current_rms = 1.0
        jitter_scaled = jitter_raw * (self.jitter_rms / current_rms)

        return jitter_scaled

    def generate_phase_noise(self, N_samples, fs):
        """
        生成残留相位噪声 (Residual Phase Noise after PLL)
        [CRITICAL FIX] 引入 PLL 高通滤波，防止 RMS 爆炸
        """
        freqs = np.fft.rfftfreq(N_samples, d=1 / fs)
        safe_freqs = np.maximum(freqs, 1.0)

        # 线性化 dB 值
        S_floor = 10 ** (self.L_floor / 10.0)
        S_1MHz = 10 ** (self.L_1MHz / 10.0)

        # Leeson Model
        S_white_fm = S_1MHz * (1e6 / safe_freqs) ** 2
        S_flicker = S_white_fm * (self.f_corner / safe_freqs)
        psd_open_loop = S_white_fm + S_floor + np.where(safe_freqs < self.f_corner, S_flicker, 0)

        # [FIX] PLL 抑制函数: 理想二阶高通滤波器 |H(f)|^2
        # 低于 pll_bw 的噪声被抑制 (Tracking)
        pll_suppression = (safe_freqs / self.pll_bw) ** 4 / (1 + (safe_freqs / self.pll_bw) ** 4)

        psd_residual = psd_open_loop * pll_suppression

        # IFFT
        amplitude = np.sqrt(psd_residual * fs * N_samples / 2)
        amplitude[0] = 0.0  # 去除 DC

        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        theta_pn = np.fft.irfft(amplitude * phase, n=N_samples)

        return theta_pn

    def apply_saleh_pa(self, input_signal, ibo_dB):
        """ 应用 Saleh PA 模型 """
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

        # SCR
        beta_r2 = self.beta_a * (r_phys ** 2)
        scr = (1.0 - beta_r2) / (1.0 + beta_r2 + 1e-12)

        return output_phys, scr, v_op_peak

    def get_pa_curves(self):
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
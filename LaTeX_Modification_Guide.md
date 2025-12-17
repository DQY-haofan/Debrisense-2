# LaTeX 修改指南：Survival Space 论述修正

## 专家指出的问题及修复方案

本文档提供论文中需要修改的 LaTeX 代码，包括**原始版本**和**修正版本**的对比。

---

## 1. [P0-1] 模板域定义修正（致命问题）

### 问题描述
原论文中的能量保留率 η 使用的模板与检测域不一致。需要明确定义 log-envelope 域模板。

### 原始版本
```latex
The energy retention ratio is defined as
\begin{equation}
\eta = \frac{\|P_\perp s\|^2}{\|s\|^2}
\end{equation}
where $s(t)$ is the debris detection template.
```

### 修正版本
```latex
The energy retention ratio is defined in the \emph{log-envelope domain} as
\begin{equation}
\eta_z = \frac{\|P_\perp s_z\|^2}{\|s_z\|^2}
\end{equation}
where the log-envelope domain template is
\begin{equation}
s_z(t) = \mathbb{E}[z_{H_1}(t)] - \mathbb{E}[z_{H_0}(t)]
\end{equation}
with $z(t) = \ln|r(t)|$ being the log-envelope transform of the received signal $r(t)$.
This definition ensures consistency between the template used for the energy retention 
analysis and the actual detection domain.
```

---

## 2. [P0-2] Survival Space 频带表述统一

### 问题描述
论文中对 Survival Space 频带的表述存在歧义：一处说 "[f_cut, f_s/2]"，另一处说 "300 Hz - 5 kHz"。

### 原始版本
```latex
The survival space is defined as the frequency band $[f_{\mathrm{cut}}, f_s/2]$ 
where the debris detection signal is preserved. The debris chirp ridge is 
preserved in the 300~Hz--5~kHz band.
```

### 修正版本
```latex
The projection operator $P_\perp$ nulls DCT coefficients corresponding to 
frequencies below $f_{\mathrm{cut}}$, thus the \emph{preserved subspace} 
spans $[f_{\mathrm{cut}}, f_s/2]$.

Meanwhile, the debris modulation signal's \emph{main energy} concentrates 
in $[0, f_{\max}]$, where 
\begin{equation}
f_{\max} = \frac{v_{\mathrm{rel}}}{2 r_F} \approx 1.7~\mathrm{kHz}
\end{equation}
for $v_{\mathrm{rel}} = 15$~km/s.

By choosing $f_{\mathrm{cut}} \in (f_{\mathrm{knee}}, f_{\max})$, we 
simultaneously suppress low-frequency 1/f jitter noise while preserving 
the target signal energy.
```

---

## 3. [P0-3] f_knee 工作范围不等式修正

### 问题描述
原论文推导出 f_knee < f_max/2 ≈ 840 Hz，但这与 f_cut = 1.5×f_knee 的设计准则矛盾。

### 原始版本
```latex
The method remains effective as long as $f_{\mathrm{knee}} < f_{\max}/2$, 
which yields approximately $f_{\mathrm{knee}} < 840$~Hz for typical parameters.
```

### 修正版本
```latex
For effective noise suppression without significant signal loss, we require:
\begin{equation}
f_{\mathrm{cut}} \leq \frac{f_{\max}}{2}
\end{equation}
With the design choice $f_{\mathrm{cut}} = 1.5 \times f_{\mathrm{knee}}$, 
this constraint becomes:
\begin{equation}
1.5 \times f_{\mathrm{knee}} \leq \frac{f_{\max}}{2} 
\quad \Rightarrow \quad 
f_{\mathrm{knee}} \leq \frac{f_{\max}}{3}
\end{equation}
For $f_{\max} \approx 1677$~Hz, this yields $f_{\mathrm{knee}} \lesssim 560$~Hz, 
which is satisfied by typical THz oscillators with $f_{\mathrm{knee}} \approx 200$~Hz.
```

---

## 4. 信号带宽上限修正

### 问题描述
原论文声称信号带宽为 "300 Hz - 5 kHz"，但正确值应为 ~1.7 kHz。

### 原始版本
```latex
The debris modulation signal is preserved in the 300~Hz--5~kHz frequency band.
```

### 修正版本
```latex
The debris modulation signal bandwidth is determined by the Fresnel zone 
crossing time:
\begin{equation}
f_{\max} = \frac{1}{T_{\mathrm{cross}}} = \frac{v_{\mathrm{rel}}}{2 r_F}
\end{equation}
where $r_F = \sqrt{\lambda L_{\mathrm{eff}}}$ is the Fresnel zone radius.

For $\lambda = 1$~mm, $L_{\mathrm{eff}} = 20$~km, and $v_{\mathrm{rel}} = 15$~km/s:
\begin{align}
r_F &= \sqrt{10^{-3} \times 2 \times 10^4} = 4.47~\mathrm{m} \\
T_{\mathrm{cross}} &= \frac{2 r_F}{v_{\mathrm{rel}}} = \frac{8.94}{15000} \approx 0.60~\mathrm{ms} \\
f_{\max} &\approx 1677~\mathrm{Hz}
\end{align}
Thus, the debris signal energy concentrates in $[0, f_{\max}] \approx [0, 1.7]$~kHz.
```

---

## 5. 1/f 噪声模型表述修正

### 问题描述
专家指出 f_knee 的定义（高频 roll-off 拐点 vs 1/f 到白噪声地板的 knee）需要明确。

### 原始版本
```latex
The oscillator phase noise exhibits 1/f characteristics with a knee frequency 
$f_{\mathrm{knee}} \approx 200$~Hz, above which the noise becomes white.
```

### 修正版本
```latex
The log-envelope jitter is modeled as parameterized $1/f^\alpha$ colored noise 
with a high-frequency roll-off:
\begin{equation}
S_J(f) \propto \frac{1}{f^\alpha} \cdot \frac{1}{1 + (f/f_{\mathrm{knee}})^p}
\end{equation}
where $f_{\mathrm{knee}}$ denotes the onset of high-frequency roll-off 
(typically $\sim$200~Hz for THz oscillators), and $\alpha \approx 0.5$--$1$ 
characterizes the low-frequency slope.

This model captures the dominant low-frequency noise behavior without 
assuming an explicit white noise floor.
```

---

## 6. 能量保留率与噪声去除率的定量结果

### 新增内容（建议添加）
```latex
\subsection{Quantitative Validation}

The effectiveness of the survival space design is validated through two metrics:

\textbf{Signal Energy Retention:} At $f_{\mathrm{cut}} = 300$~Hz, the 
log-envelope domain energy retention ratio
\begin{equation}
\eta_z = \frac{\|P_\perp s_z\|^2}{\|s_z\|^2} > 99.9\%
\end{equation}
for all $v_{\mathrm{rel}} \in [10, 20]$~km/s, confirming that the projection 
preserves essentially all debris signal energy.

\textbf{Noise Energy Removal:} The fraction of 1/f jitter noise energy 
removed by the projection is
\begin{equation}
\rho_{\mathrm{noise}} = 1 - \frac{\|P_\perp n\|^2}{\|n\|^2} \approx 73\%
\end{equation}
at $f_{\mathrm{cut}} = 300$~Hz, demonstrating effective low-frequency noise 
suppression.

These quantitative results validate the survival space design choice: 
$f_{\mathrm{cut}} = 300$~Hz achieves $>$99\% signal preservation while 
removing $\sim$73\% of the 1/f noise energy.
```

---

## 7. Figure Caption 修正

### 原始 Figure Caption
```latex
\begin{figure}
\centering
\includegraphics[width=\columnwidth]{fig5_spectrogram.png}
\caption{Spectrogram of the received signal showing the survival space 
frequency band (300~Hz--5~kHz).}
\label{fig:survival_space}
\end{figure}
```

### 修正版本（替换整个图）
```latex
\begin{figure*}
\centering
\includegraphics[width=\textwidth]{Fig_Survival_Space_Concept_FIXED.png}
\caption{Survival space processing demonstration in the log-envelope domain.
(a) Log-envelope signal $z(t) = \ln|r(t)|$ for H0 (no debris) and H1 (debris present).
(b) DCT spectrum before projection, showing 1/f noise dominance at low frequencies.
(c) DCT spectrum after $P_\perp$ projection, with $f < f_{\mathrm{cut}}$ nulled.
(d) Time-domain signal after projection.
(e) Detection statistics distribution (AUC = 0.92).
(f) Design summary showing key parameters and metrics.
All processing operates in the log-envelope domain with $f_{\mathrm{cut}} = 300$~Hz.}
\label{fig:survival_space}
\end{figure*}
```

---

## 8. 新增 Sanity Check 图

### 建议添加到论文
```latex
\begin{figure}
\centering
\includegraphics[width=\columnwidth]{Fig_Sanity_Check_Eta_FIXED.png}
\caption{Survival space validation.
(a) Energy retention $\eta_z$ versus cutoff frequency $f_{\mathrm{cut}}$ 
for different relative velocities. At $f_{\mathrm{cut}} = 300$~Hz, 
$\eta_z > 99.9\%$ for all velocities.
(b) DCT spectrum of the log-envelope domain template $s_z(t)$, showing 
signal energy distribution relative to the cutoff and noise knee frequencies.
(c) 1/f noise energy removal ratio versus $f_{\mathrm{cut}}$. At 
$f_{\mathrm{cut}} = 300$~Hz, approximately 73\% of noise energy is removed.}
\label{fig:sanity_check}
\end{figure}
```

---

## 修正总结表

| 问题编号 | 原始表述 | 修正表述 | 影响 |
|---------|---------|---------|-----|
| P0-1 | η 用非 log-envelope 域模板 | η_z 用 s_z(t) = E[z_H1 - z_H0] | **致命→已修复** |
| P0-2 | "300 Hz - 5 kHz" | 投影保留 [f_cut, f_s/2]，信号集中在 [0, f_max] | 中等→已修复 |
| P0-3 | f_knee < f_max/2 ≈ 840 Hz | f_knee < f_max/3 ≈ 560 Hz | 中等→已修复 |
| 带宽 | 5 kHz | ~1.7 kHz (from f_max = v_rel/2r_F) | 高→已修复 |
| 噪声模型 | "knee 后变白" | "1/f^α with roll-off" | 低→已修复 |

---

## 关键数值结果（用于论文）

| 参数 | 数值 | 说明 |
|------|------|------|
| r_F | 4.47 m | Fresnel zone 半径 |
| T_cross | 0.60 ms | Fresnel 穿越时间 (v=15 km/s) |
| f_max | 1677 Hz | 信号带宽上限 |
| f_cut | 300 Hz | 设计选择 |
| f_knee (typical) | 200 Hz | 典型振荡器 |
| f_knee (limit) | 560 Hz | 方法有效上限 |
| η_z (at 300 Hz) | >99.9% | 信号能量保留 |
| ρ_noise (at 300 Hz) | ~73% | 噪声能量去除 |
| AUC (at SNR=50dB) | 0.92 | 检测性能 |

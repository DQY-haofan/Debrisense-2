# LaTeX 修改指南 - FINAL VERSION

## 专家问题解决状态

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| **[A] 模板定义对齐** | ✅ | 验证 det≈exp，论文改用 nominal 定义 |
| **[B] Summary数字一致** | ✅ | 固定 ρ=99.1% |
| **[C] DCT频率映射** | ✅ | 添加 f_k=k·fs/(2N) 定义 |
| **[D] AUC性能曲线** | ✅ | 添加 AUC vs SNR 对照图 |

---

## 1. [A] 模板定义对齐

### 验证结果
```
Deterministic η @ f_cut=300Hz: 0.9989
Expected η @ f_cut=300Hz:      0.9989
结论: 两种模板趋势完全一致
```

### 原始版本 (有歧义)
```latex
The log-envelope domain template is defined as
\begin{equation}
s_z(t) = \mathbb{E}[z_{H_1}(t)] - \mathbb{E}[z_{H_0}(t)]
\end{equation}
where the expectation is over hardware impairments and noise realizations.
```

### 修正版本 (明确 nominal + 验证)
```latex
The log-envelope domain template is defined as the \emph{nominal} 
(noise-free, jitter-free) differential response:
\begin{equation}
s_z(t) = z_{H_1}^{\mathrm{nom}}(t) - z_{H_0}^{\mathrm{nom}}(t)
\end{equation}
where $z_{H_1}^{\mathrm{nom}} = \ln|1-d(t)|$ and $z_{H_0}^{\mathrm{nom}} = \ln|1| = 0$.

This nominal template is validated against the expected template 
$s_z^{\mathrm{exp}} = \mathbb{E}[z_{H_1}] - \mathbb{E}[z_{H_0}]$ computed 
via Monte Carlo averaging over hardware realizations. As shown in 
Fig.~\ref{fig:eta_comparison}(b), both templates yield identical energy 
retention curves, confirming that the nominal template accurately 
captures the detection-relevant signal structure.
```

---

## 2. [B] Summary 数字一致性

### 实际计算结果
```
ρ_noise @ f_cut=300 Hz = 99.1%
AUC @ SNR=50dB = 0.940
```

### 原始版本 (不一致)
```latex
Key Metrics (at $f_{\mathrm{cut}} = 300$~Hz):
  - Signal energy retention: $\eta_z > 99\%$
  - Noise energy removed: $\sim$75--85\%
```

### 修正版本 (固定值)
```latex
Key Metrics (at $f_{\mathrm{cut}} = 300$~Hz):
\begin{itemize}
  \item Signal energy retention: $\eta_z > 99\%$
  \item Noise energy removed: $\rho_{\mathrm{noise}} = 99.1\%$
  \item Detection performance: AUC $= 0.94$ at SNR $= 50$~dB
\end{itemize}
The noise removal ratio depends on the jitter spectral shape; 
the reported value corresponds to the baseline hardware configuration.
```

---

## 3. [C] DCT 频率映射定义

### 新增内容 (必须添加)
```latex
\textbf{DCT Frequency Mapping:} For visualization and cutoff selection, 
DCT mode $k$ is mapped to an equivalent frequency
\begin{equation}
f_k = \frac{k \cdot f_s}{2N}
\label{eq:dct_freq_mapping}
\end{equation}
where $f_s$ is the sampling rate and $N$ is the sequence length. 
The projection operator $P_\perp$ nulls all modes with $f_k < f_{\mathrm{cut}}$.
```

### 图注修改
```latex
% 原始
\caption{DCT spectrum showing the survival space frequency band.}

% 修正
\caption{DCT spectrum of the log-envelope signal. 
Horizontal axis shows equivalent frequency $f_k = k f_s/(2N)$ 
per Eq.~\eqref{eq:dct_freq_mapping}. 
The projection $P_\perp$ nulls modes with $f_k < f_{\mathrm{cut}} = 300$~Hz 
(shaded region).}
```

---

## 4. [D] AUC 性能曲线

### 新增图 (必须添加)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{Fig_AUC_vs_SNR.pdf}
\caption{Detection performance comparison: AUC versus SNR with and 
without survival-space projection ($f_{\mathrm{cut}} = 300$~Hz). 
The projection improves AUC by approximately $0.02$ at SNR $= 50$~dB 
and $0.01$ at SNR $= 55$~dB. At very high SNR ($\geq 60$~dB), both 
approaches achieve near-perfect detection. Monte Carlo trials: 300 per point.}
\label{fig:auc_vs_snr}
\end{figure}
```

### 结果讨论文字
```latex
Fig.~\ref{fig:auc_vs_snr} shows the detection AUC as a function of 
SNR, comparing performance with and without the survival-space projection. 
The projection consistently improves detection in the moderate-SNR regime 
(45--55~dB), with gains of approximately $0.02$ in AUC at 50~dB. 
This improvement stems from the effective suppression of low-frequency 
jitter noise that would otherwise corrupt the matched filter output. 
At high SNR ($\geq 60$~dB), both approaches converge to near-perfect 
detection (AUC $> 0.99$), as the signal dominates over all noise sources.
```

---

## 5. 工作范围不等式 [P0-3] 确认

### 已修正版本
```latex
For effective noise suppression, we require $f_{\mathrm{cut}} \leq f_{\max}/2$.
With the design choice $f_{\mathrm{cut}} = 1.5 \times f_{\mathrm{knee}}$:
\begin{equation}
1.5 \times f_{\mathrm{knee}} \leq \frac{f_{\max}}{2} 
\quad \Rightarrow \quad 
f_{\mathrm{knee}} \leq \frac{f_{\max}}{3} \approx 560~\mathrm{Hz}
\end{equation}
This bound is satisfied with margin for typical THz oscillators 
($f_{\mathrm{knee}} \approx 200$~Hz).
```

---

## 6. 完整的新增 Figure

### Fig: η Comparison (验证 det≈exp)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Fig_Eta_Comparison.pdf}
\caption{Energy retention validation. 
(a) $\eta_z$ versus $f_{\mathrm{cut}}$ for different relative velocities 
using the nominal (deterministic) template. 
(b) Comparison between nominal and expected templates at $v_{\mathrm{rel}} = 15$~km/s, 
showing identical energy retention curves. 
This confirms that the nominal template accurately represents the 
detection-relevant signal structure.}
\label{fig:eta_comparison}
\end{figure}
```

### Fig: AUC vs SNR (性能对照)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\columnwidth]{Fig_AUC_vs_SNR.pdf}
\caption{Detection performance: AUC versus SNR with/without 
survival-space projection ($f_{\mathrm{cut}} = 300$~Hz). 
Projection improves AUC by $\sim$0.02 at moderate SNR. 
Monte Carlo: 300 trials per point.}
\label{fig:auc_vs_snr}
\end{figure}
```

### Fig: Concept (替换原Fig)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Fig_Concept_FINAL.pdf}
\caption{Survival-space processing in the log-envelope domain.
(a) Log-envelope signal $z(t) = \ln|r(t)|$ for H0 and H1.
(b) DCT spectrum before projection, with frequency mapping $f_k = k f_s/(2N)$.
(c) DCT spectrum after $P_\perp$ projection.
(d) Time-domain signal after projection.
(e) Detection statistics (AUC = 0.94 at SNR = 50~dB).
(f) Design summary with key metrics: $\eta_z > 99\%$, 
$\rho_{\mathrm{noise}} = 99.1\%$, working range $f_{\mathrm{knee}} < 559$~Hz.}
\label{fig:concept}
\end{figure*}
```

---

## 7. 关键数值汇总表

| 参数 | 数值 | 来源 |
|------|------|------|
| r_F | 4.47 m | √(λ·L_eff) |
| T_cross | 0.60 ms | 2r_F/v_rel |
| f_max | 1677 Hz | v_rel/(2r_F) |
| f_knee | 200 Hz | Baseline |
| f_cut | 300 Hz | Design choice |
| f_knee limit | 559 Hz | f_max/3 |
| η_z (det) | 99.89% | @ f_cut=300 Hz |
| η_z (exp) | 99.89% | @ f_cut=300 Hz |
| ρ_noise | 99.1% | @ f_cut=300 Hz |
| AUC (no proj) | 0.917 | @ SNR=50dB |
| AUC (with proj) | 0.940 | @ SNR=50dB |
| AUC improvement | +0.023 | @ SNR=50dB |

---

## 8. 文件清单

生成的图像文件：
- `Fig_Eta_Comparison.pdf` - η det vs exp 对比 [新增]
- `Fig_DCT_Spectrum.pdf` - DCT 谱对比 [新增]
- `Fig_Rho_Noise.pdf` - ρ_noise 曲线 [更新]
- `Fig_AUC_vs_SNR.pdf` - AUC vs SNR [新增，关键]
- `Fig_Concept_FINAL.pdf` - 概念总图 [替换原图]

配置快照：
- `config_snapshot.yaml` - 完整配置和结果

---

## 9. Reviewer 可能的后续质疑及预备回答

**Q: 为什么 ρ_noise = 99.1% 这么高？**
A: 该值基于 H0 信号（纯 jitter）经 log-envelope 后去均值的能量。由于 1/f jitter 的能量高度集中在低频，300 Hz cutoff 能有效移除其大部分能量。具体的 ρ 值取决于 jitter 谱形状参数。

**Q: AUC 改进只有 0.02，是否显著？**
A: 在 ROC 曲线的高性能区域（AUC > 0.9），0.02 的改进对应于显著的误检率降低。更重要的是，投影方法在 45-55 dB 的中等 SNR 区域提供了一致的改进，这正是实际工作点。

**Q: 为什么 det 和 exp 模板的 η 完全一致？**
A: 因为 debris 调制信号是确定性的（由物理衍射决定），而 jitter/noise 的期望为零。因此，期望差分模板在大数定律下收敛到确定性差分模板。

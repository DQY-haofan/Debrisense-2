# THz-ISL Forward Scatter Debris Detection - Reproducibility Guide

## 📋 项目概述

本项目实现了THz星间链路前向散射碎片检测的完整仿真代码，符合IEEE顶刊可复现性标准。

### 核心特性

- ✅ **单一真源配置 (SSOT)**: 所有参数通过 `config/paper_baseline.yaml` 集中管理
- ✅ **论文版本检测链路**: log-envelope + survival-space + GLRT（非应急dip/peak方案）
- ✅ **2D ML网格搜索**: 速度-时间联合估计
- ✅ **参数化Jitter PSD**: 1/f^α 家族，支持敏感性对比
- ✅ **完整可复现证据链**: CSV/PNG/PDF + config snapshot + run.log + seed

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install numpy scipy matplotlib pyyaml pandas tqdm joblib

# 可选：安装numba加速
pip install numba
```

### 2. 验证配置

```bash
# 运行配置审计（必须通过）
python audit_config.py config/paper_baseline.yaml
```

### 3. 运行Sanity Check（关键！）

```bash
# 验证能量保留率 η > 0.01
python run_all_figures.py --sanity-check
```

### 4. 生成所有论文图

```bash
# 一条命令生成全部
python run_all_figures.py --config config/paper_baseline.yaml --seed 42
```

---

## 📁 项目结构

```
thz_isl_project/
├── config/
│   └── paper_baseline.yaml    # 单一真源配置文件
├── config_manager.py          # 配置管理器
├── audit_config.py            # 配置一致性审计工具
├── detector.py                # 检测器（论文版本）
├── estimator.py               # 2D ML估计器
├── hardware_model.py          # 硬件损伤模型
├── physics_engine.py          # 衍射物理引擎
├── run_all_figures.py         # 出图主脚本
├── REPRO.md                   # 本文件
└── outputs/                   # 输出目录
    └── thz_isl_v1/
        ├── sanity_check/
        ├── fig2/
        ├── fig3/
        ├── fig7/
        ├── fig10/
        └── alpha_sensitivity/
```

---

## 📊 输出文件说明

每个图的输出目录包含：

| 文件 | 说明 |
|------|------|
| `figure.png` | 图片预览 (300 DPI) |
| `figure.pdf` | 出版质量矢量图 |
| `data.csv` | 原始数据 |
| `config_snapshot.yaml` | 本次运行的配置快照 |
| `run.log` | 运行日志（含seed、版本、git hash）|

---

## ⚙️ 关键配置参数

### Baseline 参数（默认值）

| 参数 | 值 | 单位 | 说明 |
|------|-----|------|------|
| `fc` | 300 | GHz | 载波频率 |
| `B` | 10 | GHz | 带宽 |
| `L_eff` | 50 | km | 有效链路长度 |
| `a` | 5 | cm | 碎片半径 |
| `v_default` | 15000 | m/s | 默认相对速度 |
| `fs` | 200 | kHz | 采样频率 |
| `T_span` | 20 | ms | 观测窗口 |
| `f_cut` | 300 | Hz | DCT投影截止频率 |
| `psd_alpha` | 0.5 | - | Jitter PSD指数 |
| `sigma_j` | 1e-6 | - | Jitter RMS（无量纲）|

### 图特定参数变化

每张图仅改变 **一个** 自变量，其余保持baseline：

- **Fig 6**: Sweep IBO, 固定SNR=70dB
- **Fig 7**: SNR=50dB, jitter_sigma=2e-3
- **Fig 8**: 固定SNR=68dB, sweep碎片直径
- **Fig 10**: Ambiguity function grid
- **α敏感性**: 对比α=0.5 vs α=1.0

---

## 🔬 检测链路说明

### 论文版本（默认启用）

```
y[n] → log(|·|+ε) → P_perp投影 → [可选whitening] → GLRT统计量
         ↑              ↑                              ↑
      Step 1         Step 2                         Step 4
```

1. **Log-envelope**: `x[n] = log(|y[n]| + ε)`
2. **Survival-space**: `z = P_perp @ x`, 其中 `P_perp = I - H @ H^T`
3. **GLRT**: `T = (s_perp^T @ z)^2 / ||s_perp||^2`

### ⚠️ 已弃用方法

`_deprecated_detect_dip_peak()` 仅用于DEBUG，**禁止用于论文出图**！

---

## ✅ 验收检查清单

在提交论文前，确保：

- [ ] `audit_config.py` 通过（无ERROR）
- [ ] Sanity check: η(f_cut=300Hz) > 0.01 对所有速度
- [ ] 所有图导出 CSV/PNG/PDF
- [ ] 每张图有 config_snapshot.yaml
- [ ] run.log 包含 seed 和 git hash

---

## 🔄 常用命令

```bash
# 配置审计
python audit_config.py

# 仅运行sanity check
python run_all_figures.py --sanity-check

# 生成所有图
python run_all_figures.py

# 生成指定图
python run_all_figures.py --figures fig7 fig10

# 使用特定seed
python run_all_figures.py --seed 12345

# 严格模式审计（warnings视为error）
python audit_config.py --strict
```

---

## 🐛 故障排除

### Q: Sanity check 失败（η ≈ 0）

**原因**: 模板主要是低频"深坑"，被DCT投影消除  
**解决**: 
1. 检查 `L_eff` 是否匹配（near-terminal vs mid-ISL）
2. 降低 `f_cut`（但会引入更多jitter）
3. 确认模板包含chirp结构

### Q: 配置审计报错

运行 `python audit_config.py` 查看详细错误信息。常见问题：
- 参数缺失：补全YAML
- 参数不一致：统一到baseline
- 模式错误：确保 `mode: "paper"`

### Q: 导入错误

确保从项目根目录运行，或添加路径：
```python
import sys
sys.path.insert(0, '/path/to/thz_isl_project')
```

---

## 📚 参考文献

- 论文：THz-ISL Forward Scatter Debris Detection
- DR_algo_01：Survival space检测理论
- Saleh PA模型：AM-AM/AM-PM非线性

---

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2025-12 | 初始版本，完整实现P0-P3 |

---

**Author**: Refactored for IEEE TWC reproducibility standards

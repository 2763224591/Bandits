# 铝合金 6082 预锻仿真数据处理 → 代理模型 → 强化学习决策系统

## 项目概述

本项目实现了一个基于**上下文多臂老虎机 (Contextual Multi-Armed Bandit)**的铝合金锻造工艺优化系统。
通过代理模型学习锻造过程中的应力、损伤等物理场特征，并据此决策最优连杆比（速度）以平衡锻造质量和生产效率。

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      输入特征 (56 维)                        │
│  - 3 区域 (A 高应力/B 主体/C 突起) × 4 物理场 × 4 统计量       │
│  - 3 全局量 + 2 上下文 (欠压，摩擦系数)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Encoder 层 (可配置)                       │
│         ┌──────────────────┬──────────────────┐             │
│         │   Transformer    │       TFT        │             │
│         │  (自注意力机制)   │  (LSTM+ 注意力)   │             │
│         └──────────────────┴──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Uncertainty 层 (可配置)                      │
│    ┌──────────────┬──────────────┬──────────────┐          │
│    │ MC-Dropout   │   Ensemble   │    None      │          │
│    │ (多次采样)    │  (模型集成)   │  (确定性)     │          │
│    └──────────────┴──────────────┴──────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Policy 层 (可配置)                          │
│     ┌─────────────────────┬─────────────────────┐          │
│     │    Neural UCB       │ Thompson Sampling   │          │
│     │ Q + β*σ (置信上界)   │  从后验分布采样      │          │
│     └─────────────────────┴─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    输出：最优连杆比 r_l                       │
│              [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] │
└─────────────────────────────────────────────────────────────┘
```

## 四种实验组合

| 配置 | Encoder | Uncertainty | Policy | 配置文件 |
|------|---------|-------------|--------|----------|
| 1 | Transformer | MC-Dropout | NeuralUCB | `config_1_transformer_mcdropout_ucb.yaml` |
| 2 | Transformer | MC-Dropout | ThompsonSampling | `config_2_transformer_mcdropout_ts.yaml` |
| 3 | TFT | Ensemble | NeuralUCB | `config_3_tft_ensemble_ucb.yaml` |
| 4 | TFT | Ensemble | ThompsonSampling | `config_4_tft_ensemble_ts.yaml` |

## 目录结构

```
/workspace
├── config/                          # 配置文件目录
│   ├── experiment_config.yaml       # 默认配置
│   ├── config_1_*.yaml              # 配置 1
│   ├── config_2_*.yaml              # 配置 2
│   ├── config_3_*.yaml              # 配置 3
│   └── config_4_*.yaml              # 配置 4
├── src/                             # 源代码目录
│   ├── data_loader.py               # 数据加载与预处理
│   ├── encoders.py                  # Transformer/TFT编码器
│   ├── uncertainty.py               # MC-Dropout/Ensemble不确定性估计
│   ├── policy.py                    # NeuralUCB/ThompsonSampling策略
│   └── train.py                     # 主训练模块
├── data/                            # 数据目录
│   └── features/
│       └── features_all.csv         # 降维后的特征数据 (待放入)
├── models/                          # 保存的模型权重
├── results/                         # 训练结果和日志
└── README.md                        # 本文件
```

## 快速开始

### 1. 准备数据

将降维后的特征数据放置在 `data/features/features_all.csv`，包含以下列：
- 56 维物理特征 (PHYS_COLS)
- 2 维上下文：`underfill`, `mu`
- 1 维动作：`r_l`
- 1 维步骤：`step`

### 2. 运行实验

```bash
cd /workspace/src

# 使用默认配置训练
python train.py --config ../config/experiment_config.yaml

# 或使用特定配置
python train.py --config ../config/config_1_transformer_mcdropout_ucb.yaml
python train.py --config ../config/config_2_transformer_mcdropout_ts.yaml
python train.py --config ../config/config_3_tft_ensemble_ucb.yaml
python train.py --config ../config/config_4_tft_ensemble_ts.yaml
```

### 3. 修改配置

只需编辑 YAML 配置文件中的三个关键参数即可切换架构：

```yaml
model:
  encoder_type: "transformer"    # 或 "tft"
  uncertainty_type: "mc_dropout" # 或 "ensemble", "none"
  policy_type: "neural_ucb"      # 或 "thompson_sampling"
```

## 核心模块说明

### 数据加载器 (`data_loader.py`)

- **ForgeDataLoader**: 负责读取 CSV 数据，构建上下文老虎机数据集
- **奖励函数设计**:
  - 高质量优先：损伤/应力超标时重罚 (权重 10.0)
  - 速度奖励：连杆比越大越快 (权重 1.0)
  - 公式：`reward = speed_reward - quality_penalty`

### 编码器 (`encoders.py`)

- **TransformerEncoder**: 基于自注意力机制的序列编码
- **TemporalFusionTransformer (TFT)**: 结合 LSTM 和注意力的时间序列编码
- 统一接口：`create_encoder(encoder_type, input_dim, **kwargs)`

### 不确定性估计 (`uncertainty.py`)

- **MC-Dropout**: 推理时保持 Dropout，多次采样估计方差
- **Ensemble**: 训练多个独立模型，通过集成估计不确定性
- **QNetworkWithUncertainty**: 整合 Encoder 和 Uncertainty 的 Q 网络

### 策略模块 (`policy.py`)

- **NeuralUCB**: `UCB(a) = Q(s,a) + β * σ(s,a)`
- **ThompsonSampling**: 从 N(Q, σ²) 中采样选择动作
- 统一接口：`create_policy(policy_type, n_actions, actions, **kwargs)`

### 训练器 (`train.py`)

- **ContextualBanditTrainer**: 整合所有组件的训练框架
- 支持早停、模型保存、结果记录
- 输出 JSON 格式的训练曲线

## 奖励函数设计

根据需求"**保证锻造效果好的前提下（高优先级）速度尽可能快（相对低优先级）**"：

```python
# 质量惩罚 (高优先级)
if max_damage > damage_threshold:
    penalty += w_quality * (max_damage - threshold) ** 2
if max_stress > stress_threshold:
    penalty += w_quality * ((max_stress - threshold) / threshold) ** 2

# 速度奖励 (低优先级)
speed_reward = w_speed * (r_l - min_r_l) / (max_r_l - min_r_l)

# 总奖励
total_reward = speed_reward - quality_penalty
```

权重设置：`w_quality = 10.0`, `w_speed = 1.0`

## 依赖项

```txt
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
pyyaml>=5.4.0
scikit-learn>=0.24.0  # 可选，用于标准化
```

## 实验建议

1. **基线对比**: 先用 `uncertainty_type: "none"` + `epsilon_greedy` 建立基线
2. **不确定性收益**: 对比 MC-Dropout/Ensemble vs None 的性能提升
3. **策略对比**: 在相同 Encoder+Uncertainty 下对比 UCB vs Thompson
4. **超参调优**: 重点调整 `exploration_bonus` (UCB) 和 `prior_alpha` (Thompson)

## 注意事项

- 由于锻造过程中**连杆比全程不变**，这是典型的单步决策问题（多臂老虎机），而非多步 MDP
- 每个轨迹 (underfill, mu 组合) 的所有步骤共享相同的奖励
- 数据划分应**按轨迹**而非按样本，避免数据泄露
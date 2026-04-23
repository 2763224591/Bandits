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

## 六种实验组合 (含基线对比)

| 配置 | Encoder | Uncertainty | Policy | 说明 | 配置文件 |
|------|---------|-------------|--------|------|----------|
| 1 | Transformer | MC-Dropout | NeuralUCB | 原始实验 1 | `config_1_transformer_mcdropout_ucb.yaml` |
| 2 | Transformer | MC-Dropout | ThompsonSampling | 原始实验 2 | `config_2_transformer_mcdropout_ts.yaml` |
| 3 | TFT | Ensemble | NeuralUCB | 原始实验 3 | `config_3_tft_ensemble_ucb.yaml` |
| 4 | TFT | Ensemble | ThompsonSampling | 原始实验 4 | `config_4_tft_ensemble_ts.yaml` |
| **5** | **Transformer** | **None** | **Epsilon-Greedy** | **基线对比 1** | `config_5_transformer_none_egreedy.yaml` |
| **6** | **TFT** | **None** | **Epsilon-Greedy** | **基线对比 2** | `config_6_tft_none_egreedy.yaml` |

### 基线设计说明

- **Uncertainty=None**: 确定性模型，不估计不确定性，用于验证不确定性估计的实际收益
- **Policy=Epsilon-Greedy**: 经典探索策略，以概率 ε 随机选择动作，作为 UCB/Thompson Sampling 的对比基线
- **建议实验顺序**: 先运行配置 5/6 建立性能基线，再对比配置 1-4 评估高级模块的价值

## 目录结构

```
/workspace
├── config/                          # 配置文件目录
│   ├── experiment_config.yaml       # 默认配置
│   ├── config_1_*.yaml              # 配置 1: Transformer+MC-Dropout+UCB
│   ├── config_2_*.yaml              # 配置 2: Transformer+MC-Dropout+TS
│   ├── config_3_*.yaml              # 配置 3: TFT+Ensemble+UCB
│   ├── config_4_*.yaml              # 配置 4: TFT+Ensemble+TS
│   ├── config_5_*.yaml              # 配置 5: Transformer+None+ε-Greedy (基线)
│   ├── config_6_*.yaml              # 配置 6: TFT+None+ε-Greedy (基线)
│   └── README_experiments.md        # 实验配置详细说明
├── src/                             # 源代码目录
│   ├── data_loader.py               # 数据加载与预处理
│   ├── encoders.py                  # Transformer/TFT编码器
│   ├── uncertainty.py               # MC-Dropout/Ensemble/None 不确定性估计
│   ├── policy.py                    # NeuralUCB/ThompsonSampling/ε-Greedy 策略
│   ├── reward_utils.py              # Pareto 奖励函数与保守性评估
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

- **NeuralUCB**: `UCB(a) = Q(s,a) + β * σ(s,a)`，置信上界探索
- **ThompsonSampling**: 从 N(Q, σ²) 中采样选择动作，贝叶斯后验采样
- **EpsilonGreedy**: 以概率 ε 随机选择动作，经典基线策略 (支持衰减)
- 统一接口：`create_policy(policy_type, n_actions, actions, **kwargs)`

### 训练器 (`train.py`)

- **ContextualBanditTrainer**: 整合所有组件的训练框架
- 支持早停、模型保存、结果记录
- 输出 JSON 格式的训练曲线

## 奖励函数设计

### 核心设计理念

根据需求"**保证锻造效果好的前提下（高优先级）速度尽可能快（相对低优先级）**"，采用**Pareto 效用函数**形式：

```python
# R = Quality × Speed_Utility (非线性乘积形式)
reward = quality_score * speed_utility
```

这种形式体现了"安全优先，兼顾效率"的非线性权衡：
- **质量差 (Q≈0)**: 无论速度多快，奖励都很低 → 强制保证质量
- **质量好 (Q≈1)**: 速度效用决定最终奖励 → 在安全前提下追求效率
- **膝点设计**: 速度效用函数在 r_l=0.75 处达到峰值，模拟工业 Pareto 前沿

### 质量分数计算 (`reward_utils.py`)

```python
def calculate_quality_score(row):
    penalties = []
    
    # 1. 损伤惩罚 (权重最高，DAMAGE max > 250 时重罚)
    for region in ['A', 'B', 'C']:
        dmg_max = row[f"{region}__DAMAGE__max"]
        if dmg_max > 250:
            penalties.append(2.0 * (dmg_max - 250) / 100)
    
    # 2. 应力均匀性惩罚 (变异系数 CV > 0.5 时惩罚)
    for region in ['A', 'B', 'C']:
        cv = vm_std / (vm_mean + 1e-6)
        if cv > 0.5:
            penalties.append(0.5 * (cv - 0.5))
    
    # 3. 绝对应力水平惩罚 (global VM mean > 250 时惩罚)
    if vm_global > 250:
        penalties.append(0.3 * (vm_global - 250) / 100)
    
    return max(0.0, 1.0 - sum(penalties))
```

### 速度效用函数

```python
def compute_speed_utility(speed, knee_point=0.75):
    # 归一化到 [0, 1]
    v_norm = (speed - 0.6) / 0.4
    k_norm = (0.75 - 0.6) / 0.4
    
    if v_norm <= k_norm:
        # 上升段：0.6 → 0.75，效用递增
        utility = v_norm / k_norm
    else:
        # 下降段：0.75 → 1.0，效用递减 (模拟质量风险)
        utility = 1.0 - 0.5 * (v_norm - k_norm) / (1.0 - k_norm)
    
    return utility
```

### 保守性反事实评估 (解决数据泄露)

**问题**: 离线强化学习中，直接使用验证集真实标签计算奖励会导致数据泄露，模型可能过拟合评估函数。

**解决方案**: 采用**保守性反事实评估 (Conservative Counterfactual Evaluation)**

```python
# 方案 A+B 混合：保守性惩罚
R_final = R_oracle(a_pred) - Penalty(a_pred, a_BC)

# 惩罚项：偏离行为克隆 (BC) 分布的动作
Penalty = λ * |a_pred - a_BC_mean| / a_BC_std
```

**原理**:
1. **查找表构建**: 仅使用训练集构建 Pareto 奖励查找表和 BC 统计量表
2. **OOD 惩罚**: 对超出训练分布的动作进行惩罚，防止模型选择不可靠的 OOD 动作
3. **评估隔离**: 验证集评估时不访问真实标签，仅依赖训练集统计量

```python
# reward_utils.py 中的实现
def get_counterfactual_reward_conservative(
    lookup_table, bc_table, underfill, mu, 
    predicted_action, lambda_coeff=2.0
):
    # 1. 获取基础奖励 (来自训练集查找表)
    base_reward = get_counterfactual_reward(
        lookup_table, underfill, mu, predicted_action
    )
    
    # 2. 计算保守性惩罚
    penalty = lambda_coeff * |predicted_action - bc_mean| / bc_std
    
    # 3. 最终奖励
    return max(0.0, base_reward - penalty)
```

### 数据泄露防护措施

| 风险点 | 防护方案 | 实现位置 |
|--------|----------|----------|
| 验证集标签泄露 | 使用训练集构建查找表 | `build_pareto_lookup_table()` |
| OOD 动作过估计 | 保守性惩罚项 | `get_conservative_penalty()` |
| 轨迹级数据泄露 | 按轨迹划分训练/验证集 | `data_loader.train_val_split()` |
| 时间序列泄露 | 按 step 排序后取最终状态 | `_compute_rewards()` |

**关键实现细节**:
- `data_loader.py`: `train_val_split()` 按 `(underfill, mu)` 轨迹划分，确保同一轨迹的所有步骤在同一集合
- `reward_utils.py`: `build_conservative_lookup_table()` 仅使用训练集统计量
- `train.py`: 验证阶段调用 `get_counterfactual_reward_conservative()` 而非直接计算奖励

## 依赖项

```txt
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
pyyaml>=5.4.0
scikit-learn>=0.24.0  # 可选，用于标准化
```

## 实验建议

### 推荐实验顺序

1. **基线建立**: 运行配置 5/6 (`uncertainty_type: "none"` + `epsilon_greedy`) 建立性能基线
2. **不确定性收益评估**: 对比 MC-Dropout/Ensemble vs None 的性能提升
3. **策略对比**: 在相同 Encoder+Uncertainty 下对比 UCB vs Thompson Sampling vs ε-Greedy
4. **超参数调优**: 重点调整以下参数：
   - `exploration_bonus` (UCB): 控制探索强度，建议范围 [0.1, 2.0]
   - `prior_alpha`, `prior_beta` (Thompson): Beta 分布先验，建议 α=β=1 (均匀先验)
   - `epsilon`, `epsilon_decay` (ε-Greedy): 初始探索率及衰减，建议 ε₀=0.3, decay=0.99

### 完整配置矩阵

| 配置 | Encoder | Uncertainty | Policy | 用途 |
|------|---------|-------------|--------|------|
| 1 | Transformer | MC-Dropout | NeuralUCB | 注意力 + 采样不确定性 + 置信上界 |
| 2 | Transformer | MC-Dropout | ThompsonSampling | 注意力 + 采样不确定性 + 贝叶斯采样 |
| 3 | TFT | Ensemble | NeuralUCB | 时序融合 + 集成不确定性 + 置信上界 |
| 4 | TFT | Ensemble | ThompsonSampling | 时序融合 + 集成不确定性 + 贝叶斯采样 |
| 5 | Transformer | None | Epsilon-Greedy | **基线 1**: 确定性模型 + 经典探索 |
| 6 | TFT | None | Epsilon-Greedy | **基线 2**: 确定性模型 + 经典探索 |

### 预期分析维度

- **主效应分析**: Encoder/Uncertainty/Policy 各自主效应
- **交互效应**: 如"Transformer+MC-Dropout"是否优于"TFT+Ensemble"
- **不确定性价值**: 量化不确定性估计对探索效率的提升
- **收敛速度**: 对比不同策略达到稳定性能所需的训练轮次

## 注意事项

- 由于锻造过程中**连杆比全程不变**，这是典型的单步决策问题（多臂老虎机），而非多步 MDP
- 每个轨迹 (underfill, mu 组合) 的所有步骤共享相同的奖励
- 数据划分应**按轨迹**而非按样本，避免数据泄露
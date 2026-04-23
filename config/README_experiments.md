# 铝合金 6082 预锻仿真 - 实验配置说明

## 架构组件

本系统采用模块化设计，支持以下组件的自由组合：

### Encoder 层
- `transformer`: Transformer 编码器
- `tft`: Temporal Fusion Transformer 编码器

### Uncertainty 层
- `mc_dropout`: MC-Dropout 不确定性估计
- `ensemble`: Ensemble 模型集合不确定性估计
- `none`: 确定性模型（无不确定性估计，基线）

### Policy 层
- `neural_ucb`: Neural UCB (Upper Confidence Bound) 策略
- `thompson_sampling`: Thompson Sampling 策略
- `epsilon_greedy`: ε-Greedy 策略（基线对比）

## 实验配置矩阵

| 配置 | Encoder | Uncertainty | Policy | 说明 |
|------|---------|-------------|--------|------|
| config_1 | transformer | mc_dropout | neural_ucb | 原始实验 1 |
| config_2 | transformer | mc_dropout | thompson_sampling | 原始实验 2 |
| config_3 | tft | ensemble | neural_ucb | 原始实验 3 |
| config_4 | tft | ensemble | thompson_sampling | 原始实验 4 |
| **config_5** | **transformer** | **none** | **epsilon_greedy** | **基线对比 1** |
| **config_6** | **tft** | **none** | **epsilon_greedy** | **基线对比 2** |

## 运行实验

```bash
cd src

# 使用 experiment_config.yaml (默认配置)
python train.py

# 或使用特定配置文件
python train.py --config ../config/config_5_transformer_none_egreedy.yaml
```

## 配置说明

### experiment_config.yaml 核心参数

```yaml
model:
  encoder_type: "transformer"      # 或 "tft"
  uncertainty_type: "mc_dropout"   # 或 "ensemble", "none"
  policy_type: "neural_ucb"        # 或 "thompson_sampling", "epsilon_greedy"

hyperparameters:
  # Policy 参数
  exploration_bonus: 0.1      # UCB 探索系数
  prior_alpha: 1.0            # Thompson 先验 alpha
  prior_beta: 1.0             # Thompson 先验 beta
  epsilon: 0.1                # Epsilon-Greedy 初始探索率
  epsilon_decay: 0.995        # Epsilon 衰减率
  epsilon_min: 0.01           # Epsilon 最小值
```

## 基线对比说明

新增的基线配置用于评估：
1. **确定性 vs 不确定性**: 比较 `none` 与 `mc_dropout`/`ensemble` 的效果
2. **简单探索 vs 基于不确定性的探索**: 比较 `epsilon_greedy` 与 `neural_ucb`/`thompson_sampling` 的效果

通过对比可以验证：
- 不确定性估计是否带来性能提升
- 基于不确定性的探索策略是否优于简单随机探索

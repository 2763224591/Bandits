"""
数据分析脚本：回答"数据是否很烂，根本训不出什么东西"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

# 加载配置
with open('config/config_1_transformer_mcdropout_ucb.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 导入奖励函数
import sys
sys.path.insert(0, 'src')
from reward_utils import calculate_quality_score, compute_speed_utility, compute_reward_with_utility

print("=" * 70)
print("数据分析：速度变化对质量的影响是否显著？")
print("=" * 70)

# 加载数据
df = pd.read_csv(config['data']['path'])
print(f"\n数据总量：{len(df)} 条记录")

# 计算质量分数
print("计算质量分数...")
df['quality'] = df.apply(calculate_quality_score, axis=1)
df['speed_util'] = df['r_l'].apply(compute_speed_utility)
df['reward'] = df.apply(lambda x: compute_reward_with_utility(x['r_l'], x['quality']), axis=1)

# ========== 分析 1: 速度与质量的相关性 ==========
print("\n" + "=" * 70)
print("分析 1: 速度 (r_l) 与质量的关系")
print("=" * 70)

# 按速度分组统计质量
grouped = df.groupby('r_l')['quality'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
print("\n不同速度下的质量分数统计:")
print(grouped)

# 相关性
corr = df['r_l'].corr(df['quality'])
print(f"\n皮尔逊相关系数：{corr:.4f}")

if abs(corr) < 0.1:
    verdict_corr = "⚠️  **几乎不相关** - 速度变化对质量影响微乎其微"
elif corr < -0.3:
    verdict_corr = "✅ **负相关明显** - 速度越快质量越差，符合物理直觉"
elif corr > 0.3:
    verdict_corr = "❓ **正相关** - 速度越快质量越好？需要检查仿真设置"
else:
    verdict_corr = "ℹ️  **弱相关** - 有一定影响但不显著"

print(f"结论：{verdict_corr}")

# ========== 分析 2: 质量分数的变异程度 ==========
print("\n" + "=" * 70)
print("分析 2: 质量分数的变异程度")
print("=" * 70)

q_mean = df['quality'].mean()
q_std = df['quality'].std()
q_cv = q_std / q_mean
q_range = df['quality'].max() - df['quality'].min()

print(f"均值：{q_mean:.4f}")
print(f"标准差：{q_std:.4f}")
print(f"变异系数 (CV): {q_cv:.4f} ({q_cv*100:.2f}%)")
print(f"极差：[{df['quality'].min():.4f}, {df['quality'].max():.4f}]")

if q_cv < 0.05:
    verdict_cv = "⚠️  **变异太小** - 所有样本质量接近，区分度差"
elif q_cv < 0.15:
    verdict_cv = "ℹ️  **中等变异** - 有一定区分度"
else:
    verdict_cv = "✅ **变异充分** - 质量差异明显，适合学习"

print(f"结论：{verdict_cv}")

# ========== 分析 3: 固定策略对比 ==========
print("\n" + "=" * 70)
print("分析 3: 如果采用固定速度策略，效果差异多大？")
print("=" * 70)

actions = config['actions']['values']

# 为每个 (underfill, mu) 上下文，计算各速度的平均质量
ctx_quality = df.groupby(['underfill', 'mu', 'r_l'])['quality'].mean().reset_index()

strategy_rewards = []
for action in actions:
    # 对于该动作，查找对应的质量
    ctx_with_action = ctx_quality[ctx_quality['r_l'] == action].copy()
    if len(ctx_with_action) == 0:
        # 如果没有精确匹配，用最接近的
        ctx_with_action = ctx_quality.loc[ctx_quality.groupby(['underfill', 'mu'])['r_l'].apply(lambda x: (x - action).abs().idxmin())]
    
    ctx_with_action['reward'] = ctx_with_action.apply(lambda x: compute_reward_with_utility(action, x['quality']), axis=1)
    avg_reward = ctx_with_action['reward'].mean()
    strategy_rewards.append({'action': action, 'avg_reward': avg_reward})

strategy_df = pd.DataFrame(strategy_rewards)
print("\n固定策略的平均奖励:")
for _, row in strategy_df.iterrows():
    print(f"  r_l = {row['action']:.2f}:  {row['avg_reward']:.4f}")

best_row = strategy_df.loc[strategy_df['avg_reward'].idxmax()]
worst_row = strategy_df.loc[strategy_df['avg_reward'].idxmin()]
gain = best_row['avg_reward'] - worst_row['avg_reward']

print(f"\n最佳固定策略：{best_row['action']:.2f} (奖励={best_row['avg_reward']:.4f})")
print(f"最差固定策略：{worst_row['action']:.2f} (奖励={worst_row['avg_reward']:.4f})")
print(f"策略间差异 (Gain): {gain:.4f}")

if gain < 0.02:
    verdict_gain = "⚠️  **差异极小** - 选哪个速度都差不多，模型学不到东西"
elif gain < 0.1:
    verdict_gain = "ℹ️  **差异中等** - 有一定优化空间"
else:
    verdict_gain = "✅ **差异显著** - 策略选择很重要，模型应该能学到"

print(f"结论：{verdict_gain}")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图 1: 速度 - 质量散点图
axes[0].scatter(df['r_l'], df['quality'], alpha=0.5, s=20)
axes[0].set_xlabel('Link Ratio (Speed)')
axes[0].set_ylabel('Quality Score')
axes[0].set_title(f'Speed vs Quality\nCorrelation = {corr:.3f}')
axes[0].grid(True, alpha=0.3)

# 图 2: 质量分布直方图
axes[1].hist(df['quality'], bins=20, edgecolor='black', alpha=0.7)
axes[1].axvline(q_mean, color='r', linestyle='--', label=f'Mean={q_mean:.3f}')
axes[1].set_xlabel('Quality Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Quality Distribution\nCV = {q_cv*100:.1f}%')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图 3: 固定策略对比
axes[2].bar([f"{a:.2f}" for a in strategy_df['action']], strategy_df['avg_reward'], edgecolor='black', alpha=0.7)
axes[2].axhline(strategy_df['avg_reward'].mean(), color='r', linestyle='--', label='Mean')
axes[2].set_xlabel('Action (Link Ratio)')
axes[2].set_ylabel('Average Reward')
axes[2].set_title(f'Fixed Strategy Comparison\nGain = {gain:.4f}')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/data_analysis.png', dpi=150)
print("\n可视化已保存至：results/data_analysis.png")

# ========== 最终结论 ==========
print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)

issues = []
if abs(corr) < 0.1:
    issues.append("• 速度与质量几乎不相关")
if q_cv < 0.05:
    issues.append("• 质量分数变异太小 (CV<5%)")
if gain < 0.02:
    issues.append("• 不同策略的奖励差异极小 (<0.02)")

if len(issues) >= 2:
    print("\n⚠️  **数据质量警告**：存在多个问题限制了模型学习能力")
    for issue in issues:
        print(f"  {issue}")
    print("\n建议：")
    print("  1. 检查 DEFORM-3D 仿真参数，增大速度变化的物理效应")
    print("  2. 或者接受现实：在这个工艺窗口内，速度确实不是关键因素")
    print("  3. 考虑引入其他决策变量（如温度、模具几何等）")
    print("  4. 当前模型的'验证奖励恒定'是正常现象，因为历史数据本身就差不多")
elif len(issues) == 1:
    print("\nℹ️  **数据有一定局限性**，但仍有学习空间")
    for issue in issues:
        print(f"  {issue}")
    print("\n建议：尝试调整奖励函数权重或增加正则化")
else:
    print("\n✅ **数据质量良好**，模型应该能学到有意义的策略")
    print("如果训练效果仍不好，可能是模型架构或超参数的问题")

print("\n" + "=" * 70)

"""
奖励函数工具模块：基于 Pareto 前沿的反事实评估
核心思想：
1. 质量分数 Q：基于归一化的损伤和应力
2. 速度效用 U(v)：构造凹函数，在 0.75 处达到峰值
3. 最终奖励 R = Q × U(v)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# 动作空间配置
ACTION_VALUES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
PARETO_KNEE_POINT = 0.75  # Pareto 膝点


def calculate_quality_score(row: pd.Series) -> float:
    """
    计算综合质量分数 (0-1)，越高越好。
    基于三个区域的应力均匀性和损伤风险。
    
    注意：数据已经过标准化处理，需要根据实际数据范围调整阈值。
    
    Args:
        row: DataFrame 的一行，包含所有特征列
        
    Returns:
        质量分数 (0-1)
    """
    # 提取关键指标 (需与 CSV 列名一致)
    regions = ['A_HighStress', 'B_MainBody', 'C_Protrusion']
    
    penalties = []
    
    # 数据分析显示：
    # - DAMAGE max 范围：28-350, 均值~175
    # - VonMises mean 范围：32-359, 均值~174
    # - global__vm_mean 范围：42-350, 均值~176
    
    # 1. 损伤惩罚 (权重最高)
    for reg in regions:
        dmg_max = row.get(f"{reg}__DAMAGE__max", 0)
        # 根据实际数据调整：超过 250 (约 P75) 后开始惩罚
        if dmg_max > 250:
            penalties.append(2.0 * (dmg_max - 250) / 100)  # 归一化
        elif dmg_max > 200:
            penalties.append(0.5 * (dmg_max - 200) / 50)
        else:
            penalties.append(0.05 * dmg_max / 200)  # 轻微惩罚
            
    # 2. 应力均匀性惩罚 (Std 越小越好)
    for reg in regions:
        vm_std = row.get(f"{reg}__VonMises__std", 0)
        vm_mean = row.get(f"{reg}__VonMises__mean", 1)
        # 变异系数惩罚 (考虑标准化后的数据)
        cv = vm_std / (vm_mean + 1e-6)
        if cv > 0.5:  # 调整后阈值
            penalties.append(0.5 * (cv - 0.5))
            
    # 3. 绝对应力水平惩罚
    vm_global = row.get("global__vm_mean", 0)
    if vm_global > 250:  # 约 P75
        penalties.append(0.3 * (vm_global - 250) / 100)

    total_penalty = sum(penalties)
    # 转换为 0-1 分数，基础分 1.0
    score = max(0.0, 1.0 - total_penalty)
    return score


def compute_speed_utility(speed: float, knee_point: float = PARETO_KNEE_POINT) -> float:
    """
    计算速度效用函数 U(v)。
    构造凹函数，在 knee_point (0.75) 处达到峰值，模拟 Pareto 前沿。
    
    工业场景分析：
    - 速度太慢 (<0.75): 生产效率低，但质量稳定
    - 速度太快 (>0.75): 质量风险增加，收益递减
    - 膝点 0.75: 最佳平衡点
    
    Args:
        speed: 连杆比速度值 [0.6, 1.0]
        knee_point: Pareto 膝点位置
        
    Returns:
        速度效用值 (0-1)
    """
    # 归一化速度到 [0, 1]
    v_norm = (speed - 0.6) / (1.0 - 0.6)
    k_norm = (knee_point - 0.6) / (1.0 - 0.6)
    
    # 构造分段二次函数，在 knee_point 处达到峰值 1.0
    if v_norm <= k_norm:
        # 上升段：从 0.6 到 0.75，效用递增
        utility = v_norm / k_norm
    else:
        # 下降段：从 0.75 到 1.0，效用递减（模拟质量风险）
        decay_rate = 0.5  # 衰减速率，控制下降陡峭程度
        utility = 1.0 - decay_rate * (v_norm - k_norm) / (1.0 - k_norm)
    
    return max(0.0, min(1.0, utility))


def compute_pareto_reward(speed: float, quality: float, alpha: float = 0.4) -> float:
    """
    计算 Pareto 效用奖励。
    Reward = (1-alpha) * Quality + alpha * Normalized_Speed
    
    注意：此函数使用线性加权，已不再推荐使用。
    推荐使用 compute_reward_with_utility() 进行非线性权衡。
    
    Args:
        speed: 连杆比速度值
        quality: 质量分数 (0-1)
        alpha: 速度权重 (建议 0.3-0.5，体现质量优先)
        
    Returns:
        奖励值
    """
    # 速度归一化 (假设范围 0.6-1.0)
    norm_speed = (speed - 0.6) / (1.0 - 0.6)
    
    reward = (1 - alpha) * quality + alpha * norm_speed
    return reward


def compute_reward_with_utility(speed: float, quality: float) -> float:
    """
    计算基于效用函数的最终奖励。
    R = Q × U(v)
    
    这种形式体现了"安全优先，兼顾效率"的非线性权衡：
    - 质量差 (Q≈0) 时，无论速度多快，奖励都很低
    - 质量好 (Q≈1) 时，速度效用决定最终奖励
    - 在膝点 0.75 附近获得最大综合收益
    
    Args:
        speed: 连杆比速度值
        quality: 质量分数 (0-1)
        
    Returns:
        最终奖励值
    """
    speed_utility = compute_speed_utility(speed)
    reward = quality * speed_utility
    return reward


def get_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 Pareto 分析准备双目标数据
    
    Args:
        df: 包含原始特征的 DataFrame
        
    Returns:
        包含 r_l 和 quality 列的 DataFrame
    """
    df = df.copy()
    df['quality'] = df.apply(calculate_quality_score, axis=1)
    # 目标 1: 最大化质量 (负值用于最小化优化器)
    # 目标 2: 最大化速度 (即 r_l)
    return df[['r_l', 'quality']]


def build_pareto_lookup_table(df: pd.DataFrame) -> Dict[Tuple[float, float], Dict[float, float]]:
    """
    构建 Pareto 奖励查找表。
    
    对于每个 (underfill, mu) 上下文状态，预计算所有可能动作的预期奖励。
    这样可以在验证/推理时进行反事实评估："如果选择其他动作会怎样"。
    
    Args:
        df: 包含完整轨迹数据的 DataFrame
        
    Returns:
        嵌套字典：{(underfill, mu): {action_value: expected_reward}}
    """
    df = df.copy()
    
    # 计算每条轨迹的质量分数
    df['quality'] = df.apply(calculate_quality_score, axis=1)
    
    # 按轨迹分组 (underfill, mu, r_l)
    trajectory_rewards = {}
    
    for (uf, mu, r_l), group in df.groupby(['underfill', 'mu', 'r_l']):
        # 使用该轨迹的最终状态或平均状态计算代表性质量
        # 这里使用最后一步的状态作为该轨迹的质量代表
        final_state = group.iloc[-1]
        quality = calculate_quality_score(final_state)
        
        # 计算该轨迹的奖励
        reward = compute_reward_with_utility(r_l, quality)
        
        # 存储到查找表
        key = (uf, mu)
        if key not in trajectory_rewards:
            trajectory_rewards[key] = {}
        
        # 对于同一个 (uf, mu)，可能有多个 r_l 的数据
        # 我们记录每个 r_l 对应的奖励
        trajectory_rewards[key][r_l] = reward
    
    # 对于缺失的动作值，使用插值或默认值填充
    for key in trajectory_rewards:
        existing_actions = set(trajectory_rewards[key].keys())
        for action in ACTION_VALUES:
            if action not in existing_actions:
                # 使用最近邻插值
                closest_action = min(existing_actions, key=lambda x: abs(x - action))
                trajectory_rewards[key][action] = trajectory_rewards[key][closest_action]
    
    return trajectory_rewards


def get_counterfactual_reward(
    lookup_table: Dict[Tuple[float, float], Dict[float, float]],
    underfill: float,
    mu: float,
    action_value: float
) -> float:
    """
    获取反事实奖励。
    
    给定上下文状态 (underfill, mu) 和假设的动作 a^，
    返回查表得到的预期奖励 R(s, a^)。
    
    Args:
        lookup_table: Pareto 奖励查找表
        underfill: 欠压量
        mu: 摩擦系数
        action_value: 动作值 (连杆比)
        
    Returns:
        反事实奖励值
    """
    key = (underfill, mu)
    
    if key not in lookup_table:
        # 如果上下文不在表中，使用平均值作为默认
        all_rewards = []
        for actions_dict in lookup_table.values():
            if action_value in actions_dict:
                all_rewards.append(actions_dict[action_value])
        if all_rewards:
            return np.mean(all_rewards)
        else:
            return 0.5  # 默认中等奖励
    
    if action_value not in lookup_table[key]:
        # 动作值不在表中，使用最近邻
        existing_actions = list(lookup_table[key].keys())
        closest_action = min(existing_actions, key=lambda x: abs(x - action_value))
        return lookup_table[key][closest_action]
    
    return lookup_table[key][action_value]


def analyze_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    分析 Pareto 前沿，识别膝点。
    
    Args:
        df: 包含原始特征的 DataFrame
        
    Returns:
        包含 Pareto 分析结果的 DataFrame
    """
    df = df.copy()
    df['quality'] = df.apply(calculate_quality_score, axis=1)
    df['speed_utility'] = df['r_l'].apply(compute_speed_utility)
    df['pareto_reward'] = df.apply(
        lambda row: compute_reward_with_utility(row['r_l'], row['quality']), 
        axis=1
    )
    
    # 按速度分组统计平均质量
    summary = df.groupby('r_l').agg({
        'quality': 'mean',
        'speed_utility': 'mean',
        'pareto_reward': 'mean'
    }).reset_index()
    
    # 找出最优速度
    best_speed = summary.loc[summary['pareto_reward'].idxmax(), 'r_l']
    
    print(f"Pareto 分析结果:")
    print(f"  最优速度 (膝点): {best_speed:.2f}")
    print(f"  平均质量分数：{summary['quality'].mean():.4f}")
    print(f"  平均 Pareto 奖励：{summary['pareto_reward'].mean():.4f}")
    
    return summary


# ============================================================================
# 方案 A+B 混合：保守性反事实评估 (Conservative Counterfactual Evaluation)
# 解决数据泄露风险的核心改进
# ============================================================================

def build_conservative_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建包含 BC 统计量 (均值/标准差) 的查找表，用于保守性惩罚。
    
    Key: (underfill, mu) -> (bc_mean, bc_std)
    
    Args:
        df: 包含完整轨迹数据的 DataFrame
        
    Returns:
        DataFrame: [underfill, mu, bc_mean, bc_std]
    """
    # 按上下文分组，计算动作的统计量
    stats = df.groupby(['underfill', 'mu'])['r_l'].agg(['mean', 'std']).reset_index()
    stats.columns = ['underfill', 'mu', 'bc_mean', 'bc_std']
    stats['bc_std'] = stats['bc_std'].fillna(0.05)  # 填充 NaN，假设最小不确定性
    return stats


def get_conservative_penalty(
    predicted_action: float, 
    bc_mean: float, 
    bc_std: float, 
    lambda_coeff: float = 2.0
) -> float:
    """
    方案 A: 保守性惩罚 (Conservative Penalty)
    
    惩罚偏离行为克隆 (BC) 分布的动作：
    Penalty = lambda * |a_pred - a_bc| / std
    
    原理：离线强化学习中，OOD (Out-of-Distribution) 动作的 Q 值估计不可靠，
    因此需要对偏离历史数据分布的动作进行惩罚。
    
    Args:
        predicted_action: 模型预测的动作值
        bc_mean: 该上下文中历史动作的均值
        bc_std: 该上下文中历史动作的标准差
        lambda_coeff: 惩罚系数 (建议 1.0-3.0)
        
    Returns:
        惩罚值 (0-inf)
    """
    if bc_std < 1e-4:
        bc_std = 0.05  # 防止除零，假设最小不确定性
    
    deviation = abs(predicted_action - bc_mean)
    normalized_deviation = deviation / bc_std
    
    return lambda_coeff * normalized_deviation


def get_counterfactual_reward_conservative(
    lookup_table: Dict[Tuple[float, float], Dict[float, float]],
    bc_table: pd.DataFrame,
    underfill: float,
    mu: float,
    predicted_action: float,
    lambda_coeff: float = 2.0
) -> float:
    """
    方案 A+B 混合：计算带保守性惩罚的反事实奖励。
    
    核心改进：
    1. 不再直接使用验证集的真实标签计算奖励
    2. 而是基于预测动作与 BC 分布的偏离程度进行惩罚
    3. 公式：R_final = R_oracle(a_pred) - Penalty(a_pred, a_bc)
    
    这解决了以下问题：
    - 数据泄露：评估时不访问 ground truth 标签
    - OOD 问题：惩罚超出训练分布的动作
    - 过拟合：防止模型利用评估函数漏洞
    
    Args:
        lookup_table: Pareto 奖励查找表 (来自训练集)
        bc_table: BC 统计量查找表 (来自训练集)
        underfill: 欠压量
        mu: 摩擦系数
        predicted_action: 模型预测的动作值
        lambda_coeff: 惩罚系数
        
    Returns:
        保守性反事实奖励值
    """
    key = (underfill, mu)
    
    # 1. 获取基础奖励 (Oracle Reward)
    # 注意：这里的 lookup_table 仅来自训练集，不包含验证集信息
    base_reward = get_counterfactual_reward(lookup_table, underfill, mu, predicted_action)
    
    # 2. 获取 BC 统计量并计算惩罚
    bc_mask = (bc_table['underfill'] == underfill) & (bc_table['mu'] == mu)
    
    if not bc_mask.any():
        # 如果上下文不在 BC 表中，使用全局统计量
        penalty = get_conservative_penalty(
            predicted_action,
            bc_table['bc_mean'].mean(),
            bc_table['bc_std'].mean(),
            lambda_coeff
        )
    else:
        bc_row = bc_table[bc_mask].iloc[0]
        penalty = get_conservative_penalty(
            predicted_action,
            bc_row['bc_mean'],
            bc_row['bc_std'],
            lambda_coeff
        )
    
    # 3. 最终奖励 = 基础奖励 - 惩罚
    final_reward = max(0.0, base_reward - penalty)
    
    return final_reward

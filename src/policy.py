"""
Policy 模块：Neural UCB 和 Thompson Sampling 实现
统一接口，支持基于不确定性的探索策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
import math


class NeuralUCB:
    """Neural UCB (Upper Confidence Bound) 策略
    
    使用 Q 值预测的不确定性构建置信上界进行探索
    """
    
    def __init__(
        self,
        n_actions: int,
        actions: np.ndarray,
        exploration_bonus: float = 0.1,
        device: str = "cpu"
    ):
        """初始化 Neural UCB
        
        Args:
            n_actions: 动作数量
            actions: 实际动作值数组
            exploration_bonus: 探索系数 (beta)
            device: 计算设备
        """
        self.n_actions = n_actions
        self.actions = actions
        self.exploration_bonus = exploration_bonus
        self.device = device
        
        # 动作选择历史统计
        self.action_counts = np.zeros(n_actions)
        self.total_count = 0
        
    def select_action(
        self,
        q_values: torch.Tensor,
        uncertainty: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        """基于 UCB 选择动作
        
        UCB(a) = Q(s,a) + beta * sigma(s,a)
        
        Args:
            q_values: (n_actions,) - Q 值预测
            uncertainty: (n_actions,) - 不确定性估计 (标准差)
            context: 可选的上下文信息
            
        Returns:
            action_idx: 选择的动作索引
            action_value: 选择的动作值
        """
        q_values = q_values.to(self.device)
        uncertainty = uncertainty.to(self.device)
        
        # 计算 UCB 值
        ucb_values = q_values + self.exploration_bonus * uncertainty
        
        # 选择 UCB 最大的动作
        action_idx = torch.argmax(ucb_values).item()
        action_value = self.actions[action_idx]
        
        # 更新统计
        self.action_counts[action_idx] += 1
        self.total_count += 1
        
        return action_idx, action_value
    
    def get_exploration_bonus(self, decay: bool = True) -> float:
        """获取当前探索系数 (可选衰减)"""
        if decay and self.total_count > 0:
            # 随时间衰减
            return self.exploration_bonus / math.sqrt(self.total_count + 1)
        return self.exploration_bonus


class ThompsonSampling:
    """Thompson Sampling 策略
    
    从后验分布中采样，根据采样值选择动作
    """
    
    def __init__(
        self,
        n_actions: int,
        actions: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        device: str = "cpu"
    ):
        """初始化 Thompson Sampling
        
        Args:
            n_actions: 动作数量
            actions: 实际动作值数组
            prior_alpha: Beta 分布先验参数 alpha
            prior_beta: Beta 分布先验参数 beta
            device: 计算设备
        """
        self.n_actions = n_actions
        self.actions = actions
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.device = device
        
        # 成功/失败计数 (用于 Beta 分布后验)
        self.successes = np.ones(n_actions) * prior_alpha
        self.failures = np.ones(n_actions) * prior_beta
        
    def select_action(
        self,
        q_values: torch.Tensor,
        uncertainty: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        """基于 Thompson Sampling 选择动作
        
        从每个动作的后验分布中采样，选择采样值最大的动作
        
        Args:
            q_values: (n_actions,) - Q 值预测
            uncertainty: (n_actions,) - 不确定性估计
            context: 可选的上下文信息
            
        Returns:
            action_idx: 选择的动作索引
            action_value: 选择的动作值
        """
        # 方法 1: 使用 Beta 分布 (适用于归一化奖励)
        # theta_sample = np.random.beta(self.successes, self.failures)
        # action_idx = np.argmax(theta_sample)
        
        # 方法 2: 使用高斯分布 (更通用，基于 Q 值和不确定性)
        q_values_np = q_values.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        
        # 从 N(Q, sigma^2) 中采样
        samples = np.random.normal(q_values_np, uncertainty_np + 1e-6)
        action_idx = np.argmax(samples)
        action_value = self.actions[action_idx]
        
        return action_idx, action_value
    
    def update(self, action_idx: int, reward: float, threshold: float = 0.0):
        """更新后验分布
        
        Args:
            action_idx: 执行的动作索引
            reward: 获得的奖励
            threshold: 成功/失败的阈值
        """
        if reward >= threshold:
            self.successes[action_idx] += 1
        else:
            self.failures[action_idx] += 1
    
    def reset(self):
        """重置后验分布"""
        self.successes = np.ones(self.n_actions) * self.prior_alpha
        self.failures = np.ones(self.n_actions) * self.prior_beta


class EpsilonGreedy:
    """Epsilon-Greedy 策略 (基线对比)"""
    
    def __init__(
        self,
        n_actions: int,
        actions: np.ndarray,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        device: str = "cpu"
    ):
        self.n_actions = n_actions
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
    def select_action(
        self,
        q_values: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        """Epsilon-Greedy 动作选择"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            action_idx = np.random.randint(self.n_actions)
        else:
            # 利用：选择最优
            action_idx = torch.argmax(q_values).item()
        
        action_value = self.actions[action_idx]
        return action_idx, action_value
    
    def decay_epsilon(self):
        """衰减 epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def create_policy(
    policy_type: str,
    n_actions: int,
    actions: np.ndarray,
    **kwargs
):
    """工厂函数：创建 Policy
    
    Args:
        policy_type: "neural_ucb", "thompson_sampling", 或 "epsilon_greedy"
        n_actions: 动作数量
        actions: 动作值数组
        **kwargs: 其他参数
        
    Returns:
        Policy 对象
    """
    if policy_type == "neural_ucb":
        return NeuralUCB(
            n_actions=n_actions,
            actions=actions,
            exploration_bonus=kwargs.get('exploration_bonus', 0.1),
            device=kwargs.get('device', 'cpu')
        )
    
    elif policy_type == "thompson_sampling":
        return ThompsonSampling(
            n_actions=n_actions,
            actions=actions,
            prior_alpha=kwargs.get('prior_alpha', 1.0),
            prior_beta=kwargs.get('prior_beta', 1.0),
            device=kwargs.get('device', 'cpu')
        )
    
    elif policy_type == "epsilon_greedy":
        return EpsilonGreedy(
            n_actions=n_actions,
            actions=actions,
            epsilon=kwargs.get('epsilon', 0.1),
            epsilon_decay=kwargs.get('epsilon_decay', 0.995),
            epsilon_min=kwargs.get('epsilon_min', 0.01),
            device=kwargs.get('device', 'cpu')
        )
    
    else:
        raise ValueError(f"不支持的 Policy 类型：{policy_type}")


if __name__ == "__main__":
    # 测试 Policy 模块
    n_actions = 9
    actions = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    
    # 模拟 Q 值和不确定性
    q_values = torch.randn(n_actions)
    uncertainty = torch.abs(torch.randn(n_actions)) * 0.5
    
    print("=== 测试 Neural UCB ===")
    ucb = create_policy("neural_ucb", n_actions, actions, exploration_bonus=0.2)
    idx, val = ucb.select_action(q_values, uncertainty)
    print(f"选择动作索引：{idx}, 动作值：{val:.2f}")
    
    print("\n=== 测试 Thompson Sampling ===")
    ts = create_policy("thompson_sampling", n_actions, actions, prior_alpha=1.0)
    idx, val = ts.select_action(q_values, uncertainty)
    print(f"选择动作索引：{idx}, 动作值：{val:.2f}")
    
    # 更新 Thompson Sampling
    ts.update(idx, reward=0.8)
    print(f"更新后成功计数：{ts.successes[idx]:.1f}")
    
    print("\n=== 测试 Epsilon-Greedy ===")
    eg = create_policy("epsilon_greedy", n_actions, actions, epsilon=0.1)
    idx, val = eg.select_action(q_values, uncertainty)
    print(f"选择动作索引：{idx}, 动作值：{val:.2f}, 当前 epsilon: {eg.epsilon:.3f}")

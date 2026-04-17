"""
数据加载与预处理模块
负责读取降维后的特征数据，构建上下文、状态和动作空间
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import yaml


class ForgeDataLoader:
    """铝合金锻造数据加载器"""
    
    # 56维特征列名 (根据实际数据定义)
    PHYS_COLS = [
        "A_HighStress__VonMises__mean", "A_HighStress__VonMises__max", 
        "A_HighStress__VonMises__std", "A_HighStress__VonMises__q95",
        "A_HighStress__DAMAGE__mean", "A_HighStress__DAMAGE__max",
        "A_HighStress__DAMAGE__std", "A_HighStress__DAMAGE__q95",
        "A_HighStress__STRAIN__mean", "A_HighStress__STRAIN__max",
        "A_HighStress__STRAIN__std", "A_HighStress__STRAIN__q95",
        "A_HighStress__NDTMP__mean", "A_HighStress__NDTMP__max",
        "A_HighStress__NDTMP__std", "A_HighStress__NDTMP__q95",
        
        "B_MainBody__VonMises__mean", "B_MainBody__VonMises__max",
        "B_MainBody__VonMises__std", "B_MainBody__VonMises__q95",
        "B_MainBody__DAMAGE__mean", "B_MainBody__DAMAGE__max",
        "B_MainBody__DAMAGE__std", "B_MainBody__DAMAGE__q95",
        "B_MainBody__STRAIN__mean", "B_MainBody__STRAIN__max",
        "B_MainBody__STRAIN__std", "B_MainBody__STRAIN__q95",
        "B_MainBody__NDTMP__mean", "B_MainBody__NDTMP__max",
        "B_MainBody__NDTMP__std", "B_MainBody__NDTMP__q95",
        
        "C_Protrusion__VonMises__mean", "C_Protrusion__VonMises__max",
        "C_Protrusion__VonMises__std", "C_Protrusion__VonMises__q95",
        "C_Protrusion__DAMAGE__mean", "C_Protrusion__DAMAGE__max",
        "C_Protrusion__DAMAGE__std", "C_Protrusion__DAMAGE__q95",
        "C_Protrusion__STRAIN__mean", "C_Protrusion__STRAIN__max",
        "C_Protrusion__STRAIN__std", "C_Protrusion__STRAIN__q95",
        "C_Protrusion__NDTMP__mean", "C_Protrusion__NDTMP__max",
        "C_Protrusion__NDTMP__std", "C_Protrusion__NDTMP__q95",
        
        "global__damage_high_ratio", "global__vm_p99", "global__vm_mean",
    ]
    
    def __init__(self, config_path: str):
        """初始化数据加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['path'])
        self.context_cols = self.config['data']['context_cols']
        self.action_col = self.config['data']['action_col']
        self.step_col = self.config.get('data', {}).get('step_col', 'step')
        
        # 动作空间
        self.actions = np.array(self.config['actions']['values'])
        self.n_actions = len(self.actions)
        
        # 特征列 (如果未指定则自动推断)
        if self.config['data'].get('feature_cols') is None:
            self.feature_cols = self.PHYS_COLS.copy()
        else:
            self.feature_cols = self.config['data']['feature_cols']
        
        self.df: Optional[pd.DataFrame] = None
        self.feature_dim = len(self.feature_cols)
        self.context_dim = len(self.context_cols)
        
    def load_data(self, preprocess: bool = True) -> pd.DataFrame:
        """加载CSV数据
        
        Args:
            preprocess: 是否进行预处理 (填充、归一化等)
            
        Returns:
            加载后的DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        if preprocess:
            self._preprocess()
        
        return self.df
    
    def _preprocess(self):
        """数据预处理"""
        # 前向填充空值 (根据需求说明)
        for col in self.feature_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].ffill()
        
        # 最终仍有空值则填0
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        # 特征标准化 (使用 sklearn StandardScaler)
        self._normalize_features()
    
    def _normalize_features(self):
        """对特征进行标准化 (保存 scaler 以便推理时使用)"""
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("警告：sklearn 未安装，跳过特征标准化")
            return
        
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        print(f"特征标准化完成：均值={self.df[self.feature_cols].mean().mean():.4f}, 标准差={self.df[self.feature_cols].std().mean():.4f}")
    
    def get_trajectory(self, underfill: float, mu: float) -> pd.DataFrame:
        """获取单个锻造轨迹的数据
        
        Args:
            underfill: 欠压值
            mu: 摩擦系数
            
        Returns:
            该轨迹的所有步骤数据
        """
        mask = (self.df['underfill'] == underfill) & (self.df['mu'] == mu)
        return self.df[mask].sort_values(self.step_col)
    
    def get_all_trajectories(self) -> Dict[Tuple[float, float], pd.DataFrame]:
        """获取所有轨迹
        
        Returns:
            {(underfill, mu): trajectory_df} 字典
        """
        trajectories = {}
        for uf in self.df['underfill'].unique():
            for m in self.df['mu'].unique():
                traj = self.get_trajectory(uf, m)
                if len(traj) > 0:
                    trajectories[(uf, m)] = traj
        return trajectories
    
    def prepare_bandit_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备多臂老虎机数据集
        
        每个样本 = (context, state_features, action, reward)
        由于是离线数据，需要从完整轨迹中构造
        
        Returns:
            contexts: (N, context_dim)
            states: (N, feature_dim)  
            actions: (N,) 离散动作索引
            rewards: (N,) 计算得到的奖励
        """
        if self.df is None:
            self.load_data()
        
        # 提取上下文 (underfill, mu)
        contexts = self.df[self.context_cols].values
        
        # 提取状态特征 (56维物理特征)
        states = self.df[self.feature_cols].values
        
        # 提取动作 (r_l 连杆比)
        raw_actions = self.df[self.action_col].values
        # 映射到动作索引
        actions = np.searchsorted(self.actions, raw_actions)
        
        # 计算奖励 (需要根据最终锻造结果)
        rewards = self._compute_rewards(self.df)
        
        return contexts, states, actions, rewards
    
    def _compute_rewards(self, df: pd.DataFrame) -> np.ndarray:
        """计算奖励函数
        
        奖励设计原则:
        1. 高质量优先 (损伤/应力不超标)
        2. 速度尽可能快 (连杆比大)
        
        Args:
            df: 包含特征和参数的数据框
            
        Returns:
            奖励数组
        """
        w_quality = self.config['reward']['w_quality']
        w_speed = self.config['reward']['w_speed']
        damage_th = self.config['reward']['damage_threshold']
        stress_th = self.config['reward']['stress_threshold']
        
        rewards = []
        
        # 按轨迹分组计算 (每个轨迹对应一个固定的 underfill+mu+r_l 组合)
        for (uf, mu, r_l), group in df.groupby(['underfill', 'mu', 'r_l']):
            # 该轨迹的最终状态 (最后一步)
            final_state = group.iloc[-1]
            
            # 质量指标 (从最终状态或整个轨迹统计)
            max_damage = group['A_HighStress__DAMAGE__max'].max()
            max_stress = group['A_HighStress__VonMises__max'].max()
            
            # 质量惩罚
            quality_penalty = 0.0
            if max_damage > damage_th:
                quality_penalty += w_quality * (max_damage - damage_th) ** 2
            if max_stress > stress_th:
                quality_penalty += w_quality * ((max_stress - stress_th) / stress_th) ** 2
            
            # 速度奖励 (连杆比越大越快，归一化到 [0,1])
            speed_reward = w_speed * (r_l - self.actions[0]) / (self.actions[-1] - self.actions[0])
            
            # 总奖励 (负惩罚 + 正奖励)
            total_reward = speed_reward - quality_penalty
            
            # 该轨迹所有步骤共享相同奖励 (因为连杆比全程不变)
            for _ in range(len(group)):
                rewards.append(total_reward)
        
        return np.array(rewards)
    
    def train_val_split(self, val_ratio: float = 0.2) -> Tuple[Tuple, Tuple]:
        """按轨迹划分训练集和验证集
        
        Args:
            val_ratio: 验证集比例
            
        Returns:
            (train_data, val_data) 每个都是 (contexts, states, actions, rewards)
        """
        contexts, states, actions, rewards = self.prepare_bandit_dataset()
        
        # 获取所有唯一轨迹
        unique_trajectories = list(zip(
            self.df['underfill'].values,
            self.df['mu'].values
        ))
        
        # 随机打乱并划分
        np.random.seed(self.config['experiment']['seed'])
        n_trajectories = len(unique_trajectories)
        n_val = int(n_trajectories * val_ratio)
        
        val_indices = set(np.random.choice(n_trajectories, n_val, replace=False))
        
        train_mask = []
        val_mask = []
        
        for i, (uf, mu) in enumerate(zip(self.df['underfill'], self.df['mu'])):
            if (uf, mu) in [unique_trajectories[j] for j in val_indices]:
                val_mask.append(True)
                train_mask.append(False)
            else:
                train_mask.append(False)
                val_mask.append(True)
        
        train_mask = np.array(train_mask)
        val_mask = np.array(val_mask)
        
        train_data = (
            contexts[train_mask],
            states[train_mask],
            actions[train_mask],
            rewards[train_mask]
        )
        
        val_data = (
            contexts[val_mask],
            states[val_mask],
            actions[val_mask],
            rewards[val_mask]
        )
        
        return train_data, val_data


if __name__ == "__main__":
    # 测试数据加载
    loader = ForgeDataLoader("config/experiment_config.yaml")
    print(f"特征维度：{loader.feature_dim}")
    print(f"上下文维度：{loader.context_dim}")
    print(f"动作数量：{loader.n_actions}")
    print(f"动作空间：{loader.actions}")

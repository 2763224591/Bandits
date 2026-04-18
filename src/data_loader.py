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
    """铝合金锻造数据加载器
    
    数据格式说明:
    - CSV总列数：56列 (51物理特征 + 2上下文 + 1动作 + 2元数据)
    - 物理特征 (51维): 48区域统计 (3区×4字段×4统计量) + 3全局统计
    - 上下文 (2维): underfill, mu (轨迹级固定)
    - 动作 (1维): r_l (连杆比，轨迹级固定)
    - 元数据 (2列): step(时序索引), param_tag(标识符)
    
    模型输入分离:
    - Encoder输入：51维物理特征序列 (seq_len, 51)
    - Policy输入：2维上下文向量 (2,)
    - step列：仅用于序列排序和位置编码，不作为普通特征
    """
    
    # 51维物理特征列名 (3区域 × 4字段 × 4统计量 = 48维 + 3全局统计)
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
        
        使用新的 Pareto 效用函数计算奖励：
        R = Quality × Speed_Utility
        
        这种形式体现了"安全优先，兼顾效率"的非线性权衡。
        
        Args:
            df: 包含特征和参数的数据框
            
        Returns:
            奖励数组
        """
        # 导入新的奖励函数
        from reward_utils import calculate_quality_score, compute_reward_with_utility
        
        rewards = []
        
        # 按轨迹分组计算 (每个轨迹对应一个固定的 underfill+mu+r_l 组合)
        for (uf, mu, r_l), group in df.groupby(['underfill', 'mu', 'r_l']):
            # 该轨迹的最终状态 (最后一步)
            final_state = group.iloc[-1]
            
            # 计算质量分数
            quality = calculate_quality_score(final_state)
            
            # 计算 Pareto 奖励 (质量 × 速度效用)
            total_reward = compute_reward_with_utility(r_l, quality)
            
            # 该轨迹所有步骤共享相同奖励 (因为连杆比全程不变)
            for _ in range(len(group)):
                rewards.append(total_reward)
        
        return np.array(rewards)
    
    def prepare_sequence_data(self, seq_len: int = 51) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备序列数据用于 Transformer/TFT
        
        将数据按轨迹组织成固定长度的序列
        对于不足 seq_len 的轨迹进行 padding
        
        Args:
            seq_len: 序列长度 (默认 51 步)
            
        Returns:
            contexts: (N_trajectories, context_dim)
            states: (N_trajectories, seq_len, feature_dim)
            actions: (N_trajectories,) 离散动作索引 (每条轨迹一个固定动作)
            rewards: (N_trajectories,) 每条轨迹的奖励
        """
        if self.df is None:
            self.load_data()
        
        trajectories = []
        contexts_list = []
        states_list = []
        actions_list = []
        rewards_list = []
        
        # 按轨迹分组 (underfill, mu, r_l)
        for (uf, mu, r_l), group in self.df.groupby(['underfill', 'mu', 'r_l']):
            group = group.sort_values(self.step_col)
            
            # 提取上下文
            context = np.array([uf, mu])
            
            # 提取状态特征序列
            states = group[self.feature_cols].values
            
            # Padding 或截断到固定长度
            if len(states) < seq_len:
                # padding 用 0
                pad_width = seq_len - len(states)
                states = np.pad(states, ((0, pad_width), (0, 0)), mode='constant')
            elif len(states) > seq_len:
                states = states[:seq_len]
            
            # 动作索引
            action_idx = np.searchsorted(self.actions, r_l)
            
            # 计算轨迹奖励
            traj_df = group.copy()
            reward = self._compute_rewards(traj_df)[0]  # 取第一个 (所有步骤相同)
            
            trajectories.append((uf, mu, r_l))
            contexts_list.append(context)
            states_list.append(states)
            actions_list.append(action_idx)
            rewards_list.append(reward)
        
        return (
            np.array(contexts_list),
            np.array(states_list),
            np.array(actions_list),
            np.array(rewards_list)
        )
    
    def train_val_split(self, val_ratio: float = 0.2) -> Tuple[Tuple, Tuple]:
        """按轨迹划分训练集和验证集
        
        Args:
            val_ratio: 验证集比例
            
        Returns:
            (train_data, val_data) 每个都是 (contexts, states, actions, rewards)
        """
        contexts, states, actions, rewards = self.prepare_bandit_dataset()
        
        # 获取所有唯一轨迹 (underfill, mu 组合)
        unique_trajectories = list(set(zip(
            self.df['underfill'].values,
            self.df['mu'].values
        )))
        
        # 随机打乱并划分
        np.random.seed(self.config['experiment']['seed'])
        n_trajectories = len(unique_trajectories)
        n_val = max(1, int(n_trajectories * val_ratio))  # 至少1条验证轨迹
        
        val_indices = set(np.random.choice(n_trajectories, min(n_val, n_trajectories), replace=False))
        val_trajectories = {unique_trajectories[i] for i in val_indices}
        
        train_mask = []
        val_mask = []
        
        for uf, mu in zip(self.df['underfill'], self.df['mu']):
            if (uf, mu) in val_trajectories:
                val_mask.append(True)
                train_mask.append(False)
            else:
                train_mask.append(True)
                val_mask.append(False)
        
        train_mask = np.array(train_mask, dtype=bool)
        val_mask = np.array(val_mask, dtype=bool)
        
        # 检查划分结果
        print(f"轨迹划分：总共{n_trajectories}条，训练集{train_mask.sum()}样本，验证集{val_mask.sum()}样本")
        
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

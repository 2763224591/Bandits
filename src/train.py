"""
主训练模块：整合所有组件，支持配置驱动的实验
集成 Pareto 反事实评估系统
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

# 导入自定义模块
from data_loader import ForgeDataLoader
from encoders import create_encoder
from uncertainty import QNetworkWithUncertainty, EnsembleModel
from policy import create_policy
from reward_utils import (
    build_pareto_lookup_table,
    build_conservative_lookup_table,
    get_counterfactual_reward,
    get_counterfactual_reward_conservative,
    calculate_quality_score,
    compute_reward_with_utility,
    ACTION_VALUES
)


class ContextualBanditTrainer:
    """上下文老虎机训练器
    
    整合 Encoder + Uncertainty + Policy，支持多种组合
    """
    
    def __init__(self, config_path: str):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{self.device}")
        
        # 设置随机种子
        self._set_seed(self.config['experiment']['seed'])
        
        # 初始化数据加载器
        self.data_loader = ForgeDataLoader(config_path)
        
        # 获取动作空间
        self.actions = self.data_loader.actions
        self.n_actions = self.data_loader.n_actions
        self.feature_dim = self.data_loader.feature_dim
        self.context_dim = self.data_loader.context_dim
        
        # 超参数
        hp = self.config['hyperparameters']
        self.d_model = hp['d_model']
        self.batch_size = hp['batch_size']
        self.lr = hp['lr']
        self.epochs = hp['epochs']
        
        # 构建模型
        self._build_model()
        
        # 构建策略
        self._build_policy()
        
        # Pareto 查找表 (用于反事实评估)
        self.pareto_lookup_table: Optional[Dict] = None
        # BC 统计量表 (用于保守性惩罚)
        self.bc_lookup_table: Optional[pd.DataFrame] = None
        
        # 结果记录
        self.results = {
            'train_loss': [],
            'val_loss': [],
            'train_reward': [],
            'val_reward': [],
            'val_counterfactual_reward': []  # 新增：反事实奖励
        }
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _build_model(self):
        """构建模型"""
        model_config = self.config['model']
        hp = self.config['hyperparameters']
        
        # 创建 Encoder (始终先创建)
        from encoders import create_encoder
        self.encoder = create_encoder(
            encoder_type=model_config['encoder_type'],
            input_dim=self.feature_dim,
            context_dim=self.context_dim,
            d_model=self.d_model,
            n_heads=hp['n_heads'],
            n_layers=hp['n_layers'],
            dropout=hp['dropout']
        ).to(self.device)
        
        # 创建带不确定性的 Q 网络
        uncertainty_type = model_config['uncertainty_type']
        
        if uncertainty_type == "ensemble":
            # Ensemble: 创建多个独立的 Q 网络，每个都有完整的 encoder+head
            self.q_network = EnsembleModel(
                model_class=QNetworkWithUncertainty,
                n_models=hp['ensemble_size'],
                encoder_type=model_config['encoder_type'],
                input_dim=self.feature_dim,
                context_dim=self.context_dim,
                d_model=self.d_model,
                n_heads=hp['n_heads'],
                n_layers=hp['n_layers'],
                dropout=hp['dropout'],
                n_actions=self.n_actions,
                uncertainty_type="none"  # Ensemble 本身提供不确定性
            ).to(self.device)
        else:
            self.q_network = QNetworkWithUncertainty(
                encoder=self.encoder,
                d_model=self.d_model,
                n_actions=self.n_actions,
                uncertainty_type=uncertainty_type,
                dropout_rate=hp['dropout'] if uncertainty_type == "mc_dropout" else 0.0
            ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.lr,
            weight_decay=hp.get('weight_decay', 1e-4)
        )
        
        # 损失函数 (MSE for Q-learning)
        self.criterion = nn.MSELoss()
        
        print(f"模型架构:")
        print(f"  - Encoder: {model_config['encoder_type']}")
        print(f"  - Uncertainty: {uncertainty_type}")
        print(f"  - Policy: {model_config['policy_type']}")
        
    def _build_policy(self):
        """构建策略"""
        model_config = self.config['model']
        hp = self.config['hyperparameters']
        
        # 根据 policy_type 传递不同的参数
        policy_kwargs = {
            'policy_type': model_config['policy_type'],
            'n_actions': self.n_actions,
            'actions': self.actions,
            'device': str(self.device)
        }
        
        # 添加 UCB 参数
        if hp.get('exploration_bonus') is not None:
            policy_kwargs['exploration_bonus'] = hp.get('exploration_bonus', 0.1)
        
        # 添加 Thompson Sampling 参数
        if hp.get('prior_alpha') is not None:
            policy_kwargs['prior_alpha'] = hp.get('prior_alpha', 1.0)
            policy_kwargs['prior_beta'] = hp.get('prior_beta', 1.0)
        
        # 添加 Epsilon-Greedy 参数
        if hp.get('epsilon') is not None:
            policy_kwargs['epsilon'] = hp.get('epsilon', 0.1)
            policy_kwargs['epsilon_decay'] = hp.get('epsilon_decay', 0.995)
            policy_kwargs['epsilon_min'] = hp.get('epsilon_min', 0.01)
        
        self.policy = create_policy(**policy_kwargs)
        
    def prepare_data(self) -> Tuple:
        """准备训练和验证数据 (序列模式)"""
        # 使用序列数据格式
        contexts, states, actions, rewards = self.data_loader.prepare_sequence_data(
            seq_len=self.config['hyperparameters'].get('seq_len', 51)
        )
        
        # 转换为 Tensor (先划分再转换)
        train_idx, val_idx = self._get_trajectory_split()
        
        # 确保索引是 numpy 数组
        train_idx = np.array(train_idx, dtype=bool)
        val_idx = np.array(val_idx, dtype=bool)
        
        print(f"数据划分：训练集{train_idx.sum()}条轨迹，验证集{val_idx.sum()}条轨迹")
        
        train_tensors = (
            torch.FloatTensor(contexts[train_idx]).to(self.device),
            torch.FloatTensor(states[train_idx]).to(self.device),
            torch.LongTensor(actions[train_idx]).to(self.device),
            torch.FloatTensor(rewards[train_idx]).to(self.device)
        )
        
        val_tensors = (
            torch.FloatTensor(contexts[val_idx]).to(self.device),
            torch.FloatTensor(states[val_idx]).to(self.device),
            torch.LongTensor(actions[val_idx]).to(self.device),
            torch.FloatTensor(rewards[val_idx]).to(self.device)
        )
        
        # 创建 DataLoader
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 构建 Pareto 查找表 (用于反事实评估)
        self._build_pareto_lookup_table()
        
        return train_loader, val_loader
    
    def _build_pareto_lookup_table(self):
        """构建 Pareto 奖励查找表和 BC 统计量表"""
        if self.data_loader.df is None:
            self.data_loader.load_data()
        
        # 仅使用训练集数据构建查找表 (避免数据泄露)
        # 注意：数据已经通过_prepare_data 中的轨迹划分分离
        # 这里我们使用一个简单的方法：重新应用相同的划分逻辑
        train_idx, val_idx = self._get_trajectory_split()
        
        # 获取训练集索引对应的行
        # 由于 df 是原始数据，而 train_idx 是轨迹级别的 mask
        # 我们需要将轨迹 mask 转换为行级别的 mask
        unique_trajectories = list(set(zip(
            self.data_loader.df['underfill'].values,
            self.data_loader.df['mu'].values,
            self.data_loader.df['r_l'].values
        )))
        
        # 为每一行找到对应的轨迹索引
        trajectory_map = {traj: i for i, traj in enumerate(unique_trajectories)}
        row_to_traj_idx = [
            trajectory_map[(uf, mu, rl)] 
            for uf, mu, rl in zip(
                self.data_loader.df['underfill'].values,
                self.data_loader.df['mu'].values,
                self.data_loader.df['r_l'].values
            )
        ]
        
        # 创建行级别的训练集 mask
        train_mask = np.array([train_idx[idx] for idx in row_to_traj_idx], dtype=bool)
        train_df = self.data_loader.df[train_mask].copy()
        
        self.pareto_lookup_table = build_pareto_lookup_table(train_df)
        print(f"Pareto 查找表已构建：{len(self.pareto_lookup_table)}个上下文状态")
        
        # 构建 BC 统计量表 (用于保守性惩罚)
        self.bc_lookup_table = build_conservative_lookup_table(train_df)
        print(f"BC 统计量表已构建：{len(self.bc_lookup_table)}个上下文状态")
        
        # 分析 Pareto 前沿
        from reward_utils import analyze_pareto_frontier
        analyze_pareto_frontier(train_df)
    
    def _get_trajectory_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """按轨迹划分训练/验证索引 (基于完整轨迹数据)"""
        # 获取所有唯一轨迹 (underfill, mu, r_l)
        unique_trajectories = list(set(zip(
            self.data_loader.df['underfill'].values,
            self.data_loader.df['mu'].values,
            self.data_loader.df['r_l'].values
        )))
        
        np.random.seed(self.config['experiment']['seed'])
        n_trajectories = len(unique_trajectories)
        n_val = max(1, int(n_trajectories * self.config['data']['val_split']))
        
        val_indices = set(np.random.choice(n_trajectories, min(n_val, n_trajectories), replace=False))
        val_trajectories = {unique_trajectories[i] for i in val_indices}
        
        # 为每条轨迹生成 mask (contexts/states/actions/rewards 是按轨迹组织的)
        train_mask = []
        val_mask = []
        
        for uf, mu, r_l in unique_trajectories:
            if (uf, mu, r_l) in val_trajectories:
                val_mask.append(True)
                train_mask.append(False)
            else:
                train_mask.append(True)
                val_mask.append(False)
        
        return np.array(train_mask, dtype=bool), np.array(val_mask, dtype=bool)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个 epoch
        
        Returns:
            平均损失
        """
        self.q_network.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            contexts, states, actions, rewards = batch
            
            # 前向传播
            if isinstance(self.q_network, EnsembleModel):
                # Ensemble: 获取所有模型的预测
                q_values_list = self.q_network(states)
                # 使用平均 Q 值计算损失
                q_values_mean = torch.stack(q_values_list, dim=0).mean(dim=0)
                q_values = q_values_mean.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            else:
                # 单模型
                q_values_full = self.q_network(states, contexts)
                q_values = q_values_full.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # 计算损失 (TD error)
            loss = self.criterion(q_values, rewards)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """评估模型
        
        Returns:
            平均损失，平均奖励 (历史), 平均反事实奖励
        """
        self.q_network.eval()
        total_loss = 0.0
        total_reward_historical = 0.0
        total_reward_counterfactual = 0.0
        n_batches = 0
        n_samples = 0
        
        for batch in val_loader:
            contexts, states, actions, rewards = batch
            batch_size = len(rewards)
            
            # 预测 Q 值
            if isinstance(self.q_network, EnsembleModel):
                mean_q, std_q = self.q_network.predict_with_uncertainty(states)
            else:
                mean_q, std_q = self.q_network.predict_with_uncertainty(
                    states, contexts,
                    n_samples=self.config['hyperparameters'].get('mc_dropout_samples', 10)
                )
            
            # 计算损失
            q_values = mean_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            loss = self.criterion(q_values, rewards)
            
            # 策略选择并计算两种奖励
            selected_rewards_historical = []
            selected_rewards_counterfactual = []
            
            for i in range(batch_size):
                action_idx, _ = self.policy.select_action(mean_q[i], std_q[i])
                action_value = self.actions[action_idx]
                
                # 获取上下文 (underfill, mu)
                ctx = contexts[i].cpu().numpy()
                underfill, mu = ctx[0], ctx[1]
                
                # 1. 历史奖励 (离线评估陷阱：只是真实动作的奖励)
                selected_rewards_historical.append(rewards[i].item())
                
                # 2. 保守性反事实奖励 (方案 A+B: 查表 + BC 惩罚)
                if self.pareto_lookup_table is not None and self.bc_lookup_table is not None:
                    cf_reward = get_counterfactual_reward_conservative(
                        self.pareto_lookup_table,
                        self.bc_lookup_table,
                        underfill, mu, action_value,
                        lambda_coeff=self.config['hyperparameters'].get('conservative_lambda', 2.0)
                    )
                else:
                    cf_reward = rewards[i].item()
                
                selected_rewards_counterfactual.append(cf_reward)
            
            total_loss += loss.item() * batch_size
            total_reward_historical += sum(selected_rewards_historical)
            total_reward_counterfactual += sum(selected_rewards_counterfactual)
            n_batches += 1
            n_samples += batch_size
        
        return (
            total_loss / n_samples,
            total_reward_historical / n_samples,
            total_reward_counterfactual / n_samples
        )
    
    def train(self) -> Dict[str, Any]:
        """完整训练流程
        
        Returns:
            训练结果字典
        """
        print("\n=== 开始训练 ===")
        print(f"配置：Encoder={self.config['model']['encoder_type']}, "
              f"Uncertainty={self.config['model']['uncertainty_type']}, "
              f"Policy={self.config['model']['policy_type']}")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        print(f"训练集大小：{len(train_loader.dataset)}")
        print(f"验证集大小：{len(val_loader.dataset)}")
        
        best_val_reward = -float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_reward_historical, val_reward_counterfactual = self.evaluate(val_loader)
            
            # 记录结果
            self.results['train_loss'].append(train_loss)
            self.results['val_loss'].append(val_loss)
            self.results['train_reward'].append(train_loss)  # 近似
            self.results['val_reward'].append(val_reward_historical)
            self.results['val_counterfactual_reward'].append(val_reward_counterfactual)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Reward (Hist): {val_reward_historical:.4f} | "
                      f"Val Reward (CF): {val_reward_counterfactual:.4f}")
            
            # 早停检查 (使用反事实奖励)
            if val_reward_counterfactual > best_val_reward:
                best_val_reward = val_reward_counterfactual
                patience_counter = 0
                # 保存最佳模型
                self.save_model("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch+1}")
                    break
        
        print(f"\n训练完成！最佳验证反事实奖励：{best_val_reward:.4f}")
        
        return self.results
    
    def save_model(self, path: str):
        """保存模型"""
        save_path = Path("results") / path
        save_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'results': self.results
        }
        
        torch.save(checkpoint, save_path)
        print(f"模型已保存至：{save_path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.results = checkpoint.get('results', self.results)
        print(f"模型已从 {path} 加载")
    
    def predict(self, state: np.ndarray, context: np.ndarray) -> Tuple[int, float]:
        """预测最优动作
        
        Args:
            state: (seq_len, feature_dim) 或 (feature_dim,)
            context: (context_dim,)
            
        Returns:
            action_idx: 动作索引
            action_value: 动作值
        """
        self.q_network.eval()
        
        # 转换为 Tensor
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, seq, feat)
        context = torch.FloatTensor(context).unsqueeze(0).to(self.device)  # (1, ctx)
        
        # 预测 Q 值和不确定性
        with torch.no_grad():
            if isinstance(self.q_network, EnsembleModel):
                mean_q, std_q = self.q_network.predict_with_uncertainty(state)
            else:
                mean_q, std_q = self.q_network.predict_with_uncertainty(state, context)
        
        # 策略选择
        action_idx, action_value = self.policy.select_action(mean_q[0], std_q[0])
        
        return action_idx, action_value


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='铝合金锻造优化训练')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 初始化训练器
    trainer = ContextualBanditTrainer(args.config)
    
    # 训练
    results = trainer.train()
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / f"results_{timestamp}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        # 转换 numpy 类型为 Python 原生类型
        serializable_results = {
            k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n结果已保存至：{results_path}")


if __name__ == "__main__":
    main()

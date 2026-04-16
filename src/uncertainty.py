"""
Uncertainty 模块：MC-Dropout、Ensemble 实现
统一接口，支持不确定性估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy


class MCDropoutWrapper(nn.Module):
    """MC-Dropout 包装器
    
    在推理时保持 Dropout 激活，通过多次采样估计不确定性
    """
    
    def __init__(self, module: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.module = module
        self.dropout_rate = dropout_rate
        
        # 在推理时启用 Dropout
        self._enable_mc_dropout()
        
    def _enable_mc_dropout(self):
        """启用 MC-Dropout 模式"""
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()  # 强制设为训练模式以启用 Dropout
        self.module.apply(enable_dropout)
        
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回均值和不确定性
        
        Args:
            x: 输入张量
            n_samples: MC 采样次数
            
        Returns:
            mean: (batch_size, output_dim) - 预测均值
            std: (batch_size, output_dim) - 预测标准差 (不确定性)
        """
        self._enable_mc_dropout()
        
        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.module(x)
                outputs.append(out.unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)  # (n_samples, batch_size, output_dim)
        
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std


class EnsembleModel(nn.Module):
    """Ensemble 模型集合
    
    训练多个独立模型，通过集成估计不确定性
    架构优化：每个子模型包含完整的 Encoder+Head，确保梯度独立传播
    """
    
    def __init__(self, model_class: type, n_models: int = 5, **model_kwargs):
        super().__init__()
        self.n_models = n_models
        # 每个模型都是独立的完整 QNetworkWithUncertainty
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """前向传播，返回所有模型的输出
        
        Args:
            x: 输入张量
            context: 上下文信息
            
        Returns:
            outputs: List[Tensor] - 每个模型的输出
        """
        return [model(x, context) for model in self.models]
    
    def predict_with_uncertainty(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测并估计不确定性
        
        Args:
            x: 输入张量
            context: 上下文信息
            
        Returns:
            mean: (batch_size, output_dim) - 预测均值
            std: (batch_size, output_dim) - 预测标准差 (不确定性)
        """
        with torch.no_grad():
            outputs = self.forward(x, context)
            outputs = torch.stack(outputs, dim=0)  # (n_models, batch_size, output_dim)
            
            mean = outputs.mean(dim=0)
            std = outputs.std(dim=0)
            
            return mean, std
    
    def get_individual_predictions(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取所有个体模型的预测
        
        Returns:
            outputs: (n_models, batch_size, output_dim)
        """
        with torch.no_grad():
            outputs = [model(x, context) for model in self.models]
            return torch.stack(outputs, dim=0)


class DeterministicModel(nn.Module):
    """确定性模型包装器 (无不确定性估计)
    
    用于基线对比，直接输出预测值
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回预测值和零不确定性
        
        Args:
            x: 输入张量
            
        Returns:
            mean: (batch_size, output_dim) - 预测值
            std: (batch_size, output_dim) - 零张量
        """
        out = self.module(x)
        std = torch.zeros_like(out)
        return out, std


def create_uncertainty_wrapper(
    uncertainty_type: str,
    base_model: nn.Module,
    model_class: Optional[type] = None,
    **kwargs
) -> nn.Module:
    """工厂函数：创建 Uncertainty 包装器
    
    Args:
        uncertainty_type: "mc_dropout", "ensemble", 或 "none"
        base_model: 基础模型 (Encoder + Head)
        model_class: 模型类 (仅 ensemble 需要)
        **kwargs: 其他参数
        
    Returns:
        带不确定性估计的模型
    """
    if uncertainty_type == "mc_dropout":
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        return MCDropoutWrapper(base_model, dropout_rate)
    
    elif uncertainty_type == "ensemble":
        n_models = kwargs.get('n_models', 5)
        model_kwargs = kwargs.get('model_kwargs', {})
        return EnsembleModel(model_class, n_models, **model_kwargs)
    
    elif uncertainty_type == "none":
        return DeterministicModel(base_model)
    
    else:
        raise ValueError(f"不支持的 Uncertainty 类型：{uncertainty_type}")


class QNetworkWithUncertainty(nn.Module):
    """带不确定性估计的 Q 网络
    
    整合 Encoder 和 Uncertainty，输出 Q 值和不确定性
    支持两种初始化方式:
    1. 传入 encoder 实例 (单模型模式)
    2. 传入 encoder_type 等参数 (Ensemble 子模型模式)
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        d_model: int = 64,
        n_actions: int = 9,
        uncertainty_type: str = "mc_dropout",
        dropout_rate: float = 0.1,
        # Ensemble 子模型需要的参数
        encoder_type: Optional[str] = None,
        input_dim: Optional[int] = None,
        context_dim: Optional[int] = 2,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 100,
        **uncertainty_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_actions = n_actions
        self.uncertainty_type = uncertainty_type
        self.dropout_rate = dropout_rate
        
        # 创建 Encoder (两种方式)
        if encoder is not None:
            # 方式 1: 使用传入的 encoder 实例
            self.encoder = encoder
        elif encoder_type is not None and input_dim is not None:
            # 方式 2: 创建新的 encoder (用于 Ensemble 子模型)
            from encoders import create_encoder
            self.encoder = create_encoder(
                encoder_type=encoder_type,
                input_dim=input_dim,
                context_dim=context_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout_rate,
                max_seq_len=max_seq_len
            )
        else:
            raise ValueError("必须提供 encoder 实例或 encoder_type+input_dim 参数")
        
        # Q 值预测头 (始终包含 Dropout，MC-Dropout 时启用)
        if uncertainty_type == "mc_dropout":
            self.q_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, n_actions)
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_actions)
            )
    
    def _enable_mc_dropout(self):
        """启用 MC-Dropout 模式"""
        if self.uncertainty_type != "mc_dropout":
            return
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self.apply(enable_dropout)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码输入并预测 Q 值
        
        Args:
            x: (batch_size, seq_len, feature_dim)
            context: (batch_size, context_dim)
            
        Returns:
            q_values: (batch_size, n_actions)
        """
        # 编码
        if isinstance(self.encoder, nn.Module) and hasattr(self.encoder, 'forward'):
            import inspect
            sig = inspect.signature(self.encoder.forward)
            if 'context' in sig.parameters and context is not None:
                encoded = self.encoder(x, context)
            else:
                encoded = self.encoder(x)
        else:
            encoded = self.encoder(x)
        
        # 池化 (取最后一步或平均)
        encoded_pool = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Q 值预测
        q_values = self.q_head(encoded_pool)
        
        return q_values
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测 Q 值及其不确定性
        
        Args:
            x: 状态序列
            context: 上下文
            n_samples: 采样次数 (MC-Dropout)
            
        Returns:
            mean_q: (batch_size, n_actions) - Q 值均值
            std_q: (batch_size, n_actions) - Q 值不确定性
        """
        if self.uncertainty_type == "mc_dropout":
            # MC-Dropout: 多次前向传播
            self._enable_mc_dropout()
            
            q_values_list = []
            with torch.no_grad():
                for _ in range(n_samples):
                    q_vals = self.forward(x, context)
                    q_values_list.append(q_vals.unsqueeze(0))
            
            q_values_all = torch.cat(q_values_list, dim=0)
            mean_q = q_values_all.mean(dim=0)
            std_q = q_values_all.std(dim=0)
            
            return mean_q, std_q
        
        elif self.uncertainty_type == "ensemble":
            # Ensemble: 由外部 EnsembleModel 处理
            raise NotImplementedError("Ensemble 的不确定性估计需在外部调用")
        
        else:  # deterministic
            with torch.no_grad():
                mean_q = self.forward(x, context)
                std_q = torch.zeros_like(mean_q)
                return mean_q, std_q


if __name__ == "__main__":
    # 测试 Uncertainty 模块
    from encoders import create_encoder
    
    batch_size = 32
    seq_len = 50
    feature_dim = 51  # 51 维物理特征
    context_dim = 2
    d_model = 64
    n_actions = 9
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    context = torch.randn(batch_size, context_dim)
    
    # 创建基础 Encoder
    encoder = create_encoder("transformer", feature_dim, d_model=d_model)
    
    # 测试 MC-Dropout
    print("\n=== 测试 MC-Dropout ===")
    q_net_mc = QNetworkWithUncertainty(
        encoder=encoder,
        d_model=d_model,
        n_actions=n_actions,
        uncertainty_type="mc_dropout",
        dropout_rate=0.1
    )
    mean_q, std_q = q_net_mc.predict_with_uncertainty(x, context, n_samples=5)
    print(f"MC-Dropout - Q 值均值形状：{mean_q.shape}, 不确定性形状：{std_q.shape}")
    print(f"平均不确定性：{std_q.mean().item():.4f}")
    
    # 测试 Deterministic
    print("\n=== 测试 Deterministic ===")
    q_net_det = QNetworkWithUncertainty(
        encoder=encoder,
        d_model=d_model,
        n_actions=n_actions,
        uncertainty_type="none"
    )
    mean_q_det, std_q_det = q_net_det.predict_with_uncertainty(x, context)
    print(f"Deterministic - Q 值形状：{mean_q_det.shape}, 不确定性形状：{std_q_det.shape}")
    print(f"平均不确定性：{std_q_det.mean().item():.4f}")

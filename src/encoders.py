"""
Encoder 模块：Transformer 和 TFT 实现
统一接口，可自由切换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer Encoder
    
    用于编码锻造过程的状态特征序列
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出投影 (保持维度一致)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - 状态特征序列
            mask: 可选的注意力掩码
            
        Returns:
            encoded: (batch_size, seq_len, d_model) - 编码后的表示
        """
        # 转换为 (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # 输入投影
        x = self.input_projection(x)
        x = x * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        encoded = self.transformer_encoder(x, mask)
        
        # 转换回 (batch_size, seq_len, d_model)
        encoded = encoded.transpose(0, 1)
        
        # 输出投影
        encoded = self.output_projection(encoded)
        
        return encoded


class TFTVariableSelectionNetwork(nn.Module):
    """TFT 变量选择网络"""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            weights: (batch_size, 1) - 选择权重
            features: (batch_size, d_model) - 加权后的特征
        """
        # 先投影到 d_model
        projected = self.fc1(x)  # (batch, d_model)
        # 计算选择权重
        weights = self.fc2(self.dropout(self.elu(projected)))  # (batch, 1)
        weights = torch.softmax(weights, dim=-1)
        
        # 加权特征
        features = projected * weights
        
        return weights, features


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer (TFT)
    
    简化版 TFT，专注于时间序列编码
    """
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.d_model = d_model
        
        # 静态变量编码器 (context: underfill, mu)
        self.static_encoder = nn.Linear(context_dim, d_model)
        
        # 变量选择网络
        self.variable_selection = TFTVariableSelectionNetwork(input_dim, d_model, dropout)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # LSTM 层 (TFT 使用 LSTM 而非纯 Transformer)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # Gated Residual Network (处理双向 LSTM 输出)
        # 修改：移除 GLU，改用普通线性层避免维度问题
        self.grn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 最终投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - 状态特征序列
            context: (batch_size, context_dim) - 静态上下文 (underfill, mu)
            
        Returns:
            encoded: (batch_size, seq_len, d_model) - 编码后的表示
        """
        batch_size, seq_len, _ = x.shape
        
        # 变量选择
        selected_features = []
        for t in range(seq_len):
            _, feat = self.variable_selection(x[:, t, :])
            selected_features.append(feat.unsqueeze(1))
        x_selected = torch.cat(selected_features, dim=1)  # (batch, seq, d_model)
        
        # 静态上下文编码
        if context is not None:
            static_context = self.static_encoder(context).unsqueeze(1)  # (batch, 1, d_model)
            static_context = static_context.expand(-1, seq_len, -1)  # (batch, seq, d_model)
            x_embedded = x_selected + static_context
        else:
            x_embedded = x_selected
        
        # 位置编码
        x_embedded = self.pos_encoder.pe[:seq_len].transpose(0, 1) + x_embedded
        
        # LSTM 编码
        lstm_out, _ = self.lstm(x_embedded)  # (batch, seq, d_model*2)
        
        # Gated Residual Connection
        encoded = self.grn(lstm_out)  # (batch, seq, d_model/2)
        encoded = self.output_projection(encoded)  # (batch, seq, d_model)
        
        # 自注意力
        attn_out, _ = self.multihead_attn(encoded, encoded, encoded)
        encoded = encoded + attn_out
        
        return encoded


def create_encoder(
    encoder_type: str,
    input_dim: int,
    context_dim: int = 2,
    **kwargs
) -> nn.Module:
    """工厂函数：创建 Encoder
    
    Args:
        encoder_type: "transformer" 或 "tft"
        input_dim: 输入特征维度
        context_dim: 上下文维度
        **kwargs: 其他超参数
        
    Returns:
        Encoder 模块
    """
    if encoder_type == "transformer":
        return TransformerEncoder(
            input_dim=input_dim,
            d_model=kwargs.get('d_model', 64),
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
            max_seq_len=kwargs.get('max_seq_len', 100)
        )
    elif encoder_type == "tft":
        return TemporalFusionTransformer(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=kwargs.get('d_model', 64),
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 3),
            dropout=kwargs.get('dropout', 0.1),
            max_seq_len=kwargs.get('max_seq_len', 100)
        )
    else:
        raise ValueError(f"不支持的 Encoder 类型：{encoder_type}")


if __name__ == "__main__":
    # 测试 Encoder
    batch_size = 32
    seq_len = 50
    input_dim = 56
    context_dim = 2
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, input_dim)
    context = torch.randn(batch_size, context_dim)
    
    # 测试 Transformer
    transformer = create_encoder("transformer", input_dim)
    out_transformer = transformer(x)
    print(f"Transformer 输出形状：{out_transformer.shape}")
    
    # 测试 TFT
    tft = create_encoder("tft", input_dim, context_dim)
    out_tft = tft(x, context)
    print(f"TFT 输出形状：{out_tft.shape}")

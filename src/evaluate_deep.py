import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

# 确保能导入 src 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_loader import ForgeDataLoader
from uncertainty import QNetworkWithUncertainty, EnsembleModel
from reward_utils import calculate_quality_score, compute_speed_utility, compute_reward_with_utility

def load_config(config_path='config/config_1_transformer_mcdropout_ucb.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def deep_evaluation():
    print("=== 开始深度评估 ===")
    config = load_config()
    
    # 1. 加载数据和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 加载完整数据用于分析
    df_all = pd.read_csv(config['data']['path'])
    
    # 构建查找表逻辑 (复用 train.py 中的逻辑简化版)
    # 这里为了演示，我们直接对全量数据计算 Quality
    print("计算全量数据质量分数...")
    df_all['quality'] = df_all.apply(calculate_quality_score, axis=1)
    df_all['speed_util'] = df_all['r_l'].apply(compute_speed_utility)
    df_all['reward'] = df_all.apply(lambda x: compute_reward_with_utility(x['r_l'], x['quality']), axis=1)
    
    # 加载模型
    model_path = 'results/best_model.pth'
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('config', config)
    
    # 初始化模型结构
    feature_cols = [c for c in df_all.columns if c.startswith(('A_', 'B_', 'C_', 'global__'))]
    context_cols = ['underfill', 'mu']
    
    model = ContextualBanditModel(
        input_dim=len(feature_cols),
        context_dim=len(context_cols),
        num_actions=len(config['actions']['values']),
        action_values=config['actions']['values'],
        encoder_type=model_cfg.get('encoder', 'transformer'),
        uncertainty_type=model_cfg.get('uncertainty', 'mc_dropout'),
        policy_type=model_cfg.get('policy', 'neural_ucb'),
        dropout_rate=model_cfg.get('model', {}).get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备验证集数据 (这里用全量数据模拟，实际应取验证集划分)
    # 为了严谨，我们随机采样一部分作为"测试环境"
    test_df = df_all.sample(n=50, random_state=42).reset_index(drop=True)
    
    X_dyn = test_df[feature_cols].values.astype(np.float32)
    X_ctx = test_df[context_cols].values.astype(np.float32)
    y_true = test_df['r_l'].values
    
    # 标准化 (需使用训练时的均值方差，这里简化处理，假设模型内部已处理或使用全局统计)
    # 注意：实际推理时应使用训练集的 scaler。这里假设数据已经过类似处理或直接输入相对值
    # 如果 data_loader 做了标准化，这里需要复现。为简化，我们假设模型鲁棒性或数据本身已归一化
    # *修正*：通常 train.py 会保存 scaler，这里我们暂时直接用原始数据，观察模型反应
    # 如果模型训练时用了 Z-Score，这里必须用同样的 mean/std
    # 由于无法直接获取 train.py 中的 scaler 对象，我们假设模型内部没有强依赖特定分布，或者数据本身接近标准正态
    # *更稳妥的做法*：重新实例化 DataLoader 获取 scaler
    
    dataset = ForgeDataset(df_all, feature_cols, context_cols, config['actions']['values'])
    # 重新划分以获取 scaler
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    train_df = df_all.iloc[train_idx]
    
    mean_dyn = train_df[feature_cols].mean().values
    std_dyn = train_df[feature_cols].std().values + 1e-6
    mean_ctx = train_df[context_cols].mean().values
    std_ctx = train_df[context_cols].std().values + 1e-6
    
    # 标准化测试数据
    X_dyn_norm = (X_dyn - mean_dyn) / std_dyn
    X_ctx_norm = (X_ctx - mean_ctx) / std_ctx
    
    X_dyn_t = torch.tensor(X_dyn_norm).to(device)
    X_ctx_t = torch.tensor(X_ctx_norm).to(device)
    
    predictions = []
    q_values_all = []
    
    print("运行模型推理...")
    with torch.no_grad():
        # MC Dropout 采样 (如果启用)
        num_samples = 10 if model_cfg.get('uncertainty') == 'mc_dropout' else 1
        
        for i in range(len(test_df)):
            dyn_in = X_dyn_t[i:i+1].repeat(num_samples, 1)
            ctx_in = X_ctx_t[i:i+1].repeat(num_samples, 1)
            
            q_vals = model(dyn_in, ctx_in) # [N, num_actions]
            q_mean = q_vals.mean(dim=0).cpu().numpy()
            
            pred_idx = np.argmax(q_mean)
            pred_action = config['actions']['values'][pred_idx]
            
            predictions.append(pred_action)
            q_values_all.append(q_mean)
            
    test_df['pred_action'] = predictions
    
    # ================= 分析部分 =================
    
    # 1. 策略分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=np.arange(0.59, 1.01, 0.05), alpha=0.7, edgecolor='black')
    plt.title("Model Policy Distribution (Predicted Actions)")
    plt.xlabel("Link Ratio (Speed)")
    plt.ylabel("Frequency")
    plt.xticks(config['actions']['values'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('results/policy_dist.png')
    print("策略分布图已保存至 results/policy_dist.png")
    
    unique, counts = np.unique(predictions, return_counts=True)
    print("\n--- 策略分布统计 ---")
    for u, c in zip(unique, counts):
        print(f"动作 {u:.2f}: {c} 次 ({c/len(predictions)*100:.1f}%)")
        
    if len(unique) == 1:
        print("⚠️ 警告：模型 collapsed 到单一动作，可能未学到上下文依赖性！")
    else:
        print("✅ 模型表现出一定的策略多样性。")

    # 2. 上下文敏感性分析 (Context Sensitivity)
    print("\n--- 上下文敏感性分析 ---")
    # 按 mu 分组看预测倾向
    test_df['mu_bin'] = pd.qcut(test_df['mu'], q=3, labels=['Low', 'Mid', 'High'])
    group_stats = test_df.groupby('mu_bin')['pred_action'].mean()
    print("不同摩擦系数 (mu) 下的平均预测速度:")
    print(group_stats)
    
    # 简单判断：如果 High mu 对应低速，Low mu 对应高速，则符合物理直觉
    if group_stats['Low'] > group_stats['High']:
        print("✅ 符合物理直觉：低摩擦时倾向于高速，高摩擦时倾向于低速。")
    else:
        print("⚠️ 注意：模型未表现出明显的摩擦 - 速度权衡策略，或数据中该特征不明显。")

    # 3. 收益增益分析 (Gain Analysis)
    # 计算模型选择的奖励 vs 随机选择 vs 固定 0.75
    # 注意：这里我们需要为每个样本查找其对应的 "True Quality"
    # 由于我们的数据是离散的 (135 条轨迹)，我们需要匹配 (underfill, mu, pred_action)
    
    # 构建查找字典: (uf, mu, rl) -> quality
    # 由于连续值难以精确匹配，我们寻找最近的邻居
    def get_quality(uf, mu, rl, df_ref):
        # 简单的最近邻搜索
        diff = (df_ref['underfill'] - uf)**2 + (df_ref['mu'] - mu)**2 + (df_ref['r_l'] - rl)**2
        idx = diff.idxmin()
        return df_ref.loc[idx, 'quality']

    rewards_model = []
    rewards_fixed_75 = []
    rewards_random = []
    
    for _, row in test_df.iterrows():
        q_pred = get_quality(row['underfill'], row['mu'], row['pred_action'], df_all)
        r_mod = compute_reward_with_utility(row['pred_action'], q_pred)
        rewards_model.append(r_mod)
        
        q_75 = get_quality(row['underfill'], row['mu'], 0.75, df_all)
        r_75 = compute_reward_with_utility(0.75, q_75)
        rewards_fixed_75.append(r_75)
        
        r_rand = np.random.choice([compute_reward_with_utility(a, get_quality(row['underfill'], row['mu'], a, df_all)) for a in config['actions']['values']])
        rewards_random.append(r_rand)
        
    gain_vs_fixed = np.mean(rewards_model) - np.mean(rewards_fixed_75)
    gain_vs_rand = np.mean(rewards_model) - np.mean(rewards_random)
    
    print("\n--- 收益增益分析 ---")
    print(f"模型平均奖励：{np.mean(rewards_model):.4f}")
    print(f"固定 0.75 策略奖励：{np.mean(rewards_fixed_75):.4f}")
    print(f"随机策略平均奖励：{np.mean(rewards_random):.4f}")
    print(f"增益 (vs Fixed 0.75): {gain_vs_fixed:.4f} {'✅ 赢' if gain_vs_fixed > 0 else '❌ 输'}")
    print(f"增益 (vs Random): {gain_vs_rand:.4f}")
    
    if abs(gain_vs_fixed) < 0.01:
        print("\n⚠️ 关键结论：模型表现与‘无脑选 0.75’几乎无异。")
        print("原因可能是数据中速度对质量影响太小，导致 0.75 成为绝对主导解。")
        print("建议：检查物理仿真参数范围，或增大速度变化的物理效应。")
    else:
        print("\n✅ 模型学到了超越固定策略的知识。")

    # 4. 可视化：Q 值热力图
    # 选取一个典型的 underfill，横轴 mu，纵轴动作，颜色 Q 值
    sample_uf = test_df['underfill'].iloc[0]
    mu_range = np.linspace(test_df['mu'].min(), test_df['mu'].max(), 20)
    
    heat_q = []
    for m in mu_range:
        # 构造输入
        ctx_vec = np.array([[sample_uf, m]])
        ctx_vec_norm = (ctx_vec - mean_ctx) / std_ctx
        ctx_t = torch.tensor(ctx_vec_norm).float().to(device)
        
        # 动态特征取平均值作为代表
        dyn_vec = np.mean(X_dyn_norm, axis=0, keepdims=True)
        dyn_t = torch.tensor(dyn_vec).float().to(device).repeat(20, 1) # dummy batch
        
        # 实际只需预测一次
        with torch.no_grad():
            q_val = model(torch.tensor(dyn_vec).float().unsqueeze(0).to(device), ctx_t[0:1])[0].cpu().numpy()
        heat_q.append(q_val)
        
    heat_q = np.array(heat_q) # [20, 9]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(heat_q.T, aspect='auto', cmap='RdYlGn', origin='lower')
    plt.yticks(range(len(config['actions']['values'])), [f"{x:.2f}" for x in config['actions']['values']])
    plt.xlabel('Friction Coeff (mu)')
    plt.ylabel('Action (r_l)')
    plt.title(f'Q-Value Heatmap (underfill={sample_uf:.2f})')
    plt.colorbar(label='Q-Value')
    plt.savefig('results/q_heatmap.png')
    print("Q 值热力图已保存至 results/q_heatmap.png")

if __name__ == '__main__':
    deep_evaluation()

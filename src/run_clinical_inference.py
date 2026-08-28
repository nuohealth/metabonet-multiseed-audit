#!/usr/bin/env python3
"""
MetaboNet V2 临床评估 - 重训练缺失模型并生成逐样本预测
- 对缺失 checkpoint 的模型 (BiLSTM, LSTM-Attention) 用 seed 42 重训练并保存
- 对全部4个架构用可用 checkpoint 在测试集上生成逐样本预测
- 保存逐样本预测到 npz，供临床评估(Clarke EGA/低血糖/分层RMSE)使用
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json, os, sys, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


DATA_PATH = os.path.join(BASE_DIR, 'data', 'metabonet_v2', 'metabonet_v2_data.npz')
OUT_DIR = os.path.join(BASE_DIR, 'results', 'metabonet_v2_monitored')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3
DEVICE = 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 模型定义 (与训练脚本一致) ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1)

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(self.attention(out))

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(self.attention(out))

MODELS = {
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
    'LSTM-Attention': LSTMAttentionModel,
    'BiLSTM-Attention': BiLSTMAttentionModel,
}

# ---------- 数据 ----------
print("加载数据...")
data = np.load(DATA_PATH, allow_pickle=True)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

class CGMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = CGMDataset(X_train, y_train)
val_ds = CGMDataset(X_val, y_val)
test_ds = CGMDataset(X_test, y_test)

def train_and_save(model_name, ModelClass, ckpt_path):
    """重训练并保存模型 (用于缺失 checkpoint 的模型)"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ModelClass()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    best_val = float('inf')
    patience = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vloss += criterion(model(Xb), yb).item()
        vloss /= len(val_loader)
        if vloss < best_val:
            best_val = vloss
            patience = 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{model_name}] 已训练保存到 {ckpt_path}, 最佳val_loss={best_val:.4f}, 耗时={time.time()-t0:.1f}s")
    return model

def get_model(model_name):
    """V2 重训练。注意: V1 checkpoint (Jul 22) 与 V2 数据/config 不兼容，
    必须用 V2 config 重训以获得与报告一致的逐样本预测。"""
    ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_v2.pt')
    ModelClass = MODELS[model_name]
    model = ModelClass()
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"[{model_name}] 加载已有 V2 checkpoint {ckpt_path}")
    else:
        print(f"[{model_name}] V2 checkpoint 不存在，重训练...")
        model = train_and_save(model_name, ModelClass, ckpt_path)
    return model

# ---------- 推理：生成测试集逐样本预测 ----------
def infer(model):
    """模型预测原始目标值(与y_train同尺度)。注意: 不应对preds做 *std+mean 反标准化——
    模型直接预测raw目标，与实际值同尺度（原脚本该行是bug，results_final_corrected已修正）。"""
    model.eval()
    model.to(DEVICE)
    preds, acts = [], []
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(DEVICE))
            preds.append(out.cpu().numpy())
            acts.append(yb.numpy())
    preds = np.concatenate(preds)   # 已修正: 保持raw尺度
    acts = np.concatenate(acts)     # 实际值已是原尺度
    return preds, acts

all_preds = {}
all_acts = None
print("=== 生成逐样本预测 (seed 42, 测试集) ===")
for name in MODELS:
    model = get_model(name)
    pred, act = infer(model)
    all_preds[name] = pred
    # 验证指标
    rmse = np.sqrt(np.mean((pred - act) ** 2))
    mae = np.mean(np.abs(pred - act))
    mard = np.mean(np.abs(pred - act) / act) * 100
    print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MARD={mard:.2f}%")
    all_acts = act

np.savez(os.path.join(OUT_DIR, 'clinical_predictions.npz'),
         **{f'pred_{k}': v for k, v in all_preds.items()},
         actual=all_acts)
print("逐样本预测已保存:", os.path.join(OUT_DIR, 'clinical_predictions.npz'))
print("形状:", {k: v.shape for k, v in all_preds.items()})

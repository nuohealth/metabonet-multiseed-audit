#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaboNet 多seed消融 — 补足到20个seed (论文#5 → CMPB)
- SEEDS: 42..61 (20个)
- 4模型: LSTM / BiLSTM / LSTM-Attention / BiLSTM-Attention
- 总任务: 4 x 20 = 80 次训练 (已完成42-46的20次, 本次补47-61的60次)
- 断点续训: 每次完成后增量写结果文件, 中断后跳过已完成
- 串行训练(安全, 3G内存不OOM), 结果进度写 progress_20seed.json
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json, time, gc, os
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


# ---------- 配置 ----------
DATA_PATH=os.path.join(BASE_DIR, 'data', 'metabonet_v2', 'metabonet_v2_data.npz')
OUT_DIR=os.path.join(BASE_DIR, 'results', 'metabonet_v2_20seed')
DEVICE='cpu'
# 降内存: batch 64->32, 减少单次前向/反向峰值内存
BATCH_SIZE=32
EPOCHS=15
LEARNING_RATE=0.001
PATIENCE=3
# 20个种子: 42-61 (审稿人要求20-seed稳健性)
ALL_SEEDS=list(range(42,62))
os.makedirs(OUT_DIR, exist_ok=True)
# 数据加载模式: lazy (numpy索引) 避免整份转tensor爆内存; 3.xG内存必需
USE_LAZY_LOAD=True

# ---------- 加载数据 ----------
print("加载数据...")
data=np.load(DATA_PATH, allow_pickle=True)
X_train,y_train=data['X_train'],data['y_train']
X_val,y_val=data['X_val'],data['y_val']
X_test,y_test=data['X_test'],data['y_test']
mean,std=X_train.mean(),X_train.std()
X_train=(X_train-mean)/std; X_val=(X_val-mean)/std; X_test=(X_test-mean)/std
print(f"训练{X_train.shape} 验证{X_val.shape} 测试{X_test.shape}")

# ---------- 模型(与原文一致) ----------
class CGMDataset(Dataset):
    def __init__(self,X,y,lazy=True):
        # lazy=True: 保留numpy, 每次__getitem__转tensor, 避免一次性拷贝65万x12到GPU/CPU内存
        if lazy:
            self.X,self.y=X,y
        else:
            self.X=torch.FloatTensor(X); self.y=torch.FloatTensor(y)
        self.lazy=lazy
    def __len__(self): return len(self.X)
    def __getitem__(self,i):
        if self.lazy:
            return torch.FloatTensor(self.X[i]), torch.FloatTensor(self.y[i])
        return self.X[i],self.y[i]

class LSTMModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=128,num_layers=2,output_size=6,dropout=0.3):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout); self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        if x.dim()==2: x=x.unsqueeze(-1)
        out,_=self.lstm(x); return self.fc(out[:,-1,:])

class BiLSTMModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=6,dropout=0.3):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True,dropout=dropout); self.fc=nn.Linear(hidden_size*2,output_size)
    def forward(self,x):
        if x.dim()==2: x=x.unsqueeze(-1)
        out,_=self.lstm(x); return self.fc(out[:,-1,:])

class Attention(nn.Module):
    def __init__(self,hs): super().__init__(); self.a=nn.Sequential(nn.Linear(hs,hs),nn.Tanh(),nn.Linear(hs,1))
    def forward(self,x): w=torch.softmax(self.a(x),dim=1); return torch.sum(x*w,dim=1)

class LSTMAttentionModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=128,num_layers=2,output_size=6,dropout=0.3):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout); self.att=Attention(hidden_size); self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        if x.dim()==2: x=x.unsqueeze(-1)
        out,_=self.lstm(x); out=self.att(out); return self.fc(out)

class BiLSTMAttentionModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=6,dropout=0.3):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True,dropout=dropout); self.att=Attention(hidden_size*2); self.fc=nn.Linear(hidden_size*2,output_size)
    def forward(self,x):
        if x.dim()==2: x=x.unsqueeze(-1)
        out,_=self.lstm(x); out=self.att(out); return self.fc(out)

# ---------- 训练单个 ----------
def train_one(model_name,seed,task_idx=0,total=80):
    torch.manual_seed(seed); np.random.seed(seed)
    DL=CGMDataset
    tr=DataLoader(DL(X_train,y_train,lazy=USE_LAZY_LOAD),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    va=DataLoader(DL(X_val,y_val,lazy=USE_LAZY_LOAD),batch_size=BATCH_SIZE,num_workers=0)
    te=DataLoader(DL(X_test,y_test,lazy=USE_LAZY_LOAD),batch_size=BATCH_SIZE,num_workers=0)
    model={'LSTM':LSTMModel,'BiLSTM':BiLSTMModel,'LSTM-Attention':LSTMAttentionModel,'BiLSTM-Attention':BiLSTMAttentionModel}[model_name]()
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); crit=nn.MSELoss()
    best=float('inf'); pc=0; best_state=None; t0=time.time()
    for ep in range(EPOCHS):
        model.train(); tl=0
        ep_t0=time.time()
        for xb,yb in tr:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step(); tl+=loss.item()
        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in va: vl+=crit(model(xb),yb).item()
        tl/=len(tr); vl/=len(va)
        # 每epoch实时日志+进度(修昨天'无进度拖两天'的问题)
        ep_time=time.time()-ep_t0
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] {model_name}-seed{seed} epoch{ep+1}/{EPOCHS} tr_loss={tl:.4f} va_loss={vl:.4f} ({ep_time:.0f}s)",flush=True)
        # 实时进度写入文件(用户可随时 cat progress_20seed.json 查看)
        write_runtime_status(model_name,seed,ep+1,task_idx,total,ep+1)
        if vl<best: best=vl; pc=0; best_state=model.state_dict().copy()
        else:
            pc+=1
            if pc>=PATIENCE: break
    # 在测试集推理(此时已完成训练, 计算指标)
    model.load_state_dict(best_state); model.eval()
    pred,act=[],[]
    with torch.no_grad():
        for xb,yb in te:
            pred.append(model(xb).numpy()); act.append(yb.numpy())
    pred=np.concatenate(pred); act=np.concatenate(act)
    # ⚠️ 反标准化bug修复(2026-08-22): y从未标准化(实测38-418 mg/dL原始值), 模型输出pred即mg/dL,
    #    正确做法是pred直用, 禁止 pred*std+mean (那会把正确预测放大几十倍, RMSE=974失真)。
    #    修正后真实RMSE≈16.27(见 results_corrected.json 5-seed验证)。
    rmse=float(np.sqrt(np.mean((pred-act)**2))); mae=float(np.mean(np.abs(pred-act)))
    mard=float(np.mean(np.abs(pred-act)/act)*100)
    ss_res=np.sum((act-pred)**2); ss_tot=np.sum((act-act.mean())**2); r2=float(1-ss_res/ss_tot)
    elapsed=(time.time()-t0)/60
    del model; gc.collect()
    return {'model':model_name,'seed':int(seed),'rmse':rmse,'mae':mae,'mard':mard,'r2':r2,'time_min':elapsed,'epochs':ep+1,'ts':datetime.now().isoformat()}

def result_path(model,seed): return os.path.join(OUT_DIR,f'result_{model}_seed{seed}.json')
def progress_path(): return os.path.join(OUT_DIR,'progress_20seed.json')

def write_runtime_status(model_name,seed,ep,task_idx,total,epochs_done):
    """实时进度写入(每epoch调用), 供用户随时查看, 含预计剩余时间"""
    P=progress_path()
    st={}
    if os.path.exists(P):
        try: st=json.load(open(P))
        except: st={}
    st['current']=model_name
    st['current_seed']=int(seed)
    st['current_epoch']=ep
    st['epochs_done']=epochs_done
    st['task_idx']=task_idx+1          # 1-based
    st['task_total']=total
    st['tasks_done']=len(st.get('done',[]))
    st['last_update']=datetime.now().isoformat()
    st['status']='running' if task_idx+1<=total else 'done'
    # 预计剩余: 基于当前seed已用时间×剩余seed数
    st['eta_note']=f'已完成{st["tasks_done"]}/{total}项; 当前第{task_idx+1}项 {model_name}-seed{seed} epoch{ep}'    
    json.dump(st,open(P,'w'),indent=2)

def load_progress():
    pf=os.path.join(OUT_DIR,'progress_20seed.json')
    if os.path.exists(pf):
        try: return json.load(open(pf))
        except: return {}
    return {}

def save_progress(p):
    json.dump(p,open(os.path.join(OUT_DIR,'progress_20seed.json'),'w'),indent=2)

# ---------- 主循环(串行+断点续训) ----------
if __name__=='__main__':
    models=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
    prog=load_progress()
    total=len(models)*len(ALL_SEEDS)
    # 按模型×种子排序: 先各模型seed42,再seed43... 或按模型内顺序
    # 排序: 按(seed)优先, 同seed内4模型一起, 便于中途看各模型对比
    tasks=[(m,s) for s in ALL_SEEDS for m in models]
    done_count=0
    start_all=time.time()
    for model_name,seed in tasks:
        if result_path(model_name,seed) in prog.get('done',[]) or os.path.exists(result_path(model_name,seed)):
            done_count+=1; continue
        # 检查是否已算过(progress里的done)
        key=f'{model_name}_seed{seed}'
        if key in prog.get('done',[]): done_count+=1; continue
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 训练 {model_name}-seed{seed} ({done_count+1}/{total})...",flush=True)
        # seed开始时立即写进度(防中途崩溃无记录, 修昨天双失败教训)
        write_runtime_status(model_name,seed,0,done_count,total,0)
        try:
            res=train_one(model_name,seed,done_count,total)
            json.dump(res,open(result_path(model_name,seed),'w'))
            prog.setdefault('done',[]).append(f'{model_name}_seed{seed}')
            prog['last_update']=datetime.now().isoformat()
            prog['done_count']=len(prog['done'])
            prog['elapsed_h']=round((time.time()-start_all)/3600,2)
            save_progress(prog)
            done_count+=1
            print(f"  完成 {model_name}-seed{seed}: RMSE={res['rmse']:.2f} R²={res['r2']:.4f} {res['time_min']:.0f}min")
        except Exception as e:
            print(f"  ❌ {model_name}-seed{seed} 失败: {e}")
            open(os.path.join(OUT_DIR,'errors.log'),'a').write(f"{datetime.now()} {model_name}_seed{seed}: {e}\n")
    print(f"\n=== 全部完成! 已完成 {done_count}/{total}, 总耗时 {prog.get('elapsed_h',0)}h ===")

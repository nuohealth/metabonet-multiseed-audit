#!/usr/bin/env python3
"""
MetaboNet数据清洗与窗口生成 - 分批处理版本
内存优化：每次只处理一个row group，避免内存溢出
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pickle
import gc
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


# 配置
DATA_PATH = '/APapers/ResearchDATA/MetaboNet2026/train.parquet'
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'metabonet_v2')
ELIGIBLE_IDS_PATH = '/tmp/eligible_subjects_14_60.pkl'

# 窗口参数
LOOKBACK = 12  # 60分钟输入
HORIZON = 6    # 30分钟输出
MAX_WINDOWS_PER_SUBJECT = 1000  # 每个受试者最多窗口数（防内存溢出）

# 加载合格受试者列表
with open(ELIGIBLE_IDS_PATH, 'rb') as f:
    eligible_ids = set(pickle.load(f))

print(f"目标受试者数: {len(eligible_ids)}")

# 初始化存储
all_windows = []
all_labels = []
all_subject_ids = []

# 分批处理row groups
pf = pq.ParquetFile(DATA_PATH)
n_rgs = pf.metadata.num_row_groups

print(f"总row groups: {n_rgs}")
print("开始分批处理...")

processed_subjects = set()

for rg_idx in range(n_rgs):
    # 读取一个row group
    df = pf.read_row_group(rg_idx, columns=['id', 'date', 'CGM']).to_pandas()
    
    # 筛选合格受试者
    df = df[df['id'].isin(eligible_ids)]
    
    if len(df) == 0:
        continue
    
    # 按受试者处理
    for sid, group in df.groupby('id'):
        if sid in processed_subjects:
            continue
        
        # 排序并去重
        group = group.sort_values('date').drop_duplicates('date')
        
        # 提取有效CGM
        valid = group[group['CGM'].notna()]
        if len(valid) < LOOKBACK + HORIZON:
            continue
        
        # 生成窗口
        cgm_values = valid['CGM'].values.astype(np.float32)
        timestamps = valid['date'].values
        
        windows = []
        labels = []
        
        for i in range(LOOKBACK + HORIZON - 1, len(cgm_values)):
            # 输入窗口
            window = cgm_values[i-LOOKBACK-HORIZON+1:i-HORIZON+1]
            # 输出标签
            label = cgm_values[i-HORIZON+1:i+1]
            
            if len(window) == LOOKBACK and len(label) == HORIZON:
                windows.append(window)
                labels.append(label)
        
        # 限制每个受试者的窗口数
        if len(windows) > MAX_WINDOWS_PER_SUBJECT:
            indices = np.linspace(0, len(windows)-1, MAX_WINDOWS_PER_SUBJECT, dtype=int)
            windows = [windows[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        if len(windows) > 0:
            all_windows.extend(windows)
            all_labels.extend(labels)
            all_subject_ids.extend([sid] * len(windows))
            processed_subjects.add(sid)
    
    # 清理内存
    del df
    gc.collect()
    
    if (rg_idx + 1) % 20 == 0:
        print(f"  已处理 {rg_idx+1}/{n_rgs} RGs, {len(processed_subjects)} subjects, {len(all_windows)} windows")

print(f"\n=== 处理完成 ===")
print(f"总受试者: {len(processed_subjects)}")
print(f"总窗口数: {len(all_windows)}")

# 转换为numpy数组
X = np.array(all_windows, dtype=np.float32)
y = np.array(all_labels, dtype=np.float32)
subject_ids = np.array(all_subject_ids)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 按受试者划分train/val/test (60/20/20)
unique_subjects = np.array(list(processed_subjects))
np.random.seed(42)
np.random.shuffle(unique_subjects)

n_train = int(len(unique_subjects) * 0.6)
n_val = int(len(unique_subjects) * 0.2)

train_subjects = set(unique_subjects[:n_train])
val_subjects = set(unique_subjects[n_train:n_train+n_val])
test_subjects = set(unique_subjects[n_train+n_val:])

train_mask = np.array([sid in train_subjects for sid in subject_ids])
val_mask = np.array([sid in val_subjects for sid in subject_ids])
test_mask = np.array([sid in test_subjects for sid in subject_ids])

# 保存数据
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'metabonet_v2_data.npz'),
    X_train=X[train_mask],
    y_train=y[train_mask],
    X_val=X[val_mask],
    y_val=y[val_mask],
    X_test=X[test_mask],
    y_test=y[test_mask],
    train_subjects=list(train_subjects),
    val_subjects=list(val_subjects),
    test_subjects=list(test_subjects),
)

# 保存统计报告
report = f"""# MetaboNet V2 数据报告

## 筛选标准
- 最小记录天数: ≥14天
- 最小CGM覆盖率: ≥60%

## 样本量
- 总受试者: {len(processed_subjects)}
- 训练集: {len(train_subjects)} subjects, {train_mask.sum()} windows
- 验证集: {len(val_subjects)} subjects, {val_mask.sum()} windows
- 测试集: {len(test_subjects)} subjects, {test_mask.sum()} windows

## 窗口参数
- 输入窗口: {LOOKBACK}步 (60分钟)
- 输出窗口: {HORIZON}步 (30分钟)
- 每受试者最大窗口: {MAX_WINDOWS_PER_SUBJECT}

## 数据统计
- CGM均值: {X.mean():.1f} mg/dL
- CGM标准差: {X.std():.1f} mg/dL
- CGM范围: [{X.min():.0f}, {X.max():.0f}] mg/dL

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(os.path.join(OUTPUT_DIR, 'data_report.md'), 'w') as f:
    f.write(report)

print(f"\n数据已保存到: {OUTPUT_DIR}")
print(report)

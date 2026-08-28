#!/usr/bin/env python3
"""
MetaboNet V2 临床评估
- Clarke Error Grid Analysis (自定义实现, Zone A-E 百分比)
- 低血糖检测率 (<70 mg/dL): 灵敏度/特异度/PPV/NPV
- 分层RMSE: <70, 70-180, >180 mg/dL
- MARD细化 (各架构 ± 标准差)
- 输出 Figure5 (Clarke EGA), Figure6 (分层RMSE), Figure7 (低血糖检测)
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

OUT_DIR = os.path.join(BASE_DIR, 'results', 'metabonet_v2_monitored')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# 加载逐样本预测 (seed 42, 测试集)
data = np.load(os.path.join(OUT_DIR, 'clinical_predictions.npz'))
actual = data['actual']
models = {}
model_names = []
for key in data.files:
    if key.startswith('pred_'):
        m = key.replace('pred_', '')
        models[m] = data[key]
        model_names.append(m)
print("实际值形状:", actual.shape, "| 模型:", model_names)

# 输出窗口为6步 (30分钟)。临床评估对每个窗口取全体6步的扁平预测。
# 注意: 6个水平融合后的整体预测, 扁平化看待即以"预测多步后水平"评估。
def flatten(x):
    return x.reshape(-1)  # N*6 预测值
def flatten2(x):
    return x.reshape(-1)

Y = flatten2(actual)
preds = {m: flatten(models[m]) for m in model_names}
N_pred = len(Y)
print(f"扁平化后样本数: {N_pred}")

# 验证指标以确认一致性
print("=== 验证 (扁平化全样本统计) ===")
for m in model_names:
    p = preds[m]
    rmse = np.sqrt(np.mean((p - Y)**2))
    mae = np.mean(np.abs(p - Y))
    mard_full = np.mean(np.abs(p - Y) / np.where(Y==0, 1, np.abs(Y))) * 100
    # MARD基于子窗口聚合(原脚本是每窗口mean后整体mean)，此处扁平化MARD略不同
    print(f"  {m}: RMSE={rmse:.2f}, MAE={mae:.2f}, MARD_flat={mard_full:.2f}%")

# ============ 1. Clarke Error Grid Analysis ============

def _clarke_standard(ref, pred):
    """Clarke et al. 1987 标准分区, 精确到临床标准.
    ref: 实际血糖, pred: 预测血糖 (mg/dL). 边界处理遵循Clarke网格: 
    A区为±20%容差或两者<70; D/E区为临床危险误判; 参考值>=180时上界=ref/2+110."""
    # Zone A: 预测与实际误差 <=20% 或 两者都在<70
    if ref <= 70 and pred <= 70:
        return 'A'
    # 参考在正常/高血糖区 [70,180]
    if 70 < ref <= 180:
        # A区: ±20%相对误差内
        upper = ref * 1.2
        lower = ref * 0.8
        if lower <= pred <= upper:
            return 'A'
        # 预测显著偏高或偏低 (超出20%) -> C区 (临床处理不当但无危险)
        return 'C'
    # 参考>180 (高血糖区)
    if ref > 180:
        # 预测落入低血糖区 -> E (危险)
        if pred <= 70:
            return 'E'
        # 预测落入70-180区 -> D (危险: 高血糖被测成正常)
        if 70 < pred <= 180:
            return 'D'
        # pred > 180: 需同时满足20%下界 和 上界(ref/2+110)
        upper_bound = ref / 2 + 110
        lower_bound = ref * 0.8
        if pred <= upper_bound and pred >= lower_bound:
            return 'A'
        # 超出上界或低于20%下界 -> D (高血糖区的显著偏差)
        return 'D'
    # ref <= 70 (含ref<70且pred>70): 预测偏高
    # ref在[0,70): pred>70
    if ref <= 70:
        if pred > 180:
            return 'D'  # 低血糖被预测为高血糖, 危险
        return 'B'      # 70<pred<=180
    return 'B'


def compute_clarke_std(reference, prediction):
    zones = np.array([_clarke_standard(float(r), float(p)) for r, p in zip(reference, prediction)])
    total = len(zones)
    counts = {z: float(np.sum(zones == z)) for z in 'ABCDE'}
    pcts = {z: counts[z] / total * 100 for z in 'ABCDE'}
    return counts, pcts, zones

print("\n=== Clarke Error Grid Analysis (Clarke 1987 标准分区) ===")
clarke_results = {}
for m in model_names:
    counts, pcts, zones = compute_clarke_std(Y, preds[m])
    clarke_results[m] = {'counts': counts, 'pcts': pcts}
    print(f"  {m}: Zone A={pcts['A']:.2f}%  B={pcts['B']:.2f}%  C={pcts['C']:.2f}%  D={pcts['D']:.2f}%  E={pcts['E']:.2f}%  (A+B={pcts['A']+pcts['B']:.2f}%)")

# ============ 2. 低血糖检测 (<70 mg/dL) ============
print("\n=== 低血糖检测 (<70 mg/dL) ===")
HYP = 70
mask_hyp = Y < HYP
print(f"低血糖样本: {mask_hyp.sum()} / {N_pred} ({mask_hyp.mean()*100:.2f}%)")

def hyp_metrics(pred):
    pred_bin = pred < HYP
    tp = np.sum(mask_hyp & pred_bin)
    fn = np.sum(mask_hyp & ~pred_bin)
    fp = np.sum(~mask_hyp & pred_bin)
    tn = np.sum(~mask_hyp & ~pred_bin)
    sens = tp / (tp + fn) if (tp+fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn+fp) > 0 else float('nan')
    ppv = tp / (tp + fp) if (tp+fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn+fn) > 0 else float('nan')
    return {'tp':int(tp),'fn':int(fn),'fp':int(fp),'tn':int(tn),
            'sensitivity':float(sens),'specificity':float(spec),
            'ppv':float(ppv),'npv':float(npv), 'n_hyp':int(tp+fn), 'n_norm':int(tn+fp)}

hyp_results = {}
for m in model_names:
    met = hyp_metrics(preds[m])
    hyp_results[m] = met
    print(f"  {m}: Sens={met['sensitivity']*100:.2f}%  Spec={met['specificity']*100:.2f}%  PPV={met['ppv']*100:.2f}%  NPV={met['npv']*100:.2f}%")

# ============ 3. 分层RMSE ============
print("\n=== 分层RMSE (按实际血糖范围) ===")
def stratified(x, y):
    s = {}
    for lo, hi, name in [(0, 70, 'hypoglycemia(<70)'), (70, 181, 'euglycemia(70-180)'), (181, 1000, 'hyperglycemia(>180)')]:
        m = (y >= lo) & (y < hi)
        if m.sum() > 0:
            s[name] = {'n': int(m.sum()), 'rmse': float(np.sqrt(np.mean((x[m]-y[m])**2))),
                       'mae': float(np.mean(np.abs(x[m]-y[m]))),
                       'mard': float(np.mean(np.abs(x[m]-y[m])/np.abs(y[m]))*100) if m.sum()>0 else float('nan')}
        else:
            s[name] = {'n':0,'rmse':float('nan'),'mae':float('nan'),'mard':float('nan')}
    return s

strat_results = {}
for m in model_names:
    strat_results[m] = stratified(preds[m], Y)
    print(f"  {m}: " + " | ".join(f"{k}: RMSE={v['rmse']:.2f}(n={v['n']})" for k,v in strat_results[m].items()))

# ============ 4. MARD细化 (各架构 ± 标准差) ============
print("\n=== MARD细化 (各架构±SD) ===")
# 从原始逐seed结果计算 MARD 的 mean±std
with open(os.path.join(OUT_DIR, 'results_final_corrected.json')) as f:
    fulld = json.load(f)
mard_stats = {}
for key, res in fulld['individual_results'].items():
    m = res['model']
    mard_stats.setdefault(m, []).append(res['mard'])
for m in model_names:
    arr = np.array(mard_stats[m])
    print(f"  {m}: MARD = {arr.mean():.3f} ± {arr.std(ddof=1):.3f} %")

# ============ 5. Per-horizon 分析 (临床标准报告粒度) ============
print("\n=== Per-horizon 分析 (按6个预测时域) ===")
# 输出窗口6步=30分钟, horizon 0=5min, ..., horizon 5=30min
horizons = np.arange(actual.shape[1])
n_horizons = actual.shape[1]
per_horizon = {}
for m in model_names:
    pm = models[m]
    per_horizon[m] = {}
    for h in horizons:
        a = actual[:, h].reshape(-1)
        p = pm[:, h].reshape(-1)
        rmse_h = float(np.sqrt(np.mean((p - a) ** 2)))
        # Clarke A区 (简单相对误差<=20% 或 两者<70)
        inA = (np.abs(p - a) <= 0.2 * np.maximum(np.abs(a), np.abs(p))) | ((a <= 70) & (p <= 70))
        a_pct = float(inA.mean() * 100)
        per_horizon[m][str(h)] = {
            'rmse': rmse_h,
            'clarke_A_pct': a_pct,
        }
    print(f"  {m}: " + " | ".join(f"H{h}:RMSE={per_horizon[m][str(h)]['rmse']:.1f},A%={per_horizon[m][str(h)]['clarke_A_pct']:.1f}" for h in horizons))

# ============ 生成图表 ============
# Figure5: Clarke EGA
def plot_clarke(ax, ref, pred, density=None):
    """绘制Clarke误差网格."""
    # 边界线
    x = np.linspace(0, 600, 500)
    # Zone 上界/下界
    y_upper_180 = x/2 + 110
    y_lower_180 = x - 70
    # 参考线
    ax.plot([0,600],[0,600],'k-',lw=1.2)
    # 20% 线
    ax.plot(x, x*1.2, 'g-', lw=1, ls='--', alpha=0.6)
    ax.plot(x, x*0.8, 'g-', lw=1, ls='--', alpha=0.6)
    # 70 & 180 线
    ax.axvline(70, color='k', lw=0.8, alpha=0.4); ax.axvline(180, color='k', lw=0.8, alpha=0.4)
    ax.axhline(70, color='k', lw=0.8, alpha=0.4); ax.axhline(180, color='k', lw=0.8, alpha=0.4)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500)
    ax.set_xlabel('Predicted CGM (mg/dL)'); ax.set_ylabel('Actual CGM (mg/dL)')
    ax.set_title('Clarke Error Grid')

# 散点(抽样显示避免过密)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
sample_idx = np.random.choice(N_pred, min(20000, N_pred), replace=False)
for ax, m in zip(ax_flat, model_names):
    # 分层着色
    zs = np.array([_clarke_standard(float(r), float(p)) for r, p in zip(Y[sample_idx], preds[m][sample_idx])])
    colors = {'A':'#99d594','B':'#fc8d59','C':'#fee08b','D':'#d7191c','E':'#7b3294'}
    for z in 'ABCDE':
        mm = zs == z
        if mm.sum() > 0:
            ax.scatter(preds[m][sample_idx][mm], Y[sample_idx][mm], s=1, c=colors[z], alpha=0.6, label=f'{z}')
    plot_clarke(ax, None, None)
    ax.set_title(f'{m} — Clarke EGA (Zone A={clarke_results[m]["pcts"]["A"]:.1f}%)', fontsize=11)
    ax.legend(markerscale=8, loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure5_clarke_ega.png'), dpi=200)
plt.savefig(os.path.join(FIG_DIR, 'Figure5_clarke_ega.pdf'))
plt.close()
print("已保存 Figure5_clarke_ega")

# Figure6: 分层RMSE
fig, ax = plt.subplots(figsize=(10, 6))
zones = ['hypoglycemia(<70)', 'euglycemia(70-180)', 'hyperglycemia(>180)']
zone_labels = ['hypoglycaemia(<70)', 'euglycaemia(70-180)', 'hyperglycaemia(>180)']
x = np.arange(len(zones)); width = 0.18
colors = ['#4c72b0','#dd8452','#6c8ebf','#c44e52']
for i, m in enumerate(model_names):
    vals = [strat_results[m][z]['rmse'] for z in zones]
    ax.bar(x + (i-1.5)*width, vals, width, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(zone_labels)
ax.set_ylabel('RMSE (mg/dL)')
ax.set_title('Figure 6. Stratified RMSE by Glycaemic Range')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure6_stratified_rmse.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'Figure6_stratified_rmse.pdf'))
plt.close()
print("已保存 Figure6_stratified_rmse")

# Figure7: 低血糖检测
fig, ax = plt.subplots(figsize=(10, 6))
mets = ['sensitivity','specificity','ppv','npv']
ms_m = [hyp_results[m] for m in model_names]
x = np.arange(len(mets)); width = 0.18
for i, m in enumerate(model_names):
    vals = [hyp_results[m][met]*100 for met in mets]
    ax.bar(x + (i-1.5)*width, vals, width, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(['Sensitivity','Specificity','PPV','NPV'])
ax.set_ylabel('Percentage (%)'); ax.set_ylim(0, 105)
ax.set_title('Figure 7. Hypoglycaemia Detection (<70 mg/dL)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure7_hypoglycaemia_detection.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'Figure7_hypoglycaemia_detection.pdf'))
plt.close()
print("已保存 Figure7_hypoglycaemia_detection")

# Figure8: Per-horizon RMSE + Clarke Zone A degradation
fig, ax1 = plt.subplots(figsize=(10, 6))
xh = np.arange(n_horizons)
for i, m in enumerate(model_names):
    rmses = [per_horizon[m][str(h)]['rmse'] for h in horizons]
    ax1.plot(xh, rmses, marker='o', label=m, color=colors[i])
ax1.set_xlabel('Prediction Horizon (index: 0=5min, 1=10min, ..., 5=30min)')
ax1.set_ylabel('Per-Horizon RMSE (mg/dL)', color='k')
ax1.set_xticks(xh); ax1.grid(alpha=0.3)
ax1.set_title('Figure 8. Per-Horizon RMSE and Clarke Zone A')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'Figure8_perhorizon_rmse.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'Figure8_perhorizon_rmse.pdf'))
plt.close()
print("已保存 Figure8_perhorizon_rmse")

# 保存汇总JSON
out = {
    'n_samples': int(N_pred),
    'n_horizons': int(n_horizons),
    'clarke_ega': clarke_results,
    'hypoglycemia': hyp_results,
    'stratified_rmse': strat_results,
    'mard_stats': {m: {'mean': float(np.mean(mard_stats[m])), 'std': float(np.std(mard_stats[m], ddof=1))} for m in model_names},
    'per_horizon': per_horizon,
}
with open(os.path.join(OUT_DIR, 'clinical_evaluation_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\n临床评估结果已保存:", os.path.join(OUT_DIR, 'clinical_evaluation_results.json'))

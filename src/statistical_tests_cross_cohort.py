#!/usr/bin/env python3
"""
MetaboNet T1D vs T2D 跨人群泛化性统计检验 (CMPB核心卖点)
比较同模型在 T1D/MetaboNet 与 T2D/ShanghaiT2DM 的20-seed表现:
- 同模型跨人群 RMSE 差异 + 配对t/Wilcoxon + Cohen's dz
- 泛化性指标: R² 跨人群差异(谁能解释更多方差)
- 稳健性: 逐seed R² 离散度(CV)跨人群比较
输入: 两组 20-seed 结果
输出: stat_results_cross_cohort.json
"""
import numpy as np, json, os, glob
from scipy import stats
from itertools import combinations
import os

T1D = os.path.join(BASE_DIR, 'results', 'metabonet_v2_20seed')
T2D = os.path.join(BASE_DIR, 'results', 'metabonet_v2_20seed_t2d')
ALL_SEEDS = list(range(42,62))
MODELS = ['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']

def load(cohort_dir, model, seed):
    fp = os.path.join(cohort_dir, f'result_{model}_seed{seed}.json')
    if os.path.exists(fp):
        return json.load(open(fp))
    return None

# 收集两人群逐seed指标
data = {'T1D': {}, 'T2D': {}}
for cname, cdir in [('T1D',T1D),('T2D',T2D)]:
    for m in MODELS:
        r = {}
        for s in ALL_SEEDS:
            res = load(cdir, m, s)
            if res: r[s]=res
        data[cname][m] = r

# 共同seed
common = set(ALL_SEEDS)
for c in ['T1D','T2D']:
    for m in MODELS:
        common &= set(data[c][m].keys())
SEEDS = sorted(common)

np.random.seed(42)
results = {
    'target_journal':'CMPB','analysis':'cross-cohort generalization (T1D vs T2D)',
    'n_seeds':len(SEEDS),'seed_values':SEEDS,
    'cross_cohort':{},'generalization_summary':{}
}

def cohens_d(x,y):
    d=np.array(x)-np.array(y); sd=np.std(d,ddof=1)
    return float(d.mean()/sd) if sd>0 else float('inf')

# 1. 同模型 跨人群 比较 (T1D RMSE vs T2D RMSE)
print("="*70)
print(f"跨人群泛化性 (T1D/MetaboNet vs T2D/ShanghaiT2DM), {len(SEEDS)} seeds")
print("="*70)
for m in MODELS:
    r1 = np.array([data['T1D'][m][s]['rmse'] for s in SEEDS])
    r2 = np.array([data['T2D'][m][s]['rmse'] for s in SEEDS])
    # R²
    rr1 = np.array([data['T1D'][m][s]['r2'] for s in SEEDS])
    rr2 = np.array([data['T2D'][m][s]['r2'] for s in SEEDS])
    t_stat,t_p = stats.ttest_rel(r1,r2)
    w_stat,w_p = stats.wilcoxon(r1,r2)
    dz = cohens_d(r1,r2)
    # 稳健性(CV)
    cv1 = np.std(rr1,ddof=1)/np.mean(rr1)*100
    cv2 = np.std(rr2,ddof=1)/np.mean(rr2)*100
    results['cross_cohort'][m] = {
        'T1D_rmse_mean':float(r1.mean()), 'T1D_rmse_sd':float(r1.std(ddof=1)),
        'T2D_rmse_mean':float(r2.mean()), 'T2D_rmse_sd':float(r2.std(ddof=1)),
        'rmse_diff_T2D_minus_T1D':float(r2.mean()-r1.mean()),
        'T1D_r2_mean':float(rr1.mean()), 'T2D_r2_mean':float(rr2.mean()),
        'T1D_r2_cv_pct':float(cv1), 'T2D_r2_cv_pct':float(cv2),
        'p_ttest':float(t_p), 'p_wilcoxon':float(w_p), 'cohens_dz':dz,
        'r2_drop_T1D_to_T2D':float(rr1.mean()-rr2.mean()),
    }
    print(f"{m:<18}: T1D_RMSE={r1.mean():.3f} R²={rr1.mean():.4f}(CV{cv1:.2f}%) | "
          f"T2D_RMSE={r2.mean():.3f} R²={rr2.mean():.4f}(CV{cv2:.2f}%) | "
          f"ΔRMSE={r2.mean()-r1.mean():+.2f} t-p={t_p:.4f} dz={dz:.2f}")

# 2. 泛化性摘要: 各模型跨人群保持的排名一致性(RMSE排序)
print("\n跨人群模型排名一致性:")
t1d_rank = sorted(MODELS, key=lambda m: np.mean([data['T1D'][m][s]['rmse'] for s in SEEDS]))
t2d_rank = sorted(MODELS, key=lambda m: np.mean([data['T2D'][m][s]['rmse'] for s in SEEDS]))
print(f"  T1D RMSE排序: {t1d_rank}")
print(f"  T2D RMSE排序: {t2d_rank}")
results['generalization_summary'] = {
    'T1D_best_model':t1d_rank[0], 'T2D_best_model':t2d_rank[0],
    'T1D_rank':t1d_rank, 'T2D_rank':t2d_rank,
    'rank_consistent': t1d_rank==t2d_rank,
}

json.dump(results, open(os.path.join(BASE_DIR, 'results', 'stat_results_cross_cohort.json'),'w'), indent=2)
print("\n✅ 跨人群比较已保存: results/stat_results_cross_cohort.json")

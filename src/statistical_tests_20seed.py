#!/usr/bin/env python3
"""
MetaboNet 20-seed 统计检验 — CMPB(Computer Methods and Programs in Biomedicine)版
====================================================================================
围绕 CMPB 期刊要求的多种子稳健性推断统计：
- Diebold-Mariano (DM) 检验: 基于逐seed的RMSE差作为损失差异
- 配对 t 检验 / Wilcoxon 符号秩检验: 逐seed RMSE
- 效应量 Cohen's dz
- Holm 多重比较校正 (BM/DM、t、Wilcoxon 各家族独立校正)
- Bootstrap CI (1000次) 对 RMSE 差异

输入: 20-seed训练产物 results/metabonet_v2_20seed/result_{model}_seed{seed}.json
       (train_20seed_serial.py 每次完成一个seed增量写入)
输出: results/metabonet_v2_20seed/stat_results_20seed.json
说明: 自动发现已完成的 seed 文件(断点友好), 支持 20 seeds (ALL_SEEDS=42..61)。
      修复: 原 statistical_tests.py 硬编码 SEEDS=[42..46] 且读旧路径 monitored,
            本脚本改读 metabonet_v2_20seed 目录 + 全部已完成 seed。
"""
import numpy as np
import json
import os
import glob
from scipy import stats
from itertools import combinations

# ---------- 配置 ----------
import sys
COHORT = sys.argv[1] if len(sys.argv) > 1 else 'T1D'  # T1D 或 T2D
DIR_MAP = {
    'T1D': os.path.join(BASE_DIR, 'results', 'metabonet_v2_20seed'),
    'T2D': os.path.join(BASE_DIR, 'results', 'metabonet_v2_20seed_t2d'),
}
RESULTS_DIR = DIR_MAP[COHORT]
OUT_FILE = os.path.join(RESULTS_DIR, f'stat_results_20seed_{COHORT}.json')
ALL_SEEDS = list(range(42, 62))  # 20个种子 42..61
MODEL_NAMES = ['LSTM', 'BiLSTM', 'LSTM-Attention', 'BiLSTM-Attention']
CLINICAL_RMSE_THRESHOLD = 5.0  # mg/dL, CMPB临床意义阈值(论文: 5-10 mg/dL)

# ---------- 自动发现已完成 seed 结果(断点友好) ----------
def load_done(pred_m, seed):
    """读取单个 seed 结果文件, 不存在返回 None."""
    fp = os.path.join(RESULTS_DIR, f'result_{pred_m}_seed{seed}.json')
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)

# 收集每个模型已完成的 (seed -> res)
models = {m: {} for m in MODEL_NAMES}
n_done = 0
for m in MODEL_NAMES:
    for s in ALL_SEEDS:
        res = load_done(m, s)
        if res is not None:
            models[m][s] = res
            n_done += 1

# 取所有模型都完成的共同 seed(保证配对检验公平)
common = set(ALL_SEEDS)
for m in MODEL_NAMES:
    common &= set(models[m].keys())
SEEDS = sorted(common)
n_seeds = len(SEEDS)

print(f"已完成结果: {n_done} / {len(MODEL_NAMES)*len(ALL_SEEDS)}")
print(f"所有模型都完成的共同 seed ({n_seeds}个): {SEEDS}")

if n_seeds < 2:
    raise SystemExit("共同 seed < 2, 无法做配对检验。请等待更多 seed 训练完成。")

# ---------- 组装逐seed指标矩阵 ----------
def arr(metric):
    return {m: [models[m][s][metric] for s in SEEDS] for m in MODEL_NAMES}

rmse = arr('rmse')
mae = arr('mae')
mard = arr('mard')
r2 = arr('r2')

np.random.seed(42)

def cohens_d(x, y):
    """配对 Cohen's dz."""
    d = np.mean(x) - np.mean(y)
    sd_diff = np.std(np.array(x) - np.array(y), ddof=1)
    return float(d / sd_diff) if sd_diff > 0 else float('inf')

def dm_test(e1, e2, h=1):
    """Diebold-Mariano 检验 (逐seed RMSE差)."""
    d = np.array(e1) - np.array(e2)
    n = len(d)
    dbar = np.mean(d)
    var = np.var(d, ddof=1)
    if var <= 0:
        return (0.0, 1.0)
    if h > 1 and n > h:
        cov = np.cov(d[:-1], d[1:], ddof=1)[0, 1] if n > 2 else 0
        var += 2 * (h - 1) / h * cov
    se = np.sqrt(var / n)
    dm = dbar / se if se > 0 else 0.0
    return (float(dm), float(2 * (1 - stats.norm.cdf(abs(dm)))))

# ---------- 两两比较 ----------
pairs = list(combinations(MODEL_NAMES, 2))
results = {
    'target_journal': 'CMPB',
    'cohort': COHORT,
    'n_seeds': n_seeds,
    'seed_values': SEEDS,
    'seed_values_rmse': {m: rmse[m] for m in MODEL_NAMES},
    'clinical_rmse_threshold_mgdl': CLINICAL_RMSE_THRESHOLD,
    'comparisons': {},
}

for a, b in pairs:
    x, y = rmse[a], rmse[b]
    dm_stat, dm_p = dm_test(x, y)
    t_stat, t_p = stats.ttest_rel(x, y)
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    dz = cohens_d(x, y)
    diff = np.array(x) - np.array(y)
    se_d = np.std(diff, ddof=1) / np.sqrt(n_seeds)
    ci_t = stats.t.interval(0.95, df=n_seeds - 1, loc=np.mean(diff), scale=se_d)
    boot_rng = np.random.RandomState(20260820)
    boot_diffs = []
    xa, ya = np.array(x), np.array(y)
    for _ in range(1000):
        idx = boot_rng.choice(n_seeds, n_seeds, replace=True)
        boot_diffs.append(np.mean(xa[idx] - ya[idx]))
    boot_ci = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))
    better = a if np.mean(diff) < 0 else b
    # CMPB: 是否达到临床意义阈值(论文5-10 mg/dL, 此处用保守值5)
    clinically_meaningful = abs(np.mean(diff)) >= CLINICAL_RMSE_THRESHOLD
    results['comparisons'][f'{a} vs {b}'] = {
        'rmse_mean_a': float(np.mean(x)), 'rmse_mean_b': float(np.mean(y)),
        'rmse_diff_mean_mgdl': float(np.mean(diff)),
        'dm_statistic': dm_stat, 'dm_pvalue_raw': dm_p,
        't_statistic': float(t_stat), 'pvalue_ttest_raw': float(t_p),
        'wilcoxon_statistic': float(w_stat), 'pvalue_wilcoxon_raw': float(w_p),
        'cohens_dz': dz,
        'rmse_diff_ci95_t': [float(ci_t[0]), float(ci_t[1])],
        'rmse_diff_ci95_bootstrap': [float(boot_ci[0]), float(boot_ci[1])],
        'direction_better': better,
        'clinically_meaningful_ge5mgdl': bool(clinically_meaningful),
    }

# ---------- Holm 校正 ----------
def holm(ps):
    n = len(ps)
    sorted_p = np.sort(np.array(ps, dtype=float))
    adj_p = np.zeros(n)
    prev = 0.0
    for rank in range(n):
        factor = n - rank
        adj_p[rank] = max(prev, factor * sorted_p[rank])
        prev = adj_p[rank]
    adj_p = np.minimum(1, adj_p)
    out = np.zeros(n)
    for j, orig_idx in enumerate(np.argsort(ps)):
        out[orig_idx] = adj_p[j]
    return out

comp_keys = list(results['comparisons'].keys())
ps_t = [results['comparisons'][c]['pvalue_ttest_raw'] for c in comp_keys]
ps_w = [results['comparisons'][c]['pvalue_wilcoxon_raw'] for c in comp_keys]
ps_dm = [results['comparisons'][c]['dm_pvalue_raw'] for c in comp_keys]
h_t = holm(ps_t); h_w = holm(ps_w); h_dm = holm(ps_dm)
for i, c in enumerate(comp_keys):
    r = results['comparisons'][c]
    r['pvalue_ttest_holm'] = float(h_t[i])
    r['pvalue_wilcoxon_holm'] = float(h_w[i])
    r['dm_pvalue_holm'] = float(h_dm[i])
    r['pvalue_ttest_bonferroni'] = float(min(1, r['pvalue_ttest_raw'] * len(comp_keys)))
    r['pvalue_wilcoxon_bonferroni'] = float(min(1, r['pvalue_wilcoxon_raw'] * len(comp_keys)))
    r['dm_pvalue_bonferroni'] = float(min(1, r['dm_pvalue_raw'] * len(comp_keys)))

# ---------- 总体概要 ----------
results['summary_metrics'] = {}
for m in MODEL_NAMES:
    arr_r = np.array(rmse[m])
    results['summary_metrics'][m] = {
        'rmse_mean_mgdl': float(arr_r.mean()),
        'rmse_std': float(arr_r.std(ddof=1)),
        'rmse_min': float(arr_r.min()),
        'rmse_max': float(arr_r.max()),
        'mae_mean': float(np.mean(mae[m])),
        'mard_mean': float(np.mean(mard[m])),
        'r2_mean': float(np.mean(r2[m])),
        'r2_std': float(np.std(r2[m], ddof=1)),
    }

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

# ---------- 打印摘要 ----------
print("=" * 72)
print(f"{COHORT} 20-seed 统计检验 (CMPB版) — {n_seeds} 个共同seed")
print("=" * 72)
for m in MODEL_NAMES:
    s = results['summary_metrics'][m]
    print(f"{m:<18}: RMSE={s['rmse_mean_mgdl']:.3f}±{s['rmse_std']:.3f}  "
          f"MAE={s['mae_mean']:.3f}  MARD={s['mard_mean']:.2f}%  R²={s['r2_mean']:.4f}±{s['r2_std']:.4f}")
print("-" * 72)
print(f"{'对比':<30}{'ΔRMSE':<9}{'d_z':<7}{'t-p':<9}{'Wil-p':<9}{'DM-p':<9}{'t-Holm':<9}{'临床?'}")
for c in comp_keys:
    r = results['comparisons'][c]
    cl = '✓' if r['clinically_meaningful_ge5mgdl'] else '✗'
    print(f"{c:<30}{r['rmse_diff_mean_mgdl']:<9.3f}{r['cohens_dz']:<7.2f}"
          f"{r['pvalue_ttest_raw']:<9.4f}{r['pvalue_wilcoxon_raw']:<9.4f}"
          f"{r['dm_pvalue_raw']:<9.4f}{r['pvalue_ttest_holm']:<9.4f}{cl}")
print("\n结果已保存:", OUT_FILE)

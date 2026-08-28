#!/usr/bin/env python3
"""
论文#5 MetaboNet 20-seed 训练结果汇总分析
汇总 T1D(OhioT1DM) 与 T2D(ShanghaiT2DM) 各 80 次训练 = 4模型 × 20seed
输出: 各模型均值±SD (RMSE/MAE/MARD/R²) + seed间稳定性(变异系数CV) + 性能对比
"""
import json, os, statistics, sys

MODELS = ["LSTM", "BiLSTM", "LSTM-Attention", "BiLSTM-Attention"]

def load_results(base_dir):
    """读取目录下所有 result_*.json, 按模型分组种子结果"""
    by_model = {m: [] for m in MODELS}
    files = [f for f in os.listdir(base_dir) if f.startswith("result_") and f.endswith(".json")]
    for f in files:
        try:
            d = json.load(open(os.path.join(base_dir, f)))
        except Exception as e:
            print(f"  ⚠️ 无法解析 {f}: {e}")
            continue
        m = d.get("model")
        if m in by_model:
            by_model[m].append(d)
    return by_model, len(files)

def agg(metrics):
    """metrics: list of float -> (mean, sd, cv_percent)"""
    n = len(metrics)
    mean = statistics.mean(metrics)
    sd = statistics.stdev(metrics) if n > 1 else 0.0
    cv = (sd / abs(mean) * 100) if mean != 0 else float('nan')
    return mean, sd, cv

def analyze(base_dir, label):
    print(f"\n{'='*70}")
    print(f"📊 {label}  (20-seed × 4模型 = 80次训练)")
    print(f"{'='*70}")
    by_model, nfiles = load_results(base_dir)
    print(f"检测到 result_*.json 文件: {nfiles} 份")
    if nfiles != 80:
        print(f"  ⚠️ 数量={nfiles} ≠ 80, 检查完整性!")
    print(f"{'模型':<18} {'N':>3} | {'RMSE':>22} | {'MAE':>22} | {'MARD%':>22} | {'R²':>18}")
    print("-"*70)
    worst = None
    for m in MODELS:
        res = by_model.get(m, [])
        if not res:
            print(f"{m:<18}   0 |  无结果!")
            continue
        rmse = agg([r["rmse"] for r in res])
        mae  = agg([r["mae"] for r in res])
        mard = agg([r["mard"] for r in res])
        r2   = agg([r["r2"] for r in res])
        tmin = statistics.mean([r.get("time_min",0) for r in res])
        print(f"{m:<18} {len(res):>3} | "
              f"{rmse[0]:6.2f}±{rmse[1]:5.2f} (CV{rmse[2]:4.1f}%) | "
              f"{mae[0]:6.2f}±{mae[1]:5.2f} | "
              f"{mard[0]:6.2f}±{mard[1]:5.2f} | "
              f"{r2[0]:5.4f}±{r2[1]:.4f}")
        # 找最优(R²最高且RMSE最低)
        if worst is None or (r2[0] > worst[1][0][0]) or (r2[0] == worst[1][0][0] and rmse[0] < worst[1][0][1]):
            worst = (m, (r2, rmse))
    if worst:
        m, (r2, rmse) = worst
        print("-"*70)
        print(f"🏆 综合最优: {m}  (R²={r2[0]:.4f}, RMSE={rmse[0]:.2f})")
    # 稳定性: 各模型R²的seed间范围
    print(f"\n📈 seed间稳定性(R²极差):")
    for m in MODELS:
        res = by_model.get(m, [])
        if res:
            r2s = sorted(r["r2"] for r in res)
            print(f"  {m:<18}  R²范围 [{r2s[0]:.4f}, {r2s[-1]:.4f}]  极差={r2s[-1]-r2s[0]:.4f}")
    return by_model, nfiles

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    t1d_dir = os.path.join(base, "results", "metabonet_v2_20seed")
    t2d_dir = os.path.join(base, "results", "metabonet_v2_20seed_t2d")
    if not os.path.isdir(t1d_dir) or not os.path.isdir(t2d_dir):
        print("目录不存在:", t1d_dir, t2d_dir); sys.exit(1)
    analyze(t1d_dir, "T1D · OhioT1DM")
    analyze(t2d_dir, "T2D · ShanghaiT2DM")

    # 保存汇总JSON供论文引用
    out = {}
    for label, d in [("T1D", t1d_dir), ("T2D", t2d_dir)]:
        by_model, nfiles = load_results(d)
        out[label] = {"nfiles": nfiles, "models": {}}
        for m in MODELS:
            res = by_model.get(m, [])
            if res:
                rmse = agg([r["rmse"] for r in res])
                mae  = agg([r["mae"] for r in res])
                mard = agg([r["mard"] for r in res])
                r2   = agg([r["r2"] for r in res])
                out[label]["models"][m] = {
                    "n": len(res),
                    "rmse_mean": round(rmse[0],4), "rmse_sd": round(rmse[1],4), "rmse_cv": round(rmse[2],2),
                    "mae_mean": round(mae[0],4), "mae_sd": round(mae[1],4),
                    "mard_mean": round(mard[0],4), "mard_sd": round(mard[1],4),
                    "r2_mean": round(r2[0],4), "r2_sd": round(r2[1],4), "r2_min": round(min(r["r2"] for r in res),4), "r2_max": round(max(r["r2"] for r in res),4),
                }
    outpath = os.path.join(base, "results", "20seed_summary_authoritative.json")
    json.dump(out, open(outpath,"w"), indent=2, ensure_ascii=False)
    print(f"\n✅ 汇总JSON已保存: {outpath}")

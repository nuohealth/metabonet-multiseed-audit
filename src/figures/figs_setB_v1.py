#!/usr/bin/env python3
"""CMPB 论文#5 图 SET-B (版本二) — 完全不同的呈现思路, 与SET-A(v8)供神农对比
SET-B 各图叙事角度:
B-Fig1: 跨人群泛化代价 — 每模型 T1D vs T2D 分组, 箭头标注 +3.5mg/dL, 展示每个模型迁移到T2D都退化(泛化故事)
B-Fig2: 统计差异矩阵 — Holm校正DM p值, 直接回答"这4个模型真的不同吗?"(诚实: 部分显著但Δ<5临床可忽略)
B-Fig3: 归一化收益递减 — 相对LSTM基线的%改进 + 全种子范围, 量化"加复杂度几乎不买账"
B-Fig4: 效率前沿 — 全部160种子, RMSE vs 训练时间, 加"每单位时间精度"斜对角线参考, T2D处更高效率前沿

风格: 与SET-A一致(seaborn白底/despine/Liberation Sans/专业色), 但布局叙事全新
双栏174mm/300dpi/PNG+PDF+TIF
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import FancyArrowPatch

import os
BASE=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
MODELS=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
MODELS_SHORT=['LSTM','BiLSTM','LSTM-Atten','BiLSTM-Atten']
SEEDS=list(range(42,62)); MM=25.4; DCW=174/MM; DPI=300
C_T1D='#1B4E79'; C_T2D='#D55E00'; C_T1D_L='#A8C0DA'; C_T2D_L='#F5CFA3'
C_TEXT='#1A1A1A'; C_GRID='#E4E4E4'; CLIN=5.0
sns.set_theme(style='white',context='paper')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'legend.fontsize':7,'figure.facecolor':'white','axes.facecolor':'white','axes.linewidth':0.5,
 'grid.color':C_GRID,'grid.linewidth':0.4,'savefig.dpi':DPI})

def load_cohort(cohort):
    return {m:[json.load(open(f'results/{cohort}/result_{m}_seed{s}.json'))['rmse']
               for s in SEEDS if os.path.exists(f'results/{cohort}/result_{m}_seed{s}.json')] for m in MODELS}
T1=load_cohort('metabonet_v2_20seed'); T2=load_cohort('metabonet_v2_20seed_t2d')
TM={m:np.mean(T2[m]) for m in MODELS}; T1M={m:np.mean(T1[m]) for m in MODELS}
T1SD={m:np.std(T1[m],ddof=1) for m in MODELS}; T2SD={m:np.std(T2[m],ddof=1) for m in MODELS}

def dm_p(x,y):
    d=np.array(x)-np.array(y); n=len(d); sd=d.std(ddof=1)
    z=d.mean()/(sd/np.sqrt(n)) if sd>0 else 0
    return 2*(1-norm.cdf(abs(z)))

# ============ B-Fig1 跨人群泛化(分组+箭头) ============
fig,ax=plt.subplots(figsize=(DCW,2.4))
x=np.arange(len(MODELS)); wd=0.34
for i,m in enumerate(MODELS):
    # T1D条+种子点
    ax.scatter([i-wd/2]*20,T1[m],s=13,color=C_T1D,alpha=0.55,edgecolor='white',linewidth=0.2,zorder=2)
    ax.errorbar(i-wd/2,T1M[m],yerr=T1SD[m],fmt='o',color=C_T1D,ms=6,capsize=3,capthick=0.8,
                elinewidth=0.9,zorder=3,markeredgecolor='white',markeredgewidth=0.4)
    # T2D条+种子点
    ax.scatter([i+wd/2]*20,T2[m],s=13,color=C_T2D,alpha=0.55,edgecolor='white',linewidth=0.2,zorder=2)
    ax.errorbar(i+wd/2,TM[m],yerr=T2SD[m],fmt='s',color=C_T2D,ms=6,capsize=3,capthick=0.8,
                elinewidth=0.9,zorder=3,markeredgecolor='white',markeredgewidth=0.4)
    # 跨人群差箭头
    gap=(TM[m]-T1M[m])/2
    midy=(T1M[m]+TM[m])/2
    ax.annotate('',xy=(i+wd/2+0.0,TM[m]+1.1),xytext=(i+wd/2+0.0,T1M[m]+1.1),
                arrowprops=dict(arrowstyle='-|>',color='#777777',lw=1.0))
    ax.text(i+wd/2,T1M[m]+0.02,f'+{TM[m]-T1M[m]:.1f}',fontsize=6.5,color='#555555',
            rotation=90,va='bottom',ha='right')
ax.set_xticks(x); ax.set_xticklabels(MODELS_SHORT,fontsize=7)
ax.set_ylabel('RMSE (mg/dL, 30-min)'); ax.set_ylim(15,22.5)
ax.set_title('Cross-cohort generalization: every model degrades on T2D (+3.4–3.6 mg/dL)',
             loc='left',fontsize=8.5,fontweight='bold',color=C_TEXT,pad=6)
# 图例
import matplotlib.lines as mlines
l1=mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor=C_T1D,ms=6,label='T1D (n=1,092)')
l2=mlines.Line2D([0],[0],marker='s',color='w',markerfacecolor=C_T2D,ms=6,label='T2D (n=100)')
ax.legend(handles=[l1,l2],loc='upper left',frameon=True,framealpha=0.9,edgecolor='#DDDDDD')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
fig.tight_layout(pad=0.5)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/B_Fig1_cross_cohort_gap.{e}',facecolor='white')
plt.close(fig); print('B-Fig1 done')

# ============ B-Fig2 统计差异矩阵(Holm DM p) ============
# 每人群内两两 下三角p值
from itertools import combinations as comb
pairs=list(comb(MODELS,2))
mat_dm={'T1D':{},'T2D':{}}
for pop,data in [('T1D',T1),('T2D',T2)]:
    ps=[dm_p(data[a],data[b]) for a,b in pairs]
    hs=multipletests(ps,method='holm')[1]
    mat_dm[pop]=dict(zip(pairs,[float(h) for h in hs]))
# 画两行(每行一人群) × 列=每对模型
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(DCW,2.5),sharex=True)
pair_lbl=['LSTM–BiLSTM','LSTM–LSTM-A','LSTM–BiLSTM-A','BiLSTM–LSTM-A','BiLSTM–BiLSTM-A','LSTM-A–BiLSTM-A']
for axi,pop,cm in [(ax1,'T1D',C_T1D),(ax2,'T2D',C_T2D)]:
    vals=[mat_dm[pop][p] for p in pairs]
    cols=['#1B4E79' if v<0.05 else '#BBBBBB' for v in vals]  # 显著=深蓝, 不显著=灰
    bars=axi.bar(np.arange(len(pairs)),[-np.log10(max(v,1e-6)) for v in vals],
                 color=cols,width=0.6,edgecolor='white',linewidth=0.4)
    for i,v in enumerate(vals):
        axi.text(i,-np.log10(max(v,1e-6))+0.05,f'p={v:.3f}',ha='center',fontsize=6,
                 color=C_TEXT,va='bottom')
    # 显著性阈值线(Bonferroni 0.05/6≈0.0083 用虚线; α=0.05横线)
    thres=-np.log10(0.05)
    axi.axhline(thres,color='#999999',ls='--',lw=0.8)
    axi.text(len(pairs)-0.5,thres+0.06,'α=0.05',fontsize=6,color='#777777',ha='right')
    axi.axhline(-np.log10(0.05/6),color='#BBBBBB',ls=':',lw=0.7)
    axi.text(len(pairs)-0.5,-np.log10(0.05/6)-0.15,'Bonferroni',fontsize=6,color='#999999',ha='right')
    axi.set_xticks(np.arange(len(pairs)))
    axi.set_ylim(0,3.2)
    axi.spines[['top','right']].set_visible(False); axi.yaxis.grid(True); axi.set_axisbelow(True)
    axi.set_title(f'({ "a" if pop=="T1D" else "b" })  {pop}: Diebold–Mariano p (Holm-corrected)',loc='left',
                  fontsize=8,fontweight='bold',color=C_TEXT,pad=4)
    from matplotlib.patches import Patch
ax2.set_xticklabels(pair_lbl,fontsize=6.2,rotation=30,ha='right')
ax2.set_xlabel('Model pair (DM test of RMSE difference)')
ax1.set_ylabel('−log₁₀(p)'); ax2.set_ylabel('−log₁₀(p)')
# 顶部临床注释
fig.suptitle('Are the architectures statistically different?  (Some pairs significant, but all Δ<5 mg/dL)',
             fontsize=8,fontweight='bold',color=C_TEXT)
fig.tight_layout(pad=0.6)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/B_Fig2_dm_significance.{e}',facecolor='white')
plt.close(fig); print('B-Fig2 done')

# ============ B-Fig3 归一化收益递减(LSTM基线%) ============
fig,ax=plt.subplots(figsize=(DCW,2.2))
base1=T1M['LSTM']; base2=TM['LSTM']
x=np.arange(len(MODELS))
for pop,base,cm,cm_l,marker,label in [('T1D',base1,C_T1D,C_T1D_L,'o','T1D'),('T2D',base2,C_T2D,C_T2D_L,'s','T2D')]:
    data=T1 if pop=='T1D' else T2
    # 每模型的%改进相对该人群LSTM
    pct=[(T1M['LSTM']-T1M[m])/T1M['LSTM']*100 if pop=='T1D' else (TM['LSTM']-TM[m])/TM['LSTM']*100 for m in MODELS]
    # 全种子范围的%改进
    lo=[];hi=[]
    for m in MODELS:
        arr=np.array(data[m]); base_arr=np.array(T1['LSTM'] if pop=='T1D' else T2['LSTM'])
        pct_all=(base_arr.mean()-arr)/base_arr.mean()*100
        lo.append(pct_all.min()); hi.append(pct_all.max())
    ax.fill_between(x,lo,hi,color=cm,alpha=0.13,lw=0)
    ax.plot(x,pct,color=cm,lw=1.8,marker=marker,ms=6,label=label,markeredgecolor='white',markeredgewidth=0.4,zorder=3)
    for i,m in enumerate(MODELS):
        arr=np.array(data[m]); base_arr=np.array(T1['LSTM'] if pop=='T1D' else T2['LSTM'])
        pct_all=(base_arr.mean()-arr)/base_arr.mean()*100
        ax.scatter([i]*20,pct_all,s=13,color=cm,alpha=0.42,edgecolor='white',linewidth=0.2,zorder=2)
ax.axhline(0,color='#999999',ls='--',lw=0.8)
ax.text(0.03,0.12,'LSTM baseline',fontsize=6.5,color='#777777')
ax.set_xticks(x); ax.set_xticklabels(MODELS_SHORT,fontsize=7)
ax.set_ylabel('RMSE improvement vs LSTM (%)')
ax.set_title('What does added complexity buy?  (% improvement over LSTM, full seed range)',
             loc='left',fontsize=8.5,fontweight='bold',color=C_TEXT,pad=6)
ax.legend(loc='lower right',frameon=True,framealpha=0.9,edgecolor='#DDDDDD')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
fig.tight_layout(pad=0.5)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/B_Fig3_complexity_improvement.{e}',facecolor='white')
plt.close(fig); print('B-Fig3 done')

# ============ B-Fig4 效率前沿(RMSE vs 时间, 虚线对角线) ============
# 需要time数据
def load_time(cohort):
    return {m:[json.load(open(f'results/{cohort}/result_{m}_seed{s}.json'))['time_min']
               for s in SEEDS if os.path.exists(f'results/{cohort}/result_{m}_seed{s}.json')] for m in MODELS}
T1t=load_time('metabonet_v2_20seed'); T2t=load_time('metabonet_v2_20seed_t2d')
fig,ax=plt.subplots(figsize=(DCW,2.4))
model_cmap={'LSTM':'#1B4E79','BiLSTM':'#21918C','LSTM-Attention':'#CC79A7','BiLSTM-Attention':'#E69F00'}
# T1D效率前沿(70-83min)包络
for pop,tm,tm_r,cm,alpha in [('T1D',T1t,T1,C_T1D,0.6),('T2D',T2t,T2,C_T2D,0.85)]:
    for m in MODELS:
        ax.scatter(tm[m],tm_r[m],s=22,color=model_cmap[m],alpha=alpha,
                   marker='o' if pop=='T1D' else 's',edgecolor='white',linewidth=0.3,zorder=3,label=None)
    # 均值(大实心)
    for m in MODELS:
        ax.scatter(np.mean(tm[m]),np.mean(tm_r[m]),s=95,color='white',
                   marker='o' if pop=='T1D' else 's',edgecolor=model_cmap[m],linewidth=2.2,zorder=4)
# 效率等值线: RMSE per minute 参考(前沿线连接每人群最优)
# T1D基线线: 连接LSTM均值与BiLSTM-Attn均值(代表T1D前沿)
front_x=[np.mean(T1t['LSTM']),np.mean(T1t['BiLSTM-Attention'])]
front_y=[np.mean(T1['LSTM']),np.mean(T1['BiLSTM-Attention'])]
ax.plot(front_x,front_y,color=C_T1D,ls='--',lw=1.0,alpha=0.5)
front2_x=[np.mean(T2t['LSTM']),np.mean(T2t['BiLSTM-Attention'])]
front2_y=[np.mean(T2['LSTM']),np.mean(T2['BiLSTM-Attention'])]
ax.plot(front2_x,front2_y,color=C_T2D,ls='--',lw=1.0,alpha=0.5)
ax.annotate('T2D efficiency frontier\n(7.7–8.8 min)',xy=(np.mean(T2t['BiLSTM-Attention']),np.mean(T2['BiLSTM-Attention'])),
            xytext=(30,19.0),fontsize=7,color=C_T2D,arrowprops=dict(arrowstyle='-|>',color=C_T2D,lw=0.8))
ax.annotate('T1D frontier\n(70–83 min)',xy=(np.mean(T1t['BiLSTM-Attention']),np.mean(T1['BiLSTM-Attention'])),
            xytext=(60,16.9),fontsize=7,color=C_T1D,arrowprops=dict(arrowstyle='-|>',color=C_T1D,lw=0.8))
ax.set_xlabel('Training time (min)'); ax.set_ylabel('RMSE (mg/dL)')
ax.set_title('Efficiency frontier: accuracy per minute of training (all 160 seeds)',
             loc='left',fontsize=8.5,fontweight='bold',color=C_TEXT,pad=6)
h1=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=model_cmap[m],ms=5,label=m) for m in MODELS]
h2=[plt.Line2D([0],[0],marker=o,color='w',markerfacecolor='#555555',ms=5,label=p) for p,o in [('T1D','o'),('T2D','s')]]
leg1=ax.legend(handles=h1,loc='upper left',frameon=True,framealpha=0.9,edgecolor='#DDDDDD',title='Models',fontsize=6.5,title_fontsize=7)
ax.add_artist(leg1)
ax.legend(handles=h2,loc='center right',frameon=True,framealpha=0.9,edgecolor='#DDDDDD',title='Population',fontsize=6.5,title_fontsize=7)
for sp in ['top','right']:ax.spines[sp].set_visible(False)
ax.xaxis.grid(True); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
fig.tight_layout(pad=0.5)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/B_Fig4_efficiency_frontier.{e}',facecolor='white')
plt.close(fig); print('B-Fig4 done')
print('=== SET-B 全部完成 ===')

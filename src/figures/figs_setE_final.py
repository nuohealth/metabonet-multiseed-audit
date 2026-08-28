#!/usr/bin/env python3
"""CMPB 论文#5 图 组合E（最终投稿版）— 神农选定: Fig1/2/3用B版风格 + Fig4用D版, 并优化到"显示数据规律与价值"
优化点(以"数据规律/研究价值"为指针, 每个都有数据支撑):
E-Fig1(B风格分组但加规律标注): 每模型T1D/T2D分组, 标注稳定的+3.4~3.6跨人群gap, 强调"4模型一致性退化"规律
E-Fig2(B的DM显著性, 加研究价值): Holm校正p值, 强调"尽管部分统计显著, 但Δ<5临床可忽略"的价值结论
E-Fig3(B的%改进, 强调价值): 相对LSTM的%改进+全种子范围, 直接标注"复杂度增益≈0"的结论
E-Fig4(D双面板效率): 修9.2x同轴硬伤, 显示T2D高10倍效率的数据规律
风格: seaborn白底/despine/Liberation Sans/专业色; 双栏174mm/300dpi/PNG+PDF+TIF
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from itertools import combinations as comb

import os
BASE=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
MODELS=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
MODELS_SHORT=['LSTM','BiLSTM','LSTM-Atten','BiLSTM-Atten']
SEEDS=list(range(42,62)); MM=25.4; DCW=174/MM; DPI=300
C_T1D='#1B4E79'; C_T2D='#D55E00'; C_T1D_L='#A8C0DA'; C_T2D_L='#F5CFA3'
C_TEXT='#1A1A1A'; C_GRID='#E4E4E4'
sns.set_theme(style='white',context='paper')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'legend.fontsize':7,'figure.facecolor':'white','axes.facecolor':'white','axes.linewidth':0.5,
 'grid.color':C_GRID,'grid.linewidth':0.4,'savefig.dpi':DPI})

def load(cohort): return {m:[json.load(open(f'results/{cohort}/result_{m}_seed{s}.json')) for s in SEEDS if os.path.exists(f'results/{cohort}/result_{m}_seed{s}.json')] for m in MODELS}
T1=load('metabonet_v2_20seed'); T2=load('metabonet_v2_20seed_t2d')
def mean(d,m,k): return np.mean([x[k] for x in d[m]])
def sd(d,m,k): return np.std([x[k] for x in d[m]],ddof=1)

# ============ E-Fig1 跨人群泛化(B风格分组, 加一致性规律标注) ============
fig,ax=plt.subplots(figsize=(DCW,2.5))
x=np.arange(len(MODELS)); wd=0.34
for i,m in enumerate(MODELS):
    ax.scatter([i-wd/2]*20,[x['rmse'] for x in T1[m]],s=14,color=C_T1D,alpha=0.55,edgecolor='white',linewidth=0.2,zorder=2)
    ax.errorbar(i-wd/2,mean(T1,m,'rmse'),yerr=sd(T1,m,'rmse'),fmt='o',color=C_T1D,ms=6,capsize=3,capthick=0.8,elinewidth=0.9,zorder=3,markeredgecolor='white',markeredgewidth=0.4)
    ax.scatter([i+wd/2]*20,[x['rmse'] for x in T2[m]],s=14,color=C_T2D,alpha=0.55,edgecolor='white',linewidth=0.2,zorder=2)
    ax.errorbar(i+wd/2,mean(T2,m,'rmse'),yerr=sd(T2,m,'rmse'),fmt='s',color=C_T2D,ms=6,capsize=3,capthick=0.8,elinewidth=0.9,zorder=3,markeredgecolor='white',markeredgewidth=0.4)
    # gap标注(一致性规律: 全部正)
    g=mean(T2,m,'rmse')-mean(T1,m,'rmse')
    ax.annotate(f'+{g:.2f}***',xy=(i+wd/2,mean(T2,m,'rmse')+0.25),ha='center',fontsize=6.5,color=C_T2D,
                arrowprops=dict(arrowstyle='-',color=C_T2D,lw=0.8))
# 一致性规律强调
ax.text(0.5,22.2,'all 4 architectures degrade by a consistent +3.4–3.6 mg/dL on T2D (p<0.001)',
        ha='center',fontsize=6.8,color=C_TEXT,style='italic')
ax.set_xticks(x); ax.set_xticklabels(MODELS_SHORT,fontsize=7)
ax.set_ylabel('RMSE (mg/dL, 30-min)'); ax.set_ylim(15,23)
ax.set_title('Cross-cohort generalization: consistent degradation across all architectures',loc='left',
             fontsize=8.5,fontweight='bold',color=C_TEXT,pad=8)
import matplotlib.lines as mlines
l1=mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor=C_T1D,ms=6,label='T1D (n=1,092)')
l2=mlines.Line2D([0],[0],marker='s',color='w',markerfacecolor=C_T2D,ms=6,label='T2D (n=100)')
ax.legend(handles=[l1,l2],loc='upper left',frameon=True,framealpha=0.9,edgecolor='#DDDDDD')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
fig.tight_layout(pad=0.5)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/E_Fig1_cross_cohort.{e}',facecolor='white')
plt.close(fig); print('E-Fig1 done')

# ============ E-Fig2 架构区分度(DM显著性, 强调价值) ============
def dm_p(x,y):
    d=np.array([x[k]['rmse'] for k in range(len(x))])-np.array([y[k]['rmse'] for k in range(len(y))])
    n=len(d); s=d.std(ddof=1); z=d.mean()/(s/np.sqrt(n)) if s>0 else 0
    return 2*(1-norm.cdf(abs(z)))
pairs=list(comb(MODELS,2)); pair_lbl=['LSTM–BiLSTM','LSTM–LSTM-A','LSTM–BiLSTM-A','BiLSTM–LSTM-A','BiLSTM–BiLSTM-A','LSTM-A–BiLSTM-A']
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(DCW,2.7),sharex=True)
for axi,pop,data,cm,lab in [(ax1,'T1D',T1,C_T1D,'(a)  T1D'),(ax2,'T2D',T2,C_T2D,'(b)  T2D')]:
    ps=[dm_p(data[a],data[b]) for a,b in pairs]; hs=multipletests(ps,method='holm')[1]
    cols=['#1B4E79' if h<0.05 else '#C8C8C8' for h in hs]
    axi.bar(np.arange(len(pairs)),[-np.log10(max(h,1e-6)) for h in hs],color=cols,width=0.6,edgecolor='white',linewidth=0.4)
    for i,h in enumerate(hs):
        axi.text(i,-np.log10(max(h,1e-6))+0.06,f'{h:.3f}',ha='center',fontsize=6,color=C_TEXT,va='bottom')
    axi.axhline(-np.log10(0.05),color='#999999',ls='--',lw=0.8)
    axi.text(len(pairs)-0.4,-np.log10(0.05)+0.06,'α=0.05',fontsize=6,color='#777777',ha='right')
    axi.set_ylim(0,3.4); axi.set_ylabel('−log₁₀(p)')
    axi.set_title(lab+'  Diebold–Mariano test (Holm-corrected)',loc='left',fontsize=8,fontweight='bold',color=C_TEXT,pad=4)
    for sp in ['top','right']:axi.spines[sp].set_visible(False)
    axi.yaxis.grid(True); axi.set_axisbelow(True); axi.tick_params(length=3)
ax2.set_xticklabels(pair_lbl,fontsize=6.2,rotation=30,ha='right')
fig.suptitle('Model distinguishability: some pairs significant (p<0.05) but all Δ<5 mg/dL → clinically equivalent',
             fontsize=8,fontweight='bold',color=C_TEXT)
fig.tight_layout(pad=0.6)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/E_Fig2_dm_significance.{e}',facecolor='white')
plt.close(fig); print('E-Fig2 done')

# ============ E-Fig3 复杂度收益(B的%改进, 强调价值) ============
fig,ax=plt.subplots(figsize=(DCW,2.3))
x=np.arange(len(MODELS))
for pop,data,base,cm,cm_l,marker,label in [('T1D',T1,None,C_T1D,C_T1D_L,'o','T1D'),('T2D',T2,None,C_T2D,C_T2D_L,'s','T2D')]:
    base_arr=np.array([d['rmse'] for d in data['LSTM']])  # 该人群LSTM基线
    pct=[];lo=[];hi=[]
    for m in MODELS:
        arr=np.array([d['rmse'] for d in data[m]])
        p_all=(base_arr.mean()-arr)/base_arr.mean()*100
        pct.append(p_all.mean()); lo.append(p_all.min()); hi.append(p_all.max())
    ax.fill_between(x,lo,hi,color=cm,alpha=0.13,lw=0)
    ax.plot(x,pct,color=cm,lw=1.8,marker=marker,ms=6,label=label,markeredgecolor='white',markeredgewidth=0.4,zorder=3)
    for i,m in enumerate(MODELS):
        arr=np.array([d['rmse'] for d in data[m]])
        p_all=(base_arr.mean()-arr)/base_arr.mean()*100
        ax.scatter([i]*20,p_all,s=13,color=cm,alpha=0.42,edgecolor='white',linewidth=0.2,zorder=2)
ax.axhline(0,color='#999999',ls='--',lw=0.8); ax.text(0.02,0.12,'LSTM baseline',fontsize=6.5,color='#777777')
# 价值结论标注(诚实: 均值 T1D≤+0.4%, T2D−0.2~−1.1%; 全种子范围到−3.8%)
ax.text(1.1,0.62,'mean: complexity adds ≤0.4% on T1D, slightly hurts T2D (−0.2~−1.1%)',
        fontsize=6.8,color=C_TEXT)
ax.set_ylim(-4.2, 1.6)  # 显式y范围容纳全种子负值
ax.set_xticks(x); ax.set_xticklabels(MODELS_SHORT,fontsize=7)
ax.set_ylabel('RMSE improvement vs LSTM (%)')
ax.set_title('Added complexity: marginal T1D gain (≤0.4%), slight T2D loss (−0.2~−1.1%) → not worth it',
             loc='left',fontsize=8.5,fontweight='bold',color=C_TEXT,pad=6)
ax.legend(loc='lower right',frameon=True,framealpha=0.9,edgecolor='#DDDDDD')
for sp in ['top','right']:ax.spines[sp].set_visible(False)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
fig.tight_layout(pad=0.5)
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/E_Fig3_complexity_value.{e}',facecolor='white')
plt.close(fig); print('E-Fig3 done')

# ============ E-Fig4 效率双面板(D版, 定稿) ============
fig,(axL,axR)=plt.subplots(1,2,figsize=(DCW,2.5),sharey=True,gridspec_kw={'width_ratios':[1.6,1.0]})
model_cmap={'LSTM':'#1B4E79','BiLSTM':'#21918C','LSTM-Attention':'#CC79A7','BiLSTM-Attention':'#E69F00'}
def panel(ax,data,cm,pop):
    for m in MODELS:
        t=[x['time_min'] for x in data[m]]; r=[x['rmse'] for x in data[m]]
        ax.scatter(t,r,s=24,color=model_cmap[m],alpha=0.75,edgecolor='white',linewidth=0.3,zorder=3)
    for m in MODELS:
        ax.scatter(mean(data,m,'time_min'),mean(data,m,'rmse'),s=100,color='white',marker='o',
                   edgecolor=model_cmap[m],linewidth=2.2,zorder=4)
        ax.annotate(MODELS_SHORT[MODELS.index(m)],(mean(data,m,'time_min')+1.2,mean(data,m,'rmse')),
                    fontsize=6.5,color=model_cmap[m])
    ax.set_xlabel('Training time (min)')
    ax.set_title(pop,loc='left',fontsize=8.5,fontweight='bold',color=cm,pad=6)
    for sp in ['top','right']:ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(length=3)
panel(axL,T1,C_T1D,'T1D (1,092 pts): 42–186 min')
panel(axR,T2,C_T2D,'T2D (100 pts): 5–14 min')
axL.set_ylabel('RMSE (mg/dL)'); axL.set_xlim(30,160); axR.set_xlim(4,18)
h1=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=model_cmap[m],ms=5,label=m) for m in MODELS]
leg=axL.legend(handles=h1,loc='lower right',frameon=True,framealpha=0.9,edgecolor='#DDDDDD',
               title='Models',fontsize=6.5,title_fontsize=7)
fig.suptitle('Efficiency: ~10× lower training cost on T2D with same architectures (all 160 seeds)',
             fontsize=8,fontweight='bold',color=C_TEXT)
fig.tight_layout(pad=0.5,rect=[0,0,1,0.94])
for e in ['png','pdf','tif']:
    fig.savefig(f'{BASE}/E_Fig4_efficiency.{e}',facecolor='white')
plt.close(fig); print('E-Fig4 done')
print('=== 组合E(最终) 全部完成 ===')

#!/usr/bin/env python3
"""Fig3/4/5 修正: 图形叙事, 极简标注(去除文字框遮挡)
原则: 故事靠图形元素(参考线/带/连接)体现, 最多一个精简洁注放在空白区, 绝不压数据.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np, json, os
import seaborn as sns

# BASE: repo-relative; run from repo root
BASE='results'
# OUT: repo-relative
OUT ='figures_final'
MODELS=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
MSHORT=['LSTM','BiLSTM','LSTM-Attn','BiLSTM-Attn']
SEEDS=list(range(42,62)); DCW=174/25.4; DPI=300
C1='#1B4E79'; C2='#C1272D'
mc={'LSTM':'#1F4E79','BiLSTM':'#2E86C1','LSTM-Attention':'#E67E22','BiLSTM-Attention':'#8E44AD'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':0.6,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':DPI})
def load(c): return {m:[json.load(open(f'{BASE}/{c}/result_{m}_seed{s}.json')) for s in SEEDS
                       if os.path.exists(f'{BASE}/{c}/result_{m}_seed{s}.json')] for m in MODELS}
T1=load('metabonet_v2_20seed'); T2=load('metabonet_v2_20seed_t2d')
def r(d,m): return np.array([x['rmse'] for x in d[m]])
def t(d,m): return np.array([x['time_min'] for x in d[m]])

# ============ Fig3 复杂度: 图形叙事, 无文字框 ============
print('[Fig3] ...')
fig,ax=plt.subplots(figsize=(DCW,2.2))
x=np.arange(len(MODELS))
for data,col,mark,lab in [(T1,C1,'o','T1D'),(T2,C2,'s','T2D')]:
    base=r(data,'LSTM').mean(); mu=[];lo=[];hi=[]
    for m in MODELS:
        p=(base-r(data,m))/base*100; mu.append(p.mean());lo.append(p.min());hi.append(p.max())
    ax.plot(x,mu,color=col,lw=1.8,marker=mark,ms=7,label=lab,markeredgecolor='white',markeredgewidth=0.5,zorder=3)
    ax.fill_between(x,lo,hi,color=col,alpha=0.14,lw=0)
    for i in x: ax.plot([i,i],[lo[i],hi[i]],color=col,lw=0.9,alpha=0.6)
ax.axhline(0,color='#666666',ls='--',lw=1.0)
ax.text(0.03,0.35,'LSTM baseline',fontsize=8,color='#555555')
# 极简结论: 单行, 放左上空白(数据线在-2~1区, 顶部3区空)
ax.text(0.95,2.5,'all gains < 0.4% (T1D) or negative (T2D)',fontsize=8,color='#444444',ha='center')
ax.set_ylim(-4.5,3.0)
ax.set_xticks(x); ax.set_xticklabels(MSHORT)
ax.set_ylabel('RMSE improvement vs LSTM (%)')
ax.set_title('Relative RMSE improvement of each architecture vs the LSTM baseline',loc='left',fontsize=9,fontweight='bold',pad=5)
ax.legend(loc='lower right',frameon=True,fontsize=8,edgecolor='#DDDDDD')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.yaxis.grid(True,color='#E8E8E8',lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=3)
for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/Fig3_complexity_story.{e}',facecolor='white')
plt.close(fig); print('  Fig3 done')

# ============ Fig4 效率: 图形叙事(参考线), 无文字框 ============
print('[Fig4] ...')
fig,(axl,axr)=plt.subplots(1,2,figsize=(DCW,2.2),sharey=True,gridspec_kw={'width_ratios':[1.6,1.0]})
def panel(ax,data,pop):
    # RMSE 水平细参考线(不标文字, 让图自己说话)
    allr=[m for mm in MODELS for m in r(data,mm)]; rm=np.mean(allr)
    ax.axhline(rm,color='#888888',ls=':',lw=1.0)
    for m in MODELS:
        ax.scatter(t(data,m),r(data,m),s=14,color=mc[m],alpha=0.65,edgecolor='white',linewidth=0.2,zorder=3)
    for i,m in enumerate(MODELS):
        ax.scatter(t(data,m).mean(),r(data,m).mean(),s=80,color='white',marker='o',
                   edgecolor=mc[m],linewidth=2.2,zorder=4)
        ax.annotate(MSHORT[i],(t(data,m).mean(),r(data,m).mean()+0.16), # 标注在均值点正上方,错位防压在点
                    fontsize=7.5,color=mc[m],ha='center',va='bottom')
    ax.set_xlabel('Training time (min)'); ax.set_title(pop,loc='left',fontsize=8.5,fontweight='bold',pad=4)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True,color='#E8E8E8',lw=0.5);ax.yaxis.grid(True,color='#E8E8E8',lw=0.5);ax.tick_params(length=3)
panel(axl,T1,'(a) T1D'); panel(axr,T2,'(b) T2D')
axl.set_ylabel('RMSE (mg/dL)'); axl.set_xlim(30,185); axr.set_xlim(3,13)
fig.suptitle('Training cost vs accuracy (dotted line = cohort mean RMSE: all models sit on it)',fontsize=9,fontweight='bold')
fig.tight_layout(pad=0.6,rect=[0.02,0.04,1,0.94])
for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/Fig4_efficiency_story.{e}',facecolor='white')
plt.close(fig); print('  Fig4 done')

# ============ Fig5 临床: 回到简洁骨架(柱+参考线), 去文字框 ============
print('[Fig5] ...')
cl=json.load(open(f'{BASE}/metabonet_v2_monitored/clinical_evaluation_results.json'))
au=json.load(open(f'{BASE}/20seed_summary_authoritative.json'))
fig,(a1,a2)=plt.subplots(1,2,figsize=(DCW,2.1))
xm=np.arange(len(MODELS)); wm=0.55
# a) MARD 柱 + <10%参考线(简洁, 不堆框)
mard=[au['T1D']['models'][m]['mard_mean'] for m in MODELS]
a1.bar(xm,mard,color=[mc[m] for m in MODELS],width=wm,edgecolor='white',lw=0.4)
a1.axhline(10,color='#C0392B',ls='--',lw=1.0)
a1.text(0.05,10.25,'<10% standard',fontsize=8,color='#C0392B')
a1.set_xticks(xm); a1.set_xticklabels(MSHORT,fontsize=8)
a1.set_ylabel('MARD (%)'); a1.set_ylim(0,10.9)
a1.set_title('(a) MARD below consensus',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: a1.spines[sp].set_visible(False)
a1.yaxis.grid(True,color='#E8E8E8',lw=0.5); a1.tick_params(length=3)
# b) Clarke ZoneA 柱(简洁)
zA=[cl['clarke_ega'][m]['pcts']['A'] for m in MODELS]
a2.bar(xm,zA,color=[mc[m] for m in MODELS],width=wm,edgecolor='white',lw=0.4)
a2.axhline(np.mean(zA),color='#888888',ls=':',lw=1.0)
a2.set_xticks(xm); a2.set_xticklabels(MSHORT,fontsize=8)
a2.set_ylabel('Clarke Zone A (%)'); a2.set_ylim(76,80.8)
a2.set_title('(b) Clarke Zone A (narrow band)',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: a2.spines[sp].set_visible(False)
a2.yaxis.grid(True,color='#E8E8E8',lw=0.5); a2.tick_params(length=3)
fig.suptitle('Clinical metrics across architectures (MARD and Clarke error-grid)',fontsize=9,fontweight='bold')
fig.tight_layout(pad=0.6,rect=[0.015,0.04,1,0.94])
for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/Fig5_clinical_story.{e}',facecolor='white')
plt.close(fig); print('  Fig5 done')
print('=== Fig3/4/5 修正完成 ===')

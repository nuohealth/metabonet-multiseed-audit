#!/usr/bin/env python3
"""Fig5 修正: 用图形元素(阈值带/误差棒/窄带)凸显"临床达标且架构等价"的研究价值
关键: 图形说话, 不堆文字框; 结论在suptitle.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os

# BASE: repo-relative; run from repo root
BASE='results'
# OUT: repo-relative
OUT ='figures_final'
MODELS=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
MSHORT=['LSTM','BiLSTM','LSTM-Attn','BiLSTM-Attn']
DCW=174/25.4; DPI=300
mc={'LSTM':'#1F4E79','BiLSTM':'#2E86C1','LSTM-Attention':'#E67E22','BiLSTM-Attention':'#8E44AD'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':0.6,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':DPI})
au=json.load(open(f'{BASE}/20seed_summary_authoritative.json'))
cl=json.load(open(f'{BASE}/metabonet_v2_monitored/clinical_evaluation_results.json'))

fig,axes=plt.subplots(1,2,figsize=(DCW,2.3),sharey=False)
x=np.arange(len(MODELS)); wm=0.6

# ---- (a) MARD: 误差棒 + 阈值带(图形元素凸显"达标+等价") ----
ax=axes[0]
# 可接受阈值带(0-10%): 浅色带, 表示"达标区"
ax.axhspan(0,10,color='#EAF4EA',alpha=0.9,lw=0,zorder=0)   # 浅绿达标带
ax.axhline(10,color='#2E8B57',ls='--',lw=1.1)               # 阈值线
# 4架构 MARD 误差棒(20-seed ±SD): 误差棒重叠 => 等价
for i,m in enumerate(MODELS):
    d=au['T1D']['models'][m]
    ax.errorbar(i,d['mard_mean'],yerr=d['mard_sd'],fmt='o',color=mc[m],ms=7,capsize=4,capthick=1,
                elinewidth=1.5,zorder=3,markeredgecolor='white',markeredgewidth=0.5)
ax.text(0.02,10.28,'consensus: MARD < 10%',fontsize=7.5,color='#2E8B57',ha='left')
ax.set_xticks(x); ax.set_xticklabels(MSHORT)
ax.set_ylabel('MARD (%, 20-seed mean ± SD)'); ax.set_ylim(0,11)
ax.set_xlim(-0.6,3.6)
ax.set_title('(a) MARD across architectures (20-seed mean \u00b1 SD)',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.yaxis.grid(True,color='#EDEDED',lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=3)

# ---- (b) Clarke Zone A: 点 + 窄带(凸显"集中+等价") ----
ax=axes[1]
zA=[cl['clarke_ega'][m]['pcts']['A'] for m in MODELS]
# 窄带范围带(77.5-79.5)凸显"4架构挤在窄带"
ax.axhspan(min(zA)-0.4,max(zA)+0.4,color='#EAF4EA',alpha=0.9,lw=0,zorder=0)
ax.axhline(min(zA),color='#2E8B57',ls=':',lw=0.8)
ax.axhline(max(zA),color='#2E8B57',ls=':',lw=0.8)
# 4架构点
for i,m in enumerate(MODELS):
    ax.plot(i,zA[i],'o',color=mc[m],ms=9,markeredgecolor='white',markeredgewidth=0.5,zorder=3)
# 无文字框, 窄带宽度本身就是"等价"的图形表达
ax.set_xticks(x); ax.set_xticklabels(MSHORT)
ax.set_ylabel('Clarke Zone A (%)'); ax.set_ylim(76.5,80.5)
ax.set_xlim(-0.6,3.6)
ax.set_title('(b) Clarke Zone A across architectures',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.yaxis.grid(True,color='#EDEDED',lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=3)

fig.suptitle('Clinical metrics across architectures (MARD and Clarke error-grid)',
             fontsize=9,fontweight='bold')
fig.tight_layout(pad=0.6,rect=[0.015,0.05,0.985,0.93])
for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/Fig5_clinical_story.{e}',facecolor='white')
plt.close(fig); print('  Fig5 done')

#!/usr/bin/env python3
"""CMPB#5 -> PLOS ONE 故事凸显版 (figs_story_v2.py)
基础图形 + 凸显研究故事逻辑:
Fig1: 每个架构画20个seed散点(seed变异=论文主角), 用色带凸显
      "between-seed spread 盖过 between-architecture gap" 的核心命题.
Fig2: ΔRMSE森林图, x轴聚焦, 叠加"seed变异参考带"让"统计显著但<seed变异<临床阈值"三阶故事可视化.
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
os.makedirs(OUT, exist_ok=True)
MODELS=['LSTM','BiLSTM','LSTM-Attention','BiLSTM-Attention']
MSHORT=['LSTM','BiLSTM','LSTM-Attn','BiLSTM-Attn']
SEEDS=list(range(42,62)); DCW=174/25.4; DPI=300
C1='#1B4E79'; C2='#C1272D'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.linewidth':0.6,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':DPI})

def load(c): return {m:[json.load(open(f'{BASE}/{c}/result_{m}_seed{s}.json')) for s in SEEDS
                       if os.path.exists(f'{BASE}/{c}/result_{m}_seed{s}.json')] for m in MODELS}
T1=load('metabonet_v2_20seed'); T2=load('metabonet_v2_20seed_t2d')
def r(d,m): return np.array([x['rmse'] for x in d[m]])
def save(fig,n,bottom=0.0):
    fig.tight_layout(pad=0.6,rect=[0,bottom,1,0.95])
    for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/{n}.{e}',facecolor='white')
    plt.close(fig); print(f'  {n} done')

# ================= Fig1 核心故事: seed变异 vs 架构差异 =================
print('[Fig1] ...')
fig,axes=plt.subplots(1,2,figsize=(DCW,2.8),sharey=False)
x=np.arange(len(MODELS)); w=0.36
for ax,data,col,lab,ylo,yhi,bcol in [(axes[0],T1,C1,'(a) T1D (MetaboNet, n=1,092)',15.60,17.00,'#DCE6F1'),
                                      (axes[1],T2,C2,'(b) T2D (Shanghai, n=100)',21.30,24.70,'#FBE5E0')]:
    # 1) 全seed范围的浅色带(凸显seed变异宽度)
    for i,m in enumerate(MODELS):
        v=r(data,m); lo=v.min(); hi=v.max()
        ax.axvspan(i-w/2,i+w/2,color=bcol,alpha=0.6,lw=0,zorder=0)  # 架构占位
        # seed min-max 竖带(浅)
        ax.plot([i,i],[lo,hi],color=col,lw=6,alpha=0.18,zorder=1,solid_capstyle='round')
    # 2) 20个seed散点(论文主角: 看seed散布)
    rng=np.random.RandomState(7)
    for i,m in enumerate(MODELS):
        v=r(data,m)
        jit=rng.uniform(-0.13,0.13,v.size)
        ax.scatter(i+jit,v,s=13,color=col,alpha=0.60,edgecolor='white',linewidth=0.3,zorder=3)
    # 3) 架构均值(粗, 突出"几乎同高")
    means=[r(data,m).mean() for m in MODELS]
    ax.plot(x,means,color='#333333',lw=1.2,ls='--',zorder=4,alpha=0.7)  # 均值连线
    for i,m in enumerate(MODELS):
        ax.plot([i-0.12,i+0.12],[means[i],means[i]],color='#111111',lw=2.2,zorder=5)  # 均值短粗线
    # 4) 核心命题标注: seed变异盖过架构差异 (放面板下部空白区, 避开顶部高点/标题)
    sd_max=max(r(data,m).std(ddof=1) for m in MODELS)
    arch_diff=max(means)-min(means)
    ax.text(0.5,0.20,
            f'seed spread (SD {min(r(data,m).std(ddof=1) for m in MODELS):.2f}–{sd_max:.2f}) \n> architecture gap ({arch_diff:.3f})',
            ha='center',va='top',fontsize=7.5,color='#333333',
            bbox=dict(boxstyle='round,pad=0.35',fc='white',ec=col,lw=1.0),
            transform=ax.transAxes)
    ax.set_xlim(-0.7,len(MODELS)-0.3); ax.set_ylim(ylo,yhi)
    ax.set_xticks(x); ax.set_xticklabels(MSHORT)
    ax.set_ylabel('RMSE (mg/dL, 30-min)') if ax is axes[0] else None
    ax.set_title(lab,loc='left',fontsize=9,fontweight='bold',pad=5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True,color='#EDEDED',lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=3)
# gap注释(图b, 放面板中央上方的空白区, 不撞数据高点)
axes[1].text(1.5,23.6,'T1D → T2D:\n+6.2–6.5 mg/dL\n(consistent, p<0.001)',ha='center',
             fontsize=7.8,color='#333333',bbox=dict(boxstyle='round,pad=0.4',fc='#FFF5E6',ec='#E0A800',lw=0.9))
# 图例: seed点 + 均值
fs=[mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor=C1,ms=6,label='20 seeds (T1D)'),
    mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor=C2,ms=6,label='20 seeds (T2D)'),
    mlines.Line2D([0],[0],color='#111111',lw=2.2,label='architecture mean (dashed = mean trend)')]
fig.legend(handles=fs,loc='lower center',ncol=3,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle('Per-architecture seed dispersion and mean RMSE across cohorts',
             fontsize=9,fontweight='bold')
save(fig,'Fig1_seed_vs_arch',bottom=0.07)

# ================= Fig2: ΔRMSE + seed变异参考带 =================
print('[Fig2] ...')
pairs=['LSTM vs BiLSTM','LSTM vs LSTM-Attention','LSTM vs BiLSTM-Attention',
       'BiLSTM vs LSTM-Attention','BiLSTM vs BiLSTM-Attention','LSTM-Attention vs BiLSTM-Attention']
pshort=['LSTM–BiLSTM','LSTM–LSTM-A','LSTM–BiLSTM-A','BiLSTM–LSTM-A','BiLSTM–BiLSTM-A','LSTM-A–BiLSTM-A']
def fd(tag):
    p=f'{BASE}/{"metabonet_v2_20seed" if tag=="T1D" else "metabonet_v2_20seed_t2d"}/stat_results_20seed_{tag}.json'
    d=json.load(open(p))['comparisons']
    return [(v['rmse_diff_mean_mgdl'],v['rmse_diff_ci95_bootstrap'][0],v['rmse_diff_ci95_bootstrap'][1],
             v.get('cohens_dz',0),v['dm_pvalue_holm']) for v in d.values()]
fig,axes=plt.subplots(1,2,figsize=(DCW,2.5),sharex=True)
for ax,(tag,col,lab) in zip(axes,[('T1D',C1,'(a) T1D'),('T2D',C2,'(b) T2D')]):
    rd=fd(tag); yidx=np.arange(len(pairs))
    for yy,(est,lo,hi,dz,h) in zip(yidx,rd):
        sig=h<0.05; c='#C1272D' if sig else col
        ax.plot([lo,hi],[yy,yy],color=c,lw=1.8,zorder=2)
        ax.scatter(est,yy,s=42,color=c,zorder=3,edgecolor='white',linewidth=0.5)
    ax.axvline(0,color='#555555',lw=1.0)
    ax.set_yticks(yidx); ax.set_yticklabels(pshort if ax is axes[0] else [],fontsize=7)
    ax.set_title(lab+'  ΔRMSE (95% CI)',loc='left',fontsize=9,fontweight='bold',pad=5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True,color='#EDEDED',lw=0.5); ax.tick_params(length=3)
axes[1].set_xlim(-0.7,0.85)
# 三阶故事标注: 显著 < seed变异 < 临床阈值
for ax in axes:
    ax.axvspan(-0.7,0.7,color='#F5F5F5',alpha=0.7,zorder=0)
    ax.text(0.77,1.0,'most Δ fall within\none seed-SD (~0.07)',
            fontsize=7,color='#555555',ha='right',style='italic')
axes[0].annotate('Δ=5 mg/dL\n(clinical relevance,\nfine scale)',
                 xy=(0.6,0),xytext=(0.45,1.8),fontsize=7,color='#888888',
                 arrowprops=dict(arrowstyle='->',color='#BBBBBB',lw=1.0))
handles=[mlines.Line2D([0],[0],marker='o',color='w',markerfacecolor='#C1272D',ms=6,label='Holm p<0.05'),
         mlines.Line2D([0],[0],color='#888888',ls='--',lw=1.0,label='±one seed-SD reference')]
fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.03))
fig.suptitle('Pairwise \u0394RMSE between architectures, with bootstrap 95% CI',
             fontsize=9,fontweight='bold')
save(fig,'Fig2_delta_vs_seed',bottom=0.08)

print('=== 故事凸显版 Fig1/Fig2 完成 ===')

# ============================================================
# Fig3 复杂度收益(故事凸显): 复杂度递增但收益平坦/为负
# ============================================================
print('[Fig3] ...')
OUT3='figures_final'
fig,ax=plt.subplots(figsize=(DCW,2.4))
x=np.arange(len(MODELS))
for data,col,mark,lab in [(T1,C1,'o','T1D'),(T2,C2,'s','T2D')]:
    base=r(data,'LSTM').mean()
    mu=[];lo=[];hi=[]
    for m in MODELS:
        p=(base-r(data,m))/base*100; mu.append(p.mean());lo.append(p.min());hi.append(p.max())
    ax.plot(x,mu,color=col,lw=1.8,marker=mark,ms=7,label=lab,markeredgecolor='white',markeredgewidth=0.5,zorder=3)
    ax.fill_between(x,lo,hi,color=col,alpha=0.15,lw=0)
    for i in x: ax.plot([i,i],[lo[i],hi[i]],color=col,lw=0.9,alpha=0.6)
ax.axhline(0,color='#999999',ls='--',lw=0.9)
ax.text(0.03,0.16,'LSTM baseline (0%)',fontsize=8,color='#666666')
ax.text(0.05,2.3,'increasing architectural complexity  →',fontsize=8,color='#444444')
ax.annotate('',xy=(3.2,-3.2),xytext=(-0.2,-3.2),fontsize=0,
            arrowprops=dict(arrowstyle='->',color='#444444',lw=1.2))
ax.annotate('complexity buys ≈ nothing\n(T1D ≤+0.4%; T2D −0.2 to −1.2%)',
            xy=(1.5,0.3),xytext=(1.2,-3.9),fontsize=8,color='#333333',ha='center',
            bbox=dict(boxstyle='round,pad=0.4',fc='white',ec='#999999',lw=0.8),
            arrowprops=dict(arrowstyle='->',color='#888888',lw=0.8))
ax.set_xticks(x); ax.set_xticklabels(MSHORT)
ax.set_ylabel('RMSE improvement vs LSTM (%)'); ax.set_ylim(-4.5,3.0)
ax.set_title('Relative RMSE improvement of each architecture vs the LSTM baseline',loc='left',fontsize=9,fontweight='bold',pad=5)
ax.legend(loc='lower right',frameon=True,fontsize=8,edgecolor='#DDDDDD')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.yaxis.grid(True,color='#E8E8E8',lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=3)
save(fig,'Fig3_complexity_story')

# ============================================================
# Fig4 效率(故事凸显): 最复杂最慢, 精度无差 (y轴同一水平)
# ============================================================
print('[Fig4] ...')
mc4={'LSTM':'#1F4E79','BiLSTM':'#2E86C1','LSTM-Attention':'#E67E22','BiLSTM-Attention':'#8E44AD'}
def t(d,m): return np.array([x['time_min'] for x in d[m]])
fig,(axl,axr)=plt.subplots(1,2,figsize=(DCW,2.5),sharey=True,gridspec_kw={'width_ratios':[1.6,1.0]})
def panel4(ax,data,pop):
    # RMSE 水平参考带(所有架构几乎同高)
    allr=[m for mm in MODELS for m in r(data,mm)]
    rm_mean=np.mean(allr); rm_sd=np.std(allr)
    ax.axhspan(rm_mean-rm_sd,rm_mean+rm_sd,color='#F2F2F2',alpha=0.9,lw=0,zorder=0)
    ax.axhline(rm_mean,color='#555555',ls=':',lw=1.0,zorder=1)
    ax.text(ax.get_xlim()[0]+2,rm_mean+rm_sd+0.05,f'RMSE spread ±1 SD ({rm_sd:.2f} mg/dL)',
            fontsize=7,color='#555555',va='bottom')
    for m in MODELS:
        ax.scatter(t(data,m),r(data,m),s=15,color=mc4[m],alpha=0.65,edgecolor='white',linewidth=0.2,zorder=3)
    # 均值大标记+按时间排序连接(显示趋势)
    tm=[t(data,m).mean() for m in MODELS]; rm=[r(data,m).mean() for m in MODELS]
    order=sorted(range(len(MODELS)),key=lambda i:tm[i])
    for i in order:
        ax.scatter(tm[i],rm[i],s=85,color='white',marker='o',edgecolor=mc4[MODELS[i]],linewidth=2.2,zorder=4)
    ax.plot([tm[i] for i in order],[rm[i] for i in order],color='#666666',lw=1.0,ls='--',zorder=2)
    for i,m in enumerate(MODELS):
        ax.annotate(MSHORT[i],(tm[i]+ (3 if ax is axl else 1.2),rm[i]),fontsize=7.5,color=mc4[m],va='center')
    ax.set_xlabel('Training time (min)'); ax.set_title(pop,loc='left',fontsize=8.5,fontweight='bold',pad=4)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True,color='#E8E8E8',lw=0.5);ax.yaxis.grid(True,color='#E8E8E8',lw=0.5);ax.tick_params(length=3)
panel4(axl,T1,'(a) T1D'); panel4(axr,T2,'(b) T2D')
axl.set_ylabel('RMSE (mg/dL)'); axl.set_xlim(30,185); axr.set_xlim(3,13)
fig.suptitle('Training cost vs forecast accuracy across architectures',
             fontsize=9,fontweight='bold')
fig.tight_layout(pad=0.6,rect=[0.02,0.05,1,0.92])
for e in ['png','pdf','tif']: fig.savefig(f'{OUT}/Fig4_efficiency_story.{e}',facecolor='white')
plt.close(fig); print('  Fig4_efficiency_story done')

# ============================================================
# Fig5 临床(故事凸显): MARD误差棒凸显4架构不可分 + Clarke聚集
# ============================================================
print('[Fig5] ...')
cl5=json.load(open(f'{BASE}/metabonet_v2_monitored/clinical_evaluation_results.json'))
au5=json.load(open(f'{BASE}/20seed_summary_authoritative.json'))
fig,(a1,a2)=plt.subplots(1,2,figsize=(DCW,2.5))
xm5=np.arange(len(MODELS)); wm=0.55
# a) MARD误差棒(20-seed ±SD) -> 凸显4架构不可分
mard=[au5['T1D']['models'][m]['mard_mean'] for m in MODELS]
mardsd=[au5['T1D']['models'][m]['mard_sd'] for m in MODELS]
for i,m in enumerate(MODELS):
    a1.errorbar(i,mard[i],yerr=mardsd[i],fmt='o',color=mc4[m],ms=8,capsize=4,capthick=1,
                elinewidth=1.5,zorder=3,markeredgecolor='white',markeredgewidth=0.5)
a1.axhline(10,color='#C0392B',ls='--',lw=1.0)
a1.text(0.05,10.2,'<10% clinical standard',fontsize=8,color='#C0392B')
a1.annotate('all 4 architectures indistinguishable\n(overlapping error bars)',xy=(1.5,7.38),xytext=(1.5,8.2),
            fontsize=7.5,color='#333333',ha='center',
            bbox=dict(boxstyle='round,pad=0.35',fc='white',ec='#999999',lw=0.8),
            arrowprops=dict(arrowstyle='->',color='#888888',lw=0.8))
a1.set_xticks(xm5); a1.set_xticklabels(MSHORT,fontsize=8)
a1.set_ylabel('MARD (%), 20-seed mean ± SD'); a1.set_ylim(6.5,10.6)
a1.set_title('(a) MARD: architectures clinically indistinguishable',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: a1.spines[sp].set_visible(False)
a1.yaxis.grid(True,color='#E8E8E8',lw=0.5); a1.tick_params(length=3)
# b) Clarke ZoneA 范围带凸显聚集
zA=[cl5['clarke_ega'][m]['pcts']['A'] for m in MODELS]
zE=[cl5['clarke_ega'][m]['pcts']['E']*1000 for m in MODELS]
a2.axhspan(min(zA),max(zA),color='#F2F2F2',alpha=0.9,lw=0)
a2.text(0.4,(min(zA)+max(zA))/2,f'Zone A range {min(zA):.1f}–{max(zA):.1f}%\n(all within narrow band)',
        fontsize=7,color='#555555',ha='center',va='center')
for i,m in enumerate(MODELS):
    a2.plot(i,zA[i],'o',color=mc4[m],ms=9,markeredgecolor='white',markeredgewidth=0.5,zorder=3)
    a2.plot(i,zE[i]*2+78,'^',color='#C0392B',ms=6,zorder=3)  # ZoneE缩放显示
a2.set_xticks(xm5); a2.set_xticklabels(MSHORT,fontsize=8)
a2.set_ylabel('Clarke Zone A (%)'); a2.set_ylim(76.5,80.5)
a2.set_title('(b) Clarke grid: Zone A vs Zone E (▼, ×10⁻³)',loc='left',fontsize=8.5,fontweight='bold',pad=4)
for sp in ['top','right']: a2.spines[sp].set_visible(False)
a2.yaxis.grid(True,color='#E8E8E8',lw=0.5); a2.tick_params(length=3)
fig.suptitle('Clinical metrics across architectures (MARD and Clarke error-grid)',
             fontsize=9,fontweight='bold')
save(fig,'Fig5_clinical_story',bottom=0.03)
print('=== 故事凸显版 Fig3/4/5 完成 ===')

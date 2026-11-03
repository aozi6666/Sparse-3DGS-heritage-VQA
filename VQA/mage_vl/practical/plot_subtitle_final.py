"""使用服务器既有base matplotlib离线出图，不导入Mage或读取视频。"""
import os,json
from pathlib import Path
OUT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/eval_results/practical_mage_v1/17_subtitle_final_report/20261103_v1')
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
M=json.loads((OUT/'stage_metrics.json').read_text());P=json.loads((OUT/'paired_comparisons.json').read_text());C=json.loads((OUT/'test225_token_cost.json').read_text())
ST=['short219','dev180','val90','test225'];STL=['Short219\nHistorical coverage','Dev180\nDevelopment','Val90\nSupplementary','Test225\nFrozen evaluation']
ARMS=['no_sub','aligned','continuous'];LABEL=['Frames-32, no subtitles','Frame-aligned subtitles','Continuous + timestamps (2048)'];COL=['#64748b','#e8ad55','#287d8e']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def save(fig,name,note):
    fig.text(.5,.015,note+'\nSource: frozen stage summaries; local Video-MME subset; 2026-11-03.',ha='center',fontsize=8,color='#475569')
    fig.tight_layout(rect=(0,.1,1,.98))
    for suffix in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{suffix}',dpi=240,facecolor='white')
    plt.close(fig)
def bars(rows,ticks,title,name):
    fig,ax=plt.subplots(figsize=(11,5.2));x=np.arange(len(ticks));w=.24
    for j,a in enumerate(ARMS):
        vals=[next(m['accuracy_pct'] for m in M if m['stage']==s and m['duration']==d and m['method']==a) for s,d in rows]
        bs=ax.bar(x+(j-1)*w,vals,w,label=LABEL[j],color=COL[j]);ax.bar_label(bs,fmt='%.1f',fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=ticks,ylim=(0,100),ylabel='Question accuracy (%)',xlabel='Evaluation split / duration',title=title)
    ax.legend(loc='upper center',ncol=3,fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,name,'Split roles remain separate. Short219 / Dev180 / Val90 are not fresh independent tests.')
bars([(s,'overall') for s in ST],STL,'Frozen subtitle configurations: accuracy by split','01_accuracy_by_stage')
rows=[(s,d) for s in ST for d in ['short','medium','long'] if any(m['stage']==s and m['duration']==d for m in M)]
ticks=[f'{s} / {d}\nN='+str(next(m['N'] for m in M if m['stage']==s and m['duration']==d)) for s,d in rows]
bars(rows,ticks,'Accuracy by duration within each split','02_accuracy_by_duration')
fig,axs=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
for ax,base in zip(axs,ARMS[:2]):
    rr=[next(p for p in P if p['stage']==s and p['duration']=='overall' and p['baseline']==base) for s in ST]
    x=np.arange(4);ax.bar(x,[p['rescue'] for p in rr],color=COL[2],label='Rescued');ax.bar(x,[-p['harm'] for p in rr],color=COL[0],label='Harmed (negative)')
    for i,p in enumerate(rr):ax.text(i,p['rescue']+.6,f"+{p['rescue']} / net {p['net']:+}",ha='center',fontsize=8);ax.text(i,-p['harm']-.6,f"-{p['harm']}",va='top',ha='center',fontsize=8)
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=x,xticklabels=ST,xlabel='Evaluation split',title='Continuous vs '+('no subtitles' if base=='no_sub' else 'aligned subtitles'));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16)
axs[0].set_ylabel('Number of questions (rescue positive / harm negative)')
save(fig,'03_rescue_harm','Rescue = baseline wrong, continuous correct; harm = baseline correct, continuous wrong.')
fig,ax=plt.subplots(figsize=(9,5.2));x=np.arange(3);v=np.array([r['visual_tokens']/225 for r in C]);sub=np.array([r['subtitle_tokens']/225 for r in C]);total=np.array([r['input_tokens']/225 for r in C]);other=total-v-sub
ax.bar(x,v,label='Visual tokens',color=COL[0]);ax.bar(x,sub,bottom=v,label='Subtitle text tokens',color=COL[2]);ax.bar(x,other,bottom=v+sub,label='Other input tokens',color=COL[1])
for i,n in enumerate(total):ax.text(i,n+65,f'{n:,.1f}',ha='center')
ax.set(xticks=x,xticklabels=['No subtitles','Aligned subtitles','Continuous 2048'],ylabel='Mean logical input tokens / question',xlabel='Frozen configuration (Test225)',title='Test225 input cost: visual budget unchanged, total input increases',ylim=(0,max(total)*1.22));ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
save(fig,'04_test225_token_cost','Other input includes question, options and formatting. Reused fallback answers retain logical input cost.')
fig,axs=plt.subplots(1,2,figsize=(11,5.1),sharey=True)
for ax,base in zip(axs,ARMS[:2]):
    for i,s in enumerate(ST):
        p=next(p for p in P if p['stage']==s and p['duration']=='overall' and p['baseline']==base);lo,hi=p['video_cluster_bootstrap_95ci_pp'];v=p['delta_pp']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COL[2],capsize=4);ax.annotate(f'{v:+.2f} [{lo:.2f}, {hi:.2f}]',(v,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    ax.axvline(0,color=COL[0],ls='--',lw=1);ax.set(xlabel='Accuracy difference (percentage points)',title='Continuous vs '+('no subtitles' if base=='no_sub' else 'aligned subtitles'),ylim=(-.6,3.6));ax.grid(axis='x',alpha=.16)
axs[0].set(yticks=np.arange(4),yticklabels=ST,ylabel='Evaluation split')
save(fig,'05_paired_cluster_intervals','Points: paired accuracy differences. Bars: saved video-cluster bootstrap 95% intervals; no pooled-test claim.')
print('Saved 5 figures in PNG / PDF / SVG; no model, API or video access.')

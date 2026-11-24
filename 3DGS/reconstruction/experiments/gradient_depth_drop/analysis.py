"""CPU-only aggregation/plots of real training statistics; no fabricated rows."""
import csv,json
from pathlib import Path
import numpy as np

def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bins(z,g,valid,lo,hi,threshold,inclusive=False):
    if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid shared depth range')
    valid=valid & np.isfinite(z)&np.isfinite(g)&(z>0)
    z=z[valid];g=g[valid];u=np.clip((z-lo)/(hi-lo),0,1)
    idx=np.minimum(np.floor(u*10+1e-12).astype(int),9)
    exceeds=g>=threshold if inclusive else g>threshold
    return dict(count=np.bincount(idx,minlength=10).tolist(),sum=np.bincount(idx,weights=g,minlength=10).tolist(),
        above_count=np.bincount(idx[exceeds],minlength=10).tolist(),zero_count=np.bincount(idx[g==0],minlength=10).tolist(),
        clipped_low=int((z<lo).sum()),clipped_high=int((z>hi).sum()))

def plot(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);scene=root.name;limit={'horns':.7,'fortress':.3}.get(scene,1.)
    stats={m:json.loads((root/m/'statistics.json').read_text()) for m in ['3DGS','drop']}
    if stats['3DGS']['protocol']!=stats['drop']['protocol']:raise AssertionError('Protocol mismatch')
    schedule=lambda a:[(e['step'],e['camera']) for e in a['events']]
    if schedule(stats['3DGS'])!=schedule(stats['drop']):raise AssertionError('Training observation camera schedule mismatch')
    rows=[];ys=[];ratios=[];accrows=[]
    for method,record in stats.items():
        n=np.array(record['count']);s=np.array(record['sum']);above=np.array(record['above_count'])
        mean=np.divide(s,n,out=np.full(10,np.nan),where=n>0);ratio=np.divide(above,n,out=np.full(10,np.nan),where=n>0)
        ys.append(mean*1000);ratios.append(ratio)
        for i in range(10):
            rows.append(dict(method=method,views=record['protocol']['views'],bin_lower=i/10,bin_upper=(i+1)/10,count=int(n[i]),gradient_sum=float(s[i]),mean_gradient=float(mean[i]) if n[i] else 'NaN',mean_x1000=float(mean[i]*1000) if n[i] else 'NaN',threshold=record['protocol']['threshold'],above_count=int(above[i]),above_ratio=float(ratio[i]) if n[i] else 'NaN',zero_gradient_count=record['zero_count'][i]))
        for i in range(10):
            c=record['accumulator_count'][i];a=record['accumulator_above_count'][i]
            accrows.append(dict(method=method,views=record['protocol']['views'],bin_lower=i/10,bin_upper=(i+1)/10,denom_positive_count=c,accumulated_mean_above_count=a,accumulated_mean_above_ratio=a/c if c else 'NaN',threshold=record['protocol']['threshold'],comparison='>= (project clone/split)',unit='Gaussian-densification-check observation (not unique Gaussian)'))
        ev=[]
        for event in record['events']:
            for i in range(10):ev.append(dict(step=event['step'],camera=event['camera'],bin_lower=i/10,bin_upper=(i+1)/10,count=event['count'][i],sum=event['sum'][i],above_count=event['above_count'][i],zero_count=event['zero_count'][i]))
        table(root/method/'per_observation_bins.csv',ev)
    table(root/'gradient_bins.csv',rows);table(root/'densification_threshold_bins.csv',accrows)
    plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,3.8));x=(np.arange(10)+.5)/10
    ax.plot(x,ys[0],'o--',color='#287da5',label='3DGS (project baseline)',ms=5)
    ax.plot(x,ys[1],'s-',color='#ed3357',label='DropGaussian',ms=5)
    ax.fill_between(x,*ys,color='#ed3357',alpha=.1)
    # Display clipping only. All ten original 0.1 bins remain in CSV and plotted arrays.
    ax.set(xlabel=f'Normalized Distance from Camera (0 = near, {limit:g} = far)',ylabel='Average Gradient Magnitude (×10⁻³)',xlim=(0,limit))
    ax.set_xticks(np.arange(0,limit+.001,.05));ax.tick_params(axis='x',labelrotation=30)
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(root/'gradient_depth.png',dpi=600);fig.savefig(root/'gradient_depth.pdf');plt.close(fig)
    comparison=[];text=[f'# {scene} / {stats["3DGS"]["protocol"]["views"]}-view 真实统计解释','',
        '曲线只裁剪显示范围，0.1宽度全十箱保留；0.05仅为刻度。far指显示右边界，不是原归一化最远端。不同view数不混合。',
        '以下是单随机种子的观测描述。平均梯度和超过阈值比例提高不等价于重建质量提高，也不表示实际新增了相同数量的Gaussian。','',
        '| 区间 | Δ平均梯度×1000 | Δ瞬时超阈比例（百分点） | 判断 |','|---|---:|---:|---|']
    for i in range(10):
        b=rows[i];d=rows[10+i];ok=b['count']>0 and d['count']>0
        dm=ys[1][i]-ys[0][i];dr=(ratios[1][i]-ratios[0][i])*100
        judgment='无配对有效数据' if not ok else ('平均与超阈比例均升' if dm>0 and dr>0 else ('平均与超阈比例均未升' if dm<=0 and dr<=0 else '两项方向不一致'))
        if ok and min(b['count'],d['count'])<1000:judgment+='；样本较少（非独立重复观测）'
        comparison.append(dict(bin_lower=i/10,bin_upper=(i+1)/10,baseline_count=b['count'],drop_count=d['count'],mean_delta_x1000=float(dm) if ok else 'NaN',above_ratio_delta_pp=float(dr) if ok else 'NaN',description=judgment))
        text.append(f'| {i/10:.1f}–{(i+1)/10:.1f} | {dm:.6g} | {dr:.4g} | {judgment} |' if ok else f'| {i/10:.1f}–{(i+1)/10:.1f} | NaN | NaN | {judgment} |')
    text+=['','## 实际累计增密阈值条件（>=项目阈值）','这部分在实际增密检查前采样；范围按当次训练相机深度划分。重复Gaussian检查不是独立样本，也不是实际新增点数。','| 区间 | Baseline候选/观测 | Drop候选/观测 | 比例变化（百分点） |','|---|---:|---:|---:|']
    for i in range(10):
        b=accrows[i];d=accrows[10+i]
        ok=b['denom_positive_count']>0 and d['denom_positive_count']>0
        delta=(d['accumulated_mean_above_ratio']-b['accumulated_mean_above_ratio'])*100 if ok else 'NaN'
        text.append(f"| {i/10:.1f}–{(i+1)/10:.1f} | {b['accumulated_mean_above_count']}/{b['denom_positive_count']} | {d['accumulated_mean_above_count']}/{d['denom_positive_count']} | {delta} |")
        comparison[i].update(accumulator_above_ratio_delta_pp=delta,baseline_accumulator_count=b['denom_positive_count'],drop_accumulator_count=d['denom_positive_count'])
    text+=['','## 远区为空原因检查','固定范围取共同初始点云的正相机Z最小/最大值。离群初始点会拉大范围；训练可见点不必覆盖全部初始深度。不得为让曲线更好而改范围。']
    for method,r in stats.items():
        diag=r['diagnostics'];n=sum(r['count']);clip=diag['clipped_low']+diag['clipped_high']
        text.append(f'- {method}：有效可见观测{n}，范围外裁剪{clip}；逐箱全体正深度候选{diag["positive_depth_count"]}，可见筛选后{r["count"]}。')
    text+=['比较候选与可见数量可区分范围占用和可见性筛选；有点但零梯度的箱保留真实0，count=0的箱为NaN。阈值判定的主体为denom>0的累计均值，另见densification_threshold_bins.csv。']
    (root/'interpretation.md').write_text('\n'.join(text)+'\n')
    table(root/'bin_comparison.csv',comparison)
    return float(np.nanmean(np.abs(ys[1]-ys[0])))

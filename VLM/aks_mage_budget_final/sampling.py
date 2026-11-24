import numpy as np

def plans(payload, ranked, historical):
    positions=payload["src_positions"].reshape(len(payload["images"]),-1,3)[:,:,0]
    active=np.flatnonzero((positions>=0).all(axis=1))
    plans={0:{}}
    used={}
    scores={f:s for row in ranked for f,s in row}
    if len(active)>1:
        ts=positions[active]
        center=np.median(ts,axis=1)
        lo,hi=ts.min(axis=1),ts.max(axis=1)
    for k, ranks in [(4,[0]),(8,[1]),(16,[2,3]),(20,[4]),(24,[5]),(28,[6]),(32,[7])]:
        for rank in ranks:
            pairs=[]
            for row in ranked:
                if rank>=len(row) or len(active)<2:
                    continue
                frame=row[rank][0]
                dist=np.maximum(0,np.maximum(lo-frame,frame-hi))
                j=int(np.lexsort((abs(center-frame),dist))[0])
                if j:
                    pairs.append((abs(center[j]-frame),int(active[j]),frame))
            for _,slot,frame in sorted(pairs):
                if len(used)>=min(k,max(1,len(active)*k//32)):
                    break
                if slot not in used:
                    used[slot]=frame
        plans[k]=used.copy()
    assert list(plans[4].values())==historical, (plans[4],historical)
    n=min(2,max(1,len(active)*2//32))
    plans[2]=dict(sorted(plans[4].items(), key=lambda item:(-scores[item[1]],item[0]))[:n])
    return plans

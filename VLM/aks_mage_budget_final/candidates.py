from pathlib import Path
import collections
import json
import cv2

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "aks_mage"
VIDEO = HERE.parent / "mage_vl/data/VideoMME/video"
AKS = Path(HERE.parent / "vendor/AKS")
def main():
    def read(name):
        return json.loads((AKS/name).read_text())
    keys = read("datasets/videomme/videomme.json")
    scores = read("outscores/videomme/blip/scores.json")
    frames = read("outscores/videomme/blip/frames.json")
    picks = read("selected_frames/videomme/blip/videomme_32.json")
    assert len(keys)==len(scores)==len(frames)==len(picks)
    lookup = {(k["videoID"],k["question"]):i for i,k in enumerate(keys)}
    counts, output = {}, {}
    for q in json.loads((OLD/"questions.json").read_text()):
        vid=q["video"]
        if vid not in counts:
            cap=cv2.VideoCapture(str(VIDEO/(vid+".mp4")))
            counts[vid]=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        i=lookup[vid,q["question"]]
        assert keys[i]["question_id"]==q["id"]
        val=dict(zip(map(int,frames[i]),scores[i]))
        quarters=collections.defaultdict(list)
        for f in sorted(set(map(int,picks[i]))):
            quarters[4*f//counts[vid]].append(f)
        ranked=[sorted(quarters[j],key=lambda f:-val[f]) for j in range(4)]
        assert [v[0] for v in ranked if v]==q["anchors"]
        output[q["id"]]=[[[f,val[f]] for f in row] for row in ranked]
    assert len(output)==714 and len(counts)==238
    tmp=HERE/"candidates.tmp"
    tmp.write_text(json.dumps(output))
    tmp.replace(HERE/"candidates.json")
if __name__=="__main__":
    main()

"""Freeze the 714-question AKS evidence index from the supplied local release."""
import ast
import csv
import json
import os
import re
from pathlib import Path

import cv2
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "mage_vl/data/VideoMME"
AKS = Path(os.environ.get("AKS_ROOT", "/root/autodl-tmp/AKS"))
BOOK = Path("/root/autodl-tmp/VLMEvalKit/outputs/Mage-VL/Mage-VL_Video-MME-32frame_judge_extract_gpt-4o-mini_detail.xlsx")


def letter(raw):
    match = re.fullmatch(r"\s*([ABCD])(?:\s*[.)](?:\s+[^\n]*)?)?\s*", str(raw))
    return match[1] if match else None


def main():
    def read(path):
        return json.loads((AKS / path).read_text())
    labels = read("datasets/videomme/videomme.json")
    frames = read("outscores/videomme/blip/frames.json")
    scores = read("outscores/videomme/blip/scores.json")
    picks = read("selected_frames/videomme/blip/videomme_32.json")
    assert len(labels) == len(frames) == len(scores) == len(picks)
    lookup = {(r["videoID"], r["question"]): i for i, r in enumerate(labels)}
    sheet = load_workbook(BOOK, read_only=True).active.values
    columns = next(sheet)
    baseline = {r["index"]: r for r in (dict(zip(columns, values)) for values in sheet)}
    output = []
    geometry = {}
    with (DATA / "Video-MME.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            vid = row["video"]
            i = lookup[vid, row["question"]]
            options = ast.literal_eval(row["candidates"])
            old = baseline[int(row["index"])]
            assert old["question"] == row["question"] and old["video"] == vid
            assert ast.literal_eval(old["candidates"]) == labels[i]["options"] == options
            assert old["answer"] == row["answer"] == labels[i]["answer"]
            if vid not in geometry:
                cap = cv2.VideoCapture(str(DATA / "video" / (vid + ".mp4")))
                geometry[vid] = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
                cap.release()
            count, fps = geometry[vid]
            ids = list(map(int, frames[i]))
            assert len(ids) == len(scores[i]) and len(ids) > 0
            step = ids[1] if len(ids) > 1 else max(1, int(fps))
            if ids != list(range(0, len(ids)*step, step)) or abs(count - len(ids)*step) > max(step*2, count*.005) or abs(step-fps)>1.1:
                raise ValueError(f"AKS timeline mismatch: {vid}, {count=}, {fps=}, {step=}, {len(ids)=}")
            values = dict(zip(ids, scores[i]))
            chosen = sorted(set(map(int, picks[i])))
            assert chosen and all(n in values and n < count for n in chosen)
            # Limit semantic seats to one per temporal quarter. Empty quarters retain motion.
            anchors = []
            for quarter in range(4):
                pool = [n for n in chosen if n*4//count == quarter]
                if pool:
                    anchors.append(max(pool, key=values.get))
            output.append(dict(id=labels[i]["question_id"], video=vid, question=row["question"],
                               options=options, answer=row["answer"], duration=row["duration"],
                               anchors=anchors, baseline=int(old["score"]), old_raw=old["prediction"]))
    assert len(output) == len(baseline) == len({r["id"] for r in output}) == 714
    assert len(geometry) == 238
    tmp = HERE / "questions.tmp"
    tmp.write_text(json.dumps(output, ensure_ascii=False))
    tmp.replace(HERE / "questions.json")
    print(len(output), "questions;", len(geometry), "videos; baseline correct", sum(r["baseline"] for r in output))

if __name__ == "__main__":
    main()

import cv2
import numpy as np
from PIL import Image


def fuse(payload, cap, anchors):
    images = list(payload["images"])
    pos = payload["src_positions"].reshape(len(images), -1, 3).copy()
    ts = pos[:, :, 0]
    active = np.flatnonzero(np.all(ts >= 0, axis=1))
    if len(active) < 2:
        return payload, [], []
    centers = np.median(ts[active], axis=1)
    start = ts[active].min(axis=1)
    end = ts[active].max(axis=1)
    # Match each candidate to its closest canvas. Protect the first global view,
    # and let the closest candidate win a collision instead of displacing distant motion.
    pairs = []
    for frame in anchors:
        distance = np.maximum(0, np.maximum(start - frame, frame - end))
        j = int(np.lexsort((np.abs(centers - frame), distance))[0])
        if j:
            pairs.append((abs(centers[j]-frame), int(active[j]), frame))
    used = {}
    pairs_log = []
    for _, slot, frame in sorted(pairs):
        if slot in used or len(used) >= max(1, len(active)//8):
            continue
        aks_frame = frame
        frame = int((int(ts[slot].min()) + int(ts[slot].max())) // 2)
        pairs_log.append(dict(slot=slot, aks=aks_frame, midpoint=frame, gap_seconds=abs(frame-aks_frame)/payload["fps"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > 1:
            raise RuntimeError(f"Cannot seek to AKS frame {frame}")
        w, h = images[slot].size
        rh, rw = payload["meta"]["resize_hw"]
        if rw > w or rh > h or w % 32 or h % 32:
            raise ValueError(f"Unexpected Mage geometry: {(w,h,rw,rh)}")
        rgb = cv2.cvtColor(cv2.resize(bgr, (rw,rh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        canvas = Image.new("RGB", (w,h))
        canvas.paste(Image.fromarray(rgb))
        images[slot] = canvas
        y, x = np.indices((h//16,w//16))
        pos[slot] = np.column_stack((np.full(y.size,frame), y.ravel(), x.ravel()))
        used[slot] = frame
    return {**payload, "images":images, "src_positions":pos.reshape(-1,3)}, list(used.values()), pairs_log



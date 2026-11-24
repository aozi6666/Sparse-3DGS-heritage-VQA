import cv2
import numpy as np
from PIL import Image


from time import perf_counter


def render(payload, cap, layout):
    start = perf_counter()
    decode = 0.0
    images = list(payload["images"])
    pos = payload["src_positions"].reshape(len(images), -1, 3).copy()
    for slot, frame in layout.items():
        tic = perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > 1:
            raise RuntimeError(f"Cannot seek to AKS frame {frame}")
        decode += perf_counter()-tic
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
    return {**payload, "images": images, "src_positions": pos.reshape(-1,3)}, decode, perf_counter()-start-decode



"""CPU photogrammetry preview. Install pycolmap==3.11.1 in storage/reconstruction-runtime."""
from pathlib import Path
import sys
import cv2
import numpy as np
import pycolmap as col

def reconstruct(folder):
    images = folder / 'images'
    images.mkdir()
    cv2.setNumThreads(1)
    def save(frame, index):
        if frame is None:
            raise ValueError('无法读取照片')
        scale = min(1, 1400 / max(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images / f'{index:03}.jpg'), frame)
    sources = sorted(folder.glob('input-*'))
    if len(sources) == 1:
        cap = cv2.VideoCapture(str(sources[0]))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i, pos in enumerate(np.linspace(0,max(0,count-1),min(count,24),dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos))
            ok, frame = cap.read()
            if ok: save(frame,i)
        cap.release()
    else:
        for i, path in enumerate(sources):
            save(cv2.imread(str(path)),i)
    if len(list(images.iterdir())) < 3:
        raise ValueError('至少需要三个有效视角')
    db = str(folder / 'features.db')
    extract = col.SiftExtractionOptions()
    extract.num_threads = 1
    extract.max_num_features = 4096
    col.extract_features(db,str(images),sift_options=extract,device=col.Device.cpu)
    match = col.SiftMatchingOptions()
    match.num_threads = 1
    col.match_exhaustive(db,sift_options=match,device=col.Device.cpu)
    options = col.IncrementalPipelineOptions()
    options.num_threads = 1
    options.min_model_size = 3
    models = col.incremental_mapping(db,str(images),str(folder/'maps'),options)
    if not models:
        raise ValueError('视角间匹配不足')
    model = max(models.values(),key=lambda m:m.num_reg_images())
    points = list(model.points3D.values())
    if len(points)<32:
        raise ValueError('有效三维点过少')
    xyz = np.array([p.xyz for p in points],dtype='<f4')
    rgb = np.array([p.color for p in points],dtype='u1')
    # Trim isolated far points before choosing the display scale.
    center = np.median(xyz,axis=0)
    radii = np.linalg.norm(xyz-center,axis=1)
    keep = radii <= np.quantile(radii,.98)
    xyz, rgb = xyz[keep]-center, rgb[keep]
    size = max(float(np.linalg.norm(np.ptp(xyz,axis=0))) / np.sqrt(len(xyz)) * .32,1e-6)
    records = np.zeros(len(xyz),dtype=[('xyz','<f4',3),('scale','<f4',3),('rgba','u1',4),('rotation','u1',4)])
    records['xyz'],records['scale'] = xyz,size
    records['rgba'][:,:3],records['rgba'][:,3] = rgb,230
    records['rotation'] = [255,128,128,128]
    records.tofile(folder/'object.splat')

if __name__ == '__main__':
    reconstruct(Path(sys.argv[1]))

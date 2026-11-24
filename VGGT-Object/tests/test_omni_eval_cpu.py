import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")
import unittest,tempfile,json,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image
script=Path(__file__).resolve().parents[1]/"tools/omni_depth_eval.py"
spec=importlib.util.spec_from_file_location("omni_test_entry",script)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

class PlanTests(unittest.TestCase):
 def fixture(self,d):
  raw=Path(d)/"raw";root=raw/"fixture"/"fixture_object"/"render"
  (root/"images").mkdir(parents=True);(root/"depths").mkdir()
  frames=[]
  for i in range(24):
   rgba=np.full((16,16,4),255,dtype=np.uint8)
   Image.fromarray(rgba).save(root/"images"/f"r_{i}.png")
   depth=np.full((16,16),3.,dtype=np.float32);depth[0,0]=65504.
   self.assertTrue(cv2.imwrite(str(root/"depths"/f"r_{i}_depth.exr"),depth))
   frames.append(dict(file_path=f"r_{i}",transform_matrix=np.eye(4).tolist()))
  (root/"transforms.json").write_text(json.dumps(dict(camera_angle_x=.6911,frames=frames)))
  return raw,root
 def test_valid_plan_no_media_copy_and_frozen_hashes(self):
  with tempfile.TemporaryDirectory() as d:
   raw,root=self.fixture(d)
   a=SimpleNamespace(raw=raw,views=24,objects=1,input_views=4,resolution=392,depth_type="camera_z_depth",plan=Path(d)/"index/plan.json")
   entry.prepare(a);plan=json.loads(a.plan.read_text())
   self.assertEqual(len(plan["objects"]),1);self.assertEqual(len(plan["objects"][0]["files"]),49)
   self.assertEqual(plan["objects"][0]["frame_ids"],list(range(24)))
   self.assertEqual(list(a.plan.parent.iterdir()),[a.plan])
   with self.assertRaises(RuntimeError):entry.prepare(a)
 def test_incomplete_upload_cannot_publish_plan(self):
  with tempfile.TemporaryDirectory() as d:
   raw,root=self.fixture(d);(root/"images/r_11.png").write_bytes(b"")
   a=SimpleNamespace(raw=raw,views=24,objects=1,input_views=4,resolution=392,depth_type="camera_z_depth",plan=Path(d)/"plan.json")
   with self.assertRaises(RuntimeError):entry.prepare(a)
   self.assertFalse(a.plan.exists())
 def test_real_cpu_exr_filter_masks_finite_sentinel(self):
  with tempfile.TemporaryDirectory() as d:
   raw,root=self.fixture(d)
   meta=json.loads((root/"transforms.json").read_text())
   _,depth,mask,K,_,_=entry.read_view(root,meta["frames"][1],meta["camera_angle_x"])
   self.assertEqual(int(mask.sum()),255);self.assertFalse(mask[0,0]);self.assertEqual(depth[0,0],0.)
if __name__=="__main__":unittest.main()

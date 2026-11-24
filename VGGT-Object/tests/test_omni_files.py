import tempfile,unittest,json
from pathlib import Path
import numpy as np
from vggt_object.omni_files import exact_depth,valid_depth
class OmniFiles(unittest.TestCase):
 def test_exact_prefix_no_cross_view(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'depths';p.mkdir()
   for n in ['r_1_depth.exr','r_10_depth.exr','r_11_depth.exr']:(p/n).write_bytes(b'header')
   self.assertEqual(exact_depth(Path(d),'r_1').name,'r_1_depth.exr')
 def test_no_fuzzy_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'depths';p.mkdir();(p/'r_10_depth.exr').write_bytes(b'header')
   with self.assertRaises(ValueError):exact_depth(Path(d),'r_1')
 def test_empty_and_ambiguous(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'depths';p.mkdir();(p/'r_1_depth.exr').touch()
   with self.assertRaises(ValueError):exact_depth(Path(d),'r_1')
   (p/'r_1_depth.exr').write_bytes(b'header');(p/'r_1.npy').write_bytes(b'header')
   with self.assertRaises(ValueError):exact_depth(Path(d),'r_1')
 def test_background_before_ray_conversion(self):
  a=np.array([0.,-1.,float('nan'),float('inf'),65504.,65505.,3.])
  self.assertEqual(valid_depth(a).tolist(),[False]*6+[True])
  filtered=np.where(valid_depth(a),a,0)/1.2
  self.assertEqual(filtered[-3],0.);self.assertEqual(filtered[-1],2.5)
 def test_conversion_and_projection_math(self):
  K=np.array([[400.,0,400],[0,400.,400],[0,0,1.]])
  rays=np.array([[400,400,1],[600,400,1.]])@np.linalg.inv(K).T
  distance=np.linalg.norm(rays,axis=-1)*3.
  np.testing.assert_allclose(distance/np.linalg.norm(rays,axis=-1),3.)
if __name__=='__main__':unittest.main()

"""验证扩样零重叠、原模块恢复和未完成禁止读标签。"""
import unittest
from pathlib import Path
from unittest.mock import patch
import air_extension as e

class Extension(unittest.TestCase):
 def rows(self):return [dict(index=str(i),video_id=str(i//3)) for i in range(180)]
 def test_disjoint90(self):
  pool=self.rows();r=e.remaining(pool,pool[:90]);self.assertEqual(len(r),90);self.assertEqual(r[0]['index'],'90')
 def test_video_overlap_stops(self):
  pool=self.rows();pool[90]['video_id']='0'
  with self.assertRaises(ValueError):e.remaining(pool,pool[:90])
 def test_short_split_stops(self):
  pool=self.rows()
  with self.assertRaises(ValueError):e.remaining(pool[:-1],pool[:90])
 def test_scoped_redirect_restored_even_after_exception(self):
  context=e.p.context;out=e.h.OUT;source={'test':1}
  with self.assertRaises(RuntimeError):
   with e.hybrid_scope(source):
    self.assertIs(e.p.context('dev30'),source);self.assertNotEqual(e.h.OUT,out);raise RuntimeError('interrupted')
  self.assertIs(e.p.context,context);self.assertEqual(e.h.OUT,out)
 def test_partial_summary_no_GT(self):
  with patch.object(Path,'read_text',side_effect=AssertionError('no GT')):
   with self.assertRaises(ValueError):e.summarize({},dict(units={}))

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from analysis import bins
class BinTests(unittest.TestCase):
    def test_edges_and_first_bin(self):
        r=bins(np.arange(11)/10,np.ones(11),np.ones(11,dtype=bool),0,1,.5)
        self.assertEqual(r['count'],[0,1,1,1,1,1,1,1,1,2]) # Z=0 excluded
        a=bins(np.array([.01,.09]),np.array([0.,1.]),np.ones(2,dtype=bool),0,1,.5)
        self.assertEqual(a['count'][0],2);self.assertEqual(a['zero_count'][0],1)
    def test_empty_and_visibility(self):
        r=bins(np.array([.15,.25]),np.ones(2),np.zeros(2,dtype=bool),0,1,.5)
        self.assertEqual(sum(r['count']),0)
    def test_threshold_strict_and_project_inclusive(self):
        z=np.array([.1,.1,.1]);g=np.array([.0001,.0002,.0003]);v=np.ones(3,dtype=bool)
        self.assertEqual(sum(bins(z,g,v,0,1,.0002)['above_count']),1)
        self.assertEqual(sum(bins(z,g,v,0,1,.0002,True)['above_count']),2)
    def test_clipping_and_finite(self):
        r=bins(np.array([.01,2.,3.,np.nan,-1.]),np.array([1.,2.,np.nan,1.,1.]),np.ones(5,dtype=bool),.1,1.,.5)
        self.assertEqual(sum(r['count']),2);self.assertEqual(r['clipped_low'],1);self.assertEqual(r['clipped_high'],1)
    def test_bad_bounds(self):
        with self.assertRaises(ValueError):bins(np.ones(1),np.ones(1),np.ones(1,dtype=bool),1,1,.5)
if __name__=='__main__':unittest.main()

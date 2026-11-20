"""Critical reproducibility/resume/selection checks; no torch or GPU."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import protocol as p


class ProtocolTests(unittest.TestCase):
    def test_uniform_independent_of_scores(self):
        self.assertEqual(p.uniform(list(range(16))), [2,6,10,14])

    def test_uniform_reject_duplicates(self):
        with self.assertRaises(ValueError):p.uniform([1,1,2,3,4])

    def test_ties_deterministic(self):
        ids=list(range(0,100,10))
        self.assertEqual(p.semantic(ids,[.5]*10,100),[0,10,20,30])

    def test_separation_and_fill(self):
        self.assertEqual(len(p.semantic([0,1,2,3],[4.,3.,2.,1.],10000)),4)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):p.semantic([0,1,2,3],[1.,float('nan'),1.,1.],10)

    def test_seal_detects_corruption(self):
        x=p.seal({'a':1});x['a']=2
        with self.assertRaises(ValueError):p.unseal(x)

    def test_atomic_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'unit.json';p.atomic(f,{'a':1});p.atomic(f,{'a':1})
            with self.assertRaises(ValueError):p.atomic(f,{'a':2})
            self.assertEqual(p.read(f),{'a':1})

    def test_resume_rejects_wrong_identity(self):
        from experiment import checkpoint
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'unit.json';p.atomic(f,p.seal({'signature':'x','index':'1'}))
            with self.assertRaises(ValueError):checkpoint(f,'y',index='1')
            with self.assertRaises(ValueError):checkpoint(f,'x',index='2')

    def test_query_preserves_all_options(self):
        r={'question':'Why?', 'bodies':[' one',' two',' three',' four']}
        self.assertEqual(p.query(r),'Why?\nA. one\nB. two\nC. three\nD. four')

    def test_gpu_requires_explicit_flag(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError):p.gpu_guard()

    def test_gpu_busy_does_not_launch(self):
        class Response:stdout='8524, 13000\n'
        with patch.dict('os.environ',{'CODEC_EVIDENCE_GPU_CONFIRMED':'1'}),patch('subprocess.run',return_value=Response()):
            with self.assertRaises(ValueError):p.gpu_guard()

    def test_video_expansion_includes_initial_timestamp(self):
        from mixed_inputs import video_expansion, VS, VE, VP, IP
        raw='prefix'+VS+VP+VE+'suffix'
        insertion='<0.0 seconds>'+VS+IP*3+VE+'\n'
        self.assertEqual(video_expansion(raw,'prefix'+insertion+'suffix'),insertion)

    def test_video_expansion_preserves_trailing_text(self):
        from mixed_inputs import video_expansion, VS, VE, VP
        raw='p'+VS+VP+VE+'\nanswer'
        self.assertEqual(video_expansion(raw,'p'+'timestamp\n'+'answer'),'timestamp\n')

    def test_video_expansion_rejects_multiple_videos(self):
        from mixed_inputs import video_expansion, VS, VE, VP
        with self.assertRaises(ValueError):video_expansion((VS+VP+VE)*2,'anything')


if __name__=='__main__':unittest.main(verbosity=2)

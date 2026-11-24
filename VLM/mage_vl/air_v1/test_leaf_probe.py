"""纯CPU数值与模拟解码断点回归；不读真实视频、不加载模型、不调用API。"""
import json
from pathlib import Path
import tempfile
import unittest
import builtins
from unittest.mock import patch
import numpy as np

import air_protocol as p
import leaf_probe as k
import leaf_probe_run as r
import leaf_probe_media as media


def fixture(events=2):
    rng=np.random.default_rng(11)
    f=rng.normal(size=(301,768)).astype(np.float32)
    v=np.eye(768,dtype=np.float32)[:events]
    return dict(indices=list(range(0,9001,30)),features=f,vectors=v,fps=30.,count=9001,
                event_ids=[str(i) for i in range(events)],edges=[],relation='independent',anchors=np.linspace(0,9000,8).round().astype(int).tolist())


def pair_fixture(a,b):
    d=fixture();f=np.zeros((301,768),dtype=np.float32);f[:,767]=1
    for pos,col in [(a,0),(b,1)]:f[pos,col]=.9;f[pos,767]=.4358899
    d['features']=f;return d


class Numeric(unittest.TestCase):
    def test_budget_and_disjoint_windows(self):
        d=fixture(4);u,raw,rel=k.locate(**d)
        self.assertLessEqual(len(u['indices']),32);self.assertEqual(len(u['windows']),6)
        self.assertTrue(set(d['anchors'])<=set(u['indices']))
        for a in u['windows']:
            self.assertEqual(len(a['indices']),4)
            for b in u['windows']:
                if a is not b:self.assertLessEqual(min(a['end'],b['end'])-max(a['start'],b['start']),1e-8)
        self.assertEqual(raw.shape,(301,4));self.assertIsNone(u['semantic_grounding_pass'])

    def test_each_leaf_assigned(self):
        u,_,_=k.locate(**fixture(4))
        self.assertEqual([x['event_id'] for x in u['leaves']],['0','1','2','3'])
        self.assertTrue(all(x['assigned_window']<6 for x in u['leaves']))

    def test_top3_temporally_diverse(self):
        u,_,_=k.locate(**fixture())
        for leaf in u['leaves']:
            times=[x['second'] for x in leaf['raw_top3']]
            self.assertEqual(len(times),3)
            self.assertTrue(all(abs(a-b)>=12 for n,a in enumerate(times) for b in times[n+1:]))

    def test_unknown_order_keeps_reversed_raw_event_peaks(self):
        d=pair_fixture(200,60);d['relation']='unknown_temporal_order'
        u,_,_=k.locate(**d)
        self.assertGreater(u['leaves'][0]['assigned_hit']['frame'],u['leaves'][1]['assigned_hit']['frame'])
        self.assertEqual(u['numeric_edges'],[])

    def test_known_before_enforced_with_original_event_ids(self):
        d=pair_fixture(200,60);d['relation']='known_before_after';d['edges']=[dict(earlier='1',later='0')]
        u,_,_=k.locate(**d);edge=u['numeric_edges'][0]
        self.assertLess(edge['earlier_frame'],edge['later_frame']);self.assertEqual(edge['earlier'],'1');self.assertIsNone(edge['semantic_match'])

    def test_known_pair_repairs_inconsistent_independent_peaks(self):
        d=pair_fixture(200,60);d['relation']='known_before_after';d['edges']=[dict(earlier='0',later='1')]
        u,_,_=k.locate(**d)
        self.assertLess(u['leaves'][0]['assigned_hit']['frame'],u['leaves'][1]['assigned_hit']['frame'])
        self.assertEqual(u['leaves'][0]['raw_top3'][0]['frame'],6000)

    def test_close_known_events_share_window(self):
        d=pair_fixture(100,105);d['relation']='known_before_after';d['edges']=[dict(earlier='0',later='1')]
        u,_,_=k.locate(**d)
        self.assertEqual(u['leaves'][0]['assigned_window'],u['leaves'][1]['assigned_window'])

    def test_missing_given_edge_rejected(self):
        d=fixture();d['relation']='known_before_after'
        with self.assertRaises(ValueError):k.locate(**d)

    def test_edge_not_allowed_for_unknown_order(self):
        d=fixture();d.update(relation='unknown_temporal_order',edges=[dict(earlier='0',later='1')])
        with self.assertRaises(ValueError):k.locate(**d)

    def test_numeric_range_excludes_outside_peak(self):
        d=fixture();d['features'][250,0]=100;d['features'][200,1]=100;d['bounds']=[30,150];d['anchors']=np.linspace(900,4500,8).round().astype(int).tolist()
        u,_,_=k.locate(**d)
        self.assertTrue(all(900<=i<=4500 for i in u['indices']))
        self.assertTrue(all(30<=h['second']<=150 for x in u['leaves'] for h in x['raw_top3']))

    def test_range_anchor_outside_rejected(self):
        d=fixture();d['bounds']=[30,150]
        with self.assertRaises(ValueError):k.locate(**d)

    def test_not_enough_range_stops_without_inventing_windows(self):
        d=fixture();d['bounds']=[0,60];d['anchors']=np.linspace(0,1800,8).round().astype(int).tolist()
        with self.assertRaises(k.Unavailable):k.locate(**d)

    def test_outside_range_stops(self):
        with self.assertRaises(k.Unavailable):k.time_bounds(30.,9001,[500,600])

    def test_spatial_probe_is_not_order_claim(self):
        d=fixture(4);d['relation']='spatial_order';u,_,_=k.locate(**d)
        self.assertIsNotNone(u['joint_scene_probe']);self.assertEqual(u['numeric_edges'],[]);self.assertFalse(u['QA_selection_committed'])

    def test_flat_curve_explicitly_unavailable(self):
        d=fixture();d['features'][:]=1
        with self.assertRaises(k.Unavailable):k.locate(**d)

    def test_nonfinite_and_zero_features_rejected(self):
        for value in [0,np.nan,np.inf]:
            d=fixture();d['features'][:]=value
            with self.assertRaises(ValueError):k.locate(**d)

    def test_duplicate_indices_rejected(self):
        d=fixture();d['indices'][1]=0
        with self.assertRaises(ValueError):k.locate(**d)

    def test_duplicate_event_ids_rejected(self):
        d=fixture();d['event_ids']=['a','a']
        with self.assertRaises(ValueError):k.locate(**d)

    def test_invalid_fps_rejected(self):
        for fps in [0,float('nan'),-1]:
            with self.assertRaises(ValueError):k.time_bounds(fps,9001,None)

    def test_argmax_ties_deterministic(self):
        d=fixture();a=k.locate(**d);b=k.locate(**d)
        self.assertEqual(a[0],b[0]);self.assertTrue(np.array_equal(a[1],b[1]))


class Persistence(unittest.TestCase):
    def test_atomic_save_reuse_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data'
            k.atomic_bytes(path,b'one');k.atomic_bytes(path,b'one')
            with self.assertRaises(ValueError):k.atomic_bytes(path,b'two')
            self.assertEqual(path.read_bytes(),b'one')
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_fallback_never_reads_features_or_queries(self):
        c=dict(signature='s',protocol=dict(groups={'fallback':['1']}),meta={'1':{}},plans={'1':dict(indices=[1,2,3])})
        with patch.object(r,'feature',side_effect=AssertionError('must not encode/read fallback')):
            u=r.calculate(c,'1')
        self.assertEqual(u['original_noobs_indices'],[1,2,3]);self.assertIsNone(u['result'])

    def test_location_resume_recomputes_and_reuses_curves(self):
        import air_selection
        d=fixture(1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(air_selection,'uniform_indices',return_value=d['anchors']):
            fp=Path(tmp)/'source.npz';fp.write_bytes(b'fixture')
            c=dict(signature='s',protocol=dict(groups={'candidate':['1']},queries={'1':[dict(event_id='0',text='fixture query')]}),meta={'1':dict(fps=30.,frame_count=9001)},plans={'1':dict(indices=[1,2,3])},features={'1':str(fp)},parent=dict(source=dict(fixed={'1':dict(given_edges=[],relation='independent')})))
            text=dict(vector=d['vectors'][0].tolist());tp=r.text_path('fixture query');tp.parent.mkdir();tp.write_text('{}')
            with patch.object(r,'feature',return_value=(d['indices'],d['features'],1.)),patch.object(r,'read_text',return_value=text):
                a=r.calculate(c,'1');sha=p.sha(Path(tmp)/'curves/1.npz');b=r.calculate(c,'1')
                p.write_once(Path(tmp)/'locations/1.json',p.seal(a))
                self.assertEqual(r.read_location(c,'1'),p.seal(a))
            self.assertEqual(a,b);self.assertEqual(sha,p.sha(Path(tmp)/'curves/1.npz'))

    def test_corrupt_text_vector_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            path=r.text_path('q');p.write_once(path,p.seal(dict(signature='s',text='q',vector=[0]*768,token_chunks=[[1,2,3]],origin='new_local_clip')))
            with self.assertRaises(ValueError):r.read_text(dict(signature='s',protocol={'reusable_text':{}}),'q')

    def test_existing_text_reuse_requires_exact_vector_and_origin(self):
        v=np.eye(768)[0].tolist();old=dict(vector=v,token_chunks=[[1,2,3]],path='source',sha256='sha',row=0,historical_file_wall_s=1.)
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            path=r.text_path('q');p.write_once(path,p.seal(dict(signature='s',text='q',vector=v,token_chunks=old['token_chunks'],origin='new_local_clip')))
            with self.assertRaises(ValueError):r.read_text(dict(signature='s',protocol={'reusable_text':{'q':old}}),'q')

    def test_exact_reuse_and_completed_encode_do_not_import_models(self):
        old=dict(vector=np.eye(768)[0].tolist(),token_chunks=[[1,2,3]],path='fixture',sha256='sha',row=0,historical_file_wall_s=1.)
        c=dict(signature='s',protocol=dict(texts=['q'],reusable_text={'q':old}))
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ['torch','transformers']:raise AssertionError('must skip model import')
            return original_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(p,'disk_guard',return_value={}),patch('builtins.__import__',side_effect=guarded):
            r.encode(c);first=r.read_text(c,'q');r.encode(c)
            self.assertEqual(first,r.read_text(c,'q'));self.assertEqual(first['encoding_wall_s'],0.)

    def test_source_signature_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            p.write_once(r.text_path('q'),p.seal(dict(signature='wrong',text='q',vector=np.eye(768)[0].tolist(),token_chunks=[[1,2,3]],origin='new_local_clip')))
            with self.assertRaises(ValueError):r.read_text(dict(signature='s',protocol={'reusable_text':{}}),'q')

    def test_invalid_time_dimension_rejected(self):
        d=fixture();d['features']=d['features'][:-1]
        with self.assertRaises(ValueError):k.locate(**d)


class Media(unittest.TestCase):
    def test_large_pts_drift_stops_before_saving_preview(self):
        import decord
        d=fixture(1);result=k.locate(**d)[0]
        class Batch:
            def __init__(self,ids):self.ids=ids
            def asnumpy(self):return np.zeros((len(self.ids),8,8,3),dtype=np.uint8)
        class Reader:
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 9001
            def get_avg_fps(self):return 30.
            def get_batch(self,ids):return Batch(ids)
            def get_frame_timestamp(self,ids):return np.asarray([[x/30+1,(x+1)/30+1] for x in ids])
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(decord,'VideoReader',Reader),patch.object(p,'disk_guard',return_value={}):
            loc=dict(original_noobs_indices=d['anchors'],result=result)
            c=dict(signature='s',protocol=dict(indices=['1'],groups={'fallback':[]}),meta={'1':dict(fps=30.,frame_count=9001)},rows={'1':dict(video_path='NOT_REAL')})
            with patch.object(r,'read_location',return_value=loc),self.assertRaisesRegex(ValueError,'PTS'):
                media.render(c)
            self.assertFalse(media.preview_path('1').exists())

    def test_mock_decode_pts_and_interrupted_preview_resume(self):
        import decord
        from PIL import Image
        d=fixture(1);result=k.locate(**d)[0]
        counter=[0]
        class Batch:
            def __init__(self,ids):self.ids=ids
            def asnumpy(self):return np.asarray([np.full((16,24,3),x%255,dtype=np.uint8) for x in self.ids])
        class Reader:
            def __init__(self,*args,**kwargs):counter[0]+=1
            def __len__(self):return 9001
            def get_avg_fps(self):return 30.
            def get_batch(self,ids):return Batch(ids)
            def get_frame_timestamp(self,ids):return np.asarray([[x/30,(x+1)/30] for x in ids])
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(decord,'VideoReader',Reader),patch.object(p,'disk_guard',return_value={}):
            loc=dict(original_noobs_indices=d['anchors'],result=result)
            p.write_once(Path(tmp)/'locations/1.json',{'fixture':'only'})
            c=dict(signature='s',protocol=dict(indices=['1'],groups={'fallback':[]}),meta={'1':dict(fps=30.,frame_count=9001)},rows={'1':dict(video_path='NOT_A_REAL_VIDEO')})
            with patch.object(r,'read_location',return_value=loc):
                media.render(c);first=media.read_preview(c,'1',loc);media.render(c);self.assertEqual(counter[0],1)
                record=media.preview_path('1');record.rename(record.with_name('saved_fixture_record.json'))
                media.render(c);self.assertEqual(counter[0],2);second=media.read_preview(c,'1',loc)
                self.assertEqual(first['images'],second['images'])
                image=Path(next(iter(second['images'])));image.write_bytes(b'corrupt fixture')
                with self.assertRaises(ValueError):media.read_preview(c,'1',loc)


if __name__=='__main__':unittest.main()

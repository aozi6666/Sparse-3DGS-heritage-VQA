"""完整离线统计契约的临时合成记录，不作为实验数据。"""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import unittest
import numpy as np
import air_protocol as p
import air_analyze as a
from air_selection import AIRConfig

class CompleteSummary(unittest.TestCase):
    def build(self,root,invalid_score=False):
        rows=[dict(index=str(i),video_id='v',duration='medium',question='Q?',bodies=[' a',' b',' c',' d']) for i in range(3)]
        store=root/'cache';stage=root/'smoke3';units={};labels={r['index']:'B' for r in rows}
        for row in rows:
            idx=row['index'];raw='invalid score' if invalid_score else 'Score: 5\nReasoning: clear evidence'
            score=None if invalid_score else 5
            s=p.seal(dict(signature='sig',index=idx,analyzed_indices=[100],final_indices=[100],fallback_reason=None,
                rounds=[dict(score_records={'100':dict(score=score,raw_text=raw)})],observed_similarity={'100':.5}))
            p.write_once(store/'selections'/f'{idx}.json',s)
            p.write_once(store/'scores'/idx/'100.json',p.seal(dict(signature='sig',index=idx,frame_index=100,
                prompt_sha256=p.digest(p.ANALYSIS.format(query=p.query(row))),raw_text=raw,score=score,input_tokens=200,
                visual=dict(n_tok=144),generated_tokens=10,inference_s=.5,preprocess_s=.1,peak_gib=11.)))
            qpath=store/'clip_text_features'/f'{idx}.npz';qpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez(qpath,wall_s=np.asarray(.01))
            for arm in p.ARMS:
                u=p.seal(dict(index=idx,arm=arm,pred='A' if arm=='uniform32' else 'B',raw_text='A' if arm=='uniform32' else 'B',
                    indices=[0,100] if arm=='uniform32' else [100],visual=dict(n_tok=288 if arm=='uniform32' else 144),
                    input_tokens=400,new_calls=1,inference_s=.1,preprocess_s=.2,peak_gib=11.))
                units[f'{idx}:{arm}']=u;p.append(store/'units.jsonl',u)
        f=store/'clip_features/v/coarse.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez(f,indices=np.asarray([100]),wall_s=np.asarray(.05))
        old=root/'old';p.write_once(old/'01_manifests/evaluation_labels.json',labels)
        p.write_once(old/'01_manifests/dataset_lock.json',dict(labels_sha256=p.sha(old/'01_manifests/evaluation_labels.json')))
        ctx=dict(rows=rows,units=units,store=store,stage_dir=stage,signature='sig',protocol=dict(method='test'),expected=set(units))
        return ctx,old
    def test_complete_costs_and_single_video_not_significance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,old=self.build(root)
            with patch.object(a,'ROOT',root),patch.object(a,'OLD',old),patch('builtins.print'):a.summarize(ctx)
            s=json.loads((ctx['stage_dir']/'summary.json').read_text())
            self.assertTrue(s['audit']['passed']);self.assertEqual(s['overall']['paired']['net'],3)
            self.assertIsNone(s['overall']['paired']['video_cluster_bootstrap_95ci_pp'])
            self.assertEqual(s['cost']['analysis']['calls'],3);self.assertEqual(s['cost']['analysis']['input_tokens'],600)
            self.assertEqual(s['cost']['clip_text']['query_feature_files'],3)
            self.assertEqual(len(list((root/'archive').glob('*.json'))),1)
    def test_invalid_scoring_retained_and_audit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,old=self.build(root,invalid_score=True)
            with patch.object(a,'ROOT',root),patch.object(a,'OLD',old),patch('builtins.print'):
                with self.assertRaises(ValueError):a.summarize(ctx)
            s=json.loads((ctx['stage_dir']/'summary.json').read_text())
            self.assertFalse(s['audit']['passed']);self.assertEqual(s['audit']['invalid_scores'],3)
            self.assertEqual(s['N'],3)
    def test_later_stage_append_does_not_change_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,old=self.build(root)
            with patch.object(a,'ROOT',root),patch.object(a,'OLD',old),patch('builtins.print'):
                a.summarize(ctx)
                original=(ctx['stage_dir']/'summary.json').read_bytes()
                archive=next((root/'archive').glob('*.json'));original_archive=archive.read_bytes()
                # 模拟以后新增其他题的账本与共享特征，不能污染本阶段。
                p.append(ctx['store']/'units.jsonl',p.seal(dict(index='later',arm='uniform32')))
                np.savez(ctx['store']/'clip_features/v/extra_999.npz',wall_s=np.asarray(999.))
                a.summarize(ctx)
            self.assertEqual(original,(ctx['stage_dir']/'summary.json').read_bytes())
            self.assertEqual(original_archive,archive.read_bytes())
            self.assertEqual(len(json.loads((ctx['stage_dir']/'units_snapshot.json').read_text())),6)

if __name__=='__main__':unittest.main()

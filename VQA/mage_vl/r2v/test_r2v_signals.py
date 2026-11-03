#!/usr/bin/env python3
"""纯合成分数的单元测试；不导入 Torch、不加载模型、不启动推理。"""
import math
import unittest

from r2v_signal_score import features, logadd, normalize, token_probe
from r2v_signal_analyze import auc, choose_frames, diagnose


class Tests(unittest.TestCase):
    def test_softmax_shift_and_invalid(self):
        self.assertEqual(normalize([0,0,0,0]),[0.25]*4)
        a,b = normalize([1,2,3,4]),normalize([1001,1002,1003,1004])
        for x,y in zip(a,b):
            self.assertAlmostEqual(x,y)
        with self.assertRaises(ValueError):
            normalize([0,0,0,float('nan')])

    def test_space_variants_sum_mass(self):
        self.assertAlmostEqual(logadd(math.log(2),math.log(3)),math.log(5))
        self.assertAlmostEqual(sum(normalize([logadd(0,0)]*4)),1)

    def test_margin_wrong_argmax_stays_signed(self):
        value = features([0,4,0,0],'A')
        self.assertEqual(value['top1'],'B')
        self.assertLess(value['answer_margin'],0)
        self.assertGreater(value['margin'],0)
        self.assertFalse(choose_frames(value,value,'margin'))

    def test_auc_ties(self):
        self.assertEqual(auc([2],[1]),1)
        self.assertEqual(auc([1],[2]),0)
        self.assertEqual(auc([1],[1]),0.5)
        self.assertIsNone(auc([],[]))

    def test_token_probe_rejects_multitoken(self):
        class Fake:
            def encode(self,text,**kwargs):
                return [1,2]
        with self.assertRaises(ValueError):
            token_probe(Fake())

    def test_space_probe_roundtrip(self):
        texts = list('ABCD') + [' '+c for c in 'ABCD']
        class Fake:
            def encode(self,text,**kwargs):
                return [100+texts.index(text)] if text in texts else [198,100+texts.index(text[-1])]
            def decode(self,ids,**kwargs):
                return texts[ids[0]-100]
        values = token_probe(Fake())
        self.assertEqual(values['A']['plain'],100)
        self.assertEqual(values['A']['space'],104)
        self.assertEqual(values['A']['newline_probe'],[198,100])

    def test_labels_do_not_control_gate1(self):
        rows=[]
        for i,(c,f,cm,fm) in enumerate([(False,True,.1,.8),(True,False,.9,.1),(True,True,.5,.6)]):
            rows.append(dict(index=str(i),phase1=dict(codec_ok=c,frames_ok=f,codec_pred='A',frames_pred='B'),
                codec={'plain':dict(margin=cm,answer_margin=cm,answer_p=cm)},
                frames={'plain':dict(margin=fm,answer_margin=fm,answer_p=fm)}))
        result=diagnose(rows,'plain')
        self.assertEqual(result['selectors']['margin']['rescue'],1)
        self.assertEqual(result['selectors']['margin']['harm'],0)
        self.assertEqual(result['gate1_budget_diagnostic'][0]['triggered'],0)
        self.assertEqual(result['gate1_budget_diagnostic'][-1]['frames_only_recall'],1)
        # 预算一题时应选 margin 最低的 index=0，恰好召回唯一救回题。
        self.assertEqual(result['gate1_budget_diagnostic'][1]['frames_only_recalled'],1)
        for r in rows:
            r['phase1']['codec_ok']=not r['phase1']['codec_ok']
            r['phase1']['frames_ok']=not r['phase1']['frames_ok']
        changed=diagnose(rows,'plain')
        self.assertEqual([x['triggered'] for x in result['gate1_budget_diagnostic']],
                         [x['triggered'] for x in changed['gate1_budget_diagnostic']])
        self.assertEqual(result['selectors']['margin']['changes'],changed['selectors']['margin']['changes'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

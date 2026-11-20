"""定位到原文区间的回归保护；仅单元测试文本，不新增研究评测集。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import air_protocol as p
import condition_text_parser as b
import condition_text30 as old
import condition_text30_reaudit as k
import condition_text30_reaudit_run as r


def row(question):
    return dict(question=question, bodies=['option one','option two','option three','option four'])


def fixed(question):
    x=row(question); d=old.contract(x)
    return x,d,k.partition(x,d)


def raw(text, event_id='q1'):
    return json.dumps(dict(queries=[dict(event_id=event_id,text=text)]))


class Boundaries(unittest.TestCase):
    def test_preface_only_exemption(self):
        x,d,a=fixed('According to the content of the video, why does the boy playing with a toy car hide?')
        self.assertIsNotNone(k.validate(raw('boy playing with toy car hide'),x,d,a)[0])
        self.assertTrue(any('cause' in s for s in a['retained_structure']['unresolved']))

    def test_video_outside_preface_remains_required(self):
        x,d,a=fixed('According to the content of the video, what is the video on the screen showing?')
        self.assertIsNone(k.validate(raw('screen showing'),x,d,a)[0])
        self.assertIn('video',a['events'][0]['required_retrieval_words'])

    def test_following_action_is_not_choice_boilerplate(self):
        x,d,a=fixed('What is the boy following the red car doing?')
        self.assertIsNone(k.validate(raw('boy red car doing'),x,d,a)[0])

    def test_based_action_not_global_stop(self):
        x,d,a=fixed('What is the film based on?')
        self.assertIsNone(k.validate(raw('film'),x,d,a)[0])

    def test_nonstandard_according_preface_is_not_removed(self):
        x,d,a=fixed('According to Alice, what is the boy doing?')
        self.assertEqual(a['question_operator_spans'],[])
        self.assertIsNone(k.validate(raw('boy doing'),x,d,a)[0])

    def test_title_words_are_not_question_operator(self):
        x,d,a=fixed('What is the title Which of the Following showing?')
        self.assertEqual(a['question_operator_spans'],[])
        self.assertIsNone(k.validate(raw('title showing'),x,d,a)[0])

    def test_based_on_video_exemption(self):
        x,d,a=fixed('Based on the video, what setting is the high school theatre show trying to replicate?')
        self.assertIsNotNone(k.validate(raw('video setting high school theatre show trying to replicate'),x,d,a)[0])

    def test_information_preface_exemption(self):
        x,d,a=fixed('Based on the information provided by the video, which of the following elements takes part in the formation of the Yamal Crater?')
        self.assertIsNotNone(k.validate(raw('elements takes part formation Yamal Crater information provided video'),x,d,a)[0])

    def test_negation_required_after_choice_operator_removed(self):
        x,d,a=fixed("Which of the following hypotheses for disappearance isn't mentioned in the video?")
        self.assertIsNotNone(k.validate(raw("hypotheses disappearance isn't mentioned video"),x,d,a)[0])
        self.assertIsNone(k.validate(raw('hypotheses disappearance mentioned video'),x,d,a)[0])
        self.assertEqual(a['retained_structure']['relation'],'absence_unimplemented')

    def test_after_retained_as_unknown_successor(self):
        x,d,a=fixed('What happened after the two actresses climbed onto the high board?')
        self.assertIsNotNone(k.validate(raw('two actresses climbed onto the high board'),x,d,a)[0])
        op=a['explicit_operators'][0]
        self.assertEqual(op['kind'],'unknown_successor')
        self.assertEqual(op['cue']['quote'],'after')
        self.assertTrue(op['target_event_not_given'])
        self.assertEqual(op['known_event_order_edges'],[])

    def test_before_retained_as_unknown_predecessor(self):
        x,d,a=fixed('What happened before the boy opened the box?')
        self.assertEqual(a['explicit_operators'][0]['kind'],'unknown_predecessor')

    def test_after_not_dropped_from_other_event_descriptions(self):
        x,d,a=fixed('Why does a boy run after the car?')
        self.assertIsNone(k.validate(raw('boy run car'),x,d,a)[0])

    def test_future_operator_retained(self):
        x,d,a=fixed("When will we likely discover alien life, as anticipated by the white-haired speaker in the video's opening and closing?")
        self.assertIsNotNone(k.validate(raw('likely discover alien life anticipated white-haired speaker video opening closing'),x,d,a)[0])
        self.assertEqual(a['explicit_operators'][0]['kind'],'requested_future_time')
        self.assertIn('opening and closing',a['retained_structure']['literal_qualifiers'])

    def test_missing_opening_or_closing_not_excused(self):
        x,d,a=fixed("When will we likely discover alien life, as anticipated by the white-haired speaker in the video's opening and closing?")
        self.assertIsNone(k.validate(raw('likely discover alien life anticipated white-haired speaker video opening'),x,d,a)[0])

    def test_clothing_and_action_still_required(self):
        x,d,a=fixed('In which part of the video is a man wearing a red jersey interviewed?')
        for query in ['man wearing jersey interviewed','man wearing red jersey','man wearing blue jersey interviewed']:
            self.assertIsNone(k.validate(raw(query),x,d,a)[0])

    def test_first_condition_not_excused(self):
        x,d,a=fixed('What is the color of the car that is first to start in the race?')
        self.assertIsNone(k.validate(raw('color car start race'),x,d,a)[0])

    def test_count_not_excused(self):
        x,d,a=fixed('How many two colored birds are flying?')
        self.assertIsNone(k.validate(raw('many colored birds flying'),x,d,a)[0])

    def test_no_option_facts_added(self):
        x,d,a=fixed('Based on the video, why does a boy run away?')
        self.assertIsNone(k.validate(raw('boy run away because soldiers attacked'),x,d,a)[0])

    def test_event_ids_and_order_preserved(self):
        x,d,a=fixed('In which order?\n(a) Bend straw.\n(b) Pour snacks.')
        payload={'queries':[{'event_id':'b','text':'Pour snacks'},{'event_id':'a','text':'Bend straw'}]}
        self.assertIsNone(k.validate(json.dumps(payload),x,d,a)[0])

    def test_partition_tampering_rejected(self):
        x,d,a=fixed('What happened after the boy opened the box?')
        changed=deepcopy(a); changed['explicit_operators'][0]['kind']='unknown_predecessor'
        with self.assertRaises(ValueError):k.validate(raw('boy opened box'),x,d,changed)

    def test_range_and_known_before_edge_tampering_rejected(self):
        for question,field,value in [('Why did Alice send Bob a message in the first six minutes of the video?','time_range',None),
                                     ('Before Jane spills polish, why does Alex sit on the couch?','given_edges',[])]:
            x,d,a=fixed(question); changed=deepcopy(a); changed['retained_structure'][field]=value
            with self.assertRaises(ValueError):k.validate(raw('x'),x,d,changed)

    def test_duplicate_json_key_not_accepted(self):
        x,d,a=fixed('What is the red car doing?')
        self.assertIsNone(k.validate('{"queries":[],"queries":[]}',x,d,a)[0])

    def test_duplicate_query_key_not_accepted(self):
        x,d,a=fixed('What is the red car doing?')
        self.assertIsNone(k.validate('{"queries":[{"event_id":"q1","text":"x","text":"red car doing"}]}',x,d,a)[0])

    def test_answer_or_new_keys_not_accepted(self):
        x,d,a=fixed('What is the red car doing?')
        obj=json.loads(raw('red car doing'));obj['answer']='A'
        self.assertIsNone(k.validate(json.dumps(obj),x,d,a)[0])


class FinalCache(unittest.TestCase):
    def test_only_final_response_chosen_and_no_earlier_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'responses').mkdir()
            for slot,text in [('265_0','earlier'),('265_1','final')]:
                p.write_once(root/'responses'/f'{slot}.json',p.seal(dict(response={'choices':[{'finish_reason':'stop','message':{'content':text}}]})))
            c={'source':{'units':{'265':{'attempts':[{'slot':'265_0'},{'slot':'265_1'}],'raw_text':'final'}}}}
            with patch.object(r.source,'OUT',root):
                self.assertEqual(r.last_response(c,'265')['raw_text'],'final')
                c['source']['units']['265']['raw_text']='earlier'
                with self.assertRaises(ValueError):r.last_response(c,'265')

    def test_unfinished_response_not_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'responses').mkdir()
            p.write_once(root/'responses/265_1.json',p.seal(dict(response={'choices':[{'finish_reason':'length','message':{'content':'x'}}]})))
            c={'source':{'units':{'265':{'attempts':[{'slot':'265_1'}],'raw_text':'x'}}}}
            with patch.object(r.source,'OUT',root),self.assertRaises(ValueError):r.last_response(c,'265')

    def test_offline_guard_blocks_secrets_network_and_QA_labels(self):
        # 独立子进程安装audit hook，不污染其他单元测试；使用临时假凭据。
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['.env','deepseek.env','evaluation_labels.json']:(Path(tmp)/name).write_text('fixture only')
            script='''import sys,socket\nfrom pathlib import Path\nimport condition_text30_reaudit_run as r\nr.offline_guard()\nn=0\nfor name in ['.env','deepseek.env','evaluation_labels.json']:\n try: (Path(sys.argv[1])/name).read_text()\n except RuntimeError:n+=1\ntry:socket.getaddrinfo('network-must-not-resolve.invalid',443)\nexcept RuntimeError:n+=1\nassert n==4,n\n'''
            subprocess.run([sys.executable,'-B','-c',script,tmp],check=True,capture_output=True)


if __name__=='__main__':unittest.main()

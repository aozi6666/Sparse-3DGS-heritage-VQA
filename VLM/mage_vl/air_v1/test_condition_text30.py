"""扩展的类型/未知事实保护；无网络的CPU回归，不是新评测数据集。"""
import json
import unittest
import condition_text30 as k
import condition_text_parser as base


def row(q, opts=None):
    return dict(question=q, bodies=opts or ['a','b','c','d'])


class Extension(unittest.TestCase):
    def test_prompt_and_api_unchanged(self):
        self.assertIs(k.payload, base.payload)
        self.assertEqual(k.MAX_TOKENS, base.MAX_TOKENS)

    def test_option_only_unknown_sequence_unsupported(self):
        fixed=k.contract(row('Which option correctly sorts the sequence of events in the video?'))
        self.assertEqual(fixed['contract_status'],'unsupported')
        self.assertEqual(fixed['events'],[])
        self.assertEqual(fixed['given_edges'],[])

    def test_count_and_beginning_both_preserved(self):
        fixed=k.contract(row('At the beginning of this video, how many colors are squeezed onto a board?'))
        self.assertEqual(fixed['relation'],'count_unimplemented')
        self.assertEqual(fixed['literal_qualifiers'],['At the beginning'])

    def test_ordinal_case_is_not_numeric_time(self):
        fixed=k.contract(row('In the last case of the video, what happens when the man divorces?'))
        self.assertIsNone(fixed['time_range'])
        self.assertIn('last case', fixed['literal_qualifiers'])
        self.assertIn('fallback',fixed['grounding_policy'])

    def test_unknown_after_not_known_sequence(self):
        fixed=k.contract(row('What happened after the two actresses climbed onto the high board?'))
        self.assertEqual(fixed['relation'],'unknown_successor_or_predecessor')
        self.assertEqual(fixed['given_edges'],[])

    def test_trigger_not_option_outcome(self):
        fixed=k.contract(row('What happens when a baby elephant chases a group of ducks?',['falls','flies','jumps','crashes']))
        self.assertEqual(fixed['events'][0]['source_text'],'a baby elephant chases a group of ducks')
        self.assertNotIn('falls',str(fixed))

    def test_score_not_time_range(self):
        fixed=k.contract(row("What is the team's LAC hitting percentage when the score is 38-29?"))
        self.assertEqual(fixed['relation'],'score_conditioned_measurement')
        self.assertIsNone(fixed['time_range'])

    def test_global_not_local_certification(self):
        fixed=k.contract(row('What is the main idea of the video?'))
        self.assertEqual(fixed['relation'],'global_summary_or_comparison')
        self.assertIn('fallback',fixed['grounding_policy'])

    def test_negation_not_positive_presence(self):
        fixed=k.contract(row("Which hypothesis isn't mentioned in the video?"))
        self.assertEqual(fixed['relation'],'absence_unimplemented')

    def test_schedule_not_invented_weekdays(self):
        fixed=k.contract(row('On which days of the week does the woman take a rest?'))
        self.assertEqual(fixed['relation'],'narrated_time_or_schedule')
        self.assertNotIn('Saturday',str(fixed))

    def test_visual_query_literal_valid(self):
        fixed=k.contract(row('What is the black-white cat doing in the toilet?'))
        raw=json.dumps(dict(queries=[dict(event_id=e['event_id'],text=base.clean(e['source_text'])) for e in fixed['events']]))
        self.assertIsNone(base.validate(raw,fixed)[1])

    def test_cause_remains_unresolved(self):
        fixed=k.contract(row('Why did a boy playing with a toy car run away?'))
        self.assertTrue(any('cause' in x for x in fixed['unresolved']))


if __name__=='__main__':unittest.main()

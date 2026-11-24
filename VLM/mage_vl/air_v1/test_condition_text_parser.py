"""程序结构/拒绝失真查询的CPU回归；文本为单元测试，不是新增评测数据集。"""
import json
import unittest

import condition_text_parser as k


def row(question, options=None):
    return dict(index='0', question=question, bodies=options or ['one', 'two', 'three', 'four'])


def literal(fixed):
    return json.dumps(dict(queries=[dict(event_id=e['event_id'], text=k.clean(e['source_text'])) for e in fixed['events']]))


class Structure(unittest.TestCase):
    def test_numeric_first_is_not_first_event(self):
        c = k.contract(row('Why did Alice send Bob a text message saying "I need you" in the first six minutes of the video?'))
        self.assertEqual(c['time_range']['bounds'], [0, 360])
        self.assertEqual(c['relation'], 'independent')
        self.assertNotIn('minutes', c['events'][0]['source_text'])
        self.assertTrue(c['unresolved'])

    def test_from_numeric_range(self):
        self.assertEqual(k.numeric_range('from 2 to 3 minutes')['bounds'], [120, 180])
        self.assertEqual(k.numeric_range('between 3 and 8 seconds')['bounds'], [3, 8])

    def test_ambiguous_range_stops(self):
        with self.assertRaises(ValueError):
            k.numeric_range('first two minutes, then from 4 to 5 minutes')

    def test_bad_numeric_range_stops(self):
        with self.assertRaises(ValueError):
            k.numeric_range('from 8 to 3 minutes')

    def test_no_invented_range(self):
        c = k.contract(row('Which tool is used in the first magic?'))
        self.assertIsNone(c['time_range'])
        self.assertEqual(c['relation'], 'first_event_unimplemented')

    def test_relative_position_not_unknown_order(self):
        c = k.contract(row('In which part of the video is a man wearing a red jersey interviewed?'))
        self.assertEqual(c['relation'], 'relative_position')
        self.assertEqual(c['events'][0]['source_text'], 'a man wearing a red jersey interviewed')
        self.assertEqual(c['given_edges'], [])

    def test_before_edge_not_source_reorder(self):
        c = k.contract(row('Before Jane spills the nail polish, why does Grayson, who is wearing a black suit and a red tie, sit on the couch with a grim expression?'))
        self.assertEqual(c['relation'], 'known_before_after')
        self.assertEqual([e['event_id'] for e in c['events']], ['q1', 'q2'])
        self.assertEqual(c['events'][0]['source_text'], 'Jane spills the nail polish')
        self.assertEqual(c['given_edges'][0]['earlier'], 'q2')
        self.assertEqual(c['given_edges'][0]['later'], 'q1')
        self.assertIn('black', c['events'][1]['required_content_words'])
        self.assertIn('red', c['events'][1]['required_content_words'])

    def test_after_edge(self):
        c = k.contract(row('After Jane spills the nail polish, what does Alex do?'))
        self.assertEqual(c['given_edges'][0]['earlier'], 'q1')
        self.assertEqual(c['given_edges'][0]['later'], 'q2')

    def test_unknown_order_keeps_all_ids(self):
        q = 'In which order are the following events performed?\n(a) Bend the straw.\n(b) Pour the snacks.\n(c) Show two coins.\n(d) Drop coins into the cup.'
        c = k.contract(row(q, ['dcab', 'abcd', 'dbac', 'adbc']))
        self.assertEqual([e['event_id'] for e in c['events']], list('abcd'))
        self.assertEqual(c['relation'], 'unknown_temporal_order')
        self.assertEqual(c['given_edges'], [])
        self.assertEqual(c, k.contract(row(q, ['adbc', 'dcab', 'abcd', 'dbac'])))

    def test_spatial_entity_order_is_option_invariant(self):
        opts = ['"David", "Apollo", "Aeneas" and "Persephone".', '"Persephone", "Aeneas", "David" and "Apollo".',
                '"Aeneas", "David", "Apollo" and "Persephone".', '"Apollo", "David", "Persephone" and "Aeneas".']
        q = "In what order does the author present Bernini's four masterpieces in a single scene?"
        a, b = k.contract(row(q, opts)), k.contract(row(q, opts[::-1]))
        self.assertEqual(a['relation'], 'spatial_order')
        self.assertEqual([(e['event_id'], e['source_text']) for e in a['events']], [(e['event_id'], e['source_text']) for e in b['events']])
        self.assertEqual([e['source_text'] for e in a['events']], ['Aeneas', 'Apollo', 'David', 'Persephone'])
        self.assertEqual(a['given_edges'], [])

    def test_nonidentical_option_entities_stop(self):
        with self.assertRaises(ValueError):
            k.contract(row('In what order are four works shown in a single scene?'))

    def test_unknown_events_not_guessed_from_comma_options(self):
        with self.assertRaises(ValueError):
            k.contract(row('Which option correctly sorts the sequence of events in the video?'))


class QueryValidation(unittest.TestCase):
    def setUp(self):
        self.fixed = k.contract(row('In which part of the video is a man wearing a red jersey interviewed?'))

    def test_literal_query_is_legal_not_certified(self):
        obj, err = k.validate(literal(self.fixed), self.fixed)
        self.assertIsNone(err)
        self.assertIsNotNone(obj)
        self.assertFalse(self.fixed['semantic_certified'])

    def test_inflected_action_preserved(self):
        obj, err = k.validate(json.dumps(dict(queries=[dict(event_id='q1', text='man wearing red jersey being interviewed')])), self.fixed)
        self.assertIsNone(err)

    def test_drop_action_fails(self):
        _, err = k.validate(json.dumps(dict(queries=[dict(event_id='q1', text='man wearing red jersey')])), self.fixed)
        self.assertEqual(err, 'query_drops_required_source_condition')

    def test_drop_color_fails(self):
        _, err = k.validate(json.dumps(dict(queries=[dict(event_id='q1', text='man wearing jersey interviewed')])), self.fixed)
        self.assertEqual(err, 'query_drops_required_source_condition')

    def test_option_fact_fails(self):
        _, err = k.validate(json.dumps(dict(queries=[dict(event_id='q1', text='man wearing red jersey interviewed because he won')])), self.fixed)
        self.assertEqual(err, 'query_adds_non_source_content')

    def test_no_answer_field(self):
        obj = json.loads(literal(self.fixed)); obj['answer'] = 'B'
        self.assertIsNone(k.validate(json.dumps(obj), self.fixed)[0])

    def test_no_model_time_or_relation(self):
        for name in ['time_range', 'relation', 'events']:
            obj = json.loads(literal(self.fixed)); obj[name] = 'x'
            self.assertIsNone(k.validate(json.dumps(obj), self.fixed)[0])

    def test_no_event_reordering(self):
        fixed = k.contract(row('In what order?\n(a) Bend straw.\n(b) Pour snacks.'))
        obj = json.loads(literal(fixed)); obj['queries'].reverse()
        self.assertEqual(k.validate(json.dumps(obj), fixed)[1], 'event_identity_or_source_order_changed')

    def test_wrong_binding_fails_even_same_words_in_question(self):
        fixed = k.contract(row('Before Jane spills polish, why does Alex wear a black suit?'))
        obj = json.loads(literal(fixed)); obj['queries'][0]['text'] += ' black suit'
        self.assertEqual(k.validate(json.dumps(obj), fixed)[1], 'query_adds_non_source_content')

    def test_unclosed_json_is_not_repaired_locally(self):
        self.assertIsNone(k.validate('{"queries":[', self.fixed)[0])

    def test_prompt_has_all_options_no_vision(self):
        r = row('Where is the red car?', ['alpha', 'beta', 'gamma', 'delta'])
        body = k.payload(r, k.contract(r))
        user = json.loads(body['messages'][1]['content'])
        self.assertEqual(list(user['all_options_are_unverified_alternatives'].values()), r['bodies'])
        self.assertTrue(all(isinstance(m['content'], str) for m in body['messages']))
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(set(body) & {'tools', 'files', 'images'}, set())


if __name__ == '__main__':
    unittest.main()

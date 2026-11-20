"""模拟传输的API断点/密钥隔离测试；不访问网络、不读用户env。"""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import air_protocol as p
import condition_text_api as api
import condition_text_parser as k
import condition_text_run as run


def response(raw='{"queries":[{"event_id":"q1","text":"red car"}]}', finish='stop'):
    return dict(model='deepseek-flash', choices=[dict(message=dict(content=raw), finish_reason=finish)],
                usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20))


class Credentials(unittest.TestCase):
    def test_env_selection_and_quoted_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / 'deepseek.env').write_text('DEEPSEEK_API_KEY="fixture-key-not-a-real-credential" # comment\nUNRELATED=$(false)\n')
            self.assertEqual(api.env_path(root).name, 'deepseek.env')
            self.assertEqual(api.load_secret(root), 'fixture-key-not-a-real-credential')

    def test_dot_env_supported(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory); (root / '.env').write_text('export DEEPSEEK_API_KEY=fixture-key-not-a-real-credential\n')
            self.assertEqual(api.load_secret(root), 'fixture-key-not-a-real-credential')

    def test_ambiguous_files_require_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            for name in ['.env', 'deepseek.env']:
                (root / name).touch()
            with self.assertRaises(RuntimeError):
                api.env_path(root)
            with patch.dict(os.environ, {'DEEPSEEK_ENV_FILE': 'deepseek.env'}):
                self.assertEqual(api.env_path(root), root / 'deepseek.env')

    def test_duplicate_or_shell_value_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory); path = root / '.env'
            for value in ['DEEPSEEK_API_KEY=fixture-key-not-a-real-credential\n' * 2,
                          'DEEPSEEK_API_KEY="$(some-command-must-never-run)"\n',
                          'DEEPSEEK_API_KEY="fixture-unclosed\n']:
                path.write_text(value)
                with self.assertRaises(RuntimeError) as caught:
                    api.load_secret(root)
                self.assertNotIn('fixture-', str(caught.exception))

    def test_export_key_precedence_without_file_read(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fixture-key-not-a-real-credential'}, clear=True), patch.object(Path, 'read_text', side_effect=AssertionError('no file read')):
            self.assertEqual(api.load_secret(Path('/no-such-directory')), 'fixture-key-not-a-real-credential')


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.protocol = dict(limits=dict(max_calls=10, reported_token_stop=30000))
        self.c = dict(signature=p.digest(self.protocol), protocol=self.protocol, out=self.out)
        self.body = dict(model='deepseek-flash', messages=[dict(role='user', content='fixture question')])
        self.calls = 0
        self.guard = patch.object(p, 'disk_guard', return_value={}); self.guard.start(); self.addCleanup(self.guard.stop)

    def send(self, payload, secret):
        self.calls += 1
        return response()

    def test_saved_response_reused_zero_extra_calls(self):
        first = api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        second = api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        self.assertEqual(first, second)
        self.assertEqual(self.calls, 1)
        all_text = '\n'.join(f.read_text() for f in self.out.rglob('*') if f.is_file())
        self.assertNotIn('fixture-secret', all_text)
        self.assertNotIn('Authorization', all_text)

    def test_pending_request_not_automatically_repeated(self):
        def timeout(payload, secret):
            raise TimeoutError('fixture-secret')
        with self.assertRaises(RuntimeError):
            api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=timeout)
        with self.assertRaisesRegex(RuntimeError, 'retry-uncertain'):
            api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        self.assertEqual(self.calls, 0)
        api.call(p, self.c, '543_0', self.body, 'fixture-secret', retry_uncertain=True, send=self.send)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len([e for e in api.events(p, self.c) if e['event'] == 'start']), 2)

    def test_http_error_does_not_expose_body(self):
        def forbidden(payload, secret):
            raise urllib.error.HTTPError(api.k.ENDPOINT, 401, 'fixture-secret', {}, io.BytesIO(b'fixture-secret'))
        with self.assertRaises(RuntimeError) as caught:
            api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=forbidden)
        self.assertNotIn('fixture-secret', str(caught.exception))
        self.assertNotIn('fixture-secret', (self.out / 'attempts.jsonl').read_text())

    def test_request_drift_stops(self):
        api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        with self.assertRaises(ValueError):
            api.call(p, self.c, '543_0', dict(model='other'), 'fixture-secret', send=self.send)
        self.assertEqual(self.calls, 1)

    def test_corrupt_cache_stops(self):
        api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        path = self.out / 'responses/543_0.json'
        raw = json.loads(path.read_text()); raw['response']['model'] = 'changed'; path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)

    def test_persistent_call_cap_includes_failed_requests(self):
        self.protocol['limits']['max_calls'] = 1; self.c['signature'] = p.digest(self.protocol)
        api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        with self.assertRaisesRegex(RuntimeError, '累计API'):
            api.call(p, self.c, '625_0', self.body, 'fixture-secret', send=self.send)
        self.assertEqual(self.calls, 1)

    def test_token_stop_from_persisted_responses(self):
        self.protocol['limits']['reported_token_stop'] = 20; self.c['signature'] = p.digest(self.protocol)
        api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        with self.assertRaisesRegex(RuntimeError, 'token停止线'):
            api.call(p, self.c, '625_0', self.body, 'fixture-secret', send=self.send)

    def test_saved_response_before_end_journal_is_reused(self):
        cached = api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send)
        # 模拟崩溃点：响应已落盘，response日志事件尚未写入。
        journal = self.out / 'attempts.jsonl'; journal.write_text(journal.read_text().splitlines()[0] + '\n')
        self.assertEqual(api.call(p, self.c, '543_0', self.body, 'fixture-secret', send=self.send), cached)
        self.assertEqual(self.calls, 1)

    def test_slot_path_traversal_refused(self):
        with self.assertRaises(ValueError):
            api.call(p, self.c, '../543', self.body, 'fixture-secret', send=self.send)


class ParserReplay(unittest.TestCase):
    setUp = Checkpoints.setUp

    def test_response_finish_length_not_valid_json_success(self):
        fixed = k.contract(dict(question='What is a red car?', bodies=['x'] * 4))
        obj, error, raw = api.parse_response(response(finish='length'), fixed)
        self.assertIsNone(obj)
        self.assertEqual(error, 'response_not_finished')

    def test_compiler_replays_cached_two_attempts(self):
        row = dict(index='543', question='Where is the red car?', bodies=['x'] * 4)
        fixed = k.contract(row)
        self.c.update(rows={'543': row}, fixed={'543': fixed})
        bad = k.payload(row, fixed)
        api.call(p, self.c, '543_0', bad, 'fixture-secret', send=lambda *_: response('{"answer":"A"}'))
        error = 'exact_queries_schema_required'
        good = k.payload(row, fixed, repair=error)
        api.call(p, self.c, '543_1', good, 'fixture-secret', send=lambda *_: response())
        with patch.object(run, 'OUT', self.out):
            unit = run.compile_record(self.c, '543', require_complete=True)
        self.assertIsNotNone(unit['parsed'])
        self.assertEqual(len(unit['attempts']), 2)
        self.assertFalse(unit['answer_produced'])

    def test_cache_replay_with_missing_second_attempt_stops(self):
        row = dict(index='543', question='Where is the red car?', bodies=['x'] * 4)
        fixed = k.contract(row); self.c.update(rows={'543': row}, fixed={'543': fixed})
        api.call(p, self.c, '543_0', k.payload(row, fixed), 'fixture-secret', send=lambda *_: response('{"answer":"A"}'))
        with patch.object(run, 'OUT', self.out), self.assertRaises(ValueError):
            run.compile_record(self.c, '543', require_complete=True)


if __name__ == '__main__':
    unittest.main()

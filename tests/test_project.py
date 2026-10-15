"""Offline, credential-free tests. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evrika.memory import (
    MEMORY_SYSTEM, build_summary_request, choose_summary_plan, parse_memory_answer,
)
from evrika.privacy import suitable_for_long_memory
from evrika.humanizer import parse_bubbles, pre_message_pause, typing_seconds
from evrika.prompts import build_prompt, extract_text, load_persona, to_bedrock_messages
from evrika.settings import Settings
from evrika.store import Store

ROOT = Path(__file__).resolve().parent.parent


class PromptTests(unittest.TestCase):
    def test_curated_persona_and_ai_disclosure(self):
        obj = load_persona(ROOT / 'persona.json')
        prompt = build_prompt(obj, {"summary": "Had a fun birthday chat", "facts": []})
        self.assertIn(obj['name'], prompt)
        self.assertIn('AI', prompt)
        self.assertIn('Had a fun birthday chat', prompt)
        self.assertIn('No personal identifiers', prompt)
        self.assertIn('No forced ML analogies', prompt)

    def test_normalize_messages_merges_consecutive_and_starts_user(self):
        m = to_bedrock_messages([
            ('assistant', 'old'), ('user', 'hey'), ('user', 'again'),
            ('assistant', 'hello'), ('assistant', 'yep'), ('user', 'end'),
        ])
        self.assertEqual([e['role'] for e in m], ['user', 'assistant', 'user'])
        self.assertIn('again', m[0]['content'][0]['text'])

    def test_extract_text_from_converse(self):
        body = {'output': {'message': {'content': [{'text': 'first'}, {'text': 'second'}]}}}
        self.assertEqual(extract_text(body), 'first\nsecond')
        self.assertEqual(extract_text({}), '')

    def test_prompt_requests_optional_bubble_json_without_forced_splitting(self):
        prompt = build_prompt(load_persona(ROOT / 'persona.json'))
        self.assertIn('"messages"', prompt)
        self.assertIn('Usually return exactly ONE message', prompt)


class HumanizerTests(unittest.TestCase):
    def test_model_controls_natural_message_boundaries(self):
        raw = json.dumps({'messages': ['nu che', 'ty tam zhivoy', 'a']})
        self.assertEqual(parse_bubbles(raw, max_bubbles=3), ['nu che', 'ty tam zhivoy', 'a'])

    def test_single_message_stays_single(self):
        self.assertEqual(parse_bubbles('{"messages": ["Privet!"]}'), ['Privet!'])
        self.assertEqual(parse_bubbles('Simple ordinary answer'), ['Simple ordinary answer'])

    def test_protect_against_excessive_message_spam(self):
        raw = json.dumps({'messages': ['a', 'b', 'c', 'd', 'e', 'f']})
        self.assertEqual(parse_bubbles(raw, max_bubbles=3), ['a', 'b', 'c\nd\ne\nf'])
        self.assertEqual(parse_bubbles(raw, max_bubbles=1), ['a\nb\nc\nd\ne\nf'])

    def test_fenced_json_and_empty_replies(self):
        self.assertEqual(parse_bubbles('```json\n{"messages":["x"]}\n```'), ['x'])
        self.assertEqual(parse_bubbles(''), [])
        self.assertEqual(parse_bubbles('{"messages":[]}'), [])

    def test_realistic_timing_is_bounded(self):
        for _ in range(100):
            self.assertGreaterEqual(typing_seconds('a', min_seconds=1.2, max_seconds=7), 1.2)
            self.assertLessEqual(typing_seconds('long sentence ' * 40, min_seconds=1.2, max_seconds=7), 7)
            self.assertGreaterEqual(pre_message_pause(0), .5)
            self.assertLessEqual(pre_message_pause(1), 1.4)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'test.db')

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_add_chat_and_dedupe(self):
        self.assertTrue(self.store.add_user(100, 'hi'))
        self.assertFalse(self.store.add_user(100, 'duplicate'))
        self.store.add_assistant('hey')
        self.assertEqual(len(self.store.recent_messages(10)), 2)
        self.assertEqual(self.store.recent_messages(10)[0][2], 'hi')

    def test_budget_counts_attempts_and_usage(self):
        req = self.store.begin_request('reply', 1, 2)
        self.assertIsInstance(req, int)
        self.store.complete_request(req, {'inputTokens': 99, 'outputTokens': 13})
        self.assertIsNone(self.store.begin_request('reply', 1, 2))
        self.assertEqual(self.store.usage()['input_tokens'], 99)
        self.assertEqual(self.store.usage()['calls'], 1)

    def test_summary_plan_only_older_messages(self):
        for i in range(20):
            self.store.add_user(2000+i, 'turn ' + str(i))
        plan = choose_summary_plan(self.store, recent_limit=6, trigger=5)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.next_checkpoint, 14)
        self.assertEqual(len(plan.turns), 14)
        summary = {"summary": "We discussed work and weekend", "facts": ["Likes comedy"],
                   "open_threads": ["Choose a movie"], "tone_notes": ["Banter"]}
        self.store.save_memory(summary, plan.next_checkpoint)
        self.assertEqual(len(self.store.recent_messages(100)), 6)
        self.assertFalse(self.store.add_user(2000, 'old replay'))
        self.assertIsNone(choose_summary_plan(self.store, 6, 5))
        for i in range(20, 26):
            self.store.add_user(2000+i, 'turn ' + str(i))
        plan = choose_summary_plan(self.store, 6, 5)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.next_checkpoint, 20)
        self.assertEqual(len(plan.turns), 6)
        self.assertIn('PREVIOUS_MEMORY_JSON', build_summary_request(plan))

    def test_forget_memory_epoch_and_messages(self):
        self.store.add_user(101, 'hi')
        self.store.save_memory({"summary": "hello", "facts": [],
                                "open_threads": [], "tone_notes": []}, 1)
        self.store.clear_chat()
        self.assertEqual(self.store.recent_messages(10), [])
        self.assertEqual(self.store.load_memory()[1], 0)
        self.assertEqual(self.store.get_flag('memory_epoch'), '1')
        self.assertEqual(self.store.load_memory()[0]['summary'], '')


class SummaryTests(unittest.TestCase):
    def test_parse_valid_and_truncate_fact_list(self):
        text = json.dumps({
            'summary': 'Talked about a movie and made a joke',
            'facts': ['Likes comedies', 'mail@example.com'] + ['extra'+str(i) for i in range(20)],
            'open_threads': ['Pick a movie'], 'tone_notes': ['Banter']
        })
        result = parse_memory_answer(text)
        self.assertEqual(len(result['facts']), 8)
        self.assertNotIn('mail@example.com', result['facts'])

    def test_refuse_sensitive_summary(self):
        with self.assertRaises(ValueError):
            parse_memory_answer(json.dumps({
                'summary': 'His home address is 45 North St',
                'facts': [], 'open_threads': [], 'tone_notes': []
            }))
        self.assertFalse(suitable_for_long_memory('Contact me at a@test.com'))
        self.assertFalse(suitable_for_long_memory('Phone +12025550123'))
        self.assertTrue(suitable_for_long_memory('He likes goofy movies'))

    def test_invalid_compaction_does_not_advance_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'db.sqlite')
            for i in range(14):
                store.add_user(900+i, f'Chat turn {i}')
            plan = choose_summary_plan(store, recent_limit=6, trigger=6)
            self.assertIsNotNone(plan)
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_memory_answer('not valid JSON')
            self.assertEqual(store.load_memory()[1], 0)
            store.close()


class EndToEndMemoryTests(unittest.TestCase):
    def test_mock_bedrock_reply_and_memory_compaction(self):
        from dataclasses import replace
        from evrika.telegram_app import Engine

        class FakeBedrock:
            def __init__(self):
                self.called = []

            def generate(self, system, messages, max_tokens, temperature):
                self.called.append(system)
                if system == MEMORY_SYSTEM:
                    return (json.dumps({
                        'summary': 'They joked about movies and the birthday',
                        'facts': ['Likes clever humor'],
                        'open_threads': ['What to watch tonight'],
                        'tone_notes': ['Short replies'],
                    }), {'inputTokens': 155, 'outputTokens': 50})
                return ('Funny fictional AI reply', {'inputTokens': 100, 'outputTokens': 20})

        async def exercise():
            with tempfile.TemporaryDirectory() as folder:
                store = Store(Path(folder) / 'memory.db')
                for i in range(12):
                    store.add_user(5000+i, 'Message ' + str(i))
                fake = FakeBedrock()
                settings = replace(Settings.from_env(telegram_required=False),
                                   recent_context=6, summary_trigger=4)
                engine = Engine(settings, store, fake)
                answer = await engine.make_reply()
                self.assertEqual(answer, 'Funny fictional AI reply')
                store.add_assistant(answer)
                updated = await engine.update_memory_if_needed('0')
                self.assertTrue(updated)
                memory, until = store.load_memory()
                self.assertGreater(until, 0)
                self.assertIn('movies', memory['summary'])
                self.assertEqual([x in fake.called for x in [MEMORY_SYSTEM]], [True])
                self.assertEqual(store.usage()['calls'], 2)
                self.assertLessEqual(len(store.recent_messages(20)), 6)
                store.close()

        asyncio.run(exercise())


class SettingsTests(unittest.TestCase):
    def test_eu_region_and_model_default(self):
        with patch.dict(os.environ, {
            'AWS_REGION': 'eu-north-1',
            'BEDROCK_MODEL_ID': 'eu.anthropic.claude-sonnet-4-6',
            'TELEGRAM_API_ID': '0',
        }):
            obj = Settings.from_env(telegram_required=False)
        self.assertEqual(obj.aws_region, 'eu-north-1')
        self.assertEqual(obj.model_id, 'eu.anthropic.claude-sonnet-4-6')




class MultiRecipientTests(unittest.TestCase):
    def test_multi_target_usernames_and_ids(self):
        from evrika.settings import parse_targets
        names, ids = parse_targets(' @Alice, bob, Alice,  @carol  ', '123, 456,123')
        self.assertEqual(names, ('Alice', 'bob', 'carol'))
        self.assertEqual(ids, (123, 456))
        with self.assertRaises(ValueError):
            parse_targets('alice', '123,not-a-number')
        with self.assertRaises(ValueError):
            parse_targets('bad username', '')

    def test_explicit_opt_in_when_allowlist_is_empty(self):
        common = {
            'TARGET_USERNAMES': '', 'TARGET_USER_IDS': '',
            'TARGET_USERNAME': '', 'TARGET_USER_ID': '',
            'TELEGRAM_API_ID': '12345', 'TELEGRAM_API_HASH': 'fake_for_testing_only',
        }
        with patch.dict(os.environ, {**common, 'ALLOW_ALL_PRIVATE_CHATS': 'false'}):
            with self.assertRaisesRegex(ValueError, 'TARGET_USERNAMES'):
                Settings.from_env(telegram_required=True)
        with patch.dict(os.environ, {**common, 'ALLOW_ALL_PRIVATE_CHATS': 'true'}):
            config = Settings.from_env(telegram_required=True)
        self.assertTrue(config.allow_all_private_chats)
        self.assertEqual(config.target_usernames, ())
        self.assertEqual(config.target_user_ids, ())

    def test_old_singular_field_remains_accepted(self):
        with patch.dict(os.environ, {
            'TARGET_USERNAMES': 'second_friend', 'TARGET_USER_IDS': '456',
            'TARGET_USERNAME': 'first_friend', 'TARGET_USER_ID': '123',
            'ALLOW_ALL_PRIVATE_CHATS': 'false',
        }):
            config = Settings.from_env(telegram_required=False)
        self.assertEqual(config.target_usernames, ('second_friend', 'first_friend'))
        self.assertEqual(config.target_user_ids, (456, 123))

    def test_chat_database_paths_reject_non_ids(self):
        from evrika.telegram_app import chat_database_file
        self.assertEqual(chat_database_file(12345).name, 'chat_12345.sqlite3')
        with self.assertRaises(ValueError):
            chat_database_file(-1)
        with self.assertRaises(ValueError):
            chat_database_file('../oops')

    def test_separate_chat_memory_and_global_budget(self):
        from dataclasses import replace
        from evrika.telegram_app import Engine

        class FakeModel:
            def __init__(self):
                self.requests = []
            def generate(self, system, messages, max_tokens, temperature):
                self.requests.append(messages)
                return ('Ok', {'inputTokens': 10, 'outputTokens': 2})

        async def exercise():
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                a = Store(root / 'chat_100.sqlite3')
                b = Store(root / 'chat_200.sqlite3')
                usage = Store(root / 'usage.sqlite3')
                fake = FakeModel()
                try:
                    config = replace(Settings.from_env(telegram_required=False),
                                     max_calls_hour=10, max_calls_total=1)
                    a.add_user(1, 'Alice secret joke')
                    b.add_user(1, 'Bob entirely different joke')
                    first = await Engine(config, a, fake, budget_store=usage).make_reply()
                    second = await Engine(config, b, fake, budget_store=usage).make_reply()
                    self.assertEqual(first, 'Ok')
                    self.assertIsNone(second)  # Shared rate/budget limit.
                    self.assertEqual(usage.usage()['calls'], 1)
                    self.assertEqual(len(fake.requests), 1)
                    self.assertIn('Alice secret', str(fake.requests[0]))
                    self.assertNotIn('Bob entirely', str(fake.requests[0]))
                    self.assertEqual(b.recent_messages(20)[0][2], 'Bob entirely different joke')
                    a.set_flag('paused', '1')
                    self.assertEqual(b.get_flag('paused', '0'), '0')
                    a.clear_chat()
                    self.assertEqual(a.recent_messages(20), [])
                    self.assertEqual(len(b.recent_messages(20)), 1)
                finally:
                    a.close()
                    b.close()
                    usage.close()

        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()

"""Offline regression checks. Authored for later execution; no real provider calls."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evrika.engine import Engine, ROOT
from evrika.response import FALLBACK, parse_response
from evrika.settings import Settings
from evrika.store import Store


def envelope():
    return {'messages': ['Для какого класса подбираете?'], 'crm_state': {
        'funnel_stage': 'qualification', 'lead_category': None, 'handoff_requested': False,
        'client_data': {k: None for k in ('grade', 'goal', 'support', 'start', 'constraints')},
        'sales_data': {k: None for k in ('recommended_product_id', 'alternative_product_id', 'open_question', 'summary')}}}


class SalesTests(unittest.TestCase):
    def test_valid_state_is_separate_from_text(self):
        value = envelope()
        bubbles, state = parse_response(json.dumps(value))
        self.assertEqual(bubbles, value['messages'])
        self.assertFalse(state['handoff_requested'])

    def test_invalid_output_never_leaks_raw_state(self):
        cases = ['secret raw text', '{"crm_state":{"phone":"private"}}', '[]']
        for raw in cases:
            self.assertEqual(parse_response(raw), ([FALLBACK], None))

    def test_model_cannot_add_delivery_receipt_or_contact(self):
        for field in ['handoff_sent', 'phone', 'payment_confirmed']:
            value = envelope()
            value['crm_state'][field] = True
            self.assertIsNone(parse_response(json.dumps(value))[1])

    def test_invalid_typed_fields_rejected(self):
        for field, bad in [('handoff_requested', 'false'), ('funnel_stage', []), ('client_data', [])]:
            value = envelope()
            value['crm_state'][field] = bad
            self.assertIsNone(parse_response(json.dumps(value))[1])
        value = envelope()
        value['crm_state']['sales_data']['recommended_product_id'] = 52
        self.assertIsNone(parse_response(json.dumps(value))[1])

    def test_terminal_control_sequences_rejected(self):
        value = envelope()
        value['messages'] = ['\x1b[2J']
        self.assertIsNone(parse_response(json.dumps(value))[1])

    def test_compaction_retains_history_and_forget_clears_lead(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / 'chat.sqlite3')
            try:
                for i in range(20):
                    store.add_user(i, 'message ' + str(i))
                store.save_memory({'summary': 'Needs a mentor'}, 14)
                self.assertEqual(len(store.recent_messages(100)), 20)
                self.assertFalse(store.add_user(0, 'duplicate'))
                state = envelope()['crm_state']
                state['handoff_requested'] = True
                store.save_lead(state)
                self.assertEqual(store.conn.execute('SELECT handoff_triggered FROM leads').fetchone()[0], 0)
                store.clear_chat()
                self.assertEqual(store.load_lead(), {})
                self.assertEqual(store.recent_messages(100), [])
            finally:
                store.close()

    def test_catalog_preserves_unknown_and_conflicting_prices(self):
        kb = json.loads((ROOT / 'knowledge_base.json').read_text())
        products = {p['id']: p for p in kb['products_catalog_51_cards']}
        self.assertEqual(len(products), 51)
        self.assertIsNone(products[30]['price_usd'])
        self.assertEqual(products[23]['price_usd'], 1600)
        self.assertEqual(products[49]['price_usd'], 1500)
        self.assertTrue(products[23]['requires_review'])
        self.assertTrue(products[49]['requires_review'])
        self.assertEqual(products[5]['title'], 'Базовый 4 класс')
        self.assertTrue(products[5]['url'].endswith('базовый-3-класс'))

    def test_engine_context_and_shared_budget_without_network(self):
        class FakeModel:
            calls = []
            def generate(self, system, messages, max_tokens, temperature):
                self.calls.append((system, messages))
                return json.dumps(envelope()), {'inputTokens': 10, 'outputTokens': 5}
        async def exercise():
            with tempfile.TemporaryDirectory() as d:
                a, b, budget = [Store(Path(d) / n) for n in ('a.db', 'b.db', 'budget.db')]
                try:
                    with patch.dict('os.environ', {'MAX_MODEL_CALLS_TOTAL': '1'}, clear=True):
                        settings = Settings.from_env(telegram_required=False)
                    a.add_user(1, 'need mentoring')
                    b.add_user(1, 'other conversation')
                    a.save_lead(envelope()['crm_state'])
                    model = FakeModel()
                    self.assertIsNotNone(await Engine(settings, a, model, budget).make_reply())
                    self.assertIsNone(await Engine(settings, b, model, budget).make_reply())
                    self.assertIn('LEAD (untrusted)', model.calls[0][0])
                    self.assertNotIn('other conversation', str(model.calls[0]))
                finally:
                    for store in (a, b, budget):
                        store.close()
        asyncio.run(exercise())

    def test_polza_http_contract_without_network(self):
        from evrika.polza import Polza
        with patch.dict('os.environ', {'POLZA_API_KEY': 'fake-test-value', 'POLZA_MODEL': 'test-model'}):
            provider = Polza()
        with patch('evrika.polza.http.client.HTTPSConnection') as factory:
            response = factory.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(envelope())}}],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 7}}).encode()
            _, usage = provider.generate('system', [{'role': 'user', 'content': [{'text': 'hello'}]}], 1000, .2)
            self.assertEqual(usage, {'inputTokens': 12, 'outputTokens': 7})
            self.assertEqual(factory.call_args.args[0], 'polza.ai')
            args = factory.return_value.request.call_args.args
            self.assertEqual(args[:2], ('POST', '/api/v1/chat/completions'))
            self.assertEqual(json.loads(args[2])['messages'][1]['content'], 'hello')
            factory.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

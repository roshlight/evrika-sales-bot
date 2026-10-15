"""Explicit interactive prototype. Only LLM HTTP calls; no CRM or messaging actions."""
from __future__ import annotations
import asyncio
import json
from uuid import uuid4

from evrika.engine import Engine, ROOT
from evrika.polza import Polza
from evrika.response import parse_response
from evrika.settings import Settings
from evrika.store import Store


async def chat() -> None:
    model = Polza()  # Validate configuration before creating files.
    settings = Settings.from_env(telegram_required=False)
    store = Store(ROOT / 'data' / ('terminal_' + uuid4().hex + '.sqlite3'))
    budget = Store(ROOT / 'data' / 'usage.sqlite3')
    engine = Engine(settings, store, model, budget)
    print('Эврика: терминальный прототип. Сообщения отправляются в Polza и сохраняются локально.')
    print('Используйте вымышленные данные. /exit /state /usage /stop /start /forget')
    message_id = 0
    try:
        while True:
            try:
                text = input('Вы: ').strip()
            except EOFError:
                break
            if not text:
                continue
            if text == '/exit':
                break
            if text == '/state':
                print(json.dumps(store.load_lead(), ensure_ascii=False, indent=2))
                continue
            if text == '/usage':
                print(budget.usage())
                continue
            if text == '/stop':
                store.set_flag('paused', '1')
                print('Диалог остановлен. /start возобновляет его.')
                continue
            if text == '/start':
                store.set_flag('paused', '0')
                print('Диалог возобновлён.')
                continue
            if text == '/forget':
                store.clear_chat()
                print('История, сводка и карточка этой сессии очищены. Общий счётчик вызовов сохранён.')
                continue
            if store.get_flag('paused', '0') == '1':
                print('Диалог остановлен; используйте /start.')
                continue
            if len(text) > 4000:
                print('Разделите сообщение: максимум 4000 символов.')
                continue
            message_id += 1
            store.add_user(message_id, text)
            try:
                raw = await engine.make_reply()
            except Exception:
                print('Запрос к модели не выполнен. Автоматического повтора нет; внешние действия не совершены.')
                continue
            if raw is None:
                print('Достигнут общий лимит обращений к модели.')
                continue
            bubbles, state = parse_response(raw)
            if state is not None:
                store.save_lead(state)
                if state['handoff_requested'] or state['funnel_stage'] == 'handoff':
                    bubbles = ['В этом прототипе передача менеджеру не подключена. Могу подготовить текст обращения для вас.']
                if state['funnel_stage'] == 'closed':
                    bubbles = ['Поняла, на этом остановимся.']
                    store.set_flag('paused', '1')
            for bubble in bubbles:
                print('Эврика:', bubble)
                store.add_assistant(bubble)
            try:
                await engine.update_memory_if_needed(store.get_flag('memory_epoch', '0'))
            except Exception:
                print('[Сводка не обновлена; исходная история сохранена.]')
    finally:
        store.close()
        budget.close()

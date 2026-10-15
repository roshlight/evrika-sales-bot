"""Multi-recipient Telegram user-account client with private per-chat memory."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from evrika.bedrock import Bedrock
from evrika.humanizer import parse_bubbles, pre_message_pause, typing_seconds
from evrika.memory import MEMORY_SYSTEM, build_summary_request, choose_summary_plan, parse_memory_answer
from evrika.prompts import build_prompt, load_persona, to_bedrock_messages
from evrika.settings import Settings
from evrika.store import Store

LOG = logging.getLogger("evrika")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class Engine:
    def __init__(self, config: Settings, store: Store, model: Bedrock, budget_store: Store | None = None):
        self.config = config
        self.store = store
        self.budget_store = budget_store if budget_store is not None else store
        self.model = model
        self.persona_file = ROOT / "persona.json"

    async def ask(self, system: str, messages: list[dict], kind: str, max_tokens: int, temperature: float) -> str | None:
        req_id = self.budget_store.begin_request(kind, self.config.max_calls_hour, self.config.max_calls_total)
        if req_id is None:
            LOG.warning("LLM request budget reached (%s)", kind)
            return None
        try:
            reply, usage = await asyncio.to_thread(
                self.model.generate, system, messages, max_tokens, temperature
            )
            self.budget_store.complete_request(req_id, usage)
            return reply
        except Exception:
            self.budget_store.complete_request(req_id, status="error")
            raise

    async def make_reply(self) -> str | None:
        persona = load_persona(self.persona_file)  # Hot reload; edits take effect next turn.
        memory, _ = self.store.load_memory()
        system = build_prompt(persona, memory)
        turns = self.store.recent_messages(self.config.recent_context)
        messages = to_bedrock_messages([(role, text) for _, role, text in turns])
        if not messages:
            return None
        return await self.ask(system, messages, "reply", self.config.max_output_tokens, 0.82)

    async def update_memory_if_needed(self, epoch: str) -> bool:
        plan = choose_summary_plan(
            self.store, self.config.recent_context, self.config.summary_trigger
        )
        if plan is None:
            return False
        prompt = build_summary_request(plan)
        answer = await self.ask(
            MEMORY_SYSTEM,
            [{"role": "user", "content": [{"text": prompt}]}],
            "summary", 650, 0.0,
        )
        if answer is None or epoch != self.store.get_flag("memory_epoch", "0"):
            return False
        compact = parse_memory_answer(answer)
        self.store.save_memory(compact, plan.next_checkpoint)
        return True


@dataclass
class ChatContext:
    """Every peer has independent conversation, summary and opt-out state."""
    store: Store
    engine: Engine
    queue: asyncio.Queue
    worker_task: asyncio.Task | None = None


def chat_database_file(user_id: int) -> Path:
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError('Expected positive Telegram user ID')
    return DATA / f'chat_{user_id}.sqlite3'


async def serve(config: Settings) -> None:
    # Delayed import: preview, mock tests and Bedrock simulations work offline.
    from telethon import TelegramClient, events

    DATA.mkdir(parents=True, exist_ok=True)
    usage_store = Store(DATA / 'usage.sqlite3')
    model = Bedrock(config.aws_region, config.model_id)
    telegram = TelegramClient(str(DATA / 'evrika_user'), config.api_id, config.api_hash)
    chats: dict[int, ChatContext] = {}

    async def chat_worker(user_id: int, chat: ChatContext) -> None:
        store, engine, queue = chat.store, chat.engine, chat.queue
        carry = None
        while True:
            if carry is None:
                last_event, message, epoch = await queue.get()
            else:
                last_event, message, epoch = carry
                carry = None
            batch = [(last_event, message)]
            while True:
                try:
                    event, text, incoming_epoch = await asyncio.wait_for(
                        queue.get(), timeout=config.debounce_seconds
                    )
                    if incoming_epoch != epoch:
                        carry = (event, text, incoming_epoch)
                        break
                    batch.append((event, text))
                    last_event = event
                except asyncio.TimeoutError:
                    break
            if store.get_flag('paused', '0') == '1':
                continue
            if epoch != store.get_flag('memory_epoch', '0'):
                continue
            added = sum(bool(store.add_user(ev.id, value)) for ev, value in batch)
            if not added:
                continue
            try:
                async with telegram.action(last_event.chat_id, 'typing'):
                    response = await engine.make_reply()
                if epoch != store.get_flag('memory_epoch', '0'):
                    continue
                if store.get_flag('paused', '0') == '1':
                    continue
                if response is None:
                    await last_event.respond('Я чуть помолчу: лимит на запросы.', parse_mode=None)
                    continue
                if not queue.empty():
                    LOG.info('New incoming text from %s: waiting for fresh reply', user_id)
                    continue
                bubbles = parse_bubbles(response, config.max_reply_bubbles)
                sent = 0
                for index, bubble in enumerate(bubbles):
                    if (not queue.empty() or epoch != store.get_flag('memory_epoch', '0')
                            or store.get_flag('paused', '0') == '1'):
                        break
                    await asyncio.sleep(pre_message_pause(index))
                    if (not queue.empty() or epoch != store.get_flag('memory_epoch', '0')
                            or store.get_flag('paused', '0') == '1'):
                        break
                    async with telegram.action(last_event.chat_id, 'typing'):
                        await asyncio.sleep(typing_seconds(
                            bubble,
                            chars_per_second=config.typing_chars_per_second,
                            min_seconds=config.typing_min_seconds,
                            max_seconds=config.typing_max_seconds,
                        ))
                        if (not queue.empty() or epoch != store.get_flag('memory_epoch', '0')
                                or store.get_flag('paused', '0') == '1'):
                            break
                        await last_event.respond(bubble, parse_mode=None)
                        store.add_assistant(bubble)
                        sent += 1
                if sent and queue.empty() and epoch == store.get_flag('memory_epoch', '0'):
                    try:
                        await engine.update_memory_if_needed(epoch)
                    except Exception as exc:
                        LOG.warning('Memory compaction skipped for %s: %s', user_id, type(exc).__name__)
            except Exception as exc:
                LOG.exception('Processing error in chat %s: %s', user_id, type(exc).__name__)
                if epoch == store.get_flag('memory_epoch', '0') and store.get_flag('paused', '0') != '1':
                    await last_event.respond('Упс, я затупила. Напиши потом.', parse_mode=None)

    try:
        await telegram.start()  # Credentials/login codes only entered in local terminal.
        me = await telegram.get_me()
        my_id = int(me.id) if me is not None else 0
        allowed_ids = set(config.target_user_ids)
        for username in config.target_usernames:
            entity = await telegram.get_entity('@' + username)
            if not hasattr(entity, 'id'):
                raise ValueError(f'Username @{username} did not resolve to a Telegram user')
            allowed_ids.add(int(entity.id))
        if my_id in allowed_ids:
            raise ValueError('Target list must not include the Evrika account itself')
        if config.allow_all_private_chats:
            LOG.warning('ALLOW_ALL_PRIVATE_CHATS=true: anyone DMing this account gets an AI response')
        else:
            LOG.info('Private-chat allowlist: %s', ', '.join(map(str, sorted(allowed_ids))))
        LOG.info('No first messages are sent automatically; a recipient must write first')

        @telegram.on(events.NewMessage(incoming=True))
        async def on_incoming(event) -> None:
            if not event.is_private or event.sender_id is None:
                return
            user_id = int(event.sender_id)
            if user_id == my_id:
                return
            if not config.allow_all_private_chats and user_id not in allowed_ids:
                return
            sender = await event.get_sender()
            if sender is not None and getattr(sender, 'bot', False):
                return
            txt = (event.raw_text or '').strip()
            if not txt:
                return  # Do not send attached photos/files to Bedrock.
            chat = chats.get(user_id)
            if chat is None:
                # Independent SQLite file / paused flag / rolling memory per person.
                store = Store(chat_database_file(user_id))
                chat = ChatContext(
                    store=store,
                    engine=Engine(config, store, model, budget_store=usage_store),
                    queue=asyncio.Queue(),
                )
                chats[user_id] = chat
                chat.worker_task = asyncio.create_task(chat_worker(user_id, chat))
            store = chat.store
            cmd = txt.lower().split(maxsplit=1)[0]
            if cmd in ('/stop', '/pause'):
                store.set_flag('paused', '1')
                await event.respond('Окей, молчу. /start — вернуться.', parse_mode=None)
                return
            if cmd in ('/forget', '/reset'):
                store.clear_chat()
                await event.respond('Память нашего чата очищена.', parse_mode=None)
                return
            if cmd == '/start':
                store.set_flag('paused', '0')
                await event.respond('Я вернулась. Чё у тебя?', parse_mode=None)
                return
            if store.get_flag('paused', '0') == '1':
                return
            chat.queue.put_nowait((event, txt[:4000], store.get_flag('memory_epoch', '0')))

        await telegram.run_until_disconnected()
    finally:
        for chat in chats.values():
            if chat.worker_task is not None:
                chat.worker_task.cancel()
        for chat in chats.values():
            if chat.worker_task is not None:
                try:
                    await chat.worker_task
                except asyncio.CancelledError:
                    pass
            chat.store.close()
        await telegram.disconnect()
        usage_store.close()

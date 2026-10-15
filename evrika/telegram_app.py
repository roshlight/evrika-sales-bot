"""Multi-recipient Telegram user-account client with per-chat memory, CRM leads and follow-up."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evrika.bedrock import Bedrock
from evrika.humanizer import parse_response, pre_message_pause, typing_seconds
from evrika.memory import MEMORY_SYSTEM, build_summary_request, choose_summary_plan, parse_memory_answer
from evrika.prompts import build_prompt, load_persona, load_knowledge_base, to_bedrock_messages
from evrika.settings import Settings
from evrika.store import Store

LOG = logging.getLogger("evrika")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BITRIX_OUTBOX = DATA / "bitrix24_leads_outbox.jsonl"


# ── CRM helpers ───────────────────────────────────────────────

def _build_admin_card(user_id: int, crm_state: dict, username: str | None = None) -> str:
    """Format a rich text card for the admin chat (HTML)."""
    cd = crm_state.get("client_data") or {}
    sd = crm_state.get("sales_data") or {}
    lines = [
        f"🔔 <b>Новая заявка от клиента</b>",
        f"",
        f"<b>Telegram ID:</b> {user_id}",
    ]
    if username:
        lines.append(f"<b>Username:</b> @{username}")
    lines.append(f"<b>Категория лида:</b> {crm_state.get('lead_category', '—')}")
    lines.append(f"<b>Этап воронки:</b> {crm_state.get('funnel_stage', '—')}")
    lines.append(f"<b>SLA:</b> {crm_state.get('sla_target', '—')}")
    lines.append(f"<b>Причина передачи:</b> {crm_state.get('handoff_trigger_reason', '—')}")
    lines.append("")
    lines.append("📋 <b>Контакт:</b>")
    lines.append(f"  Имя: {cd.get('name') or '—'}")
    lines.append(f"  Телефон: {cd.get('phone') or '—'}")
    lines.append(f"  Email: {cd.get('email') or '—'}")
    lines.append(f"  Удобное время: {cd.get('preferred_contact_time') or '—'}")
    lines.append(f"  Существующий клиент: {cd.get('is_existing_client') or '—'}")
    lines.append("")
    lines.append("🎓 <b>Квалификация:</b>")
    lines.append(f"  Класс: {cd.get('grade') or '—'}")
    lines.append(f"  Детей: {cd.get('children_count') or '—'}")
    lines.append(f"  Локация / Москва: {cd.get('location_and_moscow_registration') or '—'}")
    lines.append(f"  Цель / боль: {cd.get('goal_and_pain') or '—'}")
    lines.append(f"  Срочность: {cd.get('urgency') or '—'}")
    lines.append("")
    lines.append("💼 <b>Продажа:</b>")
    lines.append(f"  Рекомендованный тариф: {sd.get('recommended_product_title') or '—'}")
    lines.append(f"  URL тарифа: {sd.get('recommended_product_url') or '—'}")
    lines.append(f"  Альтернатива: {sd.get('alternative_product_title') or '—'}")
    lines.append(f"  Возражения: {sd.get('objections_raised') or '—'}")
    lines.append(f"  Причина отказа: {sd.get('refusal_reason_code') or '—'}")
    lines.append(f"  CTA: {sd.get('next_step_cta') or '—'}")
    lines.append("")
    summary = sd.get('dialog_summary_for_manager') or '—'
    lines.append(f"📝 <b>Резюме диалога:</b>\n{summary}")
    return "\n".join(lines)


def _build_bitrix_payload(user_id: int, crm_state: dict, username: str | None = None,
                          extra_note: str | None = None) -> dict:
    cd = crm_state.get("client_data") or {}
    sd = crm_state.get("sales_data") or {}
    payload = {
        "source": "telegram_ai_bot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "telegram_user_id": user_id,
        "telegram_username": username,
        "lead_category": crm_state.get("lead_category"),
        "funnel_stage": crm_state.get("funnel_stage"),
        "sla_target": crm_state.get("sla_target"),
        "handoff_trigger_reason": crm_state.get("handoff_trigger_reason"),
        "client_data": cd,
        "sales_data": sd,
    }
    if extra_note:
        payload["extra_note"] = extra_note
    return payload


def _append_bitrix_outbox(payload: dict) -> None:
    """Append a JSON-lines record for Bitrix24 crm.lead.add integration."""
    BITRIX_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(BITRIX_OUTBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ── Engine ────────────────────────────────────────────────────

class Engine:
    def __init__(self, config: Settings, store: Store, model: Bedrock, budget_store: Store | None = None):
        self.config = config
        self.store = store
        self.budget_store = budget_store if budget_store is not None else store
        self.model = model
        self.persona_file = ROOT / "persona.json"
        self.kb_file = ROOT / "knowledge_base.json"

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
        persona = load_persona(self.persona_file)
        knowledge_base = load_knowledge_base(self.kb_file)
        memory, _ = self.store.load_memory()
        system = build_prompt(persona, knowledge_base, memory)
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


# ── Chat context ──────────────────────────────────────────────

@dataclass
class ChatContext:
    """Every peer has independent conversation, summary and opt-out state."""
    store: Store
    engine: Engine
    queue: asyncio.Queue
    user_id: int
    username: str | None = None
    worker_task: asyncio.Task | None = None
    followup_task: asyncio.Task | None = None


def chat_database_file(user_id: int) -> Path:
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError('Expected positive Telegram user ID')
    return DATA / f'chat_{user_id}.sqlite3'


# ── Main serve loop ───────────────────────────────────────────

async def serve(config: Settings) -> None:
    from telethon import TelegramClient, events

    DATA.mkdir(parents=True, exist_ok=True)
    usage_store = Store(DATA / 'usage.sqlite3')
    model = Bedrock(config.aws_region, config.model_id)
    telegram = TelegramClient(str(DATA / 'roberta_user'), config.api_id, config.api_hash)
    chats: dict[int, ChatContext] = {}

    # ── Handoff / CRM notification ────────────────────────────

    async def process_crm_state(user_id: int, crm_state: dict | None, chat: ChatContext) -> None:
        if crm_state is None:
            return
        store = chat.store
        # Read the previous state before overwriting it, to detect new contacts.
        lead_row = store.get_lead(user_id)
        store.upsert_lead(user_id, crm_state)

        handoff_requested = crm_state.get("handoff_to_manager", False)
        if not handoff_requested:
            return

        already_sent = store.was_handoff_sent(user_id)
        # Re-send if new phone/email appeared
        old_crm = json.loads(lead_row["crm_json"]) if lead_row else {}
        old_cd = old_crm.get("client_data") or {}
        new_cd = crm_state.get("client_data") or {}
        contact_updated = (
            (new_cd.get("phone") and new_cd.get("phone") != old_cd.get("phone")) or
            (new_cd.get("email") and new_cd.get("email") != old_cd.get("email"))
        )

        if already_sent and not contact_updated:
            return

        card = _build_admin_card(user_id, crm_state, username=chat.username)
        payload = _build_bitrix_payload(user_id, crm_state, username=chat.username)

        if config.admin_chat_id:
            try:
                await telegram.send_message(config.admin_chat_id, card, parse_mode='html')
                LOG.info("Handoff card sent to admin chat for user %s", user_id)
            except Exception as exc:
                LOG.error("Failed to send handoff card to admin: %s", exc)

        try:
            _append_bitrix_outbox(payload)
            LOG.info("Bitrix24 payload appended for user %s", user_id)
        except Exception as exc:
            LOG.error("Failed to write Bitrix outbox: %s", exc)

        store.mark_handoff_sent(user_id)

    # ── Follow-up for silent clients ──────────────────────────

    async def followup_watcher(user_id: int, chat: ChatContext) -> None:
        """If client is in a commercial stage, not handed off, and silent for N seconds,
        send one gentle follow-up and push to Bitrix outbox."""
        store = chat.store
        timeout = config.followup_timeout_seconds
        while True:
            await asyncio.sleep(30)  # check every 30 seconds
            try:
                if store.get_flag('paused', '0') == '1':
                    continue
                if store.was_followup_sent(user_id):
                    continue
                if store.was_handoff_sent(user_id):
                    continue
                lead = store.get_lead(user_id)
                if lead is None:
                    continue
                stage = lead.get("funnel_stage") or ""
                commercial_stages = {"qualification", "need_identification", "recommendation",
                                     "presentation", "objection_handling", "closing"}
                if stage not in commercial_stages:
                    continue
                last_user_ts = store.last_user_message_time()
                last_asst_ts = store.last_assistant_message_time()
                if not last_user_ts or not last_asst_ts:
                    continue
                # Bot must have replied after the user (waiting for user response)
                if last_asst_ts <= last_user_ts:
                    continue
                # Parse timestamps (SQLite CURRENT_TIMESTAMP = UTC)
                try:
                    last_user_dt = datetime.fromisoformat(last_user_ts.replace(' ', 'T')).replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                now_utc = datetime.now(timezone.utc)
                elapsed = (now_utc - last_user_dt).total_seconds()
                if elapsed < timeout:
                    continue

                # Send follow-up
                followup_text = (
                    "Я на связи! 😊 Если хотите, могу прямо сейчас открыть вам "
                    "<b>бесплатный гостевой доступ</b> к платформе EdPalm, чтобы вы с ребёнком "
                    "посмотрели уроки изнутри. Или, если удобнее, передам ваш вопрос нашему "
                    "старшему консультанту по образованию — он свяжется в удобное для вас время. "
                    "Что вам ближе?"
                )
                try:
                    await telegram.send_message(user_id, followup_text, parse_mode='html')
                    store.add_assistant(followup_text)
                    store.mark_followup_sent(user_id)
                    LOG.info("Follow-up sent to silent user %s after %.0f sec", user_id, elapsed)
                except Exception as exc:
                    LOG.warning("Failed to send follow-up to %s: %s", user_id, exc)
                    continue

                # Push to Bitrix outbox
                crm_json_str = lead.get("crm_json") or "{}"
                crm_state = json.loads(crm_json_str)
                payload = _build_bitrix_payload(
                    user_id, crm_state, username=chat.username,
                    extra_note=f"Follow-up / Не отвечает {int(elapsed // 60)} мин"
                )
                try:
                    _append_bitrix_outbox(payload)
                except Exception:
                    pass

                # Also notify admin
                if config.admin_chat_id:
                    try:
                        note = (
                            f"⏰ <b>Follow-up:</b> клиент {user_id}"
                            + (f" (@{chat.username})" if chat.username else "")
                            + f" не отвечает {int(elapsed // 60)} мин на этапе «{stage}».\n"
                            f"Бот отправил напоминание. Рекомендуется связаться."
                        )
                        await telegram.send_message(config.admin_chat_id, note, parse_mode='html')
                    except Exception:
                        pass

            except asyncio.CancelledError:
                return
            except Exception as exc:
                LOG.warning("Follow-up watcher error for %s: %s", user_id, type(exc).__name__)

    # ── Per-chat worker ───────────────────────────────────────

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
                    await last_event.respond(
                        'Одну секунду, я уточню информацию и вернусь к вам!',
                        parse_mode='html'
                    )
                    continue

                bubbles, crm_state = parse_response(response, config.max_reply_bubbles)

                # Process CRM state
                await process_crm_state(user_id, crm_state, chat)

                if not queue.empty():
                    LOG.info('New incoming text from %s: waiting for fresh reply', user_id)
                    continue

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
                        await last_event.respond(bubble, parse_mode='html')
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
                    await last_event.respond(
                        'Прошу прощения, произошла небольшая техническая заминка. Напишите ещё раз, я на связи!',
                        parse_mode='html'
                    )

    # ── Telegram event loop ───────────────────────────────────

    try:
        await telegram.start()
        me = await telegram.get_me()
        my_id = int(me.id) if me is not None else 0
        allowed_ids = set(config.target_user_ids)
        for username in config.target_usernames:
            entity = await telegram.get_entity('@' + username)
            if not hasattr(entity, 'id'):
                raise ValueError(f'Username @{username} did not resolve to a Telegram user')
            allowed_ids.add(int(entity.id))
        if my_id in allowed_ids:
            raise ValueError('Target list must not include the bot account itself')
        if config.allow_all_private_chats:
            LOG.warning('ALLOW_ALL_PRIVATE_CHATS=true: anyone DMing this account gets an AI response')
        else:
            LOG.info('Private-chat allowlist: %s', ', '.join(map(str, sorted(allowed_ids))))
        LOG.info('EdPalm AI consultant Эврика is online. Waiting for incoming messages.')

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
                return

            chat = chats.get(user_id)
            if chat is None:
                store = Store(chat_database_file(user_id))
                uname = getattr(sender, 'username', None) if sender else None
                chat = ChatContext(
                    store=store,
                    engine=Engine(config, store, model, budget_store=usage_store),
                    queue=asyncio.Queue(),
                    user_id=user_id,
                    username=uname,
                )
                chats[user_id] = chat
                chat.worker_task = asyncio.create_task(chat_worker(user_id, chat))
                chat.followup_task = asyncio.create_task(followup_watcher(user_id, chat))
            else:
                # Update username if it changed
                uname = getattr(sender, 'username', None) if sender else None
                if uname:
                    chat.username = uname

            store = chat.store
            cmd = txt.lower().split(maxsplit=1)[0]
            if cmd in ('/stop', '/pause'):
                store.set_flag('paused', '1')
                await event.respond('Хорошо, я на паузе. Напишите /start, когда будете готовы продолжить 🙂', parse_mode='html')
                return
            if cmd in ('/forget', '/reset'):
                store.clear_chat()
                await event.respond('Память нашего диалога очищена. Начнём сначала!', parse_mode='html')
                return
            if cmd == '/start':
                store.set_flag('paused', '0')
                await event.respond(
                    'Здравствуйте! Я <b>Эврика</b> — консультант онлайн-школы EdPalm. '
                    'Помогу подобрать программу обучения для вашего ребёнка с 1 по 11 класс. '
                    'Расскажите, что вас интересует?',
                    parse_mode='html'
                )
                return
            if store.get_flag('paused', '0') == '1':
                return

            # Reset followup flag on new user message
            store.set_flag(f"followup_sent_{user_id}", "0")

            chat.queue.put_nowait((event, txt[:4000], store.get_flag('memory_epoch', '0')))

        await telegram.run_until_disconnected()
    finally:
        for chat in chats.values():
            if chat.worker_task is not None:
                chat.worker_task.cancel()
            if chat.followup_task is not None:
                chat.followup_task.cancel()
        for chat in chats.values():
            for task in (chat.worker_task, chat.followup_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            chat.store.close()
        await telegram.disconnect()
        usage_store.close()
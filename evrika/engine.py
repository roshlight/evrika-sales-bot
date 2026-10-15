"""Channel-independent dialogue engine. No Telegram imports or outbound actions."""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from evrika.prompts import build_prompt, load_persona, load_knowledge_base, to_bedrock_messages
from evrika.memory import MEMORY_SYSTEM, build_summary_request, choose_summary_plan, parse_memory_answer
from evrika.settings import Settings
from evrika.store import Store
ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("evrika")
class Engine:
    def __init__(self, config: Settings, store: Store, model, budget_store: Store | None = None):
        self.config = config
        self.store = store
        self.budget_store = budget_store if budget_store is not None else store
        self.model = model
        self.persona_file = ROOT / "persona.example.json"

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
        system = build_prompt(persona, load_knowledge_base(ROOT / "knowledge_base.json"), memory, self.store.load_lead())
        turns = self.store.recent_messages(self.config.recent_context)
        messages = to_bedrock_messages([(role, text) for _, role, text in turns])
        if not messages:
            return None
        return await self.ask(system, messages, "reply", self.config.max_output_tokens, 0.2)

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

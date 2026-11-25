# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ClmStream: ``ClmAgent`` on ContextBench tasks under Harbor.

ContextBench pushes each op as a user message into one continuous conversation; the
agent only says when it is ready for the next one (a standalone ``READY_FOR_NEXT_OP``).
``ClmStream`` adds ContextBench's ``CBStreamMixin`` to the unmodified ``ClmAgent`` and
hooks three points of its loop instead of copying ``run()``:

1. ``run()`` loads the stream (``cb_setup``: read once, count and sha checked against
   ``task.toml``, then removed from the container), inserts the family's method skill
   into the instruction, and calls ``ClmAgent.run()``. The system prompt's finish
   paragraph is ContextBench's marker contract (``contextbench.stream.MARKER_PARAGRAPH``).
2. ``self._resume.replay`` is wrapped for that run. It returns the opening message
   list the loop then owns; the adapter keeps a reference to it and appends op 0 as a
   ``role="user"`` message.
3. ``_is_successful_submit``, which the loop calls right after appending each tool
   result, appends what the stream releases (the next op, an explanation, or the
   early-submit refusal) as ``role="user"`` messages to that list. It returns True,
   which ends the loop, once the stream has a terminal reason.

``_query_with_retry`` is wrapped only to record a server-side context overflow, so the
run's ``terminated_by`` can say ``budget_wall``. The context snapshots the graders read
are ``ClmAgent``'s own per-call snapshots.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from clm_harness.clm_agent.harness import ClmAgent, _is_context_overflow
from clm_harness.utils.finish_policy import FinishPolicy
from contextbench import stream as proto
from contextbench.harbor.mixin import CBStreamMixin

__all__ = ["ClmStream", "PAPER_KWARGS", "PAPER_SKILL_SHA16", "family_of", "load_skill",
           "place_skill"]

#: The ClmAgent settings of the paper's CLM runs on ContextBench. Each key differs from
#: the ClmAgent default or pins a default the runs depended on; see README.md.
PAPER_KWARGS: dict[str, Any] = {
    "context_budget_tokens": 32768,
    "context_budget_reserve_tokens": 2048,
    "max_steps": 100000,
    "lm_call_cap": 0,
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.95,
    "observation_max_chars": 50000,
    "nudge_ratios": "",
    "persistent_nudge_ratio": None,
    "finalize_nudge_turns": 0,
    "max_num_retry_on_limit": 0,
    "allow_edit_growth": "fit",
    "ctx_hint": "default",
    "finish_policy": "submit",
    "enable_thinking": True,
    "send_chat_template_kwargs": True,
    "cost_metric": "usd",
}

#: sha256[:16] of ContextBench's ``clm_exp/skills/clm/<family>.md``, the method text of
#: the paper's CLM runs.
PAPER_SKILL_SHA16: dict[str, str] = {
    "kv_store": "70695bdca1a7f39d",
    "log_triage": "1a071e820bcc9a93",
    "needle_retention": "e834d24bf004c7ae",
    "sudoku_sketchpad": "378c4e5d928af06f",
}

#: Where the paper's task builder put the method section: before the last match of
#: the anchor, as ``before + skill + after``.
_SKILL_ANCHORS = {
    "kv_store": ("\nThe run ends automatically", "\n", "\n\n"),
    "log_triage": ("\nThe run ends automatically", "\n", "\n\n"),
    "needle_retention": ("\nThe run ends automatically", "\n", "\n\n"),
    "sudoku_sketchpad": ("\n## Turn protocol (IMPORTANT)", "", "\n\n\n"),
}

_FAMILY_TITLES = {
    "# Key-Value Store": "kv_store",
    "# Incident Log Triage": "log_triage",
    "# Needle": "needle_retention",
    "# Sudoku": "sudoku_sketchpad",
}


def family_of(instruction: str) -> str | None:
    """The ContextBench family, from the instruction's title line."""
    head = instruction.lstrip().split("\n", 1)[0]
    return next((f for title, f in _FAMILY_TITLES.items() if head.startswith(title)), None)


def _default_skills_dir() -> Path:
    """``clm_exp/skills`` of the ContextBench source checkout on the path."""
    import contextbench

    return Path(contextbench.__file__).resolve().parents[1] / "clm_exp" / "skills"


def load_skill(family: str, skill: str = "clm", skills_dir: str | Path | None = None
               ) -> tuple[Path, str]:
    """The path and text of ``<skills_dir>/<skill>/<family>.md``.

    ``skill`` is a method directory under ``skills_dir`` (default: ContextBench's
    ``clm_exp/skills``) or an absolute path to one. ``"clm"`` is the paper's CLM text
    and must match ``PAPER_SKILL_SHA16``.
    """
    path = Path(skill) if Path(skill).is_absolute() else \
        Path(skills_dir or _default_skills_dir()) / skill
    path = path / f"{family}.md"
    text = path.read_text(encoding="utf-8")
    if skill == "clm":
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if got != PAPER_SKILL_SHA16[family]:
            raise ValueError(f"{path} has sha16 {got}, expected {PAPER_SKILL_SHA16[family]}")
    return path, text


def place_skill(instruction: str, skill_text: str, family: str | None) -> str:
    """The instruction with the skill where the paper's builder put it (else appended)."""
    anchor, before, after = _SKILL_ANCHORS.get(family or "", ("", "", ""))
    i = instruction.rfind(anchor) if anchor else -1
    if i < 0:
        return instruction.rstrip("\n") + "\n\n" + skill_text.strip("\n") + "\n"
    return instruction[:i + 1] + before + skill_text.strip("\n") + after + instruction[i + 1:]


class _MarkerFinish(FinishPolicy):
    """``submit`` mode whose system-prompt paragraph is ContextBench's marker contract."""

    __slots__ = ()

    def prompt_fragment(self) -> str:
        return proto.MARKER_PARAGRAPH


class ClmStream(CBStreamMixin, ClmAgent):
    """``ClmAgent`` fed by a ContextBench stream through ``CBStreamMixin``."""

    def __init__(self, *args: Any, skill: str | None = "clm", skills_dir: str | None = None,
                 **kwargs: Any) -> None:
        if ClmAgent._as_bool(kwargs.get("subagents", False)):
            # The stream releases ops from the main loop's submit hook; a sub-agent in the
            # same sandbox would act outside that protocol.
            raise ValueError("ClmStream does not support subagents=true; run ContextBench "
                             "tasks without sub-agents.")
        merged = {**PAPER_KWARGS, **kwargs}
        super().__init__(*args, **merged)
        self.finish_policy = _MarkerFinish(self.finish_policy.mode)
        self.cb_skill = skill if skill and str(skill).lower() != "none" else None
        self.cb_skills_dir = skills_dir
        self._cb_messages: list[dict[str, Any]] | None = None
        self._cb_turn = 0
        self._cb_overflow = False
        self._cb_record: dict[str, Any] = {
            "skill": self.cb_skill,
            "kwargs": {k: v for k, v in merged.items() if k not in ("api_base", "logs_dir")}}

    @staticmethod
    def name() -> str:
        return "clm-contextbench"

    # ----------------------------------------------------------------- hooks
    def _is_successful_submit(self, command: str, result: Any, tool_content: str) -> bool:
        if self._cb_messages is None:
            return super()._is_successful_submit(command, result, tool_content)
        self._cb_turn += 1
        rc = getattr(result, "return_code", 1)
        self._cb_messages += self.cb_user_turns(
            self.cb_pending(command, rc, turn=self._cb_turn))
        return self.cb_done

    async def _query_with_retry(self, model: str, messages: list[dict[str, Any]]) -> Any:
        try:
            return await super()._query_with_retry(model, messages)
        except Exception as exc:
            if _is_context_overflow(exc):
                self._cb_overflow = True
            raise

    # ------------------------------------------------------------------- run
    def _with_skill(self, instruction: str) -> str:
        if not self.cb_skill:
            return instruction
        family = family_of(instruction)
        if family is None:
            raise ValueError("cannot tell the ContextBench family from the instruction; "
                             "pass skill=none or package the skill into the task")
        path, text = load_skill(family, self.cb_skill, self.cb_skills_dir)
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        if first and first in instruction:
            return instruction  # the task was packaged with the skill already
        self._cb_record.update(skill_path=str(path), skill_sha16=hashlib.sha256(
            text.encode("utf-8")).hexdigest()[:16])
        return place_skill(instruction, text, family)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await self.cb_setup(environment, budget=self.context_budget_tokens,
                            reserve=self.context_budget_reserve_tokens)
        instruction = self._with_skill(instruction)
        self._cb_record["instruction_sha16"] = hashlib.sha256(
            instruction.encode("utf-8")).hexdigest()[:16]

        replay = self._resume.replay

        def _replay(messages: list[dict[str, Any]], *a: Any, **kw: Any) -> list[dict[str, Any]]:
            out = replay(messages, *a, **kw)
            self._cb_messages = out
            out += self.cb_user_turns(self.cb_initial_turns())
            return out

        self._resume.replay = _replay
        completed = False
        try:
            await super().run(instruction, environment, context)
            completed = True
        except asyncio.CancelledError:
            self.cb_finish("timeout")
            raise
        finally:
            self._resume.replay = replay
            if self._cb_overflow or self._budget.finalized:
                self.cb_finish("budget_wall")
            elif completed:
                # The loop ended without a terminal reason from the stream: a step cap.
                self.cb_finish("max_steps")
            else:
                self.cb_finish(None)
            (self.logs_dir / "clm_stream.json").write_text(
                json.dumps(self._cb_record, indent=2, default=str) + "\n")

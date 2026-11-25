# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optional parallel sub-agents for ``ClmAgent`` (off by default; ``subagents=true``).

The interface is files, like the rest of the harness. The main agent launches a
sub-agent by creating or editing an idle slot file ``<ctx_dir>/SUBCTX_<i>.txt``:

* a file containing ``[[CTX_TURN`` headers is a FORK: it is parsed back like the
  context mirror and the sub-agent continues from those turns;
* anything else is a BRIEF: the whole file becomes the sub-agent's single user message.

Each sub-agent is an asyncio task running its own bounded ``bash`` loop in the same
sandbox, with its own persistent-shell state (``<state_dir>_sub/<sub_id>``) and its own
context budget. It ends when it runs the submit command (``completed``), runs out of
commands (``max_steps``) or LM calls (``max_calls``), reaches its context budget
(``budget``; one final turn is granted first), hits an LM error (``error:<type>``), or
is stopped: deleting its slot file stops it (``cancelled_by_parent``), and any worker
still running when the main agent ends is stopped too (``cancelled_end_of_run``).

When a sub-agent ends, its slot file moves to
``<ctx_dir>/finished_subagents/<sub_id>/brief.txt`` next to ``final_message.txt`` (its
last assistant text, capped), and the slot is free again. Sub-agent transcripts never
enter the main context: the main agent sees only files and a short status note appended
to its tool result. Sub-agents cannot launch sub-agents.

Host-side, each sub-agent is recorded under ``<logs_dir>/subagents/<sub_id>/``:
``context_snapshots/turn-NNNN.json`` (one per LM call plus the final context, same
format as the main agent's), ``trajectory.json``, ``timing.json``, ``summary.json``,
``brief.txt``, ``final_message.txt`` and ``link.json`` (written at launch and rewritten
when the sub-agent ends, so a crashed run still shows what was running).

``sub_id`` is ``subctx_<slot>_<n>``: slot ``slot``'s ``n``-th launch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..agent_trajectory_format.harness_export import response_metrics
from ..context_utils.context_string import parse_back
from ..utils import pricing
from ..utils import tokens as tk
from ..utils.budget import BudgetController
from ..utils.finish_policy import SUBMIT_MARKER, is_submit_command

logger = logging.getLogger(__name__)

_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"
with _PROMPTS_PATH.open(encoding="utf-8") as _fh:
    _PROMPTS: dict[str, Any] = yaml.safe_load(_fh)

SLOT_PREFIX = "SUBCTX_"
FINISHED_DIRNAME = "finished_subagents"
FORK_MARKER = "[[CTX_TURN"
#: ``final_message.txt`` keeps at most this many characters of the last assistant text.
FINAL_MESSAGE_CHARS = 4000
#: Each sub-agent is told once, at this fraction of its context limit, to save its findings
#: (it cannot compact; at the limit it gets one last command).
BUDGET_WARNING_FRACTION = 0.75
#: A sub-agent command naming one of these is refused: sub-agents cannot launch, edit or
#: stop sub-agents, and must not write the main agent's context mirror.
_WORKER_FORBIDDEN = (SLOT_PREFIX, "LIVE_CTX_MAIN")
_USAGE_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens", "cached_tokens",
)
_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@dataclass(frozen=True)
class SubagentConfig:
    """Sub-agent limits. ``lifetime_cap`` bounds the number of launches over the whole
    run (0 = unlimited); ``n_slots`` is also the concurrency limit."""

    n_slots: int = 5
    max_steps: int = 40
    lifetime_cap: int = 24

    @classmethod
    def from_kwargs(cls, n_slots: Any = 5, max_steps: Any = 40, lifetime_cap: Any = 24,
                    self_compact: Any = False) -> "SubagentConfig":
        if _as_bool(self_compact):
            raise NotImplementedError(
                "subagent_self_compact=true is not implemented: sub-agents do not edit "
                "their own context. They run under a fixed context budget and end (status "
                "'budget') after one final turn when they reach it. Leave "
                "subagent_self_compact unset or false.")
        cfg = cls(int(n_slots), int(max_steps), int(lifetime_cap))
        if cfg.n_slots < 1 or cfg.max_steps < 1 or cfg.lifetime_cap < 0:
            raise ValueError(
                "n_subagent_slots and subagent_max_steps must be >= 1 and "
                f"subagent_lifetime_cap >= 0 (0 = unlimited), got {cfg}")
        return cfg


def add_user_note(messages: list[dict[str, Any]], text: str) -> None:
    """Append ``text`` as a user turn, or to the last turn if that is already a user turn,
    so that no two user turns are adjacent."""
    last = messages[-1] if messages else None
    if last is not None and last.get("role") == "user":
        if isinstance(last.get("content"), list):
            last["content"] = [*last["content"], {"type": "text", "text": text}]
        else:
            last["content"] = (str(last.get("content") or "").rstrip("\n") + "\n\n" + text).lstrip()
    else:
        messages.append({"role": "user", "content": text})


def _zero_usage() -> dict[str, int]:
    return {k: 0 for k in _USAGE_KEYS}


@dataclass
class _Worker:
    slot: int
    launch: int
    sub_id: str
    mode: str                      # "brief" | "fork"
    brief: str
    messages: list[dict[str, Any]]
    dir: Path
    spawn_call: int                # main LM call (1-based) whose command launched it
    spawn_time_s: float
    spawn_command: str
    status: str = "running"
    steps: int = 0
    n_lm_calls: int = 0
    usage: dict[str, int] = field(default_factory=_zero_usage)
    cost_usd: float = 0.0
    metrics: list[dict[str, Any]] = field(default_factory=list)
    timing: list[dict[str, Any]] = field(default_factory=list)
    final_message: str = ""
    end_time_s: float | None = None
    fold_call: int | None = None   # main LM call whose tool result reported the end
    fold_time_s: float | None = None
    finishing: bool = False        # sandbox finalisation started (slot file may be gone)
    budget_warning_tokens: int | None = None  # context size when the 75% warning was given
    cancel_reason: str | None = None
    recorded: bool = False         # host-side record written
    task: asyncio.Task | None = None
    _snap: int = 0


class SubagentPool:
    """Slot scanning, sub-agent loops and their records for one ``ClmAgent`` run.

    ``agent`` is the owning ``ClmAgent`` (model settings, ``logs_dir``, LM-call
    counter), ``env`` its ``ContextEnv`` (command wrapping and output formatting) and
    ``paths`` its sandbox paths."""

    def __init__(self, agent: Any, env: Any, paths: Any, cfg: SubagentConfig) -> None:
        self.agent = agent
        self.env = env
        self.paths = paths
        self.cfg = cfg
        self._environment: Any = None
        self._t0 = time.monotonic()
        self._running: dict[int, _Worker] = {}
        self._last_hash: dict[int, str] = {}
        self._launch_no: dict[int, int] = {}
        self._workers: list[_Worker] = []
        self._cap_noted: dict[int, str] = {}
        self.n_cap_refusals = 0

    # ----------------------------------------------------------------- paths
    @property
    def ctx_dir(self) -> str:
        return self.paths.ctx_dir

    @property
    def finished_dir(self) -> str:
        return f"{self.ctx_dir}/{FINISHED_DIRNAME}"

    @property
    def state_root(self) -> str:
        return f"{self.paths.state_dir}_sub"

    def slot_path(self, slot: int) -> str:
        return f"{self.ctx_dir}/{SLOT_PREFIX}{slot}.txt"

    def _now(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    # --------------------------------------------------------------- prompts
    def prompt_addendum(self) -> str:
        """Section appended to the main agent's system prompt."""
        cap = self.cfg.lifetime_cap
        text = _PROMPTS["subagent_addendum"]
        for key, val in {
            "n_slots": self.cfg.n_slots, "last_slot": self.cfg.n_slots - 1,
            "ctx_dir": self.ctx_dir, "ctx_file": self.paths.ctx_file,
            "finished_dir": self.finished_dir, "max_steps": self.cfg.max_steps,
            "lifetime_clause": f", {cap} launches over the whole run" if cap else "",
        }.items():
            text = text.replace("{{" + key + "}}", str(val))
        return text.rstrip("\n")

    def system_prompt(self) -> str:
        return self._prompt("subagent_system")

    def _prompt(self, key: str, **values: Any) -> str:
        text = _PROMPTS[key].replace("{{max_steps}}", str(self.cfg.max_steps))
        for k, v in {"submit": SUBMIT_MARKER, **values}.items():
            text = text.replace("{{" + k + "}}", str(v))
        return text.strip("\n")

    def seed_messages(self, content: str) -> tuple[list[dict[str, Any]], str]:
        """A slot file's content -> ``(starting messages, mode)``.

        A fork gets harness notes so that it knows it is a sub-agent: a header before the
        copied turns when they do not start with a user turn, and the fork note after them.
        Both are user turns (merged into an adjacent user turn when there is one), so the
        list keeps a user turn first and alternating roles at the end."""
        sys_msg = {"role": "system", "content": self.system_prompt()}
        if FORK_MARKER in content:
            try:
                msgs = parse_back(content, [sys_msg])
                if len(msgs) > 1:
                    msgs[0] = dict(sys_msg)
                    if msgs[1].get("role") != "user":
                        msgs.insert(1, {"role": "user",
                                        "content": self._prompt("subagent_fork_header")})
                    add_user_note(msgs, self._prompt("subagent_fork_note"))
                    return msgs, "fork"
            except Exception:  # noqa: BLE001 - a malformed fork is read as a brief
                logger.warning("SUBCTX fork could not be parsed; reading it as a brief")
        return [sys_msg, {"role": "user", "content": content.strip()}], "brief"

    # ----------------------------------------------------------------- setup
    async def setup(self, environment: Any) -> None:
        self._environment = environment
        self._t0 = time.monotonic()
        await environment.exec(
            command=f"mkdir -p {self.finished_dir} {self.state_root}", timeout_sec=10)

    # ------------------------------------------------------------------ poll
    async def poll(self, command: str) -> tuple[str, bool]:
        """Run after each main-agent command: report sub-agents that ended, stop those
        whose slot file was deleted, launch new or edited idle slots.

        Returns ``(note, spawned)``: the note to append to the main agent's tool result
        ("" when there is nothing to say) and whether anything was launched."""
        notes: list[str] = []
        for slot, w in list(self._running.items()):
            if w.task is not None and w.task.done():
                self._fold(w)
                notes.append(f"{SLOT_PREFIX}{slot} finished ({w.status}, {w.steps} steps) "
                             f"-> {self.finished_dir}/{w.sub_id}/")
        present = await self._present_slots()
        spawned = False
        if present is not None:
            for slot, w in list(self._running.items()):
                if slot in present or w.finishing:
                    continue
                await self._stop(w, "cancelled_by_parent")
                self._fold(w)
                notes.append(f"{SLOT_PREFIX}{slot} stopped (slot file deleted, {w.steps} "
                             f"steps) -> {self.finished_dir}/{w.sub_id}/")
            for slot in range(self.cfg.n_slots):
                if slot in self._running or slot not in present:
                    continue
                note, launched = await self._maybe_launch(slot, command)
                if note:
                    notes.append(note)
                spawned = spawned or launched
        line = self.status_line()
        if line:
            notes.append(line)
        return ("\n[" + "; ".join(notes) + "]") if notes else "", spawned

    def status_line(self) -> str:
        if not self._workers:
            return ""
        running = sum(1 for w in self._running.values() if w.task and not w.task.done())
        cap = f" (cap {self.cfg.lifetime_cap})" if self.cfg.lifetime_cap else ""
        return (f"sub-agents: {running}/{self.cfg.n_slots} running, "
                f"{len(self._workers)} launched{cap}")

    async def _present_slots(self) -> set[int] | None:
        try:
            res = await self._environment.exec(
                command=f"ls -1 {self.ctx_dir} 2>/dev/null", timeout_sec=10)
        except Exception as exc:  # noqa: BLE001 - a failed scan only delays launches
            logger.warning("sub-agent slot scan failed: %s", exc)
            return None
        present: set[int] = set()
        for name in (getattr(res, "stdout", "") or "").split():
            if name.startswith(SLOT_PREFIX) and name.endswith(".txt"):
                idx = name[len(SLOT_PREFIX):-4]
                if idx.isdigit() and int(idx) < self.cfg.n_slots:
                    present.add(int(idx))
        return present

    async def _read_slot(self, slot: int) -> str | None:
        fd, tmp = tempfile.mkstemp(prefix=f"subctx_{slot}_", suffix=".txt")
        os.close(fd)
        try:
            await self._environment.download_file(self.slot_path(slot), tmp)
            return Path(tmp).read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - vanished between scan and read
            return None
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def _maybe_launch(self, slot: int, command: str) -> tuple[str, bool]:
        """Launch the idle slot ``slot`` if its file is new or changed. Returns
        ``(note, launched)``."""
        content = await self._read_slot(slot)
        if not content or not content.strip():
            return "", False
        digest = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()
        if self._last_hash.get(slot) == digest:
            return "", False
        cap = self.cfg.lifetime_cap
        if cap and len(self._workers) >= cap:
            if self._cap_noted.get(slot) == digest:
                return "", False
            self._cap_noted[slot] = digest
            self.n_cap_refusals += 1
            return (f"{SLOT_PREFIX}{slot} not launched: lifetime cap of {cap} sub-agents "
                    "reached"), False
        self._last_hash[slot] = digest
        n = self._launch_no[slot] = self._launch_no.get(slot, 0) + 1
        sub_id = f"subctx_{slot}_{n}"
        messages, mode = self.seed_messages(content)
        w = _Worker(
            slot=slot, launch=n, sub_id=sub_id, mode=mode, brief=content, messages=messages,
            dir=Path(self.agent.logs_dir) / "subagents" / sub_id,
            spawn_call=int(getattr(self.agent, "n_lm_calls", 0)), spawn_time_s=self._now(),
            spawn_command=(command or "")[:200],
        )
        w.dir.mkdir(parents=True, exist_ok=True)
        (w.dir / "brief.txt").write_text(content, encoding="utf-8")
        self._write_link(w)
        self._workers.append(w)
        self._running[slot] = w
        w.task = asyncio.create_task(self._run(w), name=sub_id)
        return f"{SLOT_PREFIX}{slot} launched as {sub_id} ({mode})", True

    def _fold(self, w: _Worker) -> None:
        self._running.pop(w.slot, None)
        self._last_hash.pop(w.slot, None)  # an identical brief may be launched again
        w.fold_call = int(getattr(self.agent, "n_lm_calls", 0))
        w.fold_time_s = self._now()
        self._write_link(w)

    async def _stop(self, w: _Worker, reason: str) -> None:
        if w.task is None or w.task.done():
            return
        w.cancel_reason = reason
        w.task.cancel()
        try:
            await w.task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub-agent %s raised while stopping: %s", w.sub_id, exc)
        if not w.recorded:  # cancelled before its first step ran: record it here
            w.status = reason
            w.end_time_s = self._now()
            self._write_record(w)

    async def cancel_all(self) -> None:
        """Stop every running sub-agent (the main agent is done). Each still leaves its
        host-side record; the sandbox is not touched."""
        for w in list(self._running.values()):
            await self._stop(w, "cancelled_end_of_run")
        self._running.clear()

    # ------------------------------------------------------------ the worker
    async def _run(self, w: _Worker) -> dict[str, Any]:
        try:
            await self._loop(w)
        except asyncio.CancelledError:
            w.status = w.cancel_reason or "cancelled"
            # An LM call cut short has no usage; keep metrics 1:1 with calls anyway.
            w.metrics += [{}] * max(w.n_lm_calls - len(w.metrics), 0)
        except Exception as exc:  # noqa: BLE001 - a broken worker must not kill the run
            logger.warning("sub-agent %s failed: %s: %s", w.sub_id, type(exc).__name__, exc)
            w.status = f"error:{type(exc).__name__}"
        w.end_time_s = self._now()
        w.final_message = self._last_assistant_text(w.messages)
        self._write_record(w)
        if w.status != "cancelled_end_of_run":
            try:
                await self._finalize_sandbox(w)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("sub-agent %s: sandbox finalisation failed: %s", w.sub_id, exc)
        return self.summary(w)

    async def _loop(self, w: _Worker) -> None:
        from .harness import ClmAgent, _extract_tool_call, _is_context_overflow

        agent, env, environment = self.agent, self.env, self._environment
        model = agent.model_name or _DEFAULT_MODEL
        state_dir = f"{self.state_root}/{w.sub_id}"
        if env.persistent_bash:
            # Start from the main agent's working directory and exported environment.
            main = self.paths.state_dir
            await environment.exec(
                command=(f"mkdir -p {state_dir} && (cp {main}/cwd {main}/env {state_dir}/ "
                         "2>/dev/null || true)"), timeout_sec=10)
        budget = BudgetController(
            agent.context_budget_tokens or 0,
            reserve_tokens=agent.context_budget_reserve_tokens,
            nudge_ratios=[], max_num_retry_on_limit=0, protect_prefix=1,
        )
        messages = w.messages
        warned = False
        w.status = "max_steps"
        while w.steps < self.cfg.max_steps:
            if w.n_lm_calls >= 2 * self.cfg.max_steps:
                w.status = "max_calls"
                break
            dec = budget.evaluate(messages)
            messages.extend(dec.messages)
            if dec.log:
                w.timing.append({"step": w.steps + 1, **dec.log})
            if dec.action == "stop":
                w.status = "budget"
                break
            if dec.action == "continue" and not warned and budget.enabled:
                tokens = budget.count(messages)
                if tokens >= BUDGET_WARNING_FRACTION * budget.strict_target:
                    warned = True
                    w.budget_warning_tokens = tokens
                    add_user_note(messages, self._prompt(
                        "subagent_budget_warning", tokens=tokens, limit=budget.strict_target,
                        pct=round(100 * tokens / budget.strict_target)))
                    w.timing.append({"step": w.steps + 1, "budget_warning": True,
                                     "tokens": tokens})
            self._snapshot(w, messages, "agent")
            w.n_lm_calls += 1
            t0 = time.monotonic()
            try:
                # The base class's query path, never a subclass override: subclasses
                # (e.g. ContextBench's stream adapter) hook it for the main agent only.
                response = await ClmAgent._query_with_retry(agent, model, messages)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                w.metrics.append({})
                w.status = ("budget" if _is_context_overflow(exc)
                            else f"error:{type(exc).__name__}")
                break
            llm_s = round(time.monotonic() - t0, 1)
            budget.calibrate(response, messages)
            ClmAgent._accumulate_usage(response, w.usage)
            w.cost_usd += pricing.completion_cost_usd(response, model)
            w.metrics.append(response_metrics(response))

            msg = response.choices[0].message.model_dump()
            msg["content"] = msg.get("content") or ""
            action = _extract_tool_call(msg)
            if action["type"] != "command" and action["type"] != "done":
                if action["type"] == "context_status":
                    messages.append(msg)
                    messages.append({"role": "tool", "tool_call_id": action.get("tool_call_id") or "",
                                     "content": "(context_status is not available; use `bash`.)"})
                else:
                    msg.pop("tool_calls", None)
                    messages.append(msg)
                    messages.append({"role": "user", "content": (
                        "(No tool call in your last turn. Issue your next `bash` command; "
                        f"finish with `echo {SUBMIT_MARKER}`.)")})
                w.timing.append({"step": w.steps + 1, "llm_s": llm_s, "empty_turn": True})
                continue

            w.steps += 1
            command = action.get("command") or ""
            t1 = time.monotonic()
            refused = next((s for s in _WORKER_FORBIDDEN if s in command), None)
            if refused:
                rc = 1
                content = (f"[not run: sub-agents cannot touch {refused}* files]"
                           "\n\n(exit_code=1)")
            else:
                result = await environment.exec(command=env.wrap(command, state_dir=state_dir),
                                                timeout_sec=env.command_timeout)
                rc = getattr(result, "return_code", 1)
                content = env.append_submit_warning(command, env.format_tool_result(result))
                if budget.enabled:
                    content = budget.cap_newest_output(messages + [msg], content)
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": action.get("tool_call_id") or "",
                             "content": content})
            w.timing.append({"step": w.steps, "llm_s": llm_s,
                             "bash_s": round(time.monotonic() - t1, 1),
                             "return_code": rc, "cmd": command[:200],
                             **({"refused": True} if refused else {})})
            # Sub-agents always keep the submit command, whatever the main agent's
            # finish policy: finishing is how a slot is freed.
            if not refused and is_submit_command(command) and rc == 0 and SUBMIT_MARKER in content:
                w.status = "completed"
                break

    # ------------------------------------------------------------- recording
    def _snapshot(self, w: _Worker, messages: list[dict[str, Any]], kind: str) -> None:
        tk.record(w.dir, w._snap, messages, extra={"kind": kind})
        w._snap += 1

    @staticmethod
    def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
        """The last assistant turn's text; its reasoning when the content is empty (servers
        with a reasoning parser return a reply's text as ``reasoning_content``)."""
        for m in reversed(messages):
            if m.get("role") != "assistant":
                continue
            text = str(m.get("content") or "").strip()
            if text:
                if len(text) > FINAL_MESSAGE_CHARS:
                    text = text[:FINAL_MESSAGE_CHARS] + "\n[... truncated]"
                return text
            reasoning = str(m.get("reasoning_content") or "").strip()
            if reasoning:  # keep the end, where the conclusion is
                if len(reasoning) > FINAL_MESSAGE_CHARS:
                    reasoning = "[... truncated]\n" + reasoning[-FINAL_MESSAGE_CHARS:]
                return reasoning
        return ""

    def summary(self, w: _Worker) -> dict[str, Any]:
        return {
            "sub_id": w.sub_id, "slot": w.slot, "launch": w.launch, "mode": w.mode,
            "status": w.status, "steps": w.steps, "n_lm_calls": w.n_lm_calls,
            "max_steps": self.cfg.max_steps, "spawn_time_s": w.spawn_time_s,
            "end_time_s": w.end_time_s, "budget_warning_tokens": w.budget_warning_tokens,
            "cost_usd": w.cost_usd, **w.usage,
            "finished_dir": f"{self.finished_dir}/{w.sub_id}",
        }

    def _link(self, w: _Worker) -> dict[str, Any]:
        return {
            "sub_id": w.sub_id, "slot": w.slot, "launch": w.launch, "mode": w.mode,
            "status": w.status,
            "spawn": {"main_lm_call": w.spawn_call, "time_s": w.spawn_time_s,
                      "path": self.slot_path(w.slot), "command": w.spawn_command},
            "fold": (None if w.fold_call is None else
                     {"main_lm_call": w.fold_call, "time_s": w.fold_time_s,
                      "path": f"{self.finished_dir}/{w.sub_id}"}),
        }

    def _write_link(self, w: _Worker) -> None:
        try:
            w.dir.mkdir(parents=True, exist_ok=True)
            (w.dir / "link.json").write_text(json.dumps(self._link(w), indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001 - recording must not crash a run
            logger.warning("sub-agent %s: link.json not written: %s", w.sub_id, exc)

    def _write_record(self, w: _Worker) -> None:
        try:
            self._snapshot(w, w.messages, "final")
            (w.dir / "trajectory.json").write_text(
                json.dumps(w.messages, indent=2, default=str) + "\n")
            (w.dir / "timing.json").write_text(json.dumps(w.timing, indent=2) + "\n")
            (w.dir / "summary.json").write_text(json.dumps(self.summary(w), indent=2) + "\n")
            (w.dir / "final_message.txt").write_text(w.final_message, encoding="utf-8")
            self._write_link(w)
            w.recorded = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub-agent %s: record not written: %s", w.sub_id, exc)

    async def _finalize_sandbox(self, w: _Worker) -> None:
        w.finishing = True
        fin = f"{self.finished_dir}/{w.sub_id}"
        res = await self._environment.exec(
            command=(f"mkdir -p {fin} && (mv {self.slot_path(w.slot)} {fin}/brief.txt "
                     f"2>/dev/null || true) && test -f {fin}/brief.txt"), timeout_sec=10)
        if getattr(res, "return_code", 1) != 0:  # the slot file was deleted: restore it
            await self._environment.upload_file(str(w.dir / "brief.txt"), f"{fin}/brief.txt")
        await self._environment.upload_file(str(w.dir / "final_message.txt"),
                                            f"{fin}/final_message.txt")

    # ------------------------------------------------------------ accounting
    def records(self) -> list[dict[str, Any]]:
        """One dict per launched sub-agent, in launch order, for the trajectory export."""
        return [{**self.summary(w), "dir": str(w.dir), "metrics": list(w.metrics),
                 "spawn_call": w.spawn_call, "fold_call": w.fold_call,
                 "link": self._link(w)} for w in self._workers]

    def usage_report(self, main_usage: dict[str, int], main_cost: float, *, n_lm_calls: int,
                     n_bash: int) -> tuple[dict[str, int], float, dict[str, Any]]:
        """Run totals for usage.json: ``(token totals, USD total, extra keys)``.

        Token keys and ``cost_usd`` at the top level become main + sub-agents; the extra
        keys add ``"main"`` and ``"subagents"`` blocks that sum to them."""
        sub = self.usage()
        totals = {k: v + int(sub.get(k, 0)) for k, v in main_usage.items()}
        extra = {
            "usage_scope": "main+subagents",
            "main": {**main_usage, "cost_usd": main_cost, "n_lm_calls": n_lm_calls,
                     "n_bash": n_bash},
            "subagents": sub,
        }
        return totals, main_cost + sub["cost_usd"], extra

    def usage(self) -> dict[str, Any]:
        """Totals over all sub-agents (tokens, USD, calls) plus counts by status."""
        out: dict[str, Any] = {k: sum(w.usage[k] for w in self._workers) for k in _USAGE_KEYS}
        by_status: dict[str, int] = {}
        for w in self._workers:
            by_status[w.status] = by_status.get(w.status, 0) + 1
        out.update(
            cost_usd=sum(w.cost_usd for w in self._workers),
            n_lm_calls=sum(w.n_lm_calls for w in self._workers),
            n_bash=sum(w.steps for w in self._workers),
            n_launched=len(self._workers),
            n_lifetime_cap_refusals=self.n_cap_refusals,
            by_status=by_status,
            n_slots=self.cfg.n_slots, max_steps=self.cfg.max_steps,
            lifetime_cap=self.cfg.lifetime_cap,
        )
        return out

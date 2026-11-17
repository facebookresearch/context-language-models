# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ClmAgent — context editing happens naturally in bash.

A single ``bash`` tool: no dedicated compaction tool. The harness mirrors the
model's editable context to ``/tmp/.live_ctx/LIVE_CTX_MAIN.txt`` in the sandbox
before each command; the model compacts by editing that file with ordinary shell
tools (``sed``, ``python3 - <<'PY' …``, ``cat >`` …). After the command runs the
harness reads the file back and, if it changed, maps it into the conversation via
:func:`clm_harness.context_utils.context_string.reconcile_editable` (protected system/task
prefix pinned, unchanged messages preserved, broken tool exchanges summarized).

Because edits execute inside the sandbox VM (not via an in-process ``exec``), there
is no sandboxed-code-execution concern.

Run it with Harbor (with ``clm/`` on ``PYTHONPATH``)::

    harbor trial start -p <task_dir> -e docker \
      -a clm_harness.clm_agent.harness:ClmAgent \
      -m openai/<served-model> \
      --agent-kwarg api_base=http://localhost:8008/v1 \
      --agent-kwarg context_budget_tokens=32000 \
      --agent-kwarg cost_metric=usd
"""

from __future__ import annotations

import asyncio
import json

from clm_harness.utils.provenance import code_fingerprint
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import litellm
import yaml
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ..agent_trajectory_format.harness_export import export_ctx_trajectory, response_metrics
from ..agent_trajectory_format.models import Agent as CtxAgent
from ..flops_metrics import resolve_n_body
from ..utils import tokens as tk
from ..utils import pricing
from ..utils.budget import BudgetController
from ..utils.finish_policy import SUBMIT_MARKER, FinishPolicy, is_submit_command
from ..utils.resume import Checkpointer, ResumeState
from ..utils.skills import SkillMount
from ..utils.lm_compat import token_cap_key
from ..context_env.edit_gate import resolve_allow_edit_growth
from ..context_env.env import ContextEnv
from ..context_env.paths import Paths
from ..utils.tool_schemas import BASH_TOOL
from ..task_templates import instance_template as _load_instance_template

os.environ.setdefault("OPENAI_API_KEY", "dummy")
# Register public list prices so litellm can cost gateway models it doesn't know.
pricing.register_default_prices()

logger = logging.getLogger(__name__)


_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"
with _PROMPTS_PATH.open(encoding="utf-8") as _fh:
    _PROMPTS: dict[str, Any] = yaml.safe_load(_fh)
_SYSTEM_TEMPLATE: str = _PROMPTS["system_template"]
_OBS_CFG: dict[str, Any] = _PROMPTS["observation"]

_ABORT_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.ContextWindowExceededError,
    litellm.exceptions.UnsupportedParamsError,
    litellm.exceptions.PermissionDeniedError,
)
_CONTEXT_OVERFLOW_RE = re.compile(
    r"(maximum context length|context length of|reduce the (?:number of )?tokens|"
    r"decrease (?:the )?(?:input |prompt )?(?:length|tokens)|too many tokens|"
    r"exceeds? the model'?s? (?:maximum )?context|"
    # gateway phrasings, e.g. "prompt is too long: 3062409 tokens > 1000000 maximum"
    r"prompt is too long|input (?:is )?too long|context_length_exceeded|"
    r"tokens?\s*>\s*\d[\d,]*\s*maximum|too many input tokens)",
    re.IGNORECASE,
)
# Attempts per LM call on transient errors (env-overridable for flaky endpoints).
# The retry delay is jittered up to 60s, so deep retries do not stampede.
_MAX_RETRIES = int(os.environ.get("CLM_LLM_MAX_RETRIES", 5))
_RETRY_BASE_DELAY = 2.0
# Per-call LLM timeout: short so a stalled response fails fast and the retry loop recovers.
_LLM_TIMEOUT_SECONDS = 600
_LLM_OUTER_TIMEOUT_BUFFER = 30
_STATE_DIR = "/tmp/.bash_ctx_state"        # persistent-bash cwd/env snapshot
_CTX_DIR = "/tmp/.live_ctx"                # the context mirror lives here
_CTX_FILE = f"{_CTX_DIR}/LIVE_CTX_MAIN.txt"
_PROTECT = 2                               # system + initial task are pinned

# Settings of experimental features that are not part of this release. Passing one is an
# error rather than a silent no-op, so a stale launcher cannot run a different
# configuration from the one it names.
_REMOVED_KWARGS = frozenset({
    "shadow_hints", "shadow_set", "shadow_codex_ratio", "shadow_cooldown",
    "shadow_acm_ratio", "shadow_mandate_wording", "shadow_progress_guard",
    "shadow_pg_predicate", "shadow_pg_novelty_max", "shadow_pg_jaccard",
    "shadow_pg_window", "shadow_lag_fire", "shadow_refire_dedup",
    "shadow_wording_file", "shadow_wording_id", "shadow_wording_op",
    "context_plugins", "auto_hide_threshold", "gauge_inventory", "edit_echo",
    "gauge_v4", "extract_threshold", "gauge_solo", "gauge_sweep", "span_enforce",
    "seed", "cost_limit", "obs_offload", "authoring_harvest", "auto_apply",
    "research_ledger", "unlimited_ctx_turns", "enforce_budget",
    "capture_prompt_token_ids", "context_tool", "subagents", "n_subagent_slots",
    "subagent_max_steps", "subagent_lifetime_cap", "subagent_self_compact",
    "ctx_prompts", "ctx_metadata", "ctx_archive", "refresh_rollback_ledger",
})


def ctx_turn_is_free(*, ctx_changed: bool, return_code: int, task_output_empty: bool) -> bool:
    """Decide whether a completed bash turn consumes a task step: only a pure
    compaction -- mirror changed, no task output, exit 0 -- is free."""
    return bool(ctx_changed and task_output_empty and return_code == 0)


def _is_context_overflow(exc: BaseException) -> bool:
    if isinstance(exc, litellm.exceptions.ContextWindowExceededError):
        return True
    return bool(_CONTEXT_OVERFLOW_RE.search(str(exc)))


def _extract_tool_call(msg: dict) -> dict[str, Any]:
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "tool_call_id": None}
    tc = tool_calls[0]
    # Keep only the executed call; a multi tool_call turn otherwise leaves unpaired
    # tool_use ids (no tool_result) that Anthropic/Bedrock reject with a 400.
    msg["tool_calls"] = [tc]
    fn = tc.get("function", {})
    tool_call_id = tc.get("id")
    if fn.get("name", "") == "context_status":
        return {"type": "context_status", "tool_call_id": tool_call_id}
    if fn.get("name", "") != "bash":
        return {"type": "no_tool_call", "tool_call_id": tool_call_id}
    args_raw = fn.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return {"type": "no_tool_call", "tool_call_id": tool_call_id}
    else:
        args = args_raw
    if not isinstance(args, dict):
        return {"type": "no_tool_call", "tool_call_id": tool_call_id}
    # Some servers infer a type for untyped tool arguments, so a numeric-looking
    # command can arrive as an int rather than a str.
    command = str(args.get("command") or "").strip()
    if is_submit_command(command):
        return {"type": "done", "command": command, "tool_call_id": tool_call_id}
    return {"type": "command", "command": command, "tool_call_id": tool_call_id}


class ClmAgent(BaseAgent):
    """Bash-only agent that compacts context by editing a mirrored context file."""

    @staticmethod
    def name() -> str:
        return "clm-agent"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 64,
        temperature: float = 0.7,
        top_p: float | None = 0.95,
        max_tokens: int = 16384,
        api_base: str | None = None,
        command_timeout: int = 120,
        persistent_bash: bool = True,
        observation_max_chars: int = 10000,
        context_budget_tokens: int | None = None,
        context_budget_reserve_tokens: int = 2048,
        # Escalating nudges at these budget fractions (each fires once, re-arms below).
        nudge_ratios: str = "0.25,0.5,0.75",
        # Fraction of strict_target above which an URGENT nudge fires EVERY turn
        # (persistent, not one-shot). "none" turns it off. "adaptive" sizes the band from
        # recent tool outputs: it fires while the headroom is below
        # max(10% of the limit, persistent_nudge_obs_mult x the largest of the last 3
        # outputs), never below 50% of the limit (see BudgetController).
        persistent_nudge_ratio: float | str | None = "adaptive",
        persistent_nudge_obs_mult: float = 2.0,
        # On a context-limit hit, roll the newest turns back off the context until
        # `retry_edit_margin_tokens` are free and demand a compaction, up to N times,
        # instead of ending the run after one final turn. 0 = final turn, then stop.
        max_num_retry_on_limit: int = 50,
        retry_edit_margin_tokens: int = 2048,
        finalize_nudge_turns: int = 3,
        finalize_message: str | None = None,
        # Cap on the main agent's LM calls. Free context-editing turns do not use up
        # task steps, so this is what bounds the total number of calls. None means
        # 2 * max_steps + 24; a positive value sets the cap; 0 (or negative) disables it,
        # leaving only the loop's own bound of roughly 2 * max_steps calls.
        lm_call_cap: int | str | None = None,
        # Count the finalize notice down on whichever budget runs out first: task
        # steps or LM calls left under lm_call_cap. False counts task steps only.
        finalize_on_lm_call_cap: bool | str = True,
        enable_thinking: bool = True,
        send_chat_template_kwargs: bool = True,
        emit_ctx_trajectory: bool = True,
        flops_model_key: str | None = None,
        flops_n_body: float | None = None,
        flops_tokenizer: str | None = None,
        cost_metric: str = "auto",
        # Optional agent skills (task-agnostic; see clm_harness.utils.skills). Keys come from
        # each skill's own manifest; skill_env_keys only OVERRIDES that.
        skill_dirs: str | None = None,
        skill_env_keys: str | None = None,
        skill_mount_dir: str = "/tmp/harbor_skills",
        task_template: str | None = None,
        resume_from: str | None = None,  # warm restart from a trial dir (see utils.resume)
        # Mirror the sandbox and the live context to the host every N seconds, so an
        # interrupted run can be resumed from at most N seconds ago. 0 == off.
        checkpoint_interval_s: float = 0.0,
        # Edit gate. "fit" (or True, default): accept an edit if the result fits the budget.
        # "shrink" (or False): accept an edit only if it makes the context smaller. Either
        # way a rejected edit is not applied and the model gets a note it can act on.
        # None = unset: the CLM_EDIT_GATE env var (fit|shrink) decides, default fit.
        allow_edit_growth: bool | str | None = None,
        # Compaction wording in the nudges. Only "default" exists in this release;
        # "auto" resolves to it.
        ctx_hint: str = "auto",
        # "submit" (default): the agent ends the run with the submit command.
        # "open_ended": no finish command; the run ends at its step/time limit.
        finish_policy: str = "submit",
        **kwargs: Any,
    ) -> None:
        removed = sorted(k for k in kwargs if k in _REMOVED_KWARGS)
        if removed:
            raise ValueError(
                f"ClmAgent no longer accepts {', '.join(removed)}: these settings belonged to "
                "experimental features that are not part of this release. Remove them from "
                "the agent kwargs.")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.finish_policy = FinishPolicy(finish_policy)
        self.max_steps = int(max_steps)
        # temperature / top_p may be disabled by passing "none"/"" (some endpoints
        # reject requests that specify both).
        self.temperature = self._as_float_or_none(temperature)
        self.top_p = self._as_float_or_none(top_p)
        self.max_tokens = int(max_tokens)
        self.api_base = api_base
        self.command_timeout = int(command_timeout)
        self.persistent_bash = self._as_bool(persistent_bash)
        self.observation_max_chars = int(observation_max_chars)
        self.context_budget_tokens = self._parse_budget(context_budget_tokens)
        self.context_budget_reserve_tokens = int(context_budget_reserve_tokens)
        self.nudge_ratios = sorted(
            f for f in (float(x) for x in str(nudge_ratios).split(",") if x.strip()) if 0.0 < f < 1.0
        )
        # Shared context-limit mechanism (see clm_harness.utils.budget).
        self.persistent_nudge_ratio: float | str | None = (
            "adaptive"
            if isinstance(persistent_nudge_ratio, str)
            and persistent_nudge_ratio.strip().lower() == "adaptive"
            else self._as_float_or_none(persistent_nudge_ratio)
        )
        self.persistent_nudge_obs_mult = float(persistent_nudge_obs_mult)
        self.max_num_retry_on_limit = max(0, int(max_num_retry_on_limit))
        self.retry_edit_margin_tokens = int(retry_edit_margin_tokens)
        self.allow_edit_growth = resolve_allow_edit_growth(allow_edit_growth)
        self.ctx_hint = ctx_hint
        hint = (
            f"Compact {_CTX_FILE} by locating stale regions with code (match a turn by its "
            "[[CTX_TURN i ...]] header or a block by short start/end anchors) and replacing "
            "them with summaries — do not retype the text you remove. Compact settled spans; "
            "keep anything you have not finished using."
        )
        want_hint = (self.ctx_hint or "auto").strip().lower()
        if want_hint == "auto":
            want_hint = "default"
        if want_hint != "default":
            raise ValueError(f"ctx_hint must be auto|default, got {self.ctx_hint!r}")
        self.ctx_hint_used = want_hint
        self._budget = BudgetController(
            self.context_budget_tokens,
            reserve_tokens=self.context_budget_reserve_tokens,
            nudge_ratios=self.nudge_ratios,
            persistent_nudge_ratio=self.persistent_nudge_ratio,
            persistent_nudge_obs_mult=self.persistent_nudge_obs_mult,
            max_num_retry_on_limit=self.max_num_retry_on_limit,
            retry_edit_margin_tokens=self.retry_edit_margin_tokens,
            protect_prefix=_PROTECT,
            compaction_hint=hint,
        )
        self.finalize_nudge_turns = int(finalize_nudge_turns)
        self.finalize_message = finalize_message
        if lm_call_cap is None or (isinstance(lm_call_cap, str)
                                   and lm_call_cap.strip().lower() in ("", "none", "null")):
            self.lm_call_cap = 2 * self.max_steps + 24
        else:
            self.lm_call_cap = max(0, int(lm_call_cap))
        self.finalize_on_lm_call_cap = self._as_bool(finalize_on_lm_call_cap)
        self.n_lm_calls = 0
        self.hit_lm_call_cap = False
        self.enable_thinking = self._as_bool(enable_thinking)
        self.send_chat_template_kwargs = self._as_bool(send_chat_template_kwargs)
        self.emit_ctx_trajectory = self._as_bool(emit_ctx_trajectory)
        self.flops_model_key = flops_model_key
        self.flops_tokenizer = flops_tokenizer
        self.cost_metric = cost_metric
        # Fail fast (not after a long rollout) if FLOPs accounting is requested but the
        # model size can't be resolved. USD-costed API runs don't need it.
        self.flops_n_body = flops_n_body
        if self.emit_ctx_trajectory:
            flops_required = (self.cost_metric or "auto").lower() not in ("usd", "api", "closed", "cost")
            self.flops_n_body = resolve_n_body(
                self.flops_model_key, self.flops_n_body, required=flops_required
            )
        self.cost: float = 0.0
        self._skills = SkillMount(skill_dirs, skill_env_keys, skill_mount_dir)
        self._protect = _PROTECT
        self._ctx = ContextEnv(
            paths=Paths(sys.modules[__name__]),
            budget=self._budget,
            protect=self._protect,
            persistent_bash=self.persistent_bash,
            command_timeout=self.command_timeout,
            observation_max_chars=self.observation_max_chars,
            obs_cfg=_OBS_CFG,
            allow_edit_growth=self.allow_edit_growth,
            context_budget_tokens=self.context_budget_tokens or 0,
        )
        self._host_mirror = self._ctx.host_mirror
        self.instance_template = _load_instance_template(task_template)
        self._resume = ResumeState(resume_from)
        self._ckpt = Checkpointer(logs_dir, checkpoint_interval_s)
        # Rollback bookkeeping: the ledger is pinned into the protected
        # prefix the first time a rollback happens, so `_protect` grows by one.
        self._retry_ledger: dict[str, Any] | None = None
        self._ledger_index: int | None = None
        self._rolled_back_cmds: list[str] = []
        # FLOPs accounting: ONE monotonic counter over ALL LM calls (see _snap).
        self._snap_step = 0

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _as_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip().lower() in ("", "none", "null"):
                return None
            return float(value)
        return float(value)

    # ------------------------------------------------------------------
    # Per-LM-call context snapshots — EVERY LM call is charged.
    #
    # ONE monotonic counter per task over all LM calls, in true execution order,
    # each snapshot carrying a ``"kind"`` field, written to
    # ``<logs_dir>/context_snapshots/turn-<NNNN>.json``; a prefix-cache-aware
    # FLOPs replay can walk this stream directly.
    #
    # The untagged ``tk.record`` calls write to the same directory and filename
    # pattern from their own counter. ``_snap`` runs second and overwrites, so the
    # only untagged file left is a duplicate of the final context, which a replay
    # should skip because it has no ``kind``.
    #
    # Must be handed the EXACT message list about to be sent, immediately before
    # the call, so a live-context edit registers as a cache-breaking divergence.
    # ------------------------------------------------------------------
    def _snap(self, messages: list[dict[str, Any]], kind: str) -> None:
        step = self._snap_step
        self._snap_step += 1
        tk.record(self.logs_dir, step, messages, extra={"kind": kind})

    @staticmethod
    def _parse_budget(value: Any) -> int:
        """Parse context_budget_tokens. ``None`` (unset) -> FAIL FAST; ``0``/``-1`` (any
        <=0) -> unlimited = the model's full context window (stored internally as 0);
        positive -> the strict token budget."""
        if value is None:
            raise ValueError(
                "context_budget_tokens must be set explicitly: pass a positive integer for a "
                "strict token budget, or 0/-1 for unlimited (the model's full context window). "
                "Refusing to run with an unset budget so a run never accidentally uses the full "
                "window and explodes cost."
            )
        n = int(value)
        return 0 if n <= 0 else n

    # ------------------------------------------------------------------ setup
    async def setup(self, environment: BaseEnvironment) -> None:
        await self._ctx.setup_dirs(environment)
        if self._skills.enabled:
            await self._skills.install(
                environment, state_dir=_STATE_DIR, persistent_bash=self.persistent_bash
            )
        await self._resume.restore(environment)

    # ------------------------------------------------------- rollback ledger
    def _note_rollback(
        self, messages: list[dict[str, Any]], dropped: list[dict[str, Any]], after: int
    ) -> None:
        """Record the rollback in a ledger pinned just after the protected prefix.

        A rollback also erases the model's memory of what it just tried, so without a
        durable trace it re-issues the same command, blows the budget again and livelocks.
        The ledger survives every rollback and every context edit, and names the commands
        whose output caused the overflow."""
        for m in dropped:
            for tc in m.get("tool_calls") or []:
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                except Exception:  # noqa: BLE001 - malformed args are not worth failing on
                    cmd = ""
                cmd = " ".join(cmd.split())[:70]
                if cmd and cmd not in self._rolled_back_cmds:
                    self._rolled_back_cmds.append(cmd)
        del self._rolled_back_cmds[:-5]
        content = self._budget.rollback_message(len(dropped), after, self._rolled_back_cmds)
        if self._ledger_index is None:
            self._ledger_index = self._protect
            messages.insert(self._protect, {"role": "user", "content": content})
            self._protect += 1
            self._budget.protect_prefix = self._protect
            self._ctx.protect = self._protect
        else:
            # An applied context edit rebuilds the protected prefix from copies
            # (context reconciliation), so update the ledger where it sits in `messages` now.
            messages[self._ledger_index]["content"] = content
        self._retry_ledger = messages[self._ledger_index]

    # ------------------------------------------------------------------- run
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        model = self.model_name or "anthropic/claude-haiku-4-5"
        budget_str = (
            # Advertise the enforced budget (strict_target), not the raw window.
            f"{self._budget.strict_target or self.context_budget_tokens} tokens"
            if self.context_budget_tokens
            else "your model's full context window"
        )
        system_content = _SYSTEM_TEMPLATE.replace("{{context_budget}}", budget_str)
        system_content = system_content.replace(
            "{{finish_instructions}}", self.finish_policy.prompt_fragment()
        )
        if self._skills.enabled:
            system_content += "\n\n---\n\n" + self._skills.prompt_suffix()
        instance_prompt = self.instance_template.replace("{{task}}", instruction.strip())

        messages: list[dict[str, Any]] = self._resume.replay(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": instance_prompt},
            ],
            protect=_PROTECT,
            budget_tokens=self._budget.strict_target,
        )
        self._resume.write_manifest(self.logs_dir)

        timing_log: list[dict[str, Any]] = []
        agent_step_metrics: list[dict[str, Any]] = []  # per-kept-turn usage, for the ATIF export
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            # Prompt-cache accounting: these keys must exist here or
            # _accumulate_usage drops the provider fields.
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cached_tokens": 0,
        }
        n_bash = 0
        # Edit counters (syncs / grew / rejected) live on ContextEnv, where the edit
        # gate decides.
        free_ctx_turns = 0      # pure compaction turns that did NOT consume a task step

        strict_target = self._budget.strict_target

        # `step` counts TASK steps toward max_steps; a pure compaction turn is FREE (see
        # below). `total_iters` bounds ALL LLM calls so free turns can't loop unbounded.
        step = 0
        total_iters = 0
        # Each allowed retry needs at least one extra LLM call (the forced compaction), so
        # grant headroom for them — still bounded, so a thrashing run can't loop forever.
        # lm_call_cap, when set, is checked separately.
        max_total_iters = 2 * self.max_steps + 8 + min(self.max_num_retry_on_limit, self.max_steps)

        try:
            while step < self.max_steps and total_iters < max_total_iters:
                total_iters += 1
                logger.info("Step %s/%s (iter %s)", step + 1, self.max_steps, total_iters)

                fin = self._finalize_nudge(step)
                if fin is not None:
                    messages.append(fin)

                tk.record(self.logs_dir, total_iters, messages)
                await self._ckpt.maybe(environment, messages)

                # Context-limit mechanism (clm_harness.utils.budget): token gate, escalating
                # nudges, and rollback-retry or a final turn on overflow.
                dec = self._budget.evaluate(messages)
                messages.extend(dec.messages)
                if dec.log:
                    timing_log.append({"step": step + 1, **dec.log})
                if dec.action == "rollback":
                    # Discard the newest turns to free room, then demand a compaction. The
                    # dropped turns are gone from `messages`, so they never reach the final
                    # trajectory; only the retry count records that they happened.
                    dropped, after = self._budget.rollback_to_margin(
                        messages, protect=self._protect
                    )
                    self._note_rollback(messages, dropped, after)
                    logger.warning(
                        "context limit hit (%d > %d): rolled back %d turn(s) to %d tokens "
                        "(retry %d/%d)", dec.token_count, strict_target, len(dropped), after,
                        self._budget.n_retry_on_limit, self.max_num_retry_on_limit,
                    )
                    timing_log.append(
                        {"step": step + 1, "rollback_dropped": len(dropped), "tokens_after": after}
                    )
                if dec.action == "stop":
                    logger.warning(
                        "context budget reached (%d > %d); final turn taken; stopping so the "
                        "verifier can grade", dec.token_count, strict_target,
                    )
                    break
                # "final_turn": the notice is appended; fall through to let the model act.

                if self.lm_call_cap and self.n_lm_calls >= self.lm_call_cap:
                    self.hit_lm_call_cap = True
                    logger.warning("LM-call cap %d reached; stopping run", self.lm_call_cap)
                    break

                # Record the context actually sent this step (after gate/nudge).
                tk.annotate(self.logs_dir, total_iters, {"sent_tokens": tk.count_tokens(messages)[0]})
                # Replayable snapshot of that same sent context, one per LM call.
                self._snap(messages, "agent")

                t0 = time.monotonic()
                self.n_lm_calls += 1
                try:
                    response = await self._query_with_retry(model, messages)
                except litellm.exceptions.ContextWindowExceededError:
                    logger.warning("Context window exceeded; stopping run")
                    break
                except Exception as exc:
                    if _is_context_overflow(exc):
                        logger.warning("Context overflow (%s); stopping run", type(exc).__name__)
                        break
                    raise
                llm_time = time.monotonic() - t0

                # Calibrate the budget gate to the served tokenizer: the gate counts with
                # tiktoken, and the served model's tokenizer can count noticeably more.
                self._budget.calibrate(response, messages)
                self._accumulate_usage(response, usage_totals)
                turn_metrics = response_metrics(response)
                self.cost += pricing.completion_cost_usd(response, model)

                msg = response.choices[0].message.model_dump()
                msg["content"] = msg.get("content") or ""
                action = _extract_tool_call(msg)

                if action["type"] == "no_tool_call":
                    msg.pop("tool_calls", None)
                    messages.append(msg)
                    agent_step_metrics.append(turn_metrics)
                    messages.append(
                        {"role": "user", "content": "(No tool call in your last turn. When "
                         "ready, issue your next `bash` command to continue the task.)"}
                    )
                    timing_log.append({"step": step + 1, "llm_s": round(llm_time, 1), "empty_turn": True})
                    step += 1  # an empty (no-tool-call) turn counts (as before)
                    continue

                if action["type"] == "context_status":
                    messages.append(msg)
                    agent_step_metrics.append(turn_metrics)
                    messages.append({"role": "tool",
                                     "content": self._ctx.status_line(messages, dec.token_count),
                                     "tool_call_id": action.get("tool_call_id") or ""})
                    timing_log.append({"step": step + 1, "llm_s": round(llm_time, 1),
                                       "context_status": True})
                    step += 1
                    continue

                tool_call_id = action.get("tool_call_id") or ""
                command = action.get("command") or ""

                obs = await self._ctx.step(command, messages, environment=environment, pending=msg)
                result = obs.result
                exec_time = obs.exec_time
                ctx_changed = obs.ctx_changed
                touched_ctx = obs.touched_ctx

                messages.append(msg)  # this turn's bash call (after the sync rewrite)
                agent_step_metrics.append(turn_metrics)
                tool_content = obs.render()

                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_content})
                n_bash += 1
                timing_log.append(
                    {"step": step + 1, "llm_s": round(llm_time, 1), "bash_s": round(exec_time, 1),
                     "return_code": result.return_code, "ctx_touched": touched_ctx,
                     "ctx_synced": ctx_changed, "cmd": command[:200]}
                )

                if self._is_successful_submit(command, result, tool_content):
                    break

                # A pure compaction (mirror changed, no task output, exit 0) is context
                # management, not progress -> it's FREE and doesn't consume a task step.
                task_output_empty = (
                    not (result.stdout or "").strip() and not (result.stderr or "").strip()
                )
                if ctx_turn_is_free(
                    ctx_changed=bool(ctx_changed), return_code=result.return_code,
                    task_output_empty=task_output_empty,
                ):
                    free_ctx_turns += 1
                    timing_log.append(
                        {"iter": total_iters, "free_ctx_turn": True, "cmd": command[:120]}
                    )
                else:
                    step += 1
        finally:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            tk.record(self.logs_dir, total_iters + 1, messages, extra={"final": True})
            # Final retained context as one more snapshot on the same counter,
            # so a FLOPs replay sees the final context too.
            self._snap(messages, "final")
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(messages, indent=2, default=str) + "\n"
            )
            (self.logs_dir / "timing.json").write_text(json.dumps(timing_log, indent=2) + "\n")
            (self.logs_dir / "usage.json").write_text(
                json.dumps(
                    {
                        **usage_totals,
                        "cost_usd": self.cost,
                        "max_steps": self.max_steps,
                        "n_bash": n_bash,
                        "n_ctx_syncs": self._ctx.n_ctx_syncs,
                        # Syncs whose token delta clears the trivial-edit bar; raw syncs
                        # include zero-delta mirror edits.
                        "n_ctx_syncs_real": self._ctx.n_ctx_syncs_real,
                        # Subset of n_ctx_syncs that made context bigger (only possible
                        # under the fit gate).
                        "n_ctx_grew": self._ctx.n_ctx_grew,
                        "n_ctx_rejected": self._ctx.n_ctx_rejected,
                        "ctx_hint": self.ctx_hint_used,
                        "free_ctx_turns": free_ctx_turns,
                        "total_iters": total_iters,
                        "n_lm_calls": self.n_lm_calls,
                        "lm_call_cap": self.lm_call_cap,
                        "hit_lm_call_cap": self.hit_lm_call_cap,
                        "context_budget_tokens": self.context_budget_tokens,
                        "over_budget_policy": (
                            "rollback_retry" if self.max_num_retry_on_limit else "error_finalize"
                        ),
                        "n_nudges": self._budget.n_nudges,
                        "max_num_retry_on_limit": self.max_num_retry_on_limit,
                        "n_retry_on_limit": self._budget.n_retry_on_limit,
                        "n_turns_rolled_back": self._budget.n_turns_rolled_back,
                        "budget_finalized": self._budget.finalized,
                        # Hashes of the harness source, so runs can be matched to code.
                        "code_fingerprint": code_fingerprint(),
                    },
                    indent=2,
                )
                + "\n"
            )
            context.cost_usd = self.cost
            context.n_input_tokens = usage_totals["prompt_tokens"]
            context.n_output_tokens = usage_totals["completion_tokens"]

            if self.emit_ctx_trajectory:
                try:
                    export_ctx_trajectory(
                        self.logs_dir,
                        agent=CtxAgent(name=self.name(), version=self.version(), model_name=model),
                        final_messages=messages,
                        agent_step_metrics=agent_step_metrics,
                        session_id=str(self.logs_dir.name),
                        model=model,
                        model_key=self.flops_model_key,
                        n_body=self.flops_n_body,
                        tokenizer=self.flops_tokenizer,
                        cost_usd=self.cost,
                        cost_metric=self.cost_metric,
                        completion_tokens_total=usage_totals["completion_tokens"],
                        token_counter=lambda m: tk.count_tokens(m)[0],
                    )
                except Exception as exc:  # export must never crash a run
                    logger.warning("ctx-trajectory export failed: %s: %s", type(exc).__name__, exc)

            try:  # best-effort cleanup of the host-side mirror staging file
                self._host_mirror.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ budget
    def _finalize_nudge(self, step: int) -> dict[str, str] | None:
        if self.finalize_nudge_turns <= 0 or self.max_steps <= 0:
            return None
        left = self.max_steps - step
        if self.lm_call_cap and self.finalize_on_lm_call_cap:
            left = min(left, max(self.lm_call_cap - self.n_lm_calls, 0))
        if left > self.finalize_nudge_turns:
            return None
        head = (
            "FINAL TURN: this is your LAST action before the session ends."
            if left <= 1
            else f"Only {left} turns remain before this session ends."
        )
        tail = (
            "Stop exploring now: make sure you have produced your final submission "
            "exactly as the task requires (e.g. written the required "
            "output/answer/program to the location and format it specifies), then "
            "issue the submit command."
        )
        if self.finalize_message:
            tail += " " + self.finalize_message
        return {"role": "user", "content": f"[SYSTEM NOTICE] {head} {tail}"}

    # ------------------------------------------------------------------- LLM
    async def _query_with_retry(self, model: str, messages: list[dict[str, Any]]) -> Any:
        api_base = self.api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        # Claude models reject assistant prefill. A trailing assistant turn (possible
        # after the model edits the mirror) is delivered as a user-side block with the
        # same text instead. Other models get the messages unchanged.
        if messages and messages[-1].get("role") == "assistant" and (
                "claude" in model.lower() or model.startswith("anthropic/")):
            tail = str(messages[-1].get("content", ""))
            messages = messages[:-1] + [{
                "role": "user",
                "content": ("[context mirror -- your own prior assistant text, "
                            "delivered user-side (this API does not accept "
                            "assistant prefill); the text is yours verbatim:]\n"
                            + tail),
            }]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "tools": [BASH_TOOL],
                    # Some models reject max_tokens (see utils.lm_compat).
                    token_cap_key(model): self.max_tokens,
                    "api_base": api_base,
                    "timeout": _LLM_TIMEOUT_SECONDS,
                    "request_timeout": _LLM_TIMEOUT_SECONDS,
                    "num_retries": 0,
                    "drop_params": True,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.top_p is not None and not (
                    model.startswith("anthropic/") and self.temperature is not None
                ):
                    kwargs["top_p"] = self.top_p
                if self.send_chat_template_kwargs:
                    kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {
                        "enable_thinking": self.enable_thinking
                    }
                return await asyncio.wait_for(
                    asyncio.to_thread(litellm.completion, **kwargs),
                    timeout=_LLM_TIMEOUT_SECONDS + _LLM_OUTER_TIMEOUT_BUFFER,
                )
            except _ABORT_EXCEPTIONS:
                raise
            except Exception as exc:
                if _is_context_overflow(exc):
                    logger.warning("Context overflow (%s); not retrying", type(exc).__name__)
                    raise
                if attempt == _MAX_RETRIES:
                    logger.error("Max retries reached: %s: %s", type(exc).__name__, exc)
                    raise
                delay = random.uniform(0.0, min(60.0, _RETRY_BASE_DELAY * (2 ** (attempt - 1))))
                logger.warning("Retry %s/%s after %s: %s (waiting %.0fs)",
                               attempt, _MAX_RETRIES, type(exc).__name__, exc, delay)
                await asyncio.sleep(delay)

    @staticmethod
    def _accumulate_usage(response: Any, usage_totals: dict[str, int]) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for key in usage_totals:
            usage_totals[key] += getattr(usage, key, 0) or 0
        # OpenAI-style gateways nest the cache counter (prompt_tokens_details.
        # cached_tokens); Anthropic-style surfaces cache_read/creation top-level
        # (already caught above). Record both shapes.
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            usage_totals["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0

    def _is_successful_submit(self, command: str, result: Any, tool_content: str) -> bool:
        return (
            self.finish_policy.is_finish_command(command)
            and getattr(result, "return_code", None) == 0
            and SUBMIT_MARKER in (tool_content or "")
        )

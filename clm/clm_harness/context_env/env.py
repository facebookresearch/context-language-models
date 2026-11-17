# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ContextEnv — writable-context OS around a Harbor sandbox.

``step(command, messages)`` is the whole env-side turn: write the mirror, run
the command, reconcile the edit, and append the readout.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ..context_utils.context_string import reconcile_editable, render_editable
from ..utils import tokens as tk
from ..utils.budget import BudgetController
from ..utils.finish_policy import SUBMIT_MARKER, is_submit_command
from . import edit_gate
from .paths import Paths
from .types import StepResult


_DEFAULT_OBS = {"head_chars": 5000, "tail_chars": 5000,
                "too_long_hint": "The output of your last command was too long."}

#: An applied edit that moves fewer tokens than this is mirror noise (a trailing newline,
#: a line rewritten identically), not context management; n_ctx_syncs_real skips it.
TRIVIAL_EDIT_TOKENS = 32
TRIVIAL_EDIT_FRACTION = 0.01

# A command that names the context directory only through one of these path parts is not
# an attempt to edit the mirror, so it gets no "your edit matched nothing" note.
_NOT_AN_EDIT = ("SUBCTX", "finished_subagents", "subagents")


class ContextEnv:
    """The conversation environment. Harbor's BaseEnvironment is ``self.sandbox``."""

    def __init__(
        self,
        *,
        paths: Paths,
        budget: BudgetController,
        protect: int = 2,
        persistent_bash: bool = True,
        command_timeout: int = 120,
        observation_max_chars: int = 10000,
        obs_cfg: dict[str, Any] | None = None,
        allow_edit_growth: bool = True,
        context_budget_tokens: int = 0,
    ) -> None:
        self.paths = paths
        self.budget = budget
        self.protect = protect
        self.persistent_bash = persistent_bash
        self.command_timeout = command_timeout
        self.observation_max_chars = observation_max_chars
        self.obs_cfg = obs_cfg or dict(_DEFAULT_OBS)
        self.allow_edit_growth = allow_edit_growth
        self.context_budget_tokens = int(context_budget_tokens or 0)
        fd, tmp = tempfile.mkstemp(prefix="live_ctx_mirror_", suffix=".txt")
        os.close(fd)
        self.host_mirror = Path(tmp)
        # Growth since the last applied edit, for the context_status reply.
        self.tokens_at_last_edit = 0
        self.turns_since_edit = 0
        self.n_ctx_syncs = 0
        # Syncs whose token delta clears the trivial-edit bar. Raw n_ctx_syncs also
        # counts mirror edits that move ZERO tokens, so edit counts
        # should use this counter, not the raw one.
        self.n_ctx_syncs_real = 0
        self.n_ctx_grew = 0
        self.n_ctx_rejected = 0

    # ------------------------------------------------------------------ setup
    async def setup_dirs(self, environment: Any) -> None:
        p = self.paths
        cmd = f"mkdir -p {p.ctx_dir}"
        if self.persistent_bash:
            cmd = (
                f"mkdir -p {p.state_dir} {p.ctx_dir} && pwd > {p.state_dir}/cwd && "
                f"export -p > {p.state_dir}/env"
            )
        await environment.exec(command=cmd, timeout_sec=10)

    # ----------------------------------------------------------- command wrap
    def wrap(self, command: str) -> str:
        """Run ``command`` in the persistent shell: restore cwd/env, run, save them."""
        if not self.persistent_bash:
            return command
        state_dir = self.paths.state_dir
        return (
            f'cd "$(cat {state_dir}/cwd)" 2>/dev/null || true\n'
            f". {state_dir}/env 2>/dev/null || true\n"
            f"{command}\n"
            "_dr_ec=$?\n"
            f"pwd > {state_dir}/cwd\n"
            f"export -p > {state_dir}/env\n"
            "exit $_dr_ec"
        )

    async def execute(self, command: str, environment: Any) -> Any:
        return await environment.exec(command=self.wrap(command), timeout_sec=self.command_timeout)

    # ----------------------------------------------------------- mirror I/O
    async def write_mirror(self, environment: Any, messages: list[dict[str, Any]]) -> str:
        rendered = render_editable(messages, protect=self.protect)
        self.host_mirror.parent.mkdir(parents=True, exist_ok=True)
        self.host_mirror.write_text(rendered, encoding="utf-8")
        await environment.upload_file(str(self.host_mirror), self.paths.ctx_file)
        return rendered

    async def read_mirror(self, environment: Any) -> str | None:
        try:
            await environment.download_file(self.paths.ctx_file, str(self.host_mirror))
            return self.host_mirror.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None

    def status_line(self, messages: list[dict[str, Any]], tokens: int) -> str:
        """Reply to a ``context_status`` call. The tool is not advertised, but a model that
        calls it anyway gets this line rather than a rejection."""
        limit = self.budget.strict_target or self.context_budget_tokens or 0
        tokens = tokens or tk.count_tokens(messages)[0]
        if not limit:
            return f"{tokens} tokens in context (no budget set)."
        rate = (
            (tokens - self.tokens_at_last_edit) / self.turns_since_edit
            if self.turns_since_edit > 0 and tokens > self.tokens_at_last_edit
            else 0.0
        )
        free = max(limit - tokens, 0)
        turns = int(free / rate) if rate > 0 else None
        return (
            f"{tokens}/{limit} tokens = {round(100 * tokens / limit)}% of your context budget "
            f"used; {free} free, growing ~{round(rate, 1):.0f} tokens/turn recently"
            f"{f', about {turns} turns of headroom left' if turns is not None else ''}."
        )

    # ------------------------------------------------------ observation text
    def format_tool_result(self, result: Any) -> str:
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr
        output = output.rstrip()
        truncated = self.truncate_observation(output) if output else "(no output)"
        return f"{truncated}\n\n(exit_code={result.return_code})"

    def truncate_observation(self, output: str) -> str:
        max_chars = self.observation_max_chars
        if len(output) <= max_chars:
            return output
        head_n = int(self.obs_cfg.get("head_chars", 5000))
        tail_n = int(self.obs_cfg.get("tail_chars", 5000))
        if head_n + tail_n > max_chars:
            head_n = tail_n = max_chars // 2
        elided = len(output) - head_n - tail_n
        hint = self.obs_cfg.get("too_long_hint", "Output truncated.")
        return (
            f"{hint}\n\n---- HEAD ({head_n} chars) ----\n{output[:head_n]}\n"
            f"---- {elided} chars elided ----\n"
            f"---- TAIL ({tail_n} chars) ----\n{output[-tail_n:]}"
        )

    @staticmethod
    def append_submit_warning(command: str, tool_content: str) -> str:
        if SUBMIT_MARKER not in (command or "") or is_submit_command(command):
            return tool_content
        return (tool_content or "") + (
            "\n\n[submit ignored: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT must be a "
            "standalone command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.]"
        )

    # ---------------------------------------------------------------- step
    async def step(
        self,
        command: str,
        messages: list[dict[str, Any]],
        *,
        environment: Any,
        pending: dict[str, Any] | None = None,
    ) -> StepResult:
        p = self.paths
        rendered = await self.write_mirror(environment, messages)
        self.turns_since_edit += 1

        t1 = time.monotonic()
        result = await self.execute(command, environment)
        exec_time = time.monotonic() - t1

        touched_ctx = ("LIVE_CTX_MAIN" in command) or (
            p.ctx_dir in command and not any(s in command for s in _NOT_AN_EDIT)
        )
        read_back = await self.read_mirror(environment)
        before = tk.count_tokens(messages)[0]
        limit = self.budget.strict_target or self.context_budget_tokens or 0
        edit_rec: dict[str, Any] = {
            "edit": "none", "before": before, "after": before,
            "prefix": tk.count_tokens(messages[:self.protect])[0],
        }
        ctx_note = ""
        ctx_changed = False

        if read_back is not None and read_back.strip() != rendered.strip():
            ctx_note, ctx_changed = self._apply_edit(
                messages, read_back, before, limit, edit_rec
            )
        elif touched_ctx:
            rc = getattr(result, "return_code", 0)
            if rc != 0:
                ctx_note = (
                    f"\n[LIVE_CTX_MAIN.txt: NO change — your command exited "
                    f"{rc} (see error above), so nothing was compacted "
                    f"(still ~{before} tokens).]"
                )
            else:
                ctx_note = (
                    f"\n[LIVE_CTX_MAIN.txt: NO change — your edit matched nothing, so "
                    f"context is still ~{before} tokens. Match text you have already "
                    f"seen, or target the real turn headers, which look like "
                    f"`[[CTX_TURN 12 role=assistant]]` (turn index first, then role).]"
                )

        stdout_block = self.append_submit_warning(command, self.format_tool_result(result))
        readout = ""
        # Cap/readout see the assistant turn that the harness is about to append
        # (sync, then append msg, then count).
        shown = messages + ([pending] if pending else [])
        if self.context_budget_tokens:
            stdout_block = self.budget.cap_newest_output(shown, stdout_block)
            # Report the SAME number the gate enforces -- the CALIBRATED count
            # against strict_target -- rather than the raw count over the raw
            # window, so the OVER hint can fire before enforcement.
            tokens_now = self.budget.count(
                shown + [{"role": "tool", "content": stdout_block}]
            )
            shown_budget = self.budget.strict_target or self.context_budget_tokens or 0
            over = "" if tokens_now <= (self.budget.strict_target or 0) else (
                f" — OVER; compact {p.ctx_file} now"
            )
            readout = f"\n[context: ~{tokens_now}/{shown_budget} tokens{over}]"

        return StepResult(
            result=result,
            ctx_changed=ctx_changed,
            stdout_block=stdout_block,
            readout=readout,
            notes=ctx_note,
            exec_time=exec_time,
            touched_ctx=touched_ctx,
        )

    def _apply_edit(
        self,
        messages: list[dict[str, Any]],
        read_back: str,
        before: int,
        limit: int,
        edit_rec: dict[str, Any],
    ) -> tuple[str, bool]:
        try:
            candidate = reconcile_editable(read_back, messages, protect=self.protect)
        except ValueError as exc:
            self.n_ctx_rejected += 1
            return f"\n[LIVE_CTX_MAIN.txt: edit rejected — {exc}. Context unchanged.]", False
        after = tk.count_tokens(candidate)[0]
        baseline = before
        edit_rec.update(before=baseline, after=after)
        if after > baseline and not edit_gate.may_grow(
            allow_growth=self.allow_edit_growth, limit=limit, after=after
        ):
            self.n_ctx_rejected += 1
            rule = edit_gate.reject_rule(allow_growth=self.allow_edit_growth, limit=limit)
            return edit_gate.rejected_note(
                baseline=baseline, after=after, before=before, rule=rule
            ), False

        messages[:] = candidate
        self.n_ctx_syncs += 1
        # Compare actual structured contexts; a lossy round-trip is not a baseline.
        editable = max(baseline - edit_rec["prefix"], 1)
        if abs(baseline - after) >= max(TRIVIAL_EDIT_TOKENS,
                                        int(TRIVIAL_EDIT_FRACTION * editable)):
            self.n_ctx_syncs_real += 1
        grew = after > baseline
        if grew:
            self.n_ctx_grew += 1
        self.budget.note_compaction()
        self.tokens_at_last_edit = after
        self.turns_since_edit = 0
        if grew:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied but it GREW context "
                f"~{baseline}->{after} tokens (it fits, so it was kept). If you "
                f"meant to condense, you likely duplicated content instead of "
                f"replacing it.]"
            )
        elif limit and after > limit:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied — context ~{before}->{after} "
                f"tokens, but STILL OVER the ~{limit}-token limit. Compact more "
                f"NOW (delete stale turns/outputs) or you'll get one final turn "
                f"and the session ends.]"
            )
        else:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied — context ~{before}->{after} "
                f"tokens, {len(messages)} turns]"
            )
        return note, True


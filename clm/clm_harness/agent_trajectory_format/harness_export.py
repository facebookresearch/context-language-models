# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export CLM harness rollouts into ``ATIF-CTX`` trajectories.

A self-compacting rollout is reduced to per-epoch segment records
``{"messages": [...], "prompt_len": int}`` (see :func:`segment_records_from_snapshots`).
This module maps those 1:1 onto the structured format:

    messages[:prompt_len]  ->  Segment.input_context   (the masked prefix)
    messages[prompt_len:]  ->  Segment.steps           (assistant-only-trained)

so every segment boundary a compaction created becomes an explicit
:class:`Segment` with its ``input_context`` snapshot and a :class:`BranchTrigger`.
No snapshot re-diffing here: consume whatever produced the records.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from .builder import TrajectoryBuilder
from .models import (
    Agent,
    BranchTrigger,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    SpawnLink,
    Step,
    SubTrajectory,
    ToolCall,
    Trajectory,
)

logger = logging.getLogger(__name__)

_EDIT_TOOL_NAMES = ("edit_context",)
# Marks the stand-in for a reply that a rollback removed (see _discarded_reply).
_DISCARDED_KEY = "_discarded_reply"


def _content(m: dict[str, Any]) -> Any:
    return m.get("content")


def _to_tool_calls(raw: Any) -> list[ToolCall] | None:
    if not raw:
        return None
    out: list[ToolCall] = []
    for tc in raw:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        out.append(
            ToolCall(
                tool_call_id=(tc.get("id") or tc.get("tool_call_id") or "") if isinstance(tc, dict) else "",
                function_name=fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or "",
                arguments=fn.get("arguments") if "arguments" in fn else tc.get("arguments"),
            )
        )
    return out


def _is_edit_step(step: Step, edit_tool_names: tuple[str, ...]) -> bool:
    return step.source == "agent" and any(
        (tc.function_name in edit_tool_names) for tc in (step.tool_calls or [])
    )


def messages_to_steps(messages: list[dict[str, Any]]) -> list[Step]:
    """Group an OpenAI message list into ATIF steps (ids left 0 for the builder):
    each ``assistant`` turn + its trailing ``tool`` messages -> one agent step;
    standalone ``user``/``system`` messages -> their own step."""
    steps: list[Step] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "assistant":
            step = Step(
                step_id=0,
                source="agent",
                message=_content(m),
                reasoning_content=m.get("reasoning_content") or None,
                tool_calls=_to_tool_calls(m.get("tool_calls")),
            )
            if m.get(_DISCARDED_KEY):
                step.extra = {"discarded_reply": True}
            j = i + 1
            results: list[ObservationResult] = []
            while j < n and messages[j].get("role") == "tool":
                t = messages[j]
                results.append(ObservationResult(source_call_id=t.get("tool_call_id"), content=_content(t)))
                j += 1
            if results:
                step.observation = Observation(results=results)
            steps.append(step)
            i = j
        elif role in ("user", "system"):
            steps.append(Step(step_id=0, source=role, message=_content(m)))
            i += 1
        else:  # orphan tool / unknown role: keep as context (masked), don't train
            steps.append(Step(step_id=0, source="user", message=_content(m)))
            i += 1
    return steps


def trajectory_from_segment_records(
    records: list[dict[str, Any]],
    agent: Agent | dict[str, Any],
    *,
    session_id: str | None = None,
    trajectory_id: str | None = None,
    token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
    edit_tool_names: tuple[str, ...] = _EDIT_TOOL_NAMES,
) -> Trajectory:
    """Build a :class:`Trajectory` from ordered ``{"messages", "prompt_len"}``
    segment records (as produced by :func:`segment_records_from_snapshots`)."""
    if not records:
        raise ValueError("no segment records to export")
    builder = TrajectoryBuilder(
        agent, session_id=session_id, trajectory_id=trajectory_id, token_counter=token_counter
    )
    prev_trigger_id: int | None = None
    prev_kind = "context_edit"
    for idx, rec in enumerate(records):
        messages = rec.get("messages") or []
        prompt_len = int(rec.get("prompt_len", 0) or 0)
        prompt_len = max(0, min(prompt_len, len(messages)))
        input_context = messages[:prompt_len] or messages[:1]  # never empty
        tail = messages[prompt_len:]

        if idx == 0:
            builder.open_segment(input_context)
        else:
            builder.branch(
                input_context,
                trigger=BranchTrigger(
                    kind=prev_kind,
                    source_step_id=prev_trigger_id,
                    summary=f"context branch at step {prev_trigger_id}" if prev_trigger_id else "context branch",
                ),
            )

        # The turn that triggered the NEXT branch is the last agent step of this
        # segment (the bash turn that rewrote the mirror in the CLM harness, or an
        # edit tool call).
        last_agent_id: int | None = None
        saw_edit = False
        for step in messages_to_steps(tail):
            added = builder.add_step(step)
            if added.source == "agent":
                last_agent_id = added.step_id
                saw_edit = saw_edit or _is_edit_step(added, edit_tool_names)
        prev_trigger_id = last_agent_id
        prev_kind = "context_edit" if saw_edit else "context_sync"

    return builder.build()


# ---------------------------------------------------------------------------
# Snapshot reading + compaction split.
# ---------------------------------------------------------------------------
def _msg_key(msg: dict[str, Any]) -> str:
    return json.dumps(
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "tool_call_id": msg.get("tool_call_id"),
        },
        sort_keys=True,
        default=str,
    )


def _common_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    n = 0
    for x, y in zip(a, b):
        if _msg_key(x) == _msg_key(y):
            n += 1
        else:
            break
    return n


def _lead_context_len(messages: list[dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        if m.get("role") in ("system", "user"):
            n += 1
        else:
            break
    return n


def _snapshot_has_edit(msg: dict[str, Any], edit_tool_names: tuple[str, ...]) -> bool:
    if msg.get("role") != "assistant":
        return False
    for tc in msg.get("tool_calls") or []:
        name = ((tc.get("function") or {}).get("name") or tc.get("name")) if isinstance(tc, dict) else None
        if name in edit_tool_names:
            return True
    return False


def _response_after_rewrite(
    cur: list[dict[str, Any]],
    nxt: list[dict[str, Any]],
    edit_tool_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """The reply to the request ``cur`` as it appears in ``nxt`` when ``nxt`` does not
    extend ``cur`` (a context rewrite or a rollback), with its tool results.

    A dedicated edit tool is matched by name. The CLM harness edits through an ordinary
    ``bash`` call: it rewrites the context, appends that assistant turn and its tool
    result, and may then append harness notices (user role) before the next request, so
    the reply is not necessarily the last two messages. ``parse_back`` never produces
    ``tool_calls``, so the reply is the last assistant turn carrying ``tool_calls`` that
    is followed only by its tool results and user/system notices and does not already
    occur in ``cur``. Returns ``[]`` when there is none (a rollback discarded the reply)."""
    tail = nxt[-2:]
    if len(tail) == 2 and _snapshot_has_edit(tail[0], edit_tool_names):
        return list(tail)
    end = len(nxt)
    while end > 0 and nxt[end - 1].get("role") in ("user", "system"):
        end -= 1
    start = end
    while start > 0 and nxt[start - 1].get("role") == "tool":
        start -= 1
    j = start - 1
    if j < 0:
        return []
    reply = nxt[j]
    if reply.get("role") != "assistant" or not reply.get("tool_calls"):
        return []
    key = _msg_key(reply)
    if any(_msg_key(m) == key for m in cur):
        return []
    return list(nxt[j:end])


def _read_snapshot_files(agent_dir: Path | str) -> list[tuple[str | None, list[dict[str, Any]]]]:
    """``(kind, messages)`` per ``context_snapshots/turn-*.json``, in turn order.

    The harness tags one snapshot per LM call (``kind="agent"``) plus the final context
    (``kind="final"``); a second writer sharing the directory leaves untagged copies.
    When tagged snapshots are present, only those are returned."""
    d = Path(agent_dir) / "context_snapshots"
    if not d.is_dir():
        return []
    files = sorted(d.glob("turn-*.json"), key=lambda p: int(re.search(r"turn-(\d+)", p.name).group(1)))
    out: list[tuple[str | None, list[dict[str, Any]]]] = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            out.append((data.get("kind"), data["messages"]))
    if any(kind for kind, _ in out):
        out = [(kind, msgs) for kind, msgs in out if kind in ("agent", "final")]
    return out


def read_snapshots(agent_dir: Path | str) -> list[list[dict[str, Any]]]:
    """Ordered per-turn message lists from ``context_snapshots/turn-*.json``."""
    return [msgs for _, msgs in _read_snapshot_files(agent_dir)]


def _discarded_reply() -> dict[str, Any]:
    """Stand-in for a reply that was sent but is absent from every later snapshot (a
    rollback removed it). It keeps one agent step per LM call, so the request is
    still costed and per-call metrics stay aligned."""
    return {"role": "assistant", "content": None, _DISCARDED_KEY: True}


def _with_discarded(messages: list[dict[str, Any]], holes: list[int]) -> list[dict[str, Any]]:
    out = list(messages)
    for pos in sorted(holes, reverse=True):
        out.insert(pos, _discarded_reply())
    return out


def segment_records_from_snapshots(
    snapshots: list[list[dict[str, Any]]],
    final_messages: list[dict[str, Any]] | None = None,
    *,
    edit_tool_names: tuple[str, ...] = _EDIT_TOOL_NAMES,
    initial_prompt_len: int | None = None,
) -> list[dict[str, Any]]:
    """Split per-turn snapshots into ``{messages, prompt_len}`` epoch segments at
    every compaction (a turn whose next snapshot no longer extends it).

    The first segment's prompt is the leading system/user messages of the first
    snapshot, unless ``initial_prompt_len`` says how many leading messages were given
    rather than generated (e.g. a sub-agent forked from a transcript that already holds
    assistant turns).

    ``snapshots`` holds one request per LM call, in order. Every call contributes one
    assistant message: the reply that ended an epoch is located explicitly (see
    :func:`_response_after_rewrite`), and a reply that a rollback removed is kept as a
    content-less stand-in, so agent steps line up 1:1 with LM calls."""
    snaps = snapshots
    if (final_messages is not None and snaps
            and _common_prefix_len(snaps[-1], final_messages) < len(snaps[-1])):
        # The last call rewrote the context: treat the final context as one more
        # snapshot so that call's reply is found like any other.
        snaps = list(snaps) + [final_messages]
    if len(snaps) < 2:
        msgs = final_messages if final_messages is not None else (snaps[0] if snaps else [])
        return [{"messages": msgs, "prompt_len": (_lead_context_len(msgs) if initial_prompt_len
                                                  is None else min(initial_prompt_len, len(msgs)))}]

    segments: list[dict[str, Any]] = []
    epoch_prompt_len = (_lead_context_len(snaps[0]) if initial_prompt_len is None
                        else initial_prompt_len)
    # Positions (in the current epoch's message list) of replies that were sent and
    # then rolled back without changing anything else in the context.
    holes: list[int] = []
    for k in range(len(snaps) - 1):
        cur, nxt = snaps[k], snaps[k + 1]
        if _common_prefix_len(cur, nxt) < len(cur):
            reply = _response_after_rewrite(cur, nxt, edit_tool_names) or [_discarded_reply()]
            segments.append({"messages": _with_discarded(cur, holes) + reply,
                             "prompt_len": min(epoch_prompt_len, len(cur))})
            holes = []
            epoch_prompt_len = len(nxt)
        elif not any(m.get("role") == "assistant" for m in nxt[len(cur):]):
            holes.append(len(cur))

    final = final_messages if final_messages is not None else snaps[-1]
    segments.append({"messages": _with_discarded(final, holes),
                     "prompt_len": min(epoch_prompt_len, len(final))})
    return segments


# ---------------------------------------------------------------------------
# Per-turn server metrics + top-level export
# ---------------------------------------------------------------------------
def response_metrics(response: Any) -> dict[str, Any]:
    """Compact per-call usage from a litellm/OpenAI response: ``prompt_tokens``,
    ``completion_tokens`` and (for exact KV-cache-aware FLOPs) the prefix-cache
    hit count ``prompt_tokens_details.cached_tokens``."""
    out: dict[str, Any] = {}
    usage = getattr(response, "usage", None)
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is not None:
            out["prompt_tokens"] = int(pt)
        if ct is not None:
            out["completion_tokens"] = int(ct)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens")
        if cached is not None:
            out["cached_tokens"] = int(cached)
    return out


def _attach_agent_step_metrics(traj: Trajectory, per_call: list[dict[str, Any]]) -> None:
    """Attach ordered per-LLM-call metrics onto the trajectory's agent steps (1:1,
    in run order). Best-effort: on a count mismatch, attach up to the shorter."""
    agent_steps = [st for seg in traj.segments for st in seg.steps if st.source == "agent"]
    if len(per_call) != len(agent_steps):
        logger.warning(
            "ctx-export: %d per-call metrics vs %d agent steps; attaching min()",
            len(per_call), len(agent_steps),
        )
    for step, md in zip(agent_steps, per_call):
        if not md:
            continue
        step.metrics = Metrics(
            prompt_tokens=md.get("prompt_tokens"),
            completion_tokens=md.get("completion_tokens"),
            cached_tokens=md.get("cached_tokens"),
        )


def _record_export_check(
    traj: Trajectory,
    entries: list[tuple[str | None, list[dict[str, Any]]]],
    per_call: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Compare agent steps with LM calls (one ``kind="agent"`` snapshot per call) and
    with the per-call metrics; store the counts in ``trajectory.extra["export_check"]``
    and warn on any mismatch."""
    steps = [st for seg in traj.segments for st in seg.steps if st.source == "agent"]
    tagged = any(kind for kind, _ in entries)
    check: dict[str, Any] = {
        "agent_steps": len(steps),
        "lm_calls": sum(1 for kind, _ in entries if kind == "agent") if tagged else None,
        "per_call_metrics": len(per_call) if per_call is not None else None,
        "discarded_replies": sum(1 for st in steps if (st.extra or {}).get("discarded_reply")),
    }
    counts = {v for k, v in check.items() if k != "discarded_replies" and v is not None}
    check["consistent"] = len(counts) <= 1
    if not check["consistent"]:
        logger.warning(
            "ctx-export: %s agent steps vs %s LM calls vs %s per-call metrics",
            check["agent_steps"], check["lm_calls"], check["per_call_metrics"],
        )
    traj.extra = {**(traj.extra or {}), "export_check": check}
    return check


def _parent_link(traj: Trajectory, lm_call: int | None, **fields: Any) -> SpawnLink | None:
    """The parent's ``lm_call``-th agent step (1-based; agent steps line up 1:1 with LM
    calls, see :func:`segment_records_from_snapshots`) as a :class:`SpawnLink`. Clamped to
    the last agent step if the counts disagree; ``None`` if there is no agent step."""
    if lm_call is None:
        return None
    steps = [(seg, st) for seg in traj.segments for st in seg.steps if st.source == "agent"]
    if not steps:
        return None
    seg, st = steps[min(max(int(lm_call), 1), len(steps)) - 1]
    call_id = st.tool_calls[0].tool_call_id if st.tool_calls else None
    return SpawnLink(segment_id=seg.segment_id, step_id=st.step_id, tool_call_id=call_id,
                     **fields)


def attach_subtrajectories(
    traj: Trajectory,
    records: list[dict[str, Any]],
    *,
    agent: Agent | dict[str, Any],
    session_id: str | None = None,
    model: str | None = None,
    model_key: str | None = None,
    n_body: float | None = None,
    tokenizer: Any = None,
    token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
    edit_tool_names: tuple[str, ...] = _EDIT_TOOL_NAMES,
) -> list[SubTrajectory]:
    """Attach one :class:`SubTrajectory` per sub-agent record (see
    ``clm_harness.clm_agent.subagents.SubagentPool.records``).

    Each record's ``dir`` holds the sub-agent's own ``context_snapshots/`` and
    ``trajectory.json``; they are exported exactly like the main agent's, with per-call
    metrics and FLOPs under the child's ``final_metrics.extra["kv_cache_flops"]``.
    ``spawn``/``fold`` point at the main agent steps whose tool results reported the
    launch and the end (``fold`` is ``None`` when the main agent never got that report,
    e.g. the sub-agent was still running when the main agent ended)."""
    from ..flops_metrics import attach_flops

    base = agent if isinstance(agent, Agent) else Agent(**agent)
    sub_agent = Agent(name=f"{base.name}-subagent", version=base.version,
                      model_name=base.model_name)
    subs: list[SubTrajectory] = []
    for rec in records:
        sub_id = rec["sub_id"]
        d = Path(rec["dir"])
        entries = _read_snapshot_files(d)
        try:
            final = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - fall back to the last snapshot
            final = None
        if not entries and final is None:
            logger.warning("ctx-export: no record for sub-agent %s; skipped", sub_id)
            continue
        # Everything in the first request was given to the sub-agent (a brief, or the
        # forked transcript with the main agent's assistant turns), none of it generated.
        snaps = [m for _, m in entries]
        seed = snaps[0] if snaps else (final or [])
        seg_records = segment_records_from_snapshots(
            snaps, final, edit_tool_names=edit_tool_names, initial_prompt_len=len(seed))
        child = trajectory_from_segment_records(
            seg_records, sub_agent, session_id=session_id, trajectory_id=sub_id,
            token_counter=token_counter, edit_tool_names=edit_tool_names)
        metrics = rec.get("metrics")
        check = _record_export_check(child, entries, metrics)
        if metrics:
            _attach_agent_step_metrics(child, metrics)
        attach_flops(child, n_body=n_body, model_key=model_key, model=model,
                     tokenizer=tokenizer, completion_tokens_total=rec.get("completion_tokens"))
        fm = child.final_metrics or FinalMetrics()
        fm.total_prompt_tokens = rec.get("prompt_tokens")
        fm.total_completion_tokens = rec.get("completion_tokens")
        fm.total_cached_tokens = rec.get("cached_tokens")
        fm.total_cost_usd = rec.get("cost_usd")
        link = rec.get("link") or {}
        spawn = _parent_link(traj, rec.get("spawn_call"),
                             time_s=(link.get("spawn") or {}).get("time_s"),
                             path=(link.get("spawn") or {}).get("path"),
                             summary=f"launched {sub_id} ({rec.get('mode')})")
        if spawn is None:
            logger.warning("ctx-export: no parent step for sub-agent %s; skipped", sub_id)
            continue
        fold = _parent_link(traj, rec.get("fold_call"),
                            time_s=(link.get("fold") or {}).get("time_s"),
                            path=(link.get("fold") or {}).get("path"),
                            summary=f"{sub_id} ended: {rec.get('status')}")
        subs.append(SubTrajectory(
            sub_id=sub_id, parent_trajectory_id=traj.trajectory_id, spawn=spawn, fold=fold,
            agent=child.agent, segments=child.segments, final_metrics=fm,
            extra={k: rec.get(k) for k in ("slot", "launch", "mode", "status", "steps",
                                           "n_lm_calls")} | {"export_check": check},
        ))
    traj.subtrajectories = subs
    return subs


#: Additive fields of a ``kv_cache_flops`` dict (see ``compute_trajectory_flops``).
_FLOPS_SUM_KEYS = (
    "n_llm_calls", "n_aux_llm_calls", "aux_prefill_tokens", "aux_gen_tokens",
    "n_reprefill_turns", "completion_tokens", "cache_aware_prefill_tokens",
    "freeze_prefill_tokens", "naive_prefill_tokens", "cache_hit_tokens",
    "cache_aware_attn_flops", "freeze_attn_flops", "naive_attn_flops",
    "cache_aware_flops_linear", "freeze_flops_linear", "naive_flops_linear",
    "cache_aware_flops", "freeze_flops", "naive_flops",
)


def sum_flops(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum ``kv_cache_flops`` dicts of separate trajectories (e.g. sub-agents, whose
    prompts share no KV cache with each other). Additive fields are summed (``None`` if
    every part has ``None``), the saved fractions recomputed, and the descriptive fields
    taken from the first part. ``n_parts`` counts the inputs."""
    out: dict[str, Any] = dict(parts[0]) if parts else {}
    for k in _FLOPS_SUM_KEYS:
        vals = [p.get(k) for p in parts]
        out[k] = None if parts and all(v is None for v in vals) else sum(v or 0 for v in vals)
    naive = out.get("naive_prefill_tokens") or 0
    out["cache_saved_frac"] = round((out.get("cache_hit_tokens") or 0) / naive, 4) if naive else 0.0
    out["freeze_saved_frac"] = (
        round((naive - (out.get("freeze_prefill_tokens") or 0)) / naive, 4) if naive else 0.0)
    out["prefill_source"] = ",".join(sorted({str(p.get("prefill_source")) for p in parts}))
    out["n_parts"] = len(parts)
    return out


def export_ctx_trajectory(
    agent_dir: Path | str,
    *,
    agent: Agent | dict[str, Any],
    final_messages: list[dict[str, Any]] | None = None,
    agent_step_metrics: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    trajectory_id: str | None = None,
    model: str | None = None,
    model_key: str | None = None,
    n_body: float | None = None,
    tokenizer: Any = None,
    cost_usd: float | None = None,
    cost_metric: str = "auto",
    completion_tokens_total: int | None = None,
    token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
    edit_tool_names: tuple[str, ...] = _EDIT_TOOL_NAMES,
    aux_prefill_tokens: int = 0,
    aux_gen_tokens: int = 0,
    n_aux_calls: int = 0,
    filename: str | None = "trajectory.ctx.json",
    subagents: list[dict[str, Any]] | None = None,
) -> Trajectory:
    """One-call export used by the harness ``finally`` block: read the per-turn
    snapshots this run wrote, split at compactions, convert to ATIF-CTX, attach
    per-call server metrics, compute KV-cache-aware FLOPs + the unified
    ``compute_cost`` headline (USD for API models, FLOPs for open), and (if
    ``filename``) write ``agent_dir/filename``. ``subagents`` (records from
    ``SubagentPool.records``) are attached as sub-trajectories. Returns the
    :class:`Trajectory`."""
    from ..flops_metrics import attach_compute_cost, attach_flops

    entries = _read_snapshot_files(agent_dir)
    snaps = [msgs for _, msgs in entries]
    records = segment_records_from_snapshots(snaps, final_messages, edit_tool_names=edit_tool_names)
    traj = trajectory_from_segment_records(
        records, agent, session_id=session_id, trajectory_id=trajectory_id,
        token_counter=token_counter, edit_tool_names=edit_tool_names,
    )
    _record_export_check(traj, entries, agent_step_metrics)
    if agent_step_metrics:
        _attach_agent_step_metrics(traj, agent_step_metrics)
    resolved_model = model or (agent.get("model_name") if isinstance(agent, dict) else getattr(agent, "model_name", None))
    attach_flops(
        traj, n_body=n_body, model_key=model_key, model=resolved_model,
        tokenizer=tokenizer, completion_tokens_total=completion_tokens_total,
        aux_prefill_tokens=aux_prefill_tokens, aux_gen_tokens=aux_gen_tokens,
        n_aux_calls=n_aux_calls,
    )
    if subagents is not None:
        attach_subtrajectories(
            traj, subagents, agent=agent, session_id=session_id, model=resolved_model,
            model_key=model_key, n_body=n_body, tokenizer=tokenizer,
            token_counter=token_counter, edit_tool_names=edit_tool_names)
    # Rates (USD/token) let attach_compute_cost also emit a provider-INDEPENDENT USD
    # estimate from the local prefix-match token counts, alongside the provider one.
    try:
        from ..utils.pricing import rates_for
        _rates = rates_for(resolved_model)
    except Exception:  # noqa: BLE001 - pricing must never break export
        _rates = None
    _in, _out = _rates if _rates else (None, None)
    attach_compute_cost(
        traj, cost_usd=cost_usd, model=resolved_model, cost_metric=cost_metric,
        in_rate=_in, out_rate=_out,
    )
    if subagents is not None:
        # kv_cache_flops stays the main agent's; the sub-agents get their own total, and
        # compute_cost (with cost_usd, already the run total) covers both.
        from ..flops_metrics import summarize_compute_cost

        extra = traj.final_metrics.extra
        sub_flops = sum_flops([s.final_metrics.extra["kv_cache_flops"]
                               for s in traj.subtrajectories])
        extra["kv_cache_flops_subagents"] = sub_flops
        both = sum_flops([extra["kv_cache_flops"], sub_flops]) if traj.subtrajectories \
            else extra["kv_cache_flops"]
        extra["compute_cost"] = {
            **summarize_compute_cost(both, cost_usd=cost_usd, model=resolved_model,
                                     cost_metric=cost_metric, in_rate=_in, out_rate=_out),
            "scope": "main+subagents",
        }
    if filename:
        out = Path(agent_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / filename).write_text(json.dumps(traj.to_json_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    return traj

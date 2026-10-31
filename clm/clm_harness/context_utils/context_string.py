# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Context-as-a-string round-trip: render the message log to an editable string
and map an arbitrarily-edited string back to a legal chat-template message list.

Used by ``clm_agent``, which lets the model rewrite its own context as free
text by editing a mirror file on disk via bash. The editable region (turns after
the protected system+task prefix) is rendered with per-turn role headers so roles
survive arbitrary edits (turns numbered 1,2,3,... from the top of the editable
region — the hidden system+task prefix is not counted)::

    [[CTX_TURN 1 role=assistant]]
    <the turn's text, exactly as the model saw it>

    [[CTX_TURN 2 role=tool]]
    <observation ...>

The live harness uses :func:`reconcile_editable` with the original snapshot:
unchanged messages retain their fields and boundaries, and only edited blocks
(or tool exchanges broken by the edit) may become plain-text summaries.

The model edits the turns' TEXT (content it has already seen); the headers ride
along untouched. The legacy text-only :func:`parse_back` is deliberately tolerant: it re-pins the
protected prefix from the originals (so the system/task contract can't be
corrupted), recovers turns from whatever headers remain (stray leading/inter-
header text becomes a user note), drops emptied turns, and merges consecutive
same-role turns so the result is faithful to chat-template rules. History is
reconstructed as plain role+text (tool-call structure is not preserved across an
edit) — the price of "edit the raw string however you like", and always legal.

Dependency-free (stdlib only) so it is unit-testable without the serving stack.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

# Distinctive, line-anchored turn header. Unlikely to collide with real content;
# if the model damages one, parse_back still recovers the rest best-effort.
_HEADER_RE = re.compile(r"^\[\[CTX_TURN\s+(\d+)\s+role=([A-Za-z]+)\]\]\s*$", re.M)
_VALID_ROLES = {"system", "user", "assistant", "tool"}


def content_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text") and isinstance(
                part.get("text"), str
            ):
                parts.append(part["text"])
            else:
                return None
        return "".join(parts)
    return None


def rendered_text(msg: dict[str, Any]) -> str:
    """The text the model saw for a turn: reasoning, then content, then the issued
    command (tool_calls name + args). Bash observations don't echo the command, so
    this makes a turn's command part of its editable text."""
    parts: list[str] = []
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc:
        parts.append(rc)
    ct = content_text(msg.get("content"))
    if ct:
        parts.append(ct)
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            import json

            try:
                args = json.dumps(args or {})
            except Exception:
                args = ""
        piece = f"{fn.get('name', '')} {args}".strip()
        if piece:
            parts.append(piece)
    return "\n".join(p for p in parts if p)


def render_editable(messages: list[dict[str, Any]], protect: int = 2) -> str:
    """Render the editable region (turns >= ``protect``) as a headered string that
    the model may transform freely."""
    blocks: list[str] = []
    # Number editable turns 1,2,3,... from the top of the editable region (NOT the raw
    # message index): the protected system+task prefix is hidden, so a 1-based count is
    # what the model naturally references. reconcile_editable uses the number to
    # match this snapshot's originals; the legacy parse_back ignores it.
    for n, i in enumerate(range(protect, len(messages)), start=1):
        m = messages[i]
        blocks.append(f"[[CTX_TURN {n} role={m.get('role', 'user')}]]\n{rendered_text(m)}")
    return "\n\n".join(blocks)


def parse_back(edited: Any, protected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map an arbitrarily-edited context string back to a legal message list.

    ``protected`` is the untouchable prefix (system + initial task) re-pinned from
    the originals. Whatever role headers survive define the remaining turns; stray
    text before/between headers becomes a user note; ``assistant`` stays assistant
    and every other role folds to ``user`` (tool structure is not reconstructed).
    The result is normalised (drop empty, merge consecutive same-role)."""
    text = edited if isinstance(edited, str) else str(edited)
    matches = list(_HEADER_RE.finditer(text))

    sections: list[tuple[str, str]] = []
    if not matches:
        body = text.strip()
        if body:
            sections.append(("user", body))
    else:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(("user", lead))
        for k, mt in enumerate(matches):
            role = mt.group(2).lower()
            if role not in _VALID_ROLES:
                role = "user"
            start = mt.end()
            end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((role, body))

    msgs: list[dict[str, Any]] = [dict(m) for m in protected]
    for role, body in sections:
        # No tool-call structure is reconstructed: assistant stays assistant,
        # everything else (user/tool/system-in-body) becomes a plain user turn.
        msgs.append({"role": "assistant" if role == "assistant" else "user", "content": body})
    return _normalize(msgs, protect=len(protected))


def reconcile_editable(
    edited: str, original: list[dict[str, Any]], protect: int = 2,
) -> list[dict[str, Any]]:
    """Apply a mirror edit against the exact message snapshot that rendered it.

    Turn numbers identify messages within this snapshot, not across commands.
    Unchanged blocks retain ALL fields and boundaries. Plain text/tool output
    edits retain metadata; edits to flattened reasoning/calls become explicit
    plain-text summaries because their field boundaries cannot be recovered.
    Only a tool exchange made incomplete by the edit is downgraded to text.
    Duplicate identities are rejected rather than guessed. The legacy parser
    remains available for text imports without an original snapshot.
    """
    if edited == render_editable(original, protect):
        return deepcopy(original)
    # A literal turn header inside a message is ambiguous in this text format.
    # Do not silently split it and attach another message's hidden metadata.
    if any(_HEADER_RE.search(rendered_text(m)) for m in original[protect:]):
        raise ValueError("a message contains a literal CTX_TURN header")

    matches = list(_HEADER_RE.finditer(edited))
    result = deepcopy(original[:protect])
    sources: list[int | None] = list(range(len(result)))
    seen: set[int] = set()
    lead = edited[:matches[0].start()].strip() if matches else edited.strip()
    if lead:
        result.append({"role": "user", "content": lead})
        sources.append(None)
    for k, match in enumerate(matches):
        number = int(match.group(1))
        if number in seen:
            raise ValueError(f"duplicate CTX_TURN {number}; keep each header only once")
        seen.add(number)
        role = match.group(2).lower()
        end = matches[k + 1].start() if k + 1 < len(matches) else len(edited)
        body = edited[match.end():end].strip()
        index = protect + number - 1
        old = original[index] if number > 0 and index < len(original) else None
        same_role = old is not None and role == old.get("role", "user")
        if same_role and body == rendered_text(old).strip():
            message = deepcopy(old)
        elif not body:
            continue
        elif (same_role and role in ("assistant", "user", "tool")
              and isinstance(old.get("content"), str)
              and not old.get("reasoning_content") and not old.get("tool_calls")):
            message = deepcopy(old)
            message["content"] = body
        else:
            message = {"role": "assistant" if role == "assistant" else "user",
                       "content": body}
        result.append(message)
        sources.append(index if old is not None else None)

    # A call and its results must remain one contiguous exchange. Deleting,
    # rewriting or moving only part of it must not leave orphan tool messages.
    # Use original positions as identities (call IDs can be reused in history).
    positions = {source: i for i, source in enumerate(sources) if source is not None}
    for start, message in enumerate(original):
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        stop = start + 1
        while stop < len(original) and original[stop].get("role") == "tool":
            stop += 1
        members = [positions[i] for i in range(start, stop) if i in positions]
        if not members:
            continue
        call_pos = positions.get(start)
        valid = (
            call_pos is not None
            and result[call_pos].get("tool_calls") == calls
            and len(members) == stop - start
            and members == list(range(call_pos, call_pos + len(members)))
            and all(result[i].get("role") == "tool" for i in members[1:])
            and all(result[positions[i]].get("tool_call_id") == original[i].get("tool_call_id")
                    for i in range(start + 1, stop))
        )
        if valid:
            continue
        for i in members:
            if i < protect:
                raise ValueError("edit would break a tool exchange in the protected prefix")
            old = result[i]
            result[i] = {"role": "assistant" if old.get("role") == "assistant" else "user",
                         "content": rendered_text(old)}
    return result


def _normalize(messages: list[dict[str, Any]], protect: int) -> list[dict[str, Any]]:
    """Legal, template-faithful message list: drop emptied turns and merge
    consecutive same-role turns. The protected prefix is preserved verbatim."""
    kept: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i < protect:
            kept.append(m)
            continue
        c = content_text(m.get("content"))
        if c and c.strip():
            kept.append(m)

    merged: list[dict[str, Any]] = []
    for i, m in enumerate(kept):
        if (
            merged
            and i >= protect
            and len(merged) > protect
            and m.get("role") in ("user", "assistant")
            and merged[-1].get("role") == m.get("role")
        ):
            prev = content_text(merged[-1].get("content")) or ""
            cur = content_text(m.get("content")) or ""
            merged[-1] = {"role": m["role"], "content": (prev + "\n\n" + cur).strip()}
        else:
            merged.append(m)
    return merged

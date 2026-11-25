# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline stand-ins for the model and the Harbor sandbox, shared by the harness tests.

* :class:`LocalSandbox` runs each command with the host's ``bash`` inside a private
  temporary root. Absolute ``/tmp/...`` paths in commands and file transfers are mapped
  into that root and the root is mapped back out of the output, so the agent sees the
  same paths it would see in a real sandbox and recorded outputs do not depend on where
  the test ran.
* :class:`ScriptedLLM` replaces ``litellm.completion``. The main agent follows one
  script; each sub-agent follows the script named by a ``[script:<name>]`` tag in its
  first non-system message. Usage numbers are derived from the request text, so runs
  are deterministic.

Importing this module pins token counting to the chars/4 estimate (no tiktoken
vocabulary download) and keeps litellm on its bundled price map, so tests never touch
the network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.pop("CLM_EDIT_GATE", None)

from harbor.environments.base import ExecResult  # noqa: E402

from clm_harness.flops_metrics import kv_cache_flops as _kvf  # noqa: E402
from clm_harness.utils import tokens as _tk  # noqa: E402

# Deterministic, offline token counts everywhere.
_tk._ENCODER = None
_tk._ENCODER_NAME = "_failed"
_kvf._ENCODER = None
_kvf._ENCODER_TRIED = True

MODEL = "openai/fake-haiku"  # priced by the harness's family table -> non-zero cost
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
class LocalSandbox:
    """A Harbor-environment look-alike backed by the host shell (see module doc)."""

    def __init__(self, root: str | Path) -> None:
        self.root = str(Path(root).resolve())
        (Path(self.root) / "tmp").mkdir(parents=True, exist_ok=True)
        (Path(self.root) / "work").mkdir(parents=True, exist_ok=True)
        (Path(self.root) / "home").mkdir(parents=True, exist_ok=True)
        self.commands: list[str] = []

    # path mapping ----------------------------------------------------------
    def host(self, path: str) -> Path:
        return Path(self.root + path) if path.startswith("/") else Path(self.root, "work", path)

    def _to_host_cmd(self, command: str) -> str:
        return command.replace("/tmp/", f"{self.root}/tmp/")

    def _from_host(self, text: str) -> str:
        return (text or "").replace(self.root, "")

    # Harbor API ------------------------------------------------------------
    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: Any = None) -> ExecResult:
        self.commands.append(command)
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", self._to_host_cmd(command),
            cwd=str(Path(self.root, "work")),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "HOME": str(Path(self.root, "home")), "LANG": "C", "LC_ALL": "C"},
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec or 60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(stdout="", stderr="timeout", return_code=124)
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        return ExecResult(stdout=self._from_host(out.decode(errors="replace")),
                          stderr=self._from_host(err.decode(errors="replace")),
                          return_code=proc.returncode)

    async def upload_file(self, source_path: str | Path, target_path: str) -> None:
        dst = self.host(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, dst)

    async def download_file(self, source_path: str, target_path: str | Path) -> None:
        src = self.host(source_path)
        if not src.is_file():
            raise FileNotFoundError(source_path)
        shutil.copyfile(src, target_path)

    # test helpers ----------------------------------------------------------
    def read(self, path: str) -> str:
        return self.host(path).read_text()

    def exists(self, path: str) -> bool:
        return self.host(path).exists()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))


def _response(content: str, command: str | None, call_id: str, prompt_tokens: int) -> Any:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if command is not None:
        msg["tool_calls"] = [{"id": call_id, "type": "function",
                              "function": {"name": "bash",
                                           "arguments": json.dumps({"command": command})}}]
    completion = (len(content) + len(command or "")) // 4 + 1
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion,
        total_tokens=prompt_tokens + completion, reasoning_tokens=0,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(msg))], usage=usage,
                           model=MODEL)


_TAG_RE = re.compile(r"\[script:([A-Za-z0-9_-]+)\]")

#: A script step: a bash command (str), ``None`` for a turn with no tool call, a dict
#: ``{"command"|"content"|"sleep"|"raise"}``, or a callable ``(messages) -> step``.
Step = Any


class ScriptedLLM:
    """Drop-in for ``litellm.completion`` (see module doc). Thread-safe: the harness
    calls it through ``asyncio.to_thread``."""

    def __init__(self, main: list[Step], workers: dict[str, list[Step]] | None = None,
                 *, after_script: Step = SUBMIT) -> None:
        self.main = list(main)
        self.workers = dict(workers or {})
        self.after_script = after_script
        self.main_system: str | None = None
        self.calls: list[tuple[str, int]] = []  # (who, index), in call order
        self._lock = threading.Lock()
        self._n_main = 0

    def _who(self, messages: list[dict[str, Any]]) -> str:
        system = str(messages[0].get("content")) if messages else ""
        with self._lock:
            if self.main_system is None:
                self.main_system = system
        if system == self.main_system:
            return "main"
        for m in messages[1:]:
            hit = _TAG_RE.search(str(m.get("content") or ""))
            if hit:
                return hit.group(1)
        raise AssertionError("sub-agent request without a [script:<name>] tag")

    def __call__(self, **kwargs: Any) -> Any:
        messages = kwargs["messages"]
        who = self._who(messages)
        with self._lock:
            if who == "main":
                idx = self._n_main
                self._n_main += 1
                script = self.main
            else:
                idx = sum(1 for m in messages if m.get("role") == "assistant"
                          and str(m.get("content") or "").startswith(f"[{who}#"))
                script = self.workers[who]
            self.calls.append((who, idx))
        step: Step = script[idx] if idx < len(script) else self.after_script
        if callable(step):
            step = step(messages)
        if isinstance(step, dict):
            if step.get("sleep"):
                time.sleep(float(step["sleep"]))
            if step.get("raise") is not None:
                raise step["raise"]
            command = step.get("command")
            content = step.get("content")
        else:
            command, content = step, None
        content = f"[{who}#{idx}] " + (content or f"THOUGHT: step {idx}.")
        prompt_tokens = _tk.approx_tokens(messages)
        return _response(content, command, f"call_{who}_{idx}", prompt_tokens)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------
def make_agent(logs_dir: str | Path, **kwargs: Any) -> Any:
    from clm_harness.clm_agent.harness import ClmAgent

    base = {"model_name": MODEL, "context_budget_tokens": 3000,
            "context_budget_reserve_tokens": 256, "cost_metric": "usd", "max_steps": 20,
            "temperature": 0.0, "top_p": None}
    base.update(kwargs)
    return ClmAgent(logs_dir=Path(logs_dir), **base)


async def run_agent(agent: Any, sandbox: LocalSandbox, llm: Callable[..., Any],
                    instruction: str = "Write the word ready to /tmp/work/out.txt.") -> Any:
    """Run ``agent`` end to end with ``llm`` standing in for ``litellm.completion``."""
    import litellm
    from harbor.models.agent.context import AgentContext

    context = AgentContext()
    original = litellm.completion
    litellm.completion = llm
    try:
        await agent.setup(sandbox)
        await agent.run(instruction, sandbox, context)
    finally:
        litellm.completion = original
    return context


def normalized_json(path: str | Path, drop: tuple[str, ...] = ()) -> Any:
    """Load JSON and zero out wall-clock fields / drop ``drop`` keys (recursively)."""
    data = json.loads(Path(path).read_text())

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: (0 if k in ("llm_s", "bash_s") else walk(v))
                    for k, v in x.items() if k not in drop}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(data)

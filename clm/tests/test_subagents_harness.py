# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``ClmAgent`` with ``subagents=true``: settings, the env's edit detection, and a full
scripted run in which the main agent launches sub-agents and uses their results."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import clm_fakes as F

from clm_harness.clm_agent.harness import ClmAgent

CTX = "/tmp/.live_ctx"
FIN = f"{CTX}/finished_subagents"

SPAWN_TWO = (
    f"cat > {CTX}/SUBCTX_0.txt <<'EOF'\nSum 1..10 into /tmp/wa/out.txt. [script:a]\nEOF\n"
    f"cat > {CTX}/SUBCTX_1.txt <<'EOF'\nCount to 3 into /tmp/wb/out.txt. [script:b]\nEOF"
)
SPAWN_SLOW = f"echo 'Loop forever. [script:slow]' > {CTX}/SUBCTX_2.txt"
WORKERS = {
    "a": ["mkdir -p /tmp/wa && python3 -c 'print(sum(range(11)))' > /tmp/wa/out.txt",
          {"command": F.SUBMIT, "content": "Sum is in /tmp/wa/out.txt."}],
    "b": ["mkdir -p /tmp/wb && seq 1 3 > /tmp/wb/out.txt",
          {"command": F.SUBMIT, "content": "Counted to 3 in /tmp/wb/out.txt."}],
    "slow": [{"command": "echo tick", "sleep": 0.3}] * 40,
}


def wait_files(sandbox: F.LocalSandbox, paths: list[str], command: str, timeout: float = 20.0):
    """A script step that waits (in the model thread) until ``paths`` exist."""

    def step(_messages):
        end = time.monotonic() + timeout
        while not all(sandbox.exists(p) for p in paths):
            if time.monotonic() > end:
                raise AssertionError(f"sub-agent output never appeared: {paths}")
            time.sleep(0.02)
        time.sleep(0.2)  # let the worker tasks finish their sandbox bookkeeping
        return command

    return step


def run_e2e(tmp: Path, **agent_kwargs) -> tuple[F.LocalSandbox, F.ScriptedLLM, Path]:
    sandbox = F.LocalSandbox(tmp / "sandbox")
    main = [
        SPAWN_TWO,
        SPAWN_SLOW,
        wait_files(sandbox, [f"{FIN}/subctx_0_1/final_message.txt",
                             f"{FIN}/subctx_1_1/final_message.txt"],
                   "mkdir -p /tmp/work && cat /tmp/wa/out.txt /tmp/wb/out.txt > /tmp/work/out.txt && "
                   "cat /tmp/work/out.txt"),
        f"cat {FIN}/subctx_0_1/final_message.txt",
        F.SUBMIT,
    ]
    llm = F.ScriptedLLM(main, WORKERS)
    kwargs = {"subagents": "true", "n_subagent_slots": 3, "subagent_max_steps": 40,
              "context_budget_tokens": 6000}
    kwargs.update(agent_kwargs)
    agent = F.make_agent(tmp / "agent", **kwargs)
    asyncio.run(F.run_agent(agent, sandbox, llm))
    return sandbox, llm, tmp / "agent"


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="clm_sa_settings_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_off_by_default(self) -> None:
        agent = F.make_agent(self.tmp)
        self.assertFalse(agent.subagents)
        self.assertIsNone(agent._subs)
        self.assertFalse(agent._ctx.subagents)

    def test_on_with_defaults(self) -> None:
        agent = F.make_agent(self.tmp, subagents="true")
        self.assertEqual((agent._subs.cfg.n_slots, agent._subs.cfg.max_steps,
                          agent._subs.cfg.lifetime_cap), (5, 40, 24))
        self.assertTrue(agent._ctx.subagents)
        self.assertIn("up to 5 sub-agents", agent._subs.prompt_addendum())

    def test_other_removed_settings_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            F.make_agent(self.tmp, subagents="true", shadow_hints="1")

    def test_self_compact_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            F.make_agent(self.tmp, subagents="true", subagent_self_compact="true")

    def test_contextbench_refuses_subagents(self) -> None:
        stream = types.ModuleType("contextbench.stream")
        stream.MARKER_PARAGRAPH = "marker"
        mixin = types.ModuleType("contextbench.harbor.mixin")
        mixin.CBStreamMixin = type("CBStreamMixin", (), {})
        stubs = {"contextbench": types.ModuleType("contextbench"),
                 "contextbench.harbor": types.ModuleType("contextbench.harbor"),
                 "contextbench.stream": stream, "contextbench.harbor.mixin": mixin}
        saved = {k: sys.modules.get(k) for k in [*stubs, "clm_contextbench.clm_stream"]}
        sys.modules.update(stubs)
        sys.modules.pop("clm_contextbench.clm_stream", None)
        try:
            from clm_contextbench.clm_stream import ClmStream

            with self.assertRaisesRegex(ValueError, "does not support subagents"):
                ClmStream(logs_dir=self.tmp, model_name=F.MODEL, subagents="true")
            ClmStream(logs_dir=self.tmp, model_name=F.MODEL)  # unchanged without it
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class EnvEditDetectionTests(unittest.IsolatedAsyncioTestCase):
    """``cp LIVE_CTX_MAIN.txt SUBCTX_2.txt`` names the mirror but does not edit it."""

    async def _step(self, subagents: bool, command: str) -> str:
        tmp = Path(tempfile.mkdtemp(prefix="clm_sa_env_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sandbox = F.LocalSandbox(tmp / "sandbox")
        agent = F.make_agent(tmp / "agent", **({"subagents": "true"} if subagents else {}))
        await agent._ctx.setup_dirs(sandbox)
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "T"},
                {"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
        obs = await agent._ctx.step(command, msgs, environment=sandbox)
        return obs.render()

    async def test_fork_copy_gets_no_false_note(self) -> None:
        cmd = f"cp {CTX}/LIVE_CTX_MAIN.txt {CTX}/SUBCTX_2.txt"
        self.assertNotIn("NO change", await self._step(True, cmd))
        self.assertIn("NO change", await self._step(False, cmd))  # unchanged when off

    async def test_real_edit_still_detected(self) -> None:
        cmd = (f"sed -i 's/^a$/A/' {CTX}/LIVE_CTX_MAIN.txt && "
               f"cp {CTX}/LIVE_CTX_MAIN.txt {CTX}/SUBCTX_0.txt")
        self.assertIn("edit applied", await self._step(True, cmd))

    async def test_brief_with_submit_line_gets_no_submit_warning(self) -> None:
        cmd = (f"cat > {CTX}/SUBCTX_0.txt <<'B'\nDo x.\nFinish with: "
               "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nB")
        self.assertNotIn("submit ignored", await self._step(True, cmd))
        self.assertIn("submit ignored", await self._step(False, cmd))  # unchanged when off
        self.assertIn("submit ignored", await self._step(
            True, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT now"))

    async def test_missed_edit_still_reported(self) -> None:
        cmd = f"sed -i 's/nothing/x/' {CTX}/LIVE_CTX_MAIN.txt"
        self.assertIn("NO change", await self._step(True, cmd))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="clm_sa_e2e_"))
        cls.sandbox, cls.llm, cls.logs = run_e2e(cls.tmp)
        cls.traj = json.loads((cls.logs / "trajectory.json").read_text())
        cls.usage = json.loads((cls.logs / "usage.json").read_text())
        cls.timing = json.loads((cls.logs / "timing.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tool(self, i: int) -> str:
        return [m["content"] for m in self.traj if m["role"] == "tool"][i]

    def test_prompt_addendum(self) -> None:
        self.assertIn("## Sub-agents", self.traj[0]["content"])
        self.assertIn("up to 3 sub-agents", self.traj[0]["content"])

    def test_main_uses_worker_results(self) -> None:
        self.assertEqual(self.sandbox.read("/tmp/work/out.txt").split(), ["55", "1", "2", "3"])
        self.assertIn("SUBCTX_0 launched as subctx_0_1 (brief)", self.tool(0))
        self.assertIn("SUBCTX_1 launched as subctx_1_1 (brief)", self.tool(0))
        self.assertIn("SUBCTX_0 finished (completed, 2 steps)", self.tool(2))
        self.assertIn("SUBCTX_1 finished (completed, 2 steps)", self.tool(2))
        self.assertIn("Sum is in /tmp/wa/out.txt.", self.tool(3))
        self.assertTrue(self.traj[-1]["content"].startswith("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"))

    def test_spawn_turns_are_free(self) -> None:
        free = [t for t in self.timing if t.get("free_ctx_turn")]
        self.assertEqual(len(free), 2)  # both launch turns printed nothing
        self.assertEqual(self.usage["free_ctx_turns"], 2)

    def test_worker_transcripts_stay_out_of_main(self) -> None:
        text = json.dumps(self.traj)
        self.assertNotIn("[a#0]", text)
        self.assertNotIn("Counted to 3", text)

    def test_running_worker_cancelled_at_end(self) -> None:
        summary = json.loads((self.logs / "subagents" / "subctx_2_1" / "summary.json").read_text())
        self.assertEqual(summary["status"], "cancelled_end_of_run")
        self.assertLess(summary["steps"], 40)


if __name__ == "__main__":
    unittest.main()

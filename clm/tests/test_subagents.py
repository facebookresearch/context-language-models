# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``clm_harness.clm_agent.subagents.SubagentPool``.

The pool is driven directly: the test plays the main agent by writing or deleting slot
files in a local sandbox and calling ``poll``; sub-agents talk to a scripted model.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import clm_fakes as F
import litellm

from clm_harness.clm_agent.subagents import SubagentConfig, SubagentPool
from clm_harness.context_utils.context_string import render_editable

CTX = "/tmp/.live_ctx"
FIN = f"{CTX}/finished_subagents"


def slot(i: int) -> str:
    return f"{CTX}/SUBCTX_{i}.txt"


async def wait_for(pred, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("timed out waiting for sub-agents")
        await asyncio.sleep(0.02)


class PoolTestBase(unittest.IsolatedAsyncioTestCase):
    cfg = SubagentConfig(n_slots=3, max_steps=6, lifetime_cap=10)
    agent_kwargs: dict = {}

    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="clm_subagents_"))
        self.sandbox = F.LocalSandbox(self.tmp / "sandbox")
        self.agent = F.make_agent(self.tmp / "agent", **self.agent_kwargs)
        self.pool = SubagentPool(self.agent, self.agent._ctx, self.agent._ctx.paths, self.cfg)
        await self.agent._ctx.setup_dirs(self.sandbox)
        await self.pool.setup(self.sandbox)
        self._orig = litellm.completion

    async def asyncTearDown(self) -> None:
        await self.pool.cancel_all()
        litellm.completion = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def use(self, workers: dict) -> F.ScriptedLLM:
        llm = F.ScriptedLLM([], workers, main_system="(no main agent in this test)")
        litellm.completion = llm
        return llm

    def write_slot(self, i: int, text: str) -> None:
        p = self.sandbox.host(slot(i))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def worker(self, sub_id: str):
        return next(w for w in self.pool._workers if w.sub_id == sub_id)

    async def finish_all(self) -> str:
        await wait_for(lambda: all(w.task.done() for w in self.pool._workers))
        note, _ = await self.pool.poll("true")
        return note

    def record(self, sub_id: str) -> Path:
        return self.tmp / "agent" / "subagents" / sub_id


class SeedingTests(PoolTestBase):
    async def test_final_message_falls_back_to_reasoning(self) -> None:
        last = self.pool._last_assistant_text
        self.assertEqual(last([{"role": "assistant", "content": "older text"},
                               {"role": "tool", "content": "x"},
                               {"role": "assistant", "content": "", "reasoning_content": " done: see s0.md "}]),
                         "done: see s0.md")
        self.assertEqual(last([{"role": "assistant", "content": "summary",
                                "reasoning_content": "thinking"}]), "summary")
        long = last([{"role": "assistant", "content": None, "reasoning_content": "x" * 5000 + "END"}])
        self.assertTrue(long.startswith("[... truncated]") and long.endswith("END"))
        self.assertEqual(last([{"role": "assistant", "content": ""}]), "")

    async def test_brief_seed(self) -> None:
        msgs, mode = self.pool.seed_messages("  do the thing [script:a]\n")
        self.assertEqual(mode, "brief")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[1]["content"], "do the thing [script:a]")
        self.assertIn("at most 6 commands", msgs[0]["content"])
        self.assertIn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", msgs[0]["content"])

    def _fork(self, turns: list[dict]) -> list[dict]:
        main = [{"role": "system", "content": "MAIN SYSTEM"},
                {"role": "user", "content": "TASK_SENTINEL"}, *turns]
        msgs, mode = self.pool.seed_messages(render_editable(main, protect=2))
        self.assertEqual(mode, "fork")
        self.assertNotIn("TASK_SENTINEL", json.dumps(msgs))
        self.assertEqual(msgs[0]["content"], self.pool.system_prompt())
        roles = [m["role"] for m in msgs[1:]]
        self.assertEqual(roles[0], "user")  # a user turn first (Anthropic-style APIs)
        self.assertTrue(all(x != y for x, y in zip(roles, roles[1:])), roles)  # alternating
        self.assertEqual(roles[-1], "user")
        return msgs

    async def test_fork_seed(self) -> None:
        msgs = self._fork([{"role": "assistant", "content": "[script:f] looked around"},
                           {"role": "user", "content": "found x\nContinue with y; write to /w/r.md"}])
        self.assertTrue(msgs[1]["content"].startswith("[Harness note] The turns below are copied"))
        self.assertEqual(msgs[2]["content"], "[script:f] looked around")
        last = msgs[3]["content"]  # the main agent's appended instruction, then the note
        self.assertTrue(last.startswith("found x\nContinue with y; write to /w/r.md"))
        self.assertIn("You are now a FORKED SUB-AGENT", last)
        self.assertIn("cannot launch sub-agents", last)
        self.assertIn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", last)
        self.assertNotIn("{{", last)
        self.assertNotIn("answer.txt", last)

    async def test_fork_seed_ending_with_assistant(self) -> None:
        msgs = self._fork([{"role": "user", "content": "note"},
                           {"role": "assistant", "content": "[script:f] next I will search"}])
        self.assertEqual(msgs[1]["content"], "note")  # no header needed
        self.assertTrue(msgs[-1]["content"].startswith("[Harness note] You are now a FORKED"))

    async def test_addendum_follows_config(self) -> None:
        text = self.pool.prompt_addendum()
        self.assertIn("up to 3 sub-agents", text)
        self.assertIn("i = 0..2", text)
        self.assertIn("at most 6 commands", text)
        self.assertIn("10 launches over the whole run", text)
        self.assertNotIn("{{", text)

    async def test_self_compact_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            SubagentConfig.from_kwargs(5, 40, 24, "true")
        with self.assertRaises(ValueError):
            SubagentConfig.from_kwargs(0, 40, 24, False)
        self.assertEqual(SubagentConfig.from_kwargs(), SubagentConfig(5, 40, 24))


class LifecycleTests(PoolTestBase):
    async def test_spawn_by_brief_and_finish(self) -> None:
        self.use({"a": ["mkdir -p /tmp/wa && echo 42 > /tmp/wa/result.txt",
                        {"command": F.SUBMIT, "content": "Wrote 42 to /tmp/wa/result.txt."}]})
        self.write_slot(0, "Compute the answer. [script:a]")
        note, spawned = await self.pool.poll("cat > SUBCTX_0.txt")
        self.assertTrue(spawned)
        self.assertIn("SUBCTX_0 launched as subctx_0_1 (brief)", note)
        self.assertIn("sub-agents: 1/3 running, 1 launched (cap 10)", note)

        note = await self.finish_all()
        self.assertIn("SUBCTX_0 finished (completed, 2 steps) -> /tmp/.live_ctx/finished_subagents/subctx_0_1/", note)
        self.assertEqual(self.sandbox.read("/tmp/wa/result.txt").strip(), "42")
        # the slot is free again and the record moved to finished_subagents
        self.assertFalse(self.sandbox.exists(slot(0)))
        self.assertEqual(self.sandbox.read(f"{FIN}/subctx_0_1/brief.txt"), "Compute the answer. [script:a]")
        self.assertEqual(self.sandbox.read(f"{FIN}/subctx_0_1/final_message.txt"),
                         "[a#1] Wrote 42 to /tmp/wa/result.txt.")
        rec = self.record("subctx_0_1")
        for name in ("trajectory.json", "timing.json", "summary.json", "link.json",
                     "brief.txt", "final_message.txt"):
            self.assertTrue((rec / name).is_file(), name)
        summary = json.loads((rec / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["steps"], summary["n_lm_calls"]), ("completed", 2, 2))
        kinds = [json.loads(p.read_text())["kind"]
                 for p in sorted((rec / "context_snapshots").glob("turn-*.json"))]
        self.assertEqual(kinds, ["agent", "agent", "final"])
        link = json.loads((rec / "link.json").read_text())
        self.assertEqual(link["status"], "completed")
        self.assertIsNotNone(link["fold"])
        # nothing new to launch: an unchanged state is quiet apart from the status line
        note, spawned = await self.pool.poll("ls")
        self.assertFalse(spawned)
        self.assertEqual(note, "\n[sub-agents: 0/3 running, 1 launched (cap 10)]")

    async def test_relaunch_same_slot(self) -> None:
        self.use({"a": [F.SUBMIT], "b": [F.SUBMIT]})
        self.write_slot(1, "first [script:a]")
        await self.pool.poll("")
        await self.finish_all()
        self.write_slot(1, "first [script:a]")  # identical text launches again once freed
        note, spawned = await self.pool.poll("")
        self.assertTrue(spawned)
        self.assertIn("launched as subctx_1_2", note)
        await self.finish_all()
        self.assertTrue(self.sandbox.exists(f"{FIN}/subctx_1_2/brief.txt"))

    async def test_spawn_by_fork(self) -> None:
        self.use({"f": ["echo forked", F.SUBMIT]})
        main = [{"role": "system", "content": "S"}, {"role": "user", "content": "T"},
                {"role": "assistant", "content": "[script:f] plan"},
                {"role": "user", "content": "observation"}]
        await self.agent._ctx.write_mirror(self.sandbox, main)
        res = await self.sandbox.exec(f"cp {CTX}/LIVE_CTX_MAIN.txt {slot(2)}")
        self.assertEqual(res.return_code, 0)
        note, spawned = await self.pool.poll(f"cp {CTX}/LIVE_CTX_MAIN.txt {slot(2)}")
        self.assertTrue(spawned)
        self.assertIn("launched as subctx_2_1 (fork)", note)
        await self.finish_all()
        traj = json.loads((self.record("subctx_2_1") / "trajectory.json").read_text())
        self.assertTrue(traj[1]["content"].startswith("[Harness note]"))
        self.assertEqual(traj[2]["content"], "[script:f] plan")
        self.assertIn("forked", traj[-3]["content"] + traj[-4]["content"])
        self.assertEqual(self.worker("subctx_2_1").status, "completed")


class LimitTests(PoolTestBase):
    cfg = SubagentConfig(n_slots=2, max_steps=3, lifetime_cap=3)

    async def test_lifetime_cap(self) -> None:
        self.use({"a": [F.SUBMIT], "b": [F.SUBMIT], "c": [F.SUBMIT], "d": [F.SUBMIT]})
        self.write_slot(0, "[script:a]")
        self.write_slot(1, "[script:b]")
        await self.pool.poll("")
        await self.finish_all()
        self.write_slot(0, "[script:c]")
        self.write_slot(1, "[script:d]")
        note, spawned = await self.pool.poll("")
        self.assertTrue(spawned)
        self.assertIn("SUBCTX_0 launched as subctx_0_2", note)
        self.assertIn("SUBCTX_1 not launched: lifetime cap of 3 sub-agents reached", note)
        note, _ = await self.pool.poll("")  # the refusal is reported once per slot content
        self.assertNotIn("lifetime cap", note)
        await self.finish_all()
        self.assertEqual(len(self.pool._workers), 3)
        self.assertEqual(self.pool.usage()["n_lifetime_cap_refusals"], 1)
        self.assertTrue(self.sandbox.exists(slot(1)))  # left in place, never launched

    async def test_max_steps(self) -> None:
        self.use({"a": ["echo 1", "echo 2", "echo 3", "echo 4"]})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        note = await self.finish_all()
        self.assertIn("SUBCTX_0 finished (max_steps, 3 steps)", note)
        self.assertEqual(self.worker("subctx_0_1").n_lm_calls, 3)

    async def test_max_calls(self) -> None:
        self.use({"a": [None] * 10})  # never calls a tool
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        note = await self.finish_all()
        self.assertIn("SUBCTX_0 finished (max_calls, 0 steps)", note)
        self.assertEqual(self.worker("subctx_0_1").n_lm_calls, 6)

    async def test_concurrency_limited_to_slots(self) -> None:
        slow = [{"command": "echo busy", "sleep": 0.3}] * 3
        self.use({"a": slow, "b": slow, "c": [F.SUBMIT]})
        self.write_slot(0, "[script:a]")
        self.write_slot(1, "[script:b]")
        self.write_slot(2, "[script:c]")  # beyond n_slots: not a slot
        note, _ = await self.pool.poll("")
        self.assertIn("sub-agents: 2/2 running, 2 launched", note)
        self.assertNotIn("SUBCTX_2", note)
        self.write_slot(0, "edited while busy [script:c]")
        note, spawned = await self.pool.poll("")
        self.assertFalse(spawned)
        self.assertEqual(len(self.pool._running), 2)
        await self.finish_all()

    async def test_delete_to_cancel(self) -> None:
        self.use({"a": [{"command": "echo tick", "sleep": 0.2}] * 3})
        self.write_slot(0, "long job [script:a]")
        await self.pool.poll("")
        await wait_for(lambda: self.worker("subctx_0_1").steps >= 1)
        self.sandbox.host(slot(0)).unlink()
        note, spawned = await self.pool.poll(f"rm {slot(0)}")
        self.assertFalse(spawned)
        self.assertIn("SUBCTX_0 stopped (slot file deleted", note)
        w = self.worker("subctx_0_1")
        self.assertEqual(w.status, "cancelled_by_parent")
        self.assertLess(w.steps, 3)
        self.assertEqual(self.sandbox.read(f"{FIN}/subctx_0_1/brief.txt"), "long job [script:a]")
        self.assertTrue(self.sandbox.exists(f"{FIN}/subctx_0_1/final_message.txt"))
        summary = json.loads((self.record("subctx_0_1") / "summary.json").read_text())
        self.assertEqual(summary["status"], "cancelled_by_parent")
        self.assertFalse(self.pool._running)

    async def test_cancel_all_leaves_record(self) -> None:
        self.use({"a": [{"command": "echo tick", "sleep": 0.2}] * 3})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        await wait_for(lambda: self.worker("subctx_0_1").n_lm_calls >= 1)
        await self.pool.cancel_all()
        rec = self.record("subctx_0_1")
        self.assertEqual(json.loads((rec / "summary.json").read_text())["status"],
                         "cancelled_end_of_run")
        self.assertTrue((rec / "trajectory.json").is_file())
        self.assertIsNone(json.loads((rec / "link.json").read_text())["fold"])
        self.assertFalse(self.sandbox.exists(f"{FIN}/subctx_0_1"))  # sandbox left alone

    async def test_cancel_before_first_step_leaves_record(self) -> None:
        self.use({"a": [F.SUBMIT]})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")  # launched, but the task has not run yet
        await self.pool.cancel_all()
        summary = json.loads((self.record("subctx_0_1") / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["n_lm_calls"]), ("cancelled_end_of_run", 0))
        self.assertTrue((self.record("subctx_0_1") / "trajectory.json").is_file())


class WorkerBehaviourTests(PoolTestBase):
    cfg = SubagentConfig(n_slots=2, max_steps=8, lifetime_cap=0)
    agent_kwargs = {"context_budget_tokens": 900, "context_budget_reserve_tokens": 100}

    async def test_budget_stop(self) -> None:
        self.use({"a": ["seq 1 150"] * 8})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        note = await self.finish_all()
        self.assertIn("finished (budget", note)
        traj = json.loads((self.record("subctx_0_1") / "trajectory.json").read_text())
        notices = [m["content"] for m in traj if m["role"] == "user"
                   and "FINAL turn" in str(m["content"])]
        self.assertEqual(len(notices), 1)
        w = self.worker("subctx_0_1")
        self.assertLess(w.steps, 8)
        # the notice is followed by exactly one more turn
        last_user = max(i for i, m in enumerate(traj) if m in [{"role": "user", "content": notices[0]}])
        self.assertEqual([m["role"] for m in traj[last_user + 1:]], ["assistant", "tool"])

    async def test_budget_warning_once_before_final_turn(self) -> None:
        self.use({"a": ["seq 1 150"] * 8})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        await self.finish_all()
        traj = json.loads((self.record("subctx_0_1") / "trajectory.json").read_text())
        text = [str(m.get("content")) for m in traj]
        warn = [i for i, t in enumerate(text) if "Write your findings to your result file NOW" in t]
        final = [i for i, t in enumerate(text) if "FINAL turn" in t]
        self.assertEqual(len(warn), 1)
        self.assertEqual(len(final), 1)
        self.assertLess(warn[0], final[0])
        self.assertEqual(traj[warn[0]]["role"], "user")
        self.assertEqual(traj[warn[0] + 1]["role"], "assistant")  # the worker acted on it
        roles = [m["role"] for m in traj]
        self.assertFalse(any(a == b == "user" for a, b in zip(roles, roles[1:])), roles)
        summary = json.loads((self.record("subctx_0_1") / "summary.json").read_text())
        self.assertGreaterEqual(summary["budget_warning_tokens"], 0.75 * 800)
        timing = json.loads((self.record("subctx_0_1") / "timing.json").read_text())
        self.assertEqual(sum(1 for t in timing if t.get("budget_warning")), 1)

    async def test_no_budget_warning_for_small_work(self) -> None:
        self.use({"a": ["echo hi", F.SUBMIT]})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        await self.finish_all()
        summary = json.loads((self.record("subctx_0_1") / "summary.json").read_text())
        self.assertIsNone(summary["budget_warning_tokens"])

    async def test_workers_cannot_spawn(self) -> None:
        self.use({"a": [f"echo '[script:b]' > {slot(1)}", f"rm -f {slot(0)}",
                        f"sed -i s/a/b/ {CTX}/LIVE_CTX_MAIN.txt", F.SUBMIT],
                  "b": [F.SUBMIT]})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        await self.finish_all()
        self.assertFalse(self.sandbox.exists(slot(1)))
        self.assertEqual(len(self.pool._workers), 1)
        traj = json.loads((self.record("subctx_0_1") / "trajectory.json").read_text())
        refused = [m["content"] for m in traj if m["role"] == "tool" and "[not run" in m["content"]]
        self.assertEqual(len(refused), 3)
        self.assertEqual(self.worker("subctx_0_1").status, "completed")
        self.assertNotIn("sub-agents:", json.dumps(traj))  # no status notes for workers

    async def test_worker_uses_base_query_and_own_budget(self) -> None:
        from clm_harness.clm_agent.harness import ClmAgent

        class Hooked(ClmAgent):  # a subclass hooking the main loop, like ContextBench's
            async def _query_with_retry(self, model, messages):
                raise AssertionError("subclass query hook used by a sub-agent")

            def _is_successful_submit(self, command, result, tool_content):
                raise AssertionError("subclass submit hook used by a sub-agent")

        agent = Hooked(logs_dir=self.tmp / "hooked", model_name=F.MODEL,
                       context_budget_tokens=900, context_budget_reserve_tokens=100,
                       cost_metric="usd")
        pool = SubagentPool(agent, agent._ctx, agent._ctx.paths, self.cfg)
        await pool.setup(self.sandbox)
        self.use({"a": ["echo hi", F.SUBMIT]})
        with mock.patch.object(agent._budget, "calibrate") as calibrate:
            self.write_slot(0, "[script:a]")
            await pool.poll("")
            await wait_for(lambda: all(w.task.done() for w in pool._workers))
        self.assertEqual(pool._workers[0].status, "completed")
        calibrate.assert_not_called()

    async def test_lm_error_ends_worker(self) -> None:
        err = litellm.exceptions.AuthenticationError("bad key", llm_provider="openai", model=F.MODEL)
        self.use({"a": [{"raise": err}]})
        self.write_slot(0, "[script:a]")
        await self.pool.poll("")
        note = await self.finish_all()
        self.assertIn("finished (error:AuthenticationError, 0 steps)", note)
        rec = self.record("subctx_0_1")
        self.assertEqual(len(self.worker("subctx_0_1").metrics), 1)
        self.assertTrue((rec / "trajectory.json").is_file())


if __name__ == "__main__":
    unittest.main()

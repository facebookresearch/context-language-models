# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Golden-trace regression test for ``ClmAgent`` with optional features at their defaults.

A scripted model drives one run through a local sandbox (see ``clm_fakes``): ordinary
commands, a turn without a tool call, a context edit, an edit that matches nothing, a
large output and a submit. The run's ``trajectory.json``, ``context_snapshots/`` and
``trajectory.ctx.json`` must match ``golden/off_trace/`` byte for byte;
``usage.json`` and ``timing.json`` must match after dropping the code fingerprint and
zeroing wall-clock fields.

Regenerate the golden files (only when an intended change alters the trace)::

    CLM_REGEN_GOLDEN=1 PYTHONPATH=clm python -m pytest clm/tests/test_golden_off_trace.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import clm_fakes as F

GOLDEN = Path(__file__).resolve().parent / "golden" / "off_trace"
BYTE_EQUAL = ("trajectory.json", "trajectory.ctx.json")
NORMALIZED = {"usage.json": ("code_fingerprint",), "timing.json": ()}

EDIT_TURN_2 = (
    "python3 - <<'PY'\n"
    "import re; p='/tmp/.live_ctx/LIVE_CTX_MAIN.txt'; s=open(p).read()\n"
    "s=re.sub(r'\\n1\\n2\\n3\\n(?:\\d+\\n)*300\\n', '\\n[seq output dropped: 1..300]\\n', s)\n"
    "open(p,'w').write(s)\n"
    "PY"
)

MAIN_SCRIPT = [
    "mkdir -p /tmp/work && cd /tmp/work && printf 'alpha\\nbeta\\n' > notes.txt && cat notes.txt",
    {"content": "THOUGHT: thinking out loud without a tool call."},
    "seq 1 300",
    EDIT_TURN_2,
    "sed -i 's/no-such-text/x/' /tmp/.live_ctx/LIVE_CTX_MAIN.txt",
    "pwd && wc -l < notes.txt && echo ready > /tmp/work/out.txt",
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT extra",
    F.SUBMIT,
]


def run_off_trace(logs_dir: Path, sandbox_root: Path, **agent_kwargs) -> F.LocalSandbox:
    import asyncio

    sandbox = F.LocalSandbox(sandbox_root)
    agent = F.make_agent(logs_dir, **agent_kwargs)
    asyncio.run(F.run_agent(agent, sandbox, F.ScriptedLLM(MAIN_SCRIPT)))
    return sandbox


class GoldenOffTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="clm_golden_"))
        self.logs = self.tmp / "agent"
        run_off_trace(self.logs, self.tmp / "sandbox")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _files(self) -> list[str]:
        snaps = sorted(p.name for p in (self.logs / "context_snapshots").glob("turn-*.json"))
        return [*BYTE_EQUAL, *NORMALIZED, *(f"context_snapshots/{n}" for n in snaps)]

    def test_matches_golden(self) -> None:
        if os.environ.get("CLM_REGEN_GOLDEN") == "1":
            shutil.rmtree(GOLDEN, ignore_errors=True)
            (GOLDEN / "context_snapshots").mkdir(parents=True)
            for rel in self._files():
                if rel in NORMALIZED:
                    data = F.normalized_json(self.logs / rel, NORMALIZED[rel])
                    (GOLDEN / rel).write_text(json.dumps(data, indent=2) + "\n")
                else:
                    shutil.copyfile(self.logs / rel, GOLDEN / rel)
            self.skipTest("golden files regenerated")

        want = sorted(p.relative_to(GOLDEN).as_posix() for p in GOLDEN.rglob("*.json"))
        self.assertEqual(sorted(self._files()), want, "set of recorded files changed")
        self.assertFalse((self.logs / "subagents").exists())
        for rel in want:
            with self.subTest(file=rel):
                if rel in NORMALIZED:
                    got = F.normalized_json(self.logs / rel, NORMALIZED[rel])
                    self.assertEqual(got, json.loads((GOLDEN / rel).read_text()))
                else:
                    self.assertEqual((self.logs / rel).read_bytes(), (GOLDEN / rel).read_bytes())

    def test_trace_exercises_the_loop(self) -> None:
        """Guard against a golden file that silently stopped covering the edit paths."""
        traj = json.loads((self.logs / "trajectory.json").read_text())
        tools = "\n".join(str(m.get("content")) for m in traj if m.get("role") == "tool")
        self.assertIn("edit applied", tools)
        self.assertIn("NO change", tools)
        self.assertIn("submit ignored", tools)
        self.assertIn("(No tool call", "\n".join(str(m.get("content")) for m in traj))
        usage = json.loads((self.logs / "usage.json").read_text())
        self.assertGreater(usage["cost_usd"], 0)
        self.assertEqual(usage["n_ctx_syncs"], 1)


if __name__ == "__main__":
    unittest.main()

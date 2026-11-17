# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise mirror reconciliation through the environment's real step/gate."""

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.paths import Paths
from clm_harness.context_utils.context_string import render_editable


class ContextEnvTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Task"},
            {"role": "assistant", "content": None, "reasoning_content": "Think",
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "bash", "arguments": '{"command":"ls"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "logs"},
            {"role": "assistant", "content": "Old note"},
        ]
        self.budget = SimpleNamespace(strict_target=0, note_compaction=Mock())
        self.env = ContextEnv(paths=Paths(None), budget=self.budget, allow_edit_growth=False)
        self.addCleanup(self.env.host_mirror.unlink)
        self.env.write_mirror = AsyncMock(return_value=render_editable(self.messages))
        self.env.execute = AsyncMock(return_value=SimpleNamespace(
            stdout="", stderr="", return_code=0))
        # Deterministic counter with a cost for message structure. No tokenizer or
        # serving dependency is needed to catch use of a normalized-text baseline.
        counter = patch("clm_harness.context_env.env.tk.count_tokens",
                        side_effect=lambda messages: (len(json.dumps(messages)), "test"))
        counter.start()
        self.addCleanup(counter.stop)

    async def run_edit(self, edited):
        self.env.read_mirror = AsyncMock(return_value=edited)
        return await self.env.step("edit LIVE_CTX_MAIN", self.messages, environment=None)

    async def test_local_edit_uses_actual_structured_baseline(self):
        expected = deepcopy(self.messages)
        expected[-1]["content"] = "Note"
        result = await self.run_edit(render_editable(self.messages).replace("Old note", "Note"))
        self.assertTrue(result.ctx_changed)
        self.assertEqual(self.messages, expected)
        self.assertEqual(self.env.n_ctx_syncs, 1)
        self.assertEqual(self.env.n_ctx_rejected, 0)
        self.budget.note_compaction.assert_called_once_with()

    async def test_rejected_growth_is_atomic(self):
        original = deepcopy(self.messages)
        result = await self.run_edit(render_editable(self.messages).replace("Old note", "x" * 1000))
        self.assertFalse(result.ctx_changed)
        self.assertEqual(self.messages, original)
        self.assertEqual(self.env.n_ctx_rejected, 1)
        self.assertEqual(self.env.n_ctx_syncs, 0)
        self.budget.note_compaction.assert_not_called()

    async def test_ambiguous_edit_is_reported_without_crashing(self):
        original = deepcopy(self.messages)
        result = await self.run_edit(render_editable(self.messages) +
                                     "\n[[CTX_TURN 1 role=assistant]]\ncopy")
        self.assertFalse(result.ctx_changed)
        self.assertIn("duplicate CTX_TURN", result.notes)
        self.assertEqual(self.messages, original)
        self.assertEqual(self.env.n_ctx_rejected, 1)
        self.budget.note_compaction.assert_not_called()

    async def test_untouched_file_does_not_reconcile(self):
        original = deepcopy(self.messages)
        result = await self.run_edit(render_editable(self.messages))
        self.assertFalse(result.ctx_changed)
        self.assertEqual(self.messages, original)
        self.assertEqual(self.env.n_ctx_syncs, 0)
        self.budget.note_compaction.assert_not_called()


if __name__ == "__main__":
    unittest.main()

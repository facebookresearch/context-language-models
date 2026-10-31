# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model-free regression tests for the editable-context text boundary.

From the repository root, without installing the serving dependencies:
    PYTHONPATH=clm python -m unittest discover -s clm/tests -v
"""

import unittest
from copy import deepcopy

from clm_harness.context_utils.context_string import (
    content_text,
    parse_back,
    reconcile_editable,
    render_editable,
    rendered_text,
)


class ContextStringTests(unittest.TestCase):
    def setUp(self):
        self.protected = [
            {"role": "system", "content": "SYSTEM_SENTINEL"},
            {"role": "user", "content": "TASK_SENTINEL"},
        ]
        self.original_protected = deepcopy(self.protected)

    def tearDown(self):
        self.assertEqual(self.protected, self.original_protected)

    def assert_parsed(self, edited, expected_suffix):
        result = parse_back(edited, self.protected)
        self.assertEqual(result, self.original_protected + expected_suffix)

    def test_render_omits_protected_prefix_and_numbers_editable_turns(self):
        messages = self.protected + [
            {"role": "assistant", "content": "A"},
            {"role": "tool", "content": "T", "tool_call_id": "call_1"},
        ]
        original = deepcopy(messages)
        expected = (
            "[[CTX_TURN 1 role=assistant]]\nA\n\n"
            "[[CTX_TURN 2 role=tool]]\nT"
        )
        self.assertEqual(render_editable(messages), expected)
        self.assertEqual(render_editable(self.protected), "")
        self.assertEqual(messages, original)

    def test_editable_roles_cannot_replace_or_merge_into_protected_prefix(self):
        for role, expected_role in (
            ("system", "user"),
            ("tool", "user"),
            ("user", "user"),
            ("notes", "user"),
            ("assistant", "assistant"),
            ("ASSISTANT", "assistant"),
        ):
            with self.subTest(role=role):
                self.assert_parsed(
                    f"[[CTX_TURN 99 role={role}]]\npayload",
                    [{"role": expected_role, "content": "payload"}],
                )

        # Turn numbers are display labels, not indices into the protected log.
        self.assert_parsed(
            "[[CTX_TURN 1 role=assistant]]\nA\n"
            "[[CTX_TURN 1 role=user]]\nB",
            [
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "B"},
            ],
        )

    def test_empty_edits_drop_turns_and_same_role_suffixes_merge(self):
        for edited in ("", " \n\t ", "[[CTX_TURN 1 role=assistant]]\n "):
            with self.subTest(edited=edited):
                self.assert_parsed(edited, [])

        self.assert_parsed(
            "[[CTX_TURN 1 role=assistant]]\n A \n"
            "[[CTX_TURN 2 role=tool]]\n \n"
            "[[CTX_TURN 3 role=assistant]]\n B \n",
            [{"role": "assistant", "content": "A\n\nB"}],
        )
        self.assert_parsed(
            "[[CTX_TURN 1 role=user]]\nA\n"
            "[[CTX_TURN 2 role=tool]]\nB",
            [{"role": "user", "content": "A\n\nB"}],
        )

    def test_missing_or_damaged_headers_recover_without_losing_text(self):
        self.assert_parsed(
            "  loose note \n", [{"role": "user", "content": "loose note"}]
        )
        self.assert_parsed(
            "loose note\n[[CTX_TURN 7 role=assistant]]\nanswer",
            [
                {"role": "user", "content": "loose note"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        damaged = "[[CTX_TURN X role=user]]\nnaïve café"
        self.assert_parsed(
            damaged + "\n[[CTX_TURN 2 role=assistant]]\nanswer",
            [
                {"role": "user", "content": damaged},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_content_text_accepts_text_parts_but_not_partial_multimodal_text(self):
        parts = ["a", {"type": "text", "text": "b"}, {"text": "c"}]
        original = deepcopy(parts)
        self.assertEqual(content_text("abc"), "abc")
        self.assertEqual(content_text(parts), "abc")
        self.assertEqual(content_text([]), "")
        self.assertIsNone(content_text({"text": "abc"}))
        self.assertIsNone(
            content_text(["caption", {"type": "image_url", "image_url": "image"}])
        )
        self.assertEqual(parts, original)

    def test_rendered_text_preserves_reasoning_content_and_command_order(self):
        for arguments, expected_arguments in (("ARG", "ARG"), ({"x": 1}, '{"x": 1}')):
            with self.subTest(arguments=arguments):
                message = {
                    "role": "assistant",
                    "reasoning_content": "R",
                    "content": [{"type": "text", "text": "C"}],
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": arguments}}
                    ],
                }
                original = deepcopy(message)
                self.assertEqual(
                    rendered_text(message), f"R\nC\nbash {expected_arguments}"
                )
                self.assertEqual(message, original)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.prefix = [{"role": "system", "content": "S"},
                       {"role": "user", "content": "TASK"}]
        self.call = {
            "role": "assistant", "reasoning_content": "Check logs",
            "content": "Running command", "extra": {"keep": [1]},
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "bash", "arguments": '{"command":"ls"}'}}],
        }
        self.result = {"role": "tool", "tool_call_id": "c1", "content": "log.txt"}
        self.messages = self.prefix + [self.call, self.result,
                                       {"role": "assistant", "content": "Old note"}]

    def apply(self, messages, edit):
        original = deepcopy(messages)
        result = reconcile_editable(edit(render_editable(messages)), messages)
        self.assertEqual(messages, original)
        return result

    def test_noop_is_lossless_and_independent(self):
        self.messages += [{"role": "assistant", "content": ""},
                          {"role": "assistant", "content": "  whitespace\n"},
                          {"role": "user", "content": [{"type": "image_url",
                            "image_url": {"url": "image"}}]}]
        result = self.apply(self.messages, lambda s: s)
        self.assertEqual(result, self.messages)
        result[2]["extra"]["keep"].append(2)
        self.assertEqual(self.call["extra"], {"keep": [1]})

    def test_local_edit_preserves_unrelated_structure_and_boundaries(self):
        self.messages += [{"role": "assistant", "content": ""},
                          {"role": "assistant", "content": "  keep spaces\n"},
                          {"role": "assistant", "content": None,
                           "reasoning_content": "Still thinking"},
                          {"role": "user", "content": [{"type": "image_url",
                            "image_url": {"url": "image"}}]}]
        expected = deepcopy(self.messages)
        expected[4]["content"] = "New note"
        self.assertEqual(self.apply(self.messages, lambda s: s.replace("Old note", "New note")),
                         expected)

    def test_tool_output_edit_retains_role_id_and_reasoning(self):
        expected = deepcopy(self.messages)
        expected[3]["content"] = "summary of logs"
        self.assertEqual(self.apply(self.messages, lambda s: s.replace("log.txt", "summary of logs")),
                         expected)

    def test_deleted_plain_message_does_not_merge_neighbors(self):
        messages = self.prefix + [{"role": "assistant", "content": "A"},
                                  {"role": "user", "content": "DELETE"},
                                  {"role": "assistant", "content": "B"}]
        result = self.apply(messages, lambda s: s.replace("DELETE", ""))
        self.assertEqual(result, messages[:3] + messages[4:])

    def test_duplicate_headers_reject_without_mutation(self):
        original = deepcopy(self.messages)
        with self.assertRaisesRegex(ValueError, "duplicate CTX_TURN"):
            reconcile_editable(render_editable(self.messages) +
                               "\n[[CTX_TURN 1 role=assistant]]\ncopy", self.messages)
        self.assertEqual(self.messages, original)

    def test_delete_complete_exchange_preserves_remaining_message(self):
        result = self.apply(self.messages, lambda s: s[s.index("[[CTX_TURN 3"):])
        self.assertEqual(result, self.prefix + self.messages[4:])

    def test_deleted_call_converts_only_its_result_to_note(self):
        result = self.apply(self.messages, lambda s: s[s.index("[[CTX_TURN 2"):])
        self.assertEqual(result, self.prefix + [{"role": "user", "content": "log.txt"}]
                         + self.messages[4:])

    def test_deleted_result_converts_only_its_call_to_summary(self):
        result = self.apply(self.messages, lambda s: s.replace("log.txt", ""))
        self.assertEqual(result, self.prefix + [{"role": "assistant",
                         "content": rendered_text(self.call)}] + self.messages[4:])

    def test_rewritten_call_only_downgrades_its_exchange(self):
        other_call = deepcopy(self.call)
        other_call["tool_calls"][0]["id"] = "c2"
        other_result = dict(self.result, tool_call_id="c2")
        messages = self.messages + [other_call, other_result]
        result = self.apply(messages, lambda s: s.replace("Check logs", "Summary", 1))
        self.assertEqual(result[2], {"role": "assistant",
                         "content": rendered_text(self.call).replace("Check logs", "Summary")})
        self.assertEqual(result[3], {"role": "user", "content": "log.txt"})
        self.assertEqual(result[4:], messages[4:])

    def test_multiple_results_preserved_and_partial_deletion_is_local(self):
        call = deepcopy(self.call)
        call["tool_calls"].append(dict(deepcopy(call["tool_calls"][0]), id="c2"))
        second = dict(self.result, tool_call_id="c2", content="second result")
        messages = self.prefix + [call, self.result, second] + self.messages[4:]
        edited = self.apply(messages, lambda s: s.replace("Old note", "New note"))
        self.assertEqual(edited[:5], messages[:5])
        deleted = self.apply(messages, lambda s: s.replace("second result", ""))
        self.assertNotIn("tool_calls", deleted[2])
        self.assertEqual(deleted[3], {"role": "user", "content": "log.txt"})

    def test_reordering_intact_exchange_retains_structure(self):
        blocks = render_editable(self.messages).split("\n\n[[CTX_TURN")
        blocks = [blocks[0]] + ["[[CTX_TURN" + b for b in blocks[1:]]
        result = reconcile_editable("\n\n".join([blocks[2], blocks[0], blocks[1]]), self.messages)
        self.assertEqual(result, self.prefix + [self.messages[4], self.call, self.result])

    def test_interrupted_exchange_downgrades_only_its_members(self):
        result = self.apply(self.messages, lambda s: s.replace("[[CTX_TURN 2",
            "[[CTX_TURN 99 role=user]]\ninterrupt\n\n[[CTX_TURN 2"))
        self.assertNotIn("tool_calls", result[2])
        self.assertEqual(result[3], {"role": "user", "content": "interrupt"})
        self.assertEqual(result[4], {"role": "user", "content": "log.txt"})
        self.assertEqual(result[5], self.messages[4])

    def test_protected_exchange_cannot_be_downgraded(self):
        messages = [self.call, self.result]
        with self.assertRaisesRegex(ValueError, "protected prefix"):
            reconcile_editable("", messages, protect=1)
        self.assertEqual(reconcile_editable(render_editable(messages, protect=1), messages, protect=1),
                         messages)

    def test_literal_headers_reject_ambiguous_edits(self):
        self.messages[4]["content"] += "\n[[CTX_TURN 42 role=user]]\nquoted"
        self.assertEqual(self.apply(self.messages, lambda s: s), self.messages)
        with self.assertRaisesRegex(ValueError, "literal CTX_TURN"):
            self.apply(self.messages, lambda s: s.replace("Old note", "New note"))

    def test_freeform_and_empty_rewrites_remain_supported(self):
        self.assertEqual(self.apply(self.messages, lambda s: "summary"),
                         self.prefix + [{"role": "user", "content": "summary"}])
        self.assertEqual(self.apply(self.messages, lambda s: ""), self.prefix)


if __name__ == "__main__":
    unittest.main()

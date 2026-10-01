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


if __name__ == "__main__":
    unittest.main()

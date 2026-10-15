# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU/model-free checks of transcript suppression at ContextEnv.step.

Run with PYTHONPATH=clm python -m unittest discover -s clm/tests -v.
The shell smoke test uses an existing Bash, and skips when unavailable.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clm_harness.context_env.env import ContextEnv, _ECHO_NOTE, _exact_match_spans
from clm_harness.context_env.paths import Paths
from clm_harness.context_utils.context_string import render_editable
from clm_harness.utils import tokens as tk
from clm_harness.utils.budget import BudgetController


class FakeSandbox:
    """Harbor's three operations; execute a callback against the uploaded mirror."""

    def __init__(self, action):
        self.action = action
        self.text = ""
        self.uploads = self.downloads = 0
        self.result = None
        self.fail_download = False

    async def upload_file(self, source, destination):
        self.uploads += 1
        self.uploaded_bytes = Path(source).read_bytes()
        self.text = self.uploaded_bytes.decode("utf-8")

    async def download_file(self, source, destination):
        self.downloads += 1
        if self.fail_download:
            raise OSError("mirror unavailable")
        Path(destination).write_text(self.text, encoding="utf-8", newline="")

    async def exec(self, command, timeout_sec):
        self.result = self.action(self, command)
        return self.result


def result(stdout="", stderr="", code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=code)


class ContextEchoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "protected system"},
            {"role": "user", "content": "protected task"},
            {"role": "assistant", "content": "original transcript payload"},
        ]
        # No tokenizer download or model dependency in these tests.
        self.token_patch = patch.object(tk, "get_encoder", return_value=None)
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)

    def env(self, **kwargs):
        env = ContextEnv(paths=Paths(SimpleNamespace()), budget=BudgetController(0), **kwargs)
        self.addCleanup(env.host_mirror.unlink, missing_ok=True)
        return env

    async def step(self, env, action, command="cat /tmp/.live_ctx/LIVE_CTX_MAIN.txt"):
        sandbox = FakeSandbox(action)
        observation = await env.step(command, self.messages, environment=sandbox)
        self.assertEqual((sandbox.uploads, sandbox.downloads), (1, 1))
        self.assertIs(observation.result, sandbox.result)
        return observation, sandbox

    async def test_full_echo_both_streams_repeats_and_diagnostics(self):
        before = render_editable(self.messages)
        original = deepcopy(self.messages)
        obs, sandbox = await self.step(
            self.env(), lambda s, c: result("before\n" + s.text * 2 + "\nafter", s.text + "\nerror", 7)
        )
        self.assertNotIn(before, obs.render())
        self.assertEqual(obs.stdout_block.count(_ECHO_NOTE), 3)
        self.assertIn("before\n", obs.render())
        self.assertIn("\nafter\n", obs.render())
        self.assertIn("\nerror\n\n(exit_code=7)", obs.render())
        self.assertEqual(sandbox.result.stdout, "before\n" + before * 2 + "\nafter")
        self.assertEqual(sandbox.result.stderr, before + "\nerror")
        self.assertEqual(sandbox.result.return_code, 7)
        self.assertEqual(self.messages, original)

    async def test_contained_pre_and_post_edit_snapshots(self):
        self.messages[-1]["content"] = "summary\nlong obsolete body"
        before = render_editable(self.messages)
        after = before.split("\nlong obsolete body")[0]

        def edit(sandbox, command):
            sandbox.text = after
            return result(before + "\n" + after + "\nedit diagnostic")

        obs, _ = await self.step(self.env(), edit)
        self.assertTrue(obs.ctx_changed)
        self.assertEqual(obs.stdout_block.count(_ECHO_NOTE), 2)
        self.assertNotIn("long obsolete body", obs.render())
        self.assertIn("edit diagnostic", obs.render())
        self.assertEqual(self.messages[-1]["content"], "summary")

    async def test_rejected_growth_still_suppresses_post_snapshot(self):
        before = render_editable(self.messages)
        after = before + "\n" + "larger edited body " * 100

        def edit(sandbox, command):
            sandbox.text = after
            return result(before, after)

        original = deepcopy(self.messages)
        obs, _ = await self.step(self.env(allow_edit_growth=False), edit)
        self.assertFalse(obs.ctx_changed)
        self.assertEqual(self.messages, original)
        self.assertEqual(obs.stdout_block.count(_ECHO_NOTE), 2)
        self.assertIn("REJECTED", obs.notes)

    async def test_partially_overlapping_pre_post_echoes_use_original_stream(self):
        header = "[[CTX_TURN 1 role=assistant]]\n"
        self.messages[-1]["content"] = "old\n" + header + "shared"
        before = render_editable(self.messages)
        after = header + "shared\nnew"

        def edit(sandbox, command):
            sandbox.text = after
            return result("diagnostic before\n" + before + "\nnew\ndiagnostic after")

        obs, _ = await self.step(self.env(), edit)
        self.assertEqual(
            obs.stdout_block,
            "diagnostic before\n" + _ECHO_NOTE + "\ndiagnostic after\n\n(exit_code=0)",
        )

    async def test_self_overlapping_echoes_and_disjoint_diagnostics(self):
        header = "[[CTX_TURN 1 role=assistant]]\n"
        unit = header + "x\n"
        self.messages[-1]["content"] = "x\n" + unit
        before = render_editable(self.messages)
        self.assertEqual(before, unit * 2)
        obs, _ = await self.step(
            self.env(), lambda s, c: result(unit * 3 + "DIAGNOSTIC\n" + before)
        )
        self.assertEqual(obs.stdout_block, _ECHO_NOTE + "DIAGNOSTIC\n" + _ECHO_NOTE + "\n\n(exit_code=0)")

    async def test_periodic_serialized_transcript_overlap_suppresses_union(self):
        header = "[[CTX_TURN 1 role=assistant]]\n"
        unit = header + "x\n"
        self.messages[-1]["content"] = "x\n" + unit * 199
        self.assertEqual(render_editable(self.messages), unit * 200)
        obs, _ = await self.step(
            self.env(), lambda s, c: result("before\n" + unit * 400 + "after")
        )
        self.assertEqual(obs.stdout_block, "before\n" + _ECHO_NOTE + "after\n\n(exit_code=0)")

    async def test_replacement_does_not_synthesize_match_consuming_diagnostic(self):
        header = "[[CTX_TURN 1 role=assistant]]\n"
        self.messages[-1]["content"] = "x" * 200
        before = render_editable(self.messages)
        after = header + _ECHO_NOTE

        def edit(sandbox, command):
            sandbox.text = after
            return result(header + before)

        obs, _ = await self.step(self.env(), edit)
        self.assertEqual(obs.stdout_block, header + _ECHO_NOTE + "\n\n(exit_code=0)")

    async def test_upload_bytes_and_raw_crlf_post_echo_preserve_edit_semantics(self):
        self.messages[-1]["content"] = "Unicode: caf\u00e9 \u96ea \U0001f642\nsecond line"
        before = render_editable(self.messages)
        original = deepcopy(self.messages)

        def crlf_only(sandbox, command):
            self.assertEqual(sandbox.uploaded_bytes, before.encode("utf-8"))
            self.assertNotIn(b"\r\n", sandbox.uploaded_bytes)
            sandbox.text = sandbox.text.replace("\n", "\r\n")
            return result(sandbox.text + "\r\nUnicode diagnostic: \u96ea")

        obs, sandbox = await self.step(self.env(), crlf_only)
        self.assertFalse(obs.ctx_changed)
        self.assertEqual(self.messages, original)
        self.assertIn(_ECHO_NOTE, obs.stdout_block)
        self.assertIn("Unicode diagnostic: \u96ea", obs.stdout_block)
        self.assertIn("\r\n", obs.result.stdout)
        self.assertEqual(obs.result.stdout, sandbox.text + "\r\nUnicode diagnostic: \u96ea")

    async def test_crlf_post_edit_echo_and_lf_pre_echo_are_both_suppressed(self):
        before = render_editable(self.messages)
        after = "[[CTX_TURN 1 role=assistant]]\r\nUnicode summary: \u96ea\r\n"

        def edit(sandbox, command):
            sandbox.text = after
            return result(before + "\n" + after)

        obs, _ = await self.step(self.env(), edit)
        self.assertTrue(obs.ctx_changed)
        self.assertEqual(self.messages[-1]["content"], "Unicode summary: \u96ea")
        self.assertEqual(obs.stdout_block.count(_ECHO_NOTE), 2)

    async def test_newline_transformed_output_does_not_match_lf_only_snapshot(self):
        obs, _ = await self.step(self.env(), lambda s, c: result(s.text.replace("\n", "\r\n")))
        self.assertNotIn(_ECHO_NOTE, obs.stdout_block)

    async def test_lone_cr_post_echo_preserves_normalized_parsing(self):
        original = deepcopy(self.messages)
        for changed in (False, True):
            with self.subTest(changed=changed):
                self.messages = deepcopy(original)
                after = render_editable(self.messages).replace("\n", "\r")
                if changed:
                    after = after.replace("original transcript payload", "Unicode summary: \u96ea")

                def edit(sandbox, command):
                    sandbox.text = after
                    return result(after)

                obs, _ = await self.step(self.env(), edit)
                self.assertEqual(obs.ctx_changed, changed)
                self.assertEqual(obs.result.stdout, after)
                self.assertEqual(obs.stdout_block, _ECHO_NOTE + "\n\n(exit_code=0)")
                expected = deepcopy(original)
                if changed:
                    expected[-1]["content"] = "Unicode summary: \u96ea"
                self.assertEqual(self.messages, expected)

    async def test_read_mirror_default_retains_universal_newline_behavior(self):
        env = self.env()
        sandbox = FakeSandbox(lambda s, c: result())
        sandbox.text = "a\r\nb\rc\n"
        self.assertEqual(await env.read_mirror(sandbox), "a\nb\nc\n")
        self.assertEqual(await env.read_mirror(sandbox, preserve_newlines=True), sandbox.text)

    async def test_suppression_precedes_character_and_budget_caps(self):
        self.messages[-1]["content"] = "SENSITIVE-HEAD" + "x" * 20000 + "SENSITIVE-TAIL"
        env = self.env(observation_max_chars=200)
        env.context_budget_tokens = 100000
        with patch.object(env.budget, "cap_newest_output", side_effect=lambda shown, text: text) as cap:
            obs, _ = await self.step(env, lambda s, c: result(s.text + "\nretained diagnostic"))
        self.assertIn(_ECHO_NOTE, obs.stdout_block)
        self.assertIn("retained diagnostic", obs.stdout_block)
        self.assertNotIn("SENSITIVE-", cap.call_args.args[1])
        self.assertNotIn("HEAD (", obs.stdout_block)

    async def test_empty_header_only_and_headerless_edits_preserve_output(self):
        for text in ("", "0", "ok", "[[CTX_TURN 1 role=assistant]]\n", "not a header\nok"):
            with self.subTest(text=text):
                def edit(sandbox, command):
                    sandbox.text = text
                    return result("ordinary ok 0 diagnostic\n" + text)

                obs, _ = await self.step(self.env(), edit)
                self.assertNotIn(_ECHO_NOTE, obs.stdout_block)
                self.assertIn("ordinary ok 0 diagnostic", obs.stdout_block)

        self.messages = self.messages[:2]
        obs, _ = await self.step(self.env(), lambda s, c: result("ordinary diagnostic"))
        self.assertEqual(obs.stdout_block, "ordinary diagnostic\n\n(exit_code=0)")

    async def test_short_serialized_body_and_exact_trailing_whitespace(self):
        for body in ("x", "x\n", "x \n\n"):
            with self.subTest(body=body):
                self.messages[-1]["content"] = body
                obs, _ = await self.step(self.env(), lambda s, c: result(s.text))
                self.assertEqual(obs.stdout_block, _ECHO_NOTE + "\n\n(exit_code=0)")

    async def test_partial_transformed_and_cross_stream_copies_are_outside_scope(self):
        before = render_editable(self.messages)
        outputs = (before[:15], before.replace("payload", "PAYLOAD"), before.encode().hex())
        for output in outputs:
            with self.subTest(output=output):
                obs, _ = await self.step(self.env(), lambda s, c: result(output))
                self.assertEqual(obs.stdout_block, output + "\n\n(exit_code=0)")
        split = len(before) // 2
        obs, _ = await self.step(self.env(), lambda s, c: result(before[:split], before[split:]))
        self.assertNotIn(_ECHO_NOTE, obs.stdout_block)

    async def test_failed_download_still_uses_pre_snapshot(self):
        def echo(sandbox, command):
            sandbox.fail_download = True
            return result(sandbox.text)

        obs, _ = await self.step(self.env(), echo)
        self.assertEqual(obs.stdout_block, _ECHO_NOTE + "\n\n(exit_code=0)")

    async def test_echo_removed_at_observation_limit_boundaries(self):
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                before = render_editable(self.messages)
                env = self.env(observation_max_chars=len(before) + delta)
                obs, _ = await self.step(env, lambda s, c: result(s.text))
                self.assertNotIn("original transcript payload", obs.stdout_block)
                self.assertIn("exit_code=0", obs.stdout_block)

    async def test_unrelated_long_output_still_uses_existing_truncation(self):
        diagnostic = "diagnostic-head" + "x" * 200 + "diagnostic-tail"
        env = self.env(observation_max_chars=40)
        obs, _ = await self.step(env, lambda s, c: result(diagnostic))
        self.assertEqual(obs.stdout_block, env.truncate_observation(diagnostic) + "\n\n(exit_code=0)")

    async def test_silent_internal_edit_keeps_empty_raw_output(self):
        def edit(sandbox, command):
            sandbox.text = sandbox.text.replace("original transcript payload", "summary")
            return result()

        obs, _ = await self.step(self.env(), edit, "python3 edit.py")
        self.assertTrue(obs.ctx_changed)
        self.assertEqual(obs.stdout_block, "(no output)\n\n(exit_code=0)")
        self.assertEqual((obs.result.stdout, obs.result.stderr), ("", ""))

    async def test_content_matching_is_independent_of_command_or_path(self):
        obs, _ = await self.step(
            self.env(), lambda s, c: result("copied file:\n" + s.text), "cat unrelated-copy.txt"
        )
        self.assertIn("copied file:\n" + _ECHO_NOTE, obs.stdout_block)
        obs, _ = await self.step(
            self.env(), lambda s, c: result("LIVE_CTX_MAIN.txt: ordinary diagnostic"),
            "echo LIVE_CTX_MAIN.txt",
        )
        self.assertNotIn(_ECHO_NOTE, obs.stdout_block)


class ExactMatchSpanTests(unittest.TestCase):
    def test_matches_bounded_simple_oracle_including_unicode_and_empty_inputs(self):
        for output_size in range(5):
            for output_chars in product("a\u96ea", repeat=output_size):
                output = "".join(output_chars)
                for snapshot_size in range(4):
                    for snapshot_chars in product("a\u96ea", repeat=snapshot_size):
                        snapshot = "".join(snapshot_chars)
                        expected = [
                            (start, start + len(snapshot))
                            for start in range(len(output) - len(snapshot) + 1)
                            if snapshot and output.startswith(snapshot, start)
                        ]
                        with self.subTest(output=output, snapshot=snapshot):
                            self.assertEqual(list(_exact_match_spans(output, snapshot)), expected)

    def test_periodic_headered_transcript_matches_every_overlapping_start(self):
        unit = "[[CTX_TURN 1 role=assistant]]\nshared\n"
        snapshot = unit * 200
        output = unit * 400
        self.assertEqual(
            list(_exact_match_spans(output, snapshot)),
            [(start * len(unit), (start + 200) * len(unit)) for start in range(201)],
        )


_BASH = shutil.which("bash")
if os.name == "nt":
    _BASH = next((str(p) for p in (
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
    ) if p.is_file()), None)


class LocalShellSandbox:
    def __init__(self, cwd):
        self.cwd = cwd

    async def upload_file(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download_file(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec(self, command, timeout_sec):
        completed = subprocess.run(
            [_BASH, "--noprofile", "--norc", "-c", command], cwd=self.cwd,
            capture_output=True, timeout=timeout_sec,
        )
        # Harbor decodes bytes without universal-newline conversion.
        return result(
            completed.stdout.decode("utf-8", errors="replace"),
            completed.stderr.decode("utf-8", errors="replace"), completed.returncode,
        )


@unittest.skipUnless(_BASH, "Bash is unavailable; no runtime is installed by tests")
class ContextEchoShellTests(unittest.IsolatedAsyncioTestCase):
    setUp = ContextEchoTests.setUp

    async def test_real_symlink_readback_when_supported(self):
        with tempfile.TemporaryDirectory(prefix="echo_link_") as directory:
            root = Path(directory)
            mirror = root / "mirror.txt"
            mirror.write_text("", encoding="utf-8")
            link = root / "linked.txt"
            try:
                link.symlink_to(mirror)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            env = ContextEnv(
                paths=Paths(SimpleNamespace(_CTX_FILE=mirror.as_posix())),
                budget=BudgetController(0), persistent_bash=False,
            )
            self.addCleanup(env.host_mirror.unlink, missing_ok=True)
            obs = await env.step("cat linked.txt", self.messages, environment=LocalShellSandbox(root))
            self.assertEqual(obs.result.return_code, 0)
            self.assertIn(_ECHO_NOTE, obs.stdout_block)

    async def test_real_shell_paths_wrappers_and_allowed_operations(self):
        with tempfile.TemporaryDirectory(prefix="context echo ") as directory, \
                tempfile.TemporaryDirectory(prefix="echo_state_") as state_directory:
            root = Path(directory)
            ns = SimpleNamespace(
                _CTX_DIR=(root / "context").as_posix(),
                _CTX_FILE=(root / "context" / "mirror.txt").as_posix(),
                _STATE_DIR=Path(state_directory).as_posix(),
            )
            # Existing setup_dirs assumes space-free custom paths. The mirror
            # contains spaces here; initialize the shell state independently.
            Path(ns._STATE_DIR, "cwd").write_text(root.as_posix(), encoding="utf-8")
            Path(ns._STATE_DIR, "env").write_text("", encoding="utf-8")
            for persistent in (False, True):
                env = ContextEnv(paths=Paths(ns), budget=BudgetController(0), persistent_bash=persistent)
                self.addCleanup(env.host_mirror.unlink, missing_ok=True)
                sandbox = LocalShellSandbox(root)
                p = shlex.quote(ns._CTX_FILE)
                py = shlex.quote(Path(sys.executable).as_posix())
                code = shlex.quote(f"import sys; sys.stdout.buffer.write(open({ns._CTX_FILE!r}, 'rb').read())")
                for command in (
                    f"cat {p}", f"P={p}; cat \"$P\"", "cd context && cat mirror.txt",
                    f"cat < {p}", f"cat {p} | cat", f"{py} -c {code}",
                    f"cat {p} >&2; exit 9",
                ):
                    with self.subTest(command=command):
                        Path(ns._STATE_DIR, "cwd").write_text(root.as_posix(), encoding="utf-8")
                        before = render_editable(self.messages)
                        obs = await env.step(command, self.messages, environment=sandbox)
                        self.assertIn(_ECHO_NOTE, obs.stdout_block)
                        self.assertNotIn(before, obs.stdout_block)
                        self.assertEqual(obs.result.return_code, 9 if "exit 9" in command else 0)

                other = root / "other.txt"
                Path(ns._STATE_DIR, "cwd").write_text(root.as_posix(), encoding="utf-8")
                other.write_text("ordinary file diagnostic", encoding="utf-8")
                obs = await env.step("cat other.txt", self.messages, environment=sandbox)
                self.assertIn("ordinary file diagnostic", obs.stdout_block)
                obs = await env.step(f"head -c 8 {p}", self.messages, environment=sandbox)
                self.assertIn("[[CTX_TU", obs.stdout_block)
                obs = await env.step(f"printf 'summary' | cat > {p}", self.messages, environment=sandbox)
                self.assertTrue(obs.ctx_changed)
                self.assertEqual(obs.result.stdout, "")
                self.messages[-1]["content"] = "original transcript payload"
                edit = shlex.quote(
                    f"p={ns._CTX_FILE!r}; s=open(p, encoding='utf-8').read(); "
                    "open(p, 'w', encoding='utf-8').write(s.replace('original transcript payload', 'summary'))"
                )
                obs = await env.step(f"{py} -c {edit}", self.messages, environment=sandbox)
                self.assertTrue(obs.ctx_changed)
                self.assertEqual(obs.result.stdout, "")
                original = deepcopy(self.messages)
                crlf_echo = shlex.quote(
                    f"import sys; p={ns._CTX_FILE!r}; b=open(p, 'rb').read(); "
                    "b=b.replace(b'\\n', b'\\r\\n'); open(p, 'wb').write(b); "
                    "sys.stdout.buffer.write(b)"
                )
                obs = await env.step(f"{py} -c {crlf_echo}", self.messages, environment=sandbox)
                self.assertFalse(obs.ctx_changed)
                self.assertEqual(self.messages, original)
                self.assertIn("\r\n", obs.result.stdout)
                self.assertIn(_ECHO_NOTE, obs.stdout_block)

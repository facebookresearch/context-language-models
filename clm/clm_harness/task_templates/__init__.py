# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared task-instruction (instance) templates, selected by name.

The instance prompt describes the TASK DOMAIN (what to do, workflow, rules), which is
independent of the context-management strategy, so only the system-prompt context
section differs across configurations. Pick a template with the
`task_template` agent kwarg (default ``terminal_agent_tasks``):

  * terminal_agent_tasks   — SWE-bench / terminal-bench style (solve-via-bash + workflow).
  * open_problems          — open-ended research tasks (e.g. DeepSearchQA).
  * edgebench              — EdgeBench tasks.
  * edgebench_sa           — EdgeBench tasks with parallel sub-agents (needs subagents=true).

Each ``<name>.yaml`` defines a single ``instance_template`` (with a ``{{task}}`` slot).
"""
from __future__ import annotations

from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_TEMPLATE = "terminal_agent_tasks"


def instance_template(name: str | None = None) -> str:
    """Return the ``instance_template`` string for task-template ``name`` (default
    ``terminal_agent_tasks``). Raises if the template file is missing."""
    name = name or DEFAULT_TASK_TEMPLATE
    path = _DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _DIR.glob("*.yaml"))
        raise ValueError(f"unknown task_template {name!r}; available: {available}")
    return yaml.safe_load(path.read_text())["instance_template"]

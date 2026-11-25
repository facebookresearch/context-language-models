#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run one Harbor trial of ClmAgent on a local Harbor task directory.
#
#   API_BASE=http://localhost:8000/v1 clm/examples/run_harbor.sh <task_dir> [config] [extra harbor args...]
#
#   task_dir   a Harbor task (task.toml, instruction.md, environment/, tests/)
#   config     bcp | edgebench | edgebench_sa | path to a config YAML (default: none,
#              class defaults)
#
# Environment:
#   API_BASE    OpenAI-compatible endpoint (required)
#   MODEL       litellm model name (default: openai/qwen36-27b)
#   HARBOR_ENV  Harbor environment type (default: docker; e.g. singularity)
#   TRIALS_DIR  where trial outputs go (default: ./trials)
#   BUDGET      context_budget_tokens when no config sets it (default: 32000)
#   COST_METRIC cost_metric when no config sets it (default: usd)
#   PYTHON      interpreter with pyyaml, used to expand the config (default: python3)
#   HARBOR      harbor executable (default: harbor)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,19p' "$0"; exit 2
fi
TASK_DIR=$(cd "$1" && pwd); shift
CONFIG=${1:-}; [[ $# -gt 0 ]] && shift

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${API_BASE:?set API_BASE to an OpenAI-compatible endpoint, e.g. http://localhost:8000/v1}"
MODEL=${MODEL:-openai/qwen36-27b}
HARBOR_ENV=${HARBOR_ENV:-docker}
TRIALS_DIR=${TRIALS_DIR:-$PWD/trials}
PYTHON=${PYTHON:-python3}
HARBOR=${HARBOR:-harbor}

case "$CONFIG" in
  "") CFG_FILE="" ;;
  bcp|edgebench|edgebench_sa) CFG_FILE="$REPO/clm/clm_harness/configs/$CONFIG.yaml" ;;
  *) CFG_FILE="$CONFIG" ;;
esac

KWARGS=(--agent-kwarg "api_base=$API_BASE")
if [[ -n "$CFG_FILE" ]]; then
  # One line per item: "env KEY=VALUE" or "kw key=value". Parsed into a variable first
  # so a parse failure stops the script instead of launching with defaults.
  if ! CFG_LINES=$("$PYTHON" - "$CFG_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
if not isinstance(cfg, dict):
    sys.exit(f"{sys.argv[1]}: expected a mapping at the top level")
def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
for k, v in (cfg.get("env") or {}).items():
    print(f"env {k}={fmt(v)}")
for k, v in (cfg.get("agent_kwargs") or {}).items():
    print(f"kw {k}={fmt(v)}")
PY
  ); then
    echo "run_harbor.sh: could not read config $CFG_FILE" >&2
    exit 1
  fi
  while IFS= read -r line; do
    [[ -z $line ]] && continue
    kind=${line%% *}; item=${line#* }
    if [[ $kind == env ]]; then export "$item"; else KWARGS+=(--agent-kwarg "$item"); fi
  done <<< "$CFG_LINES"
fi
if ! printf '%s\n' "${KWARGS[@]}" | grep -q '^context_budget_tokens='; then
  KWARGS+=(--agent-kwarg "context_budget_tokens=${BUDGET:-32000}")
fi
# Without a FLOPs model size, cost_metric=auto refuses to start; fall back to usd.
if ! printf '%s\n' "${KWARGS[@]}" | grep -q '^cost_metric='; then
  KWARGS+=(--agent-kwarg "cost_metric=${COST_METRIC:-usd}")
fi

export PYTHONPATH="$REPO/clm${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

exec "$HARBOR" trial start \
  -p "$TASK_DIR" \
  -e "$HARBOR_ENV" \
  -a clm_harness.clm_agent.harness:ClmAgent \
  -m "$MODEL" \
  "${KWARGS[@]}" \
  --trials-dir "$TRIALS_DIR" \
  "$@"

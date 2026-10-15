# clm_harness

The CLM agent for [Harbor](https://github.com/laude-institute/harbor). `ClmAgent` gives the
model one `bash` tool and lets it manage its own context by editing a file.

## How a turn works

1. **Mirror.** Before each command, the harness writes the model's editable context to
   `/tmp/.live_ctx/LIVE_CTX_MAIN.txt` in the sandbox. This is every turn after the system
   prompt and the task, one `[[CTX_TURN i role=…]]` block per message.
2. **Budget check.** The harness counts the tokens of the next request.
   - It adds a nudge when the context crosses 25%, 50% or 75% of the budget.
   - It adds an urgent nudge on every turn when the room left is too small for the next
     output (`persistent_nudge_ratio`).
   - Above the limit (budget minus reserve), it rolls back the newest turns and asks the
     model to condense (`max_num_retry_on_limit`).
3. **Command.** The model runs one bash command. To compact, it edits the mirror with
   ordinary tools such as `sed`, `python3` or `cat >`.
4. **Read back.** If the file changed, the harness parses it back into a valid message list.
   The system prompt and the task stay pinned. The edit gate decides whether to accept the
   edit, and the model gets a one-line receipt. A turn that only edits the mirror and prints
   nothing does not count as a task step.
5. **Finish.** Near the step or LM-call limit, the model is told to write its final
   output. The run ends on the submit command (`finish_policy: submit`), at the limits, or
   when the context is still over budget after the final turn.

The prompt asks the model to edit the mirror without printing its transcript back into
context. As a best-effort guard against duplicate observations, the harness replaces
complete, exact copies of the pre-command or post-command mirror with an omission note
in stdout and stderr, before observation truncation and the newest-output budget cap.
Uploads avoid host-platform newline translation; existing CR or CRLF characters in
message content are preserved. Post-command echoes are matched against raw downloaded
text, while edit reconciliation retains universal-newline conversion.
Only snapshots starting with a serialized `[[CTX_TURN i role=...]]` header and a nonempty
body qualify; empty or headerless edits such as `ok` are left alone. Raw execution
results and exit codes remain intact, so printing a transcript still costs a task step.

This guard matches content, not paths: an identical transcript printed from another
file is omitted too. It permits internal reads, read-modify-write edits, redirects and
ordinary file output. It does not deny filesystem access or provide a security boundary.
Partial, transformed, encoded, intermediate or older transcript copies, copies split
between stdout and stderr, and output truncated by a sandbox provider before reaching
the harness are outside its scope. Diagnostics around an exact copy are retained and
still subject to the usual output caps.

## Layout

| directory | contents |
|---|---|
| [`clm_agent/`](clm_agent/) | `ClmAgent`, the agent loop, and its system prompt |
| [`context_env/`](context_env/) | the context file: writing it to the sandbox, reading edits back, the edit gate |
| [`context_utils/`](context_utils/) | rendering the context to text and parsing an edited file back into messages |
| [`utils/`](utils/) | token budget (nudges, rollback-and-retry, final turn), finish policies, token counting, checkpoint and resume, skill mounting |
| [`configs/`](configs/) | configs used for zero-shot evaluation |
| [`task_templates/`](task_templates/) | how the task is presented to the model |
| [`agent_trajectory_format/`](agent_trajectory_format/) | ATIF-CTX, our extension of Harbor's [ATIF](https://docs.harborframework.com/core-concepts/agents/atif) that records intermediate context segments and multi-agent sub-trajectories |
| [`flops_metrics/`](flops_metrics/) | inference FLOPs with prefix-cache reuse |

## Outputs

Each trial's agent log directory holds these files:

- `trajectory.json`
- `usage.json` (steps, LM calls, edits, nudges, retries, cost)
- `timing.json`
- per-call `context_snapshots/`
- `trajectory.ctx.json`

## Install

```bash
git clone https://github.com/facebookresearch/context-language-models.git && cd context-language-models
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .            # harbor==0.16.1, litellm, tiktoken, pyyaml
```

## Run with Harbor

After `pip install -e .`, the `clm-harbor` command is the Harbor CLI with CLM available as
`-a clm-minimal`. All other arguments go to Harbor unchanged:

```bash
clm-harbor run -p path/to/task -a clm-minimal -m openai/qwen36-27b \
  --agent-kwarg api_base=http://localhost:8000/v1
clm-harbor trial start -p path/to/task -e docker -a clm-minimal -m openai/qwen36-27b \
  --agent-kwarg api_base=http://localhost:8000/v1 --clm-config bcp
```

- `--clm-config bcp|edgebench|<path.yaml>` expands a config into `--agent-kwarg` flags;
  explicit `--agent-kwarg` values take precedence.
- Unless given, `context_budget_tokens=32000` and `cost_metric=usd` are added.
- With plain `harbor`, the same agent is `-a clm_harness.clm_agent.harness:ClmAgent`.

### Example script and configs

[`clm/examples/run_harbor.sh`](../examples/run_harbor.sh) runs one trial on any Harbor task directory. It can also expand one
of the configs in [`configs/`](configs/) into `--agent-kwarg` flags:

```bash
export API_BASE=http://localhost:8000/v1   # OpenAI-compatible endpoint
export MODEL=openai/qwen36-27b             # litellm model name (openai/<served name>)
clm/examples/run_harbor.sh path/to/task              # class defaults, 32000-token budget
clm/examples/run_harbor.sh path/to/task bcp          # BrowseComp-Plus settings
clm/examples/run_harbor.sh path/to/task edgebench    # EdgeBench settings
HARBOR_ENV=singularity clm/examples/run_harbor.sh path/to/task bcp --trial-name t1
```

Arguments after the config are passed to `harbor trial start`. See the header of the script
for the other environment variables.

| config | used for | notable settings |
|---|---|---|
| `configs/bcp.yaml` | BrowseComp-Plus (deep research) | 28672-token budget, 500 task steps (100 and 2000 were also used), 60k-char observations, persistent 90% nudge, 6 rollback retries, shrink edit gate |
| `configs/edgebench.yaml` | EdgeBench (long-horizon improvement) | 32000-token budget, no step limit in practice, no finish command (`finish_policy: open_ended`), no persistent nudge, 50 retries, hourly checkpoints, EdgeBench finalize message |

A config is a YAML file with `agent_kwargs` (passed as `--agent-kwarg`) and `env` (exported
before Harbor starts).

### Key settings

| kwarg | default | what it does |
|---|---|---|
| `context_budget_tokens` | required | Token budget for the context. The enforced limit is the budget minus `context_budget_reserve_tokens` (2048). |
| `max_num_retry_on_limit` | 50 | Rollback-retry. When the context crosses the limit, the newest turns are rolled back until `retry_edit_margin_tokens` (2048) are free. The model is then told to compact, and a pinned note lists the commands whose output overflowed. Once N retries are used up, an overflow gives the model one final turn, and the run stops if the context is still over the limit on the next turn. `0` skips the retries. |
| `nudge_ratios` | `0.25,0.5,0.75` | Budget fractions at which the model gets an escalating reminder to compact. Each fires once and re-arms after the context shrinks. |
| `persistent_nudge_ratio` | `adaptive` | Urgent nudge, repeated on every turn near the limit. `adaptive` fires while the room left is below max(10% of the limit, `persistent_nudge_obs_mult` (default 2) × the largest of the last 3 tool outputs), and never below 50% of the limit. With small outputs it behaves like a 90% threshold; with outputs that are large relative to the budget it fires early enough to compact before a single output overflows. A number (e.g. `0.9`) uses a fixed fraction of the limit; `none` turns it off. |
| `max_steps` | 64 | Task-step budget. A turn that only edits the context (mirror changed, no output, exit 0) is free. |
| `lm_call_cap` | `2*max_steps+24` | Cap on the main agent's LM calls. Free context-editing turns still count here. `0` disables it. |
| `finalize_nudge_turns`, `finalize_message` | 3, none | In the last N turns, a notice tells the model to write its final output and submit. The count uses whichever runs out first, task steps or LM calls under the cap (`finalize_on_lm_call_cap`, default on). `finalize_message` is appended to the notice. |
| `allow_edit_growth` / `CLM_EDIT_GATE` | fit | Edit gate. `fit` accepts an edit if the result fits the limit. `shrink` accepts it only if the context gets smaller. Both take `fit` or `shrink`; the kwarg also takes `true` (fit) or `false` (shrink). A rejected edit is not applied, and the model is told why. The kwarg overrides the env var. |
| `finish_policy` | `submit` | `submit`: the model ends the run with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. `open_ended`: there is no finish command. |
| `task_template` | `terminal_agent_tasks` | How the task is presented: `terminal_agent_tasks`, `open_problems` (the instruction as is), or `edgebench`. |
| `temperature`, `top_p`, `max_tokens` | 0.7, 0.95, 16384 | Sampling. Pass `none` to omit `temperature` or `top_p`. |
| `cost_metric`, `flops_model_key` | auto | `usd` for priced API models. For local models, use `flops` with a model key (`9b` or `27b`) or `flops_n_body`. With `auto` (the default) the agent needs a FLOPs model size and refuses to start without one, whatever the model; pass `cost_metric=usd` to skip FLOPs accounting. The example script passes `usd` when no config sets it. |

### Serving

Any OpenAI-compatible server works; tool calling must be enabled. For Qwen with vLLM:

```bash
vllm serve Qwen/Qwen3.6-27B --served-model-name qwen36-27b --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

The server's context window must hold `context_budget_tokens + max_tokens`. For API models,
pass the litellm model name (e.g. `anthropic/...`) and leave `api_base` unset.

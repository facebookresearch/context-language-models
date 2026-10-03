# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Suffix Cache Reuse (SCR): KV reuse for context-editing agents on SGLang.

An agent that edits its own context (deleting or rewriting a span in the middle
of the transcript) changes the prompt mid-sequence. SGLang's radix cache matches
only the longest common prefix, so every token after the edit point is prefilled
again, even when most of it is unchanged from the previous turn.

SCR keeps those unchanged tokens. For each conversation it holds a copy of the
previous turn's KV in a side buffer, diffs the new prompt against the previous
one, and for up to K surviving blocks after the edit point:

  1. extends the new text in front of the block as a normal (chunked) prefill,
  2. loads the block's K/V from the side buffer into fresh pool slots,
     re-rotates the RoPE keys from their old positions to their new positions,
     and appends those slots to the request's prefix ("relocation"),
  3. continues with the next stretch of new text, and so on; the final extend
     always forwards at least KVREUSE_V6_MIN_EXTEND tokens before sampling.

Hybrid models with linear-attention layers (for example Qwen3.6, whose
GatedDeltaNet layers carry a recurrent state instead of KV) are handled with
KVREUSE_SSM_MODE=fork: the recurrent state saved at the end of the previous
prompt is restored into the request's state slot at the first relocation.

The module patches SGLang in place when imported with KVREUSE_ENABLED=1. It is
meant to be imported in every SGLang process (see suffix_cache_reuse._site),
and no SGLang source file is modified. With KVREUSE_ENABLED unset or 0 the
module defines its helpers and patches nothing.

Environment knobs (read once at import time):
  KVREUSE_ENABLED        1 installs the patches.
  KVREUSE_MAX_BLOCKS     K, the number of surviving blocks relocated per request.
  KVREUSE_SPLICE_RETRY   scheduling rounds a ready plan waits for SGLang's single
                         chunked-prefill slot.
  KVREUSE_SSM_MODE       fork | strict | none, recurrent-state handling.
  KVREUSE_V6_MIN_EXTEND  tokens of each block forwarded (not relocated) so the
                         model always runs on real tokens before the next step.
  KVREUSE_SIDE_SESSIONS  conversations held in the side buffer.
  KVREUSE_SIDE_TOKENS    tokens held per conversation in the side buffer.
  KVREUSE_MAX_SESSIONS   conversations tracked for session matching (LRU).
  KVREUSE_TRACE          1 logs one `kv6trace ev=req` line per request.
  KVREUSE_DEBUG          1 logs the per-request planning decisions.
suffix_cache_reuse.config.DEFAULTS holds the values used with this module.

Per-request log lines (prefix `[kvreuse]`), used by analysis/scr_report.py:
  kv6trace ev=req rid=<rid> sid=<session> ... n_new=<L> n_old=<L_prev> ...
  v6 splice OK: rid=<rid> sid=<session> ... C'=<relocated rows> ... blocks=<i>/<K>
Startup banner: `[kvreuse-site] ... max_blocks=<K> ... splice_retry=<N>`.
"""
import os
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ON = os.environ.get("KVREUSE_ENABLED") == "1"
_DEBUG = os.environ.get("KVREUSE_DEBUG") == "1"
_NO_REROT = os.environ.get("KVREUSE_NO_REROTATE") == "1"
# Two-phase relocation (extend the new text, then splice the surviving block).
_V6 = os.environ.get("KVREUSE_V6", "1") == "1"
# Control mode: keep only the leading run, relocate nothing.
_PIN_ONLY = os.environ.get("KVREUSE_PIN_ONLY") == "1"
# Smallest surviving block (in tokens) worth relocating.
_V6_MIN_TAIL = int(os.environ.get("KVREUSE_V6_MIN_TAIL", "64"))
# The last KVREUSE_V6_MIN_EXTEND tokens of every relocated block are forwarded
# instead of relocated, so the linear-attention layers always run on real tokens
# before the next relocated block and before the first sampled token.
_V6_MIN_EXTEND = int(os.environ.get("KVREUSE_V6_MIN_EXTEND", "16"))
# K: plan up to K of the largest surviving blocks after the edit point. Each is
# spliced at its own new position with its own RoPE shift; the new text between
# blocks is forwarded in between. K=1 relocates the single largest block.
_MAX_BLOCKS = int(os.environ.get("KVREUSE_MAX_BLOCKS", "3"))
# Optional floor (tokens) on the forwarded stretch between two relocated blocks.
# A block that would create a shorter stretch is left out of the plan. 0 = off.
_MB_MIN_GAP = int(os.environ.get("KVREUSE_MB_MIN_GAP", "0"))
# SGLang runs at most one chunked prefill per scheduler. When that slot is busy,
# keep the plan for up to N later scheduling rounds instead of dropping it.
_SPLICE_RETRY = int(os.environ.get("KVREUSE_SPLICE_RETRY", "0"))
# When the recurrent state is snapshotted: at the end of the prompt's prefill
# (prefill_end), or at request release (release).
_SSM_SNAP = os.environ.get("KVREUSE_SSM_SNAP", "prefill_end").lower()
# 1 serves the leading run from the side buffer too; by default the radix cache
# serves it (with its exact recurrent-state checkpoint).
_LEAD_FREEZE = os.environ.get("KVREUSE_LEAD_FREEZE", "0") == "1"
# Optional per-request switch, read from `kvreuse_session_id` when the request
# object carries it, else from a `cache_salt` of the form "kvreuse:<switch>":
#   "off"                          plain SGLang for this request
#   "knobs:maxblocks=3,ssm=none"   per-request overrides of the module defaults
def _req_switch(req):
    """Per-request switch string, or None."""
    sid = getattr(req, "kvreuse_session_id", None)
    if isinstance(sid, str) and sid:
        return sid
    ek = getattr(req, "extra_key", None)
    if isinstance(ek, str) and ek.startswith("kvreuse:"):
        return ek[len("kvreuse:"):]
    return None
def _req_knobs(req):
    k = getattr(req, "_kvreuse_knobs", None)
    if k is not None:
        return k
    sw = _req_switch(req)
    k = {}
    if isinstance(sw, str) and sw.startswith("knobs:"):
        for part in sw[6:].split(","):
            if "=" in part:
                a, b = part.split("=", 1); k[a.strip()] = b.strip()
    req._kvreuse_knobs = k
    return k
def _kb(req, name, default):
    """Boolean knob: per-request override, else the module default."""
    v = _req_knobs(req).get(name)
    return default if v is None else v in ("1", "true", "on", "yes")
def _ks(req, name, default):
    """String knob: per-request override, else the module default."""
    v = _req_knobs(req).get(name)
    return default if v is None else v
# Conversations tracked for session matching (LRU).
_MAX_SESS = int(os.environ.get("KVREUSE_MAX_SESSIONS", "256"))
# Pool-fraction budget reported in the init line.
_BUDGET_FRAC = float(os.environ.get("KVREUSE_BUDGET_FRAC", "0.30"))
_MIN_FREE_FRAC = float(os.environ.get("KVREUSE_MIN_FREE_FRAC", "0.25"))
_SNAP_MAX = int(os.environ.get("KVREUSE_SNAP_MAX", "12000"))
_ALLOC_MARGIN = int(os.environ.get("KVREUSE_ALLOC_MARGIN", "16384"))
# Allocation for relocated rows: when the pool is short, evict the shortfall
# from the radix tree and retry once, keeping this fraction of the pool
# reclaimable.
_EVICT_FLOOR_FRAC = float(os.environ.get("KVREUSE_EVICT_FLOOR_FRAC", "0.20"))
_EVICT_ON_FAIL = os.environ.get("KVREUSE_EVICT_ON_ALLOC_FAIL", "1") == "1"
_EVICT_MARGIN = int(os.environ.get("KVREUSE_EVICT_MARGIN", "2048"))

# Token diff. Long sequences are diffed at the level of content-defined chunks
# (about KVREUSE_CDC_TARGET tokens each) and the equal blocks are then extended
# token by token; every plan is verified token for token before use.
_FASTDIFF = os.environ.get("KVREUSE_FASTDIFF", "1") == "1"
_CDC_TARGET = int(os.environ.get("KVREUSE_CDC_TARGET", "64"))
_FASTDIFF_MIN = int(os.environ.get("KVREUSE_FASTDIFF_MIN", "2048"))
_CDC_EXTEND = os.environ.get("KVREUSE_CDC_EXTEND", "1") == "1"

# Runtime handles, populated on the first request.
_RT = {"alloc": None, "pool": None, "rotary": None, "layers": None,
       "sessions": None, "budget": None}
_STATS = {"held": 0, "frozen_turns": 0, "alloc_fail": 0, "sessions": 0,
          "tokens_frozen": 0, "tokens_prefilled": 0,
          "cold_turns": 0, "passthrough_turns": 0,
          "budget_skip": 0, "evicted_for_budget": 0,
          "too_long": 0, "low_free_skip": 0, "emergency_release": 0}


def _log(msg):
    print(f"[kvreuse] {msg}", flush=True)


# Optional profiling (KVREUSE_TIMING=1).
_TIMING = os.environ.get("KVREUSE_TIMING") == "1"
_TIMING_EVERY = int(os.environ.get("KVREUSE_TIMING_EVERY", "50"))
_TM = {}
_TC = {}


def _tick():
    if not _TIMING:
        return None
    import time
    return time.perf_counter()


def _tock(name, t0):
    if t0 is None:
        return
    import time
    dt = time.perf_counter() - t0
    _TM[name] = _TM.get(name, 0.0) + dt
    _TC[name] = _TC.get(name, 0) + 1


def _timing_report(tag):
    if not _TIMING:
        return
    n = max(1, _TC.get("intercept", 0))
    parts = []
    for k in sorted(_TM, key=lambda x: -_TM[x]):
        parts.append(f"{k}={_TM[k]*1000/n:.2f}ms/req(tot={_TM[k]:.1f}s,"
                     f"n={_TC[k]})")
    _log(f"TIMING[{tag}] over {n} intercepts: " + " ".join(parts))


# ---------------------------------------------------------------------------
# KV pool helpers
# ---------------------------------------------------------------------------
def _full_layer_ids(kv_pool, model_config=None):
    """Global ids of the full-attention layers (the layers that hold KV)."""
    m = getattr(kv_pool, "full_attention_layer_id_mapping", None)
    if m:
        return sorted(m.keys())
    for attr in ("full_attention_layer_ids", "full_layers", "full_attn_layers"):
        ids = getattr(kv_pool, attr, None)
        if ids:
            return sorted(int(x) for x in ids)
    for cfg in (getattr(model_config, "hf_text_config", None),
                getattr(model_config, "hf_config", None), model_config):
        lt = getattr(cfg, "layer_types", None)
        if lt:
            ids = [i for i, t in enumerate(lt) if "sliding" not in str(t)]
            if ids:
                return ids
    start = getattr(kv_pool, "start_layer", 0) or 0
    n = getattr(kv_pool, "layer_num", None)
    if n is None:
        n = getattr(getattr(kv_pool, "full_kv_pool", kv_pool), "layer_num", 0)
    return list(range(start, start + n))


def _copy_kv(kv_pool, src, dst):
    """Copy K and V for every full-attention layer, src slots -> dst slots."""
    for lid in _RT["layers"]:
        kv_pool.get_key_buffer(lid)[dst] = kv_pool.get_key_buffer(lid)[src]
        kv_pool.get_value_buffer(lid)[dst] = kv_pool.get_value_buffer(lid)[src]


# ---------------------------------------------------------------------------
# Side buffer: the previous turn's KV per conversation, held in its own
# allocation outside SGLang's paged KV pool, so it never competes with the
# scheduler for pool slots. Relocated rows are copied into ordinary per-request
# pool slots, which SGLang frees at release as usual.
# ---------------------------------------------------------------------------
_SIDE_SESSIONS = int(os.environ.get("KVREUSE_SIDE_SESSIONS", "8"))
_SIDE_TOKENS = int(os.environ.get("KVREUSE_SIDE_TOKENS", "34000"))
# Session matching (see _find_session): a request is attached to the tracked
# conversation with the longest common prefix, subject to these thresholds.
_MATCH_WINDOW = int(os.environ.get("KVREUSE_MATCH_WINDOW", "8192"))
_MATCH_MIN_TOKENS = int(os.environ.get("KVREUSE_MATCH_MIN_TOKENS", "1024"))
_MATCH_MIN_FRAC = float(os.environ.get("KVREUSE_MATCH_MIN_FRAC", "0.02"))
_MATCH_MIN_RATIO = float(os.environ.get("KVREUSE_MATCH_MIN_RATIO", "0.55"))
_MATCH_MIN_PREFIX_FRAC = float(os.environ.get("KVREUSE_MATCH_MIN_PREFIX_FRAC", "0.5"))
_MATCH_MIN_OVERLAP = float(os.environ.get("KVREUSE_MATCH_MIN_OVERLAP", "0.5"))
_MATCH_AMBIG_MARGIN = float(os.environ.get("KVREUSE_MATCH_AMBIG_MARGIN", "0.15"))
# Minimum gain over the radix match for the single-phase path (unless FORCE=1).
_MIN_EXTRA_TOKENS = int(os.environ.get("KVREUSE_MIN_EXTRA_TOKENS", "1024"))
# Re-check side-slot ownership and token identity right before every splice.
# KVREUSE_TRACE=1 logs session, slot and splice events per request.
_T6_GUARD = os.environ.get("KVREUSE_T6_GUARD", "1") == "1"
_TRACE = os.environ.get("KVREUSE_TRACE", "0") == "1"


def _tr(ev, **kw):
    if _TRACE:
        _log("kv6trace ev=" + ev + " " + " ".join(f"{k}={v}" for k, v in kw.items()))
_MIN_EXTRA_FRAC = float(os.environ.get("KVREUSE_MIN_EXTRA_FRAC", "0.15"))


def _side_init(pool):
    """Allocate per-layer K/V side storage for SIDE_SESSIONS x SIDE_TOKENS."""
    import torch
    cap = _SIDE_SESSIONS * _SIDE_TOKENS
    kbuf, vbuf = [], []
    for lid in _RT["layers"]:
        kb = pool.get_key_buffer(lid)
        vb = pool.get_value_buffer(lid)
        kbuf.append(torch.empty((cap, *kb.shape[1:]), dtype=kb.dtype, device=kb.device))
        vbuf.append(torch.empty((cap, *vb.shape[1:]), dtype=vb.dtype, device=vb.device))
    _RT["side_k"], _RT["side_v"] = kbuf, vbuf
    _RT["side_free"] = list(range(_SIDE_SESSIONS))
    per_tok = sum(x.element_size() * x.shape[1] * x.shape[2] for x in kbuf) * 2
    _log(f"side buffer: {_SIDE_SESSIONS} slots x {_SIDE_TOKENS} tokens "
         f"({cap * per_tok / 2**30:.2f} GiB, {per_tok / 1024:.1f} KiB/token)")


def _side_acquire(sid=None):
    f = _RT.get("side_free")
    idx = f.pop(0) if f else None
    if idx is not None:
        _RT.setdefault("side_owner", {})[idx] = sid
    _tr("side_acq", slot=idx, sid=sid, free=len(f or []))
    return idx


def _side_release(idx):
    if idx is not None:
        _tr("side_rel", slot=idx, sid=_RT.get("side_owner", {}).get(idx))
        _RT["side_free"].append(idx)
        _RT.setdefault("side_owner", {}).pop(idx, None)


def _side_store(slot, src, keep=0):
    """Copy pool slots into the side buffer, writing only rows [keep:n]."""
    n = min(int(src.numel()), _SIDE_TOKENS)
    keep = max(0, min(int(keep), n))
    if keep >= n:
        return n
    base = slot * _SIDE_TOKENS
    sub = src[keep:n]
    _t0 = _tick()
    for i, lid in enumerate(_RT["layers"]):
        _RT["side_k"][i][base + keep:base + n] = _RT["pool"].get_key_buffer(lid)[sub]
        _RT["side_v"][i][base + keep:base + n] = _RT["pool"].get_value_buffer(lid)[sub]
    _tock("side_store", _t0)
    _STATS["side_tokens_copied"] = _STATS.get("side_tokens_copied", 0) + (n - keep)
    _STATS["side_tokens_skipped"] = _STATS.get("side_tokens_skipped", 0) + keep
    return n


def _side_load(slot, rows, dst):
    """Copy selected side-buffer rows into freshly allocated pool slots."""
    base = slot * _SIDE_TOKENS
    idx = rows + base
    _t0 = _tick()
    for i, lid in enumerate(_RT["layers"]):
        _RT["pool"].get_key_buffer(lid)[dst] = _RT["side_k"][i][idx]
        _RT["pool"].get_value_buffer(lid)[dst] = _RT["side_v"][i][idx]
    _tock("side_load", _t0)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
_ALLOCED = {}


def _track_alloc(t, tag):
    _ALLOCED[id(t)] = (int(t.numel()), tag)


def _track_free(t):
    _ALLOCED.pop(id(t), None)


def _outstanding():
    return sum(v[0] for v in _ALLOCED.values()), len(_ALLOCED)


def _drop_private(sess):
    """Release a conversation's side-buffer copy; its next turn prefills normally."""
    _tr("drop_private", sid=sess.get("sid"), slot=sess.get("side_slot"),
        n_side=sess.get("n_side"), inflight=sess.get("inflight", 0))
    _side_release(sess.get("side_slot"))
    if sess.get("side_slot") is not None:
        _STATS["held"] -= int(sess.get("n_side", 0) or 0)
    sess["side_slot"] = None
    sess["n_side"] = 0
    sess["kv_slots"] = None
    sess["token_ids"] = []
    sess["positions"] = []


def _reclaim(need):
    budget, sessions = _RT["budget"], _RT["sessions"]
    if budget is None or sessions is None:
        return True
    if _STATS["held"] + need <= budget:
        return True
    for sid in [s for s in sessions]:
        if _STATS["held"] + need <= budget:
            break
        s_ = sessions[sid]
        if s_ is sess_guard[0] or s_.get("inflight", 0) > 0 or s_.get("side_slot") is None:
            continue
        _drop_private(s_)
        _STATS["evicted_for_budget"] += 1
    return _STATS["held"] + need <= budget


sess_guard = [None]


def _snapshot_private(sess, src):
    """Store this turn's committed KV in the conversation's side-buffer slot."""
    if _RT.get("side_k") is None:
        return
    slot = sess.get("side_slot")
    if slot is None:
        slot = _side_acquire(sess.get("sid"))
        if slot is None:
            _STATS["budget_skip"] += 1
            _drop_private(sess)
            return
        sess["side_slot"] = slot
    prev = int(sess.get("n_side", 0) or 0)
    keep = 0
    prev_ids = sess.get("side_ids")
    ids = sess.get("commit_ids")
    if prev_ids and ids:
        lim = min(len(prev_ids), len(ids), prev)
        while keep < lim and prev_ids[keep] == ids[keep]:
            keep += 1
    try:
        n = _side_store(slot, src, keep=keep)
        if ids:
            sess["side_ids"] = list(ids[:n])
    except Exception as e:
        _log(f"side store failed: {e}")
        _drop_private(sess)
        return
    sess["n_side"] = n
    _tr("snap", sid=sess.get("sid"), slot=slot, keep=keep, n=n, prev=prev)
    _STATS["held"] += n - prev
    if n < int(src.numel()):
        _STATS["budget_skip"] += 1
        _drop_private(sess)


def _evict_sessions(scheduler):
    """LRU-cap tracked conversations, skipping any with a request in flight."""
    sessions = scheduler._kvreuse_sessions
    if len(sessions) <= _MAX_SESS:
        _STATS["sessions"] = len(sessions)
        return
    for sid in [s for s in sessions]:
        if len(sessions) <= _MAX_SESS:
            break
        sess = sessions[sid]
        if sess.get("inflight", 0) > 0:
            continue
        _drop_private(sess)
        sessions.pop(sid, None)
    _STATS["sessions"] = len(sessions)


# ---------------------------------------------------------------------------
# SGLang patches (installed only with KVREUSE_ENABLED=1)
#
#   release_kv_cache                  snapshot the finished turn into the side buffer
#   process_batch_result_prefill      snapshot the recurrent state at prefill end
#   HybridReqToTokenPool.alloc,
#   ScheduleBatch.prepare_for_extend  restore the recurrent state into the new slot
#   Scheduler._add_request_to_queue   diff the new prompt, build the relocation plan
#   PrefillAdder.add_one_req,
#   PrefillAdder.add_chunked_req      keep a planned request chunked at each block
#   Req.init_next_round_input         run the plan: extend, splice, next block
#   invariant checker, cache_unfinished_req  bookkeeping
# ---------------------------------------------------------------------------
if _ON:
    try:
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker as _SIC,
        )
        _orig_report_leak = _SIC._report_leak

        def _drift_snapshot(token_msg):
            import re as _re
            g = {k: int(v) for k, v in _re.findall(r"(\w+)=(\d+)", token_msg or "")}
            tot = g.get("total"); av = g.get("available"); ev = g.get("evictable")
            if tot is None or av is None or ev is None:
                return None
            return tot - av - ev

        def _patched_report_leak(self, pool_name, token_msg):
            drift = _drift_snapshot(token_msg)
            _STATS["drift_last"] = drift if drift is not None else -1
            if drift is not None:
                first = _STATS.setdefault("drift_first", drift)
                _STATS["drift_max"] = max(_STATS.get("drift_max", drift), drift)
                _log(f"DRIFT: {drift} slots unaccounted "
                     f"(first={first} max={_STATS['drift_max']}) after "
                     f"{_STATS.get('freeze_alloc_n', 0)} freeze allocs / "
                     f"{_STATS.get('freeze_alloc_slots', 0)} slots | "
                     f"side_held={_STATS['held']}")
            held = _STATS["held"]
            if held > 0:
                if _DEBUG:
                    _log(f"pool check: {held} slots held privately by kvreuse sessions")
                return
            return _orig_report_leak(self, pool_name, token_msg)

        _SIC._report_leak = _patched_report_leak

        from sglang.srt.mem_cache import common as _cache_common
        _orig_release = _cache_common.release_kv_cache

        def _patched_release(req, tree_cache, is_insert=True):
            sess = getattr(req, "_kvreuse_session", None)
            if sess is not None:
                _rt0 = _tick()
                try:
                    if req.req_pool_idx is not None:
                        kv_len = req.effective_kv_committed_len()
                        src = tree_cache.req_to_token_pool.req_to_token[
                            req.req_pool_idx, :kv_len
                        ]
                        _STATS["rel_seen"] = _STATS.get("rel_seen", 0) + 1
                        if _STATS["rel_seen"] % 25 == 1:
                            try:
                                cpl = int(getattr(req, "cache_protected_len", 0) or 0)
                                _log(f"acct: kv_len={kv_len} cpl={cpl} "
                                     f"will_free={max(0, kv_len - cpl)} "
                                     f"held={_STATS['held']} "
                                     f"avail={_RT['alloc'].available_size()} "
                                     f"evictable={getattr(tree_cache,'evictable_size',lambda:-1)()} "
                                     f"frozen={getattr(req,'_kvreuse_prefix_set',False)}")
                            except Exception:
                                pass
                        _ct0 = _tick()
                        try:
                            sess["commit_ids"] = list(
                                req.origin_input_ids + req.output_ids)[:kv_len]
                        except Exception:
                            sess["commit_ids"] = None
                        _tock("commit_ids", _ct0)
                        _sp0 = _tick()
                        _snapshot_private(sess, src)
                        _tock("snapshot_private", _sp0)
                        if _SSM_MODE != "strict" and (_SSM_SNAP == "release" or _PIN_ONLY or _LEAD_FREEZE):
                            _ssm_save(sess, req, kv_len)
                        elif not getattr(req, "_kvreuse_ssm_saved", False):
                            # Finished before the prefill-end snapshot (e.g. EOS as the
                            # first token): the saved state is from an earlier turn.
                            sess["ssm_tokens"] = -1
                            _STATS["ssm_stale_dropped"] = _STATS.get("ssm_stale_dropped", 0) + 1
                            _log(f"ssm snapshot dropped: rid={getattr(req, 'rid', '?')} "
                                 f"sid={sess.get('sid')} finished before prefill end")
                except Exception as e:
                    _log(f"snapshot failed: {e}")
                    _drop_private(sess)
                finally:
                    sess["inflight"] = max(0, sess.get("inflight", 1) - 1)
                    req._kvreuse_session = None
                    _tock("release_hook", _rt0)
            return _orig_release(req, tree_cache, is_insert)

        _cache_common.release_kv_cache = _patched_release

        try:
            from sglang.srt.managers.scheduler_components.batch_result_processor \
                import BatchResultProcessor as _BRP
            _orig_prefill_done = _BRP.process_batch_result_prefill

            def _patched_prefill_done(self, batch, result, *a, **kw):
                out = _orig_prefill_done(self, batch, result, *a, **kw)
                try:
                    for _r in getattr(batch, "reqs", []) or []:
                        _sess = getattr(_r, "_kvreuse_session", None)
                        if _sess is not None and getattr(_r, "_kvreuse_v6", None) is None:
                            _ssm_save(_sess, _r, _v6_full_len(_r))
                except Exception as _e:
                    _log(f"prefill-end ssm snapshot failed: {_e}")
                return out

            _BRP.process_batch_result_prefill = _patched_prefill_done
        except Exception:
            pass

        try:
            from sglang.srt.managers.scheduler_components import batch_result_processor as _brpm
            _SBRP = getattr(_brpm, "SchedulerBatchResultProcessor", None)
            if _SBRP is not None and not getattr(_SBRP, "_kvreuse_pe_wrapped", False):
                _orig_sbrp_prefill = _SBRP.process_batch_result_prefill

                def _sbrp_prefill(self, batch, result, *a, **kw):
                    out = _orig_sbrp_prefill(self, batch, result, *a, **kw)
                    try:
                        for _r in getattr(batch, "reqs", []) or []:
                            _sess = getattr(_r, "_kvreuse_session", None)
                            if _sess is None:
                                continue
                            if int(getattr(_r, "inflight_middle_chunks", 0) or 0) > 0:
                                continue
                            if getattr(_r, "_kvreuse_v6", None) is not None:
                                _STATS["ssm_save_skipped_midplan"] = _STATS.get("ssm_save_skipped_midplan", 0) + 1
                                if _DEBUG:
                                    _log(f"kv6 rid={getattr(_r,'rid','?')} prefill-end snapshot SKIPPED (v6 phase {_r._kvreuse_v6.get('phase')} in flight)")
                                continue
                            _n = _v6_full_len(_r) if getattr(_r, "extend_range", None) is None else int(_r.extend_range.end)
                            if _sess.get("side_slot") is None and _RT.get("side_k") is not None and _RT.get("ssm"):
                                _slot = _side_acquire(_sess.get("sid"))
                                if _slot is not None:
                                    _sess["side_slot"] = _slot
                                    _STATS["side_slot_at_prefill"] = _STATS.get("side_slot_at_prefill", 0) + 1
                                    if _DEBUG:
                                        _log(f"kv6 rid={getattr(_r,'rid','?')} side slot {_slot} acquired at prefill end (cold session)")
                            _ssm_save(_sess, _r, _n)
                            if _DEBUG:
                                _log(f"kv6 rid={getattr(_r,'rid','?')} prefill-end ssm snapshot saved n={_n} slot={_sess.get('side_slot')}")
                    except Exception as _e:
                        _log(f"prefill-end ssm snapshot (installed hook) failed: {_e}")
                    return out

                _SBRP.process_batch_result_prefill = _sbrp_prefill
                _SBRP._kvreuse_pe_wrapped = True
                print("[kvreuse] v6.3: prefill-end snapshot hook attached to SchedulerBatchResultProcessor", flush=True)
        except Exception as _e:
            print(f"[kvreuse] v6.3: could not attach the installed prefill-end hook: {_e}", flush=True)

        try:
            from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
            _orig_hyb_alloc = HybridReqToTokenPool.alloc

            def _patched_hyb_alloc(self, reqs):
                out = _orig_hyb_alloc(self, reqs)
                if out is not None:
                    for _r in reqs:
                        if getattr(_r, "_kvreuse_ssm_shadow", None) is not None:
                            _ssm_restore(_r)
                return out

            HybridReqToTokenPool.alloc = _patched_hyb_alloc
        except Exception as _e:
            print(f"[kvreuse] could not patch HybridReqToTokenPool.alloc: {_e}",
                  flush=True)

        try:
            from sglang.srt.managers.schedule_batch import ScheduleBatch as _SB
            _orig_pfe = _SB.prepare_for_extend

            def _patched_pfe(self, *a, **kw):
                out = _orig_pfe(self, *a, **kw)
                try:
                    _rs = getattr(self, "reqs", []) or []
                    if _DEBUG:
                        _log("pfe: n=%d armed=%d idx=%s" % (
                            len(_rs),
                            sum(1 for _r in _rs
                                if getattr(_r, "_kvreuse_ssm_shadow", None) is not None),
                            [getattr(_r, "mamba_pool_idx", None) for _r in _rs][:3]))
                    for _r in _rs:
                        if getattr(_r, "_kvreuse_ssm_shadow", None) is not None:
                            _ssm_restore(_r)
                except Exception as _e:
                    _log(f"prepare_for_extend restore failed: {_e}")
                return out

            _SB.prepare_for_extend = _patched_pfe
        except Exception as _e:
            print(f"[kvreuse] could not patch prepare_for_extend: {_e}", flush=True)

        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.schedule_policy import SchedulePolicy
        from sglang.srt.managers.schedule_batch import Req

        _orig_add = Scheduler._add_request_to_queue

        def _patched_add(self, req, is_retracted=False):
            if not is_retracted:
                _t0 = _tick()
                try:
                    _kvreuse_intercept(self, req)
                except Exception as e:
                    _log(f"intercept failed, falling back to re-prefill: {e}")
                _tock("intercept", _t0)
                if _TIMING and _TC.get("intercept", 0) % _TIMING_EVERY == 0:
                    _timing_report("add")
            return _orig_add(self, req, is_retracted)

        Scheduler._add_request_to_queue = _patched_add

        from sglang.srt.managers.schedule_policy import PrefillAdder as _PA
        from sglang.srt.managers.schedule_policy import AddReqResult as _AddReqResult
        _orig_pa_add_one = _PA.add_one_req
        _orig_pa_add_chunked = _PA.add_chunked_req

        def _pa_add_one(self, req, has_chunked_req, *a, **kw):
            if getattr(self, "_kvreuse_forced_chunk", False):
                _STATS["v6_admit_blocked"] = _STATS.get("v6_admit_blocked", 0) + 1
                return _AddReqResult.OTHER
            v6 = getattr(req, "_kvreuse_v6", None)
            if v6 is not None and v6.get("phase") == 1 and (has_chunked_req or self.new_chunked_req is not None):
                _v6_abort_phase1(req, "another request is already chunked")
            res = _orig_pa_add_one(self, req, has_chunked_req, *a, **kw)
            v6 = getattr(req, "_kvreuse_v6", None)
            if v6 is not None and v6.get("phase") == 1 and req in self.can_run_list:
                end_here = _v6_end(req)
                if end_here > v6["bprime_end"]:
                    _v6_abort_phase1(req, f"admitted extend ends at {end_here} > |A|+|B'| {v6['bprime_end']}", restore_lengths=False)
                elif self.new_chunked_req is None or self.new_chunked_req is req:
                    self.new_chunked_req = req
                    self._kvreuse_forced_chunk = True
                    _STATS["v6_forced_chunk"] = _STATS.get("v6_forced_chunk", 0) + 1
                    if end_here == v6["bprime_end"]:
                        v6["phase"] = 2
                    if _DEBUG:
                        _log(f"v6 adder: forced chunk; prefix={len(req.prefix_indices)} end_here={end_here} "
                             f"full={_v6_full_len(req)} res={res} phase={v6['phase']}")
                else:
                    _v6_abort_phase1(req, "another request became the chunked request", restore_lengths=False)
            return res

        def _pa_add_chunked(self, req):
            v6 = getattr(req, "_kvreuse_v6", None)
            if v6 is not None and v6.get("phase") == 1:
                end = v6["bprime_end"]; pl = len(req.prefix_indices)
                if pl < end:
                    if not hasattr(req, "full_untruncated_fill_ids"):
                        req.fill_ids = req.fill_ids[:end]
                        req.extend_input_len = end - pl
                    _up = _orig_pa_add_chunked(self, req)
                    if _v6_end(req) >= end:
                        v6["phase"] = 2
                    self._kvreuse_forced_chunk = True
                    if _DEBUG:
                        _log(f"v6 invariant: force-keep chunked rid={getattr(req,'rid','?')} "
                             f"upstream_returned={'req' if _up is not None else 'None'} "
                             f"prefix={len(req.prefix_indices)} end={end} "
                             f"new_chunked_req={'set' if self.new_chunked_req is not None else 'None'} "
                             f"-> admitting nothing else this round")
                    return req
                v6["phase"] = 2
            return _orig_pa_add_chunked(self, req)

        _PA.add_one_req = _pa_add_one
        _PA.add_chunked_req = _pa_add_chunked

        try:
            from sglang.srt.mem_cache import mamba_radix_cache as _mrc, radix_cache as _rc
            for _cls in (getattr(_mrc, "MambaRadixCache", None), getattr(_rc, "RadixCache", None)):
                if _cls is None or getattr(_cls, "_kvreuse_cur_wrapped", False):
                    continue
                _orig_cur = _cls.cache_unfinished_req

                def _p_cur(self, req, *a, _o=_orig_cur, **kw):
                    watch = _DEBUG and (getattr(req, "_kvreuse_v6", None) is not None or getattr(req, "_kvreuse_prefix_set", False))
                    if watch:
                        _log(f"cache_unfinished_req IN: fill={_v6_full_len(req)} prefix={len(req.prefix_indices)} "
                             f"is_chunked={getattr(req,'is_chunked',None)} chunked_kw={kw.get('chunked', a[0] if a else None)} "
                             f"v6={(getattr(req,'_kvreuse_v6',None) or {}).get('phase')}")
                    out = _o(self, req, *a, **kw)
                    if watch:
                        _log(f"cache_unfinished_req OUT: fill={_v6_full_len(req)} prefix={len(req.prefix_indices)}")
                    return out

                _cls.cache_unfinished_req = _p_cur
                _cls._kvreuse_cur_wrapped = True
        except Exception as _e:
            print(f"[kvreuse] could not wrap cache_unfinished_req: {_e}", flush=True)
        try:
            from sglang.srt.managers import scheduler as _schedmod
            if hasattr(_schedmod, "maybe_cache_unfinished_req") and not getattr(_schedmod, "_kvreuse_mcur_wrapped", False):
                _orig_mcur = _schedmod.maybe_cache_unfinished_req

                def _p_mcur(req, tree_cache, **kw):
                    watch = _DEBUG and getattr(req, "_kvreuse_v6", None) is not None
                    if watch:
                        _log(f"stash IN: full={_v6_full_len(req)} range={getattr(req,'extend_range',None)} prefix={len(req.prefix_indices)} phase={(req._kvreuse_v6 or {}).get('phase')}")
                    out = _orig_mcur(req, tree_cache, **kw)
                    if watch:
                        _log(f"stash OUT: prefix={len(req.prefix_indices)}")
                    return out

                _schedmod.maybe_cache_unfinished_req = _p_mcur
                _schedmod._kvreuse_mcur_wrapped = True
        except Exception as _e:
            print(f"[kvreuse] could not wrap maybe_cache_unfinished_req: {_e}", flush=True)


        _orig_init_round = Req.init_next_round_input

        def _patched_init_round(self, tree_cache=None, cow_mamba=None):
            v6 = getattr(self, "_kvreuse_v6", None)
            if v6 is not None and v6.get("phase") == 2:
                if _DEBUG:
                    _log(f"v6 phase2 entry: full={_v6_full_len(self)} prefix={len(self.prefix_indices)} "
                         f"is_chunked={getattr(self,'is_chunked',None)} tree_cache={'set' if tree_cache is not None else 'None'} "
                         f"out={len(self.output_ids)}")
                out = _orig_init_round(self, tree_cache, cow_mamba)
                self._kvreuse_v6 = None
                _v6_splice(self, v6)
                return out
            if v6 is not None and v6.get("phase") == 1:
                out = _orig_init_round(self, tree_cache, cow_mamba)
                pl = len(self.prefix_indices)
                end = v6["bprime_end"]
                if pl < end:
                    from array import array as _arr6
                    _cut = _arr6("q", v6["new_ids"][:end])
                    if hasattr(self, "full_untruncated_fill_ids"):
                        self.full_untruncated_fill_ids = _cut
                        if hasattr(self, "set_extend_range"):
                            self.set_extend_range(pl, end)
                    else:
                        self.fill_ids = _cut; self.extend_input_len = end - pl
                    _STATS["v6_reentry_kept"] = _STATS.get("v6_reentry_kept", 0) + 1
                    if _DEBUG:
                        _sch = _RT.get("scheduler")
                        _log(f"kv6 rid={getattr(self,'rid','?')} phase1 re-entry ({'chunked path' if tree_cache is None else 'waiting queue'}): prefix={pl} end={end} -> chunk continues "
                             f"| invariant: scheduler.chunked_req={'self' if getattr(_sch,'chunked_req',None) is self else ('other' if getattr(_sch,'chunked_req',None) is not None else 'None')}")
                    return out
                if pl == end:
                    v6["phase"] = 2
                    _STATS["v6_reentry_phase2"] = _STATS.get("v6_reentry_phase2", 0) + 1
                    self._kvreuse_v6 = None
                    _v6_splice(self, v6)
                    return out
                _v6_abort_phase1(self, f"re-entered with prefix {pl} beyond |A|+|B'| {end}", restore_lengths=False)
                self.extend_input_len = _v6_full_len(self) - len(self.prefix_indices)
                return out
            out = _orig_init_round(self, tree_cache, cow_mamba)
            plan = getattr(self, "_kvreuse_plan", None)
            if plan is None:
                return out
            _pin = _kb(self, "pin", _PIN_ONLY); _v6 = _kb(self, "v6", _V6); _leadfz = _kb(self, "lead", _LEAD_FREEZE)
            if _pin and plan.get("lead_run", 0) < len(plan["rows"]):
                _lead = plan.get("lead_run", 0)
                plan["rows"], plan["old_pos"], plan["new_pos"] = plan["rows"][:_lead], plan["old_pos"][:_lead], plan["new_pos"][:_lead]
                plan["suffix_only"] = True; plan["tail"] = None
                _STATS["pin_only_trunc"] = _STATS.get("pin_only_trunc", 0) + 1
                if _lead == 0:
                    self._kvreuse_plan = None
                    return out
            if _v6 and not _pin and not _leadfz:
                self._kvreuse_plan = None
                tail = plan.get("tail")
                radix_n0 = len(self.prefix_indices) if self.prefix_indices is not None else 0
                lead = plan.get("lead_run", 0)
                if _DEBUG:
                    _log(f"kv6 rid={getattr(self,'rid','?')} decide: radix_n0={radix_n0} lead={lead} rows={len(plan['rows'])} "
                         f"tail={'none' if tail is None else (tail['n_c'], tail['bprime_end'], tail['lead_new_end'])} "
                         f"chunked_req={'set' if getattr(_RT.get('scheduler'),'chunked_req',None) is not None else 'None'} "
                         f"ssm_tokens={plan.get('ssm_tokens')} side_slot={plan.get('side_slot')} live={len(plan['new_ids'])}")
                if tail is None:
                    _STATS["v61_lead_only_deferred"] = _STATS.get("v61_lead_only_deferred", 0) + 1
                    return out
                sch = _RT.get("scheduler")
                if sch is None or getattr(sch, "chunked_req", None) is not None or radix_n0 > tail["lead_new_end"]:
                    _STATS["v61_tail_deferred"] = _STATS.get("v61_tail_deferred", 0) + 1
                    if _SPLICE_RETRY > 0 and sch is not None and getattr(sch, "chunked_req", None) is not None:
                        _nr = int(getattr(self, "_kvreuse_retry", 0) or 0)
                        if _nr < _SPLICE_RETRY:
                            self._kvreuse_retry = _nr + 1
                            self._kvreuse_plan = plan
                            _STATS["v61_tail_retry"] = _STATS.get("v61_tail_retry", 0) + 1
                            _log(f"kv6 rid={getattr(self,'rid','?')} tail DEFERRED-RETRY {_nr + 1}/{_SPLICE_RETRY} (chunked slot busy)")
                    if _DEBUG:
                        _log(f"kv6 rid={getattr(self,'rid','?')} tail DEFERRED (sch={sch is not None}, chunked={getattr(sch,'chunked_req',None) is not None}, radix_n0={radix_n0} > lead_new_end={tail['lead_new_end']})")
                    return out
                if not _ssm_ready(plan, self):
                    _STATS["ssm_gated"] = _STATS.get("ssm_gated", 0) + 1
                    if _DEBUG:
                        _log(f"kv6 rid={getattr(self,'rid','?')} SSM GATE declined (ssm_tokens={plan.get('ssm_tokens')} side_slot={plan.get('side_slot')} mode={_ssm_mode_for(self)})")
                    return out
                bprime = tail["bprime_end"] - radix_n0
                if bprime >= 1:
                    from array import array as _arr6
                    _cut = _arr6("q", plan["new_ids"][:tail["bprime_end"]])
                    if hasattr(self, "full_untruncated_fill_ids"):
                        self.full_untruncated_fill_ids = _cut
                        if hasattr(self, "set_extend_range"):
                            self.set_extend_range(radix_n0, tail["bprime_end"])
                    else:
                        self.fill_ids = _cut; self.extend_input_len = bprime
                    self._kvreuse_v6 = {"phase": 1, "rows": tail["rows"], "old_pos": tail["old_pos"], "bprime_end": tail["bprime_end"],
                                        "new_ids": plan["new_ids"], "side_slot": plan.get("side_slot"), "ssm_restore": bool(plan.get("ssm_restore")),
                                        "n_c": tail["n_c"], "radix_a": True, "radix_n0": radix_n0,
                                        "more": list(tail.get("more") or []),
                                        "bi": 1, "bn": 1 + len(tail.get("more") or [])}
                    _log(f"kv6 rid={getattr(self,'rid','?')} phase1 radix_a={radix_n0} chunk1={bprime} tail={tail['n_c']} total={len(plan['new_ids'])} "
                         f"ssm_restore={bool(plan.get('ssm_restore'))} ssm_mode={_ssm_mode_for(self)} ssm_na={_RT.get('ssm_na')} ssm_obj={_RT.get('ssm') is not None}")
                    self._kvreuse_ssm_shadow = None
                    _STATS["v6_phase1"] = _STATS.get("v6_phase1", 0) + 1
                    _STATS["frozen_turns"] += 1
                    if _DEBUG or _STATS["v6_phase1"] % 25 == 1:
                        _log(f"v6.1 phase1 (radix A={radix_n0}, lead={lead}): B'={bprime} C={tail['n_c']} live={len(plan['new_ids'])} [n={_STATS['v6_phase1']}]")
                    return out
                v6 = {"phase": 2, "rows": tail["rows"], "old_pos": tail["old_pos"], "bprime_end": tail["bprime_end"], "new_ids": plan["new_ids"],
                      "side_slot": plan.get("side_slot"), "ssm_restore": bool(plan.get("ssm_restore")), "n_c": tail["n_c"], "radix_a": True,
                      "more": list(tail.get("more") or []), "bi": 1, "bn": 1 + len(tail.get("more") or [])}
                if _v6_splice(self, v6):
                    _STATS["v61_direct_splice"] = _STATS.get("v61_direct_splice", 0) + 1
                    _STATS["frozen_turns"] += 1
                return out
            self._kvreuse_plan = None
            radix_n = len(self.prefix_indices) if self.prefix_indices is not None else 0
            gain = len(plan["rows"]) - radix_n
            _STATS["cmp_seen"] = _STATS.get("cmp_seen", 0) + 1
            if _STATS["cmp_seen"] % 20 == 1:
                _log(f"cmp: radix={radix_n} rows={len(plan['rows'])} "
                     f"kept={plan.get('kept')} n_side={plan.get('n_side')} "
                     f"len_pos={plan.get('len_pos')} old_len={plan.get('old_len')} "
                     f"live={len(plan['new_ids'])}")
            _force = _kb(self, "force", _FORCE)
            _min_extra = int(_ks(self, "min_extra", _MIN_EXTRA_TOKENS) or _MIN_EXTRA_TOKENS)
            if (not _force) and gain < max(_min_extra,
                                           int(_MIN_EXTRA_FRAC * len(plan["new_ids"]))):
                _STATS["radix_better"] = _STATS.get("radix_better", 0) + 1
                _thr = max(_MIN_EXTRA_TOKENS,
                           int(_MIN_EXTRA_FRAC * len(plan["new_ids"])))
                _r = (gain / _thr) if _thr > 0 else 0.0
                _b = ("0-25%" if _r < .25 else "25-50%" if _r < .5 else
                      "50-75%" if _r < .75 else "75-100%")
                _h = _STATS.setdefault("defer_hist", {})
                _h[_b] = _h.get(_b, 0) + 1
                _STATS["defer_gain_sum"] = _STATS.get("defer_gain_sum", 0) + max(gain, 0)
                if _STATS["radix_better"] % 250 == 1:
                    _log(f"defer headroom: {_h} | mean_gain="
                         f"{_STATS['defer_gain_sum'] / max(_STATS['radix_better'],1):.0f} "
                         f"thr~{_thr}")
                if _STATS["radix_better"] % 100 == 1:
                    _log(f"radix match wins (radix={radix_n} frozen={len(plan['rows'])}); "
                         f"keeping radix [deferred={_STATS['radix_better']}]")
                return out
            if _SUFFIX_GATE and not plan.get("suffix_only", False):
                _lead = plan.get("lead_run", 0)
                if _lead > 0:
                    plan["rows"] = plan["rows"][:_lead]
                    plan["old_pos"] = plan["old_pos"][:_lead]
                    plan["new_pos"] = plan["new_pos"][:_lead]
                    plan["suffix_only"] = True
                    _STATS["lead_trunc"] = _STATS.get("lead_trunc", 0) + 1
            if _SUFFIX_GATE and not plan.get("suffix_only", False):
                _STATS["nonsuffix_gated"] = _STATS.get("nonsuffix_gated", 0) + 1
                if _STATS["nonsuffix_gated"] % 100 == 1:
                    _log(f"suffix gate: declined mid-sequence insert "
                         f"(rows={len(plan['rows'])}) "
                         f"[gated={_STATS['nonsuffix_gated']}]")
                return out
            if not _ssm_ready(plan, self):
                _STATS["ssm_gated"] = _STATS.get("ssm_gated", 0) + 1
                if _STATS["ssm_gated"] % 100 == 1:
                    _log(f"ssm gate: declined freeze (rows={len(plan['rows'])} "
                         f"ssm_tokens={plan.get('ssm_tokens')}) "
                         f"[gated={_STATS['ssm_gated']}]")
                return out
            if _materialise(self, plan, tree_cache):
                tail = plan.get("tail")
                sch = _RT.get("scheduler")
                if (tail is not None and _v6 and not _pin and sch is not None
                        and getattr(sch, "chunked_req", None) is None
                        and len(self.prefix_indices) == tail["lead_new_end"]
                        and tail["bprime_end"] - tail["lead_new_end"] >= 1):
                    from array import array as _arr6
                    _cut = _arr6("q", plan["new_ids"][:tail["bprime_end"]])
                    if hasattr(self, "full_untruncated_fill_ids"):
                        self.full_untruncated_fill_ids = _cut
                        if hasattr(self, "set_extend_range"):
                            self.set_extend_range(len(self.prefix_indices), tail["bprime_end"])
                    else:
                        self.fill_ids = _cut
                        self.extend_input_len = tail["bprime_end"] - len(self.prefix_indices)
                    self._kvreuse_v6 = {"phase": 1, "rows": tail["rows"], "old_pos": tail["old_pos"],
                                        "bprime_end": tail["bprime_end"], "new_ids": plan["new_ids"],
                                        "side_slot": plan.get("side_slot"), "ssm_restore": bool(plan.get("ssm_restore")),
                                        "n_c": tail["n_c"]}
                    self._kvreuse_ssm_shadow = None
                    _STATS["v6_phase1"] = _STATS.get("v6_phase1", 0) + 1
                    if _DEBUG or _STATS["v6_phase1"] % 25 == 1:
                        _log(f"v6 phase1: A={len(self.prefix_indices)} B'={tail['bprime_end']-len(self.prefix_indices)} "
                             f"C={tail['n_c']} live={len(plan['new_ids'])} full={len(getattr(self,'full_untruncated_fill_ids',[]))} "
                             f"range={getattr(self,'extend_range',None)} [n={_STATS['v6_phase1']}]")
                elif plan.get("ssm_restore"):
                    self._kvreuse_ssm_shadow = plan.get("side_slot")
                    if _DEBUG:
                        _log(f"arm: rid={id(self)} idx={getattr(self,'mamba_pool_idx',None)} "
                             f"slot={plan.get('side_slot')}")
                    _ssm_restore(self)
                _STATS["frozen_turns"] += 1
                _STATS["tokens_frozen"] += len(plan["rows"])
                _STATS["tokens_prefilled"] += len(plan["new_ids"]) - len(plan["rows"])
                if _STATS["frozen_turns"] % 25 == 1:
                    tf, tp = _STATS["tokens_frozen"], _STATS["tokens_prefilled"]
                    _log(f"cum: frozen={_STATS['frozen_turns']} "
                         f"deferred={_STATS.get('radix_better', 0)} "
                         f"saved_vs_radix={_STATS.get('gain_tokens', 0)} "
                         f"tokens_frozen={tf} tokens_prefilled={tp} "
                         f"ssm_saved={_STATS.get('ssm_saved', 0)} "
                         f"ssm_restored={_STATS.get('ssm_restored', 0)} "
                         f"ssm_gated={_STATS.get('ssm_gated', 0)} "
                         f"nonsuffix={_STATS.get('nonsuffix_gated', 0)} "
                         f"ambig={_STATS.get('match_ambiguous', 0)} "
                         f"r_noslot={_STATS.get('ssm_r_noslot', 0)} "
                         f"r_noidx={_STATS.get('ssm_r_noidx', 0)} "
                         f"weak_match={_STATS.get('match_weak', 0)} "
                         f"swa_short={_STATS.get('swa_too_short', 0)} "
                         f"lead_trunc={_STATS.get('lead_trunc', 0)} "
                         f"v6_plans={_STATS.get('v6_tail_plans', 0)} v6_p1={_STATS.get('v6_phase1', 0)} "
                         f"v6_spliced={_STATS.get('v6_spliced', 0)} v6_tok={_STATS.get('v6_spliced_tokens', 0)} "
                         f"v6_abort={_STATS.get('v6_abort', 0)} v6_mismatch={_STATS.get('v6_phase2_mismatch', 0)} "
                         f"pin_only={_STATS.get('pin_only_trunc', 0)} "
                         f"v61_lead_def={_STATS.get('v61_lead_only_deferred', 0)} v61_tail_def={_STATS.get('v61_tail_deferred', 0)} "
                         f"v61_direct={_STATS.get('v61_direct_splice', 0)}")
                _STATS["gain_tokens"] = _STATS.get("gain_tokens", 0) + gain
            return out

        Req.init_next_round_input = _patched_init_round

        print(f"[kvreuse-site] v6.10 splice-after-extend patches applied (per-request knobs via kvreuse_session_id) (v5 base) v6={_V6} pin_only={_PIN_ONLY} lead_freeze={_LEAD_FREEZE} ssm_snap={_SSM_SNAP} pid={os.getpid()} "
              f"no_rerotate={_NO_REROT} max_sessions={_MAX_SESS} "
              f"fastdiff={_FASTDIFF} cdc_target={_CDC_TARGET} "
              f"fastdiff_min={_FASTDIFF_MIN} timing={_TIMING} min_extend={_V6_MIN_EXTEND} t6_guard={_T6_GUARD} trace={_TRACE} "
              f"evict_on_fail={_EVICT_ON_FAIL} evict_margin={_EVICT_MARGIN} "
              f"| v6.10k multiblock=1 max_blocks={_MAX_BLOCKS} mb_min_gap={_MB_MIN_GAP} splice_retry={_SPLICE_RETRY}",
              flush=True)

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Token diff with content-defined chunking
# ---------------------------------------------------------------------------
def _cdc_bounds(ids, target):
    """Content-defined chunk boundaries (a function of the token ids only)."""
    mask = target - 1
    bounds = [0]
    h = 0
    for i, t in enumerate(ids):
        h = ((h << 1) ^ ((t * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF
        if ((h >> 8) & mask) == 0:
            bounds.append(i + 1)
    if bounds[-1] != len(ids):
        bounds.append(len(ids))
    return bounds


def _chunk_ids(ids, bounds, interner):
    out = []
    for k in range(len(bounds) - 1):
        key = tuple(ids[bounds[k]:bounds[k + 1]])
        v = interner.get(key)
        if v is None:
            v = len(interner)
            interner[key] = v
        out.append(v)
    return out


def _verify_ops(a, b, ops):
    """True iff ops tile both sequences and every 'equal' block is really equal."""
    pi = pj = 0
    for t, i1, i2, j1, j2 in ops:
        if i1 != pi or j1 != pj:
            return False
        if t == "equal" and a[i1:i2] != b[j1:j2]:
            return False
        pi, pj = i2, j2
    return pi == len(a) and pj == len(b)


def _fast_opcodes(a, b):
    """difflib-compatible opcodes via chunk-level diffing; None means fall back."""
    ba = _cdc_bounds(a, _CDC_TARGET)
    bb = _cdc_bounds(b, _CDC_TARGET)
    interner = {}
    ca = _chunk_ids(a, ba, interner)
    cb = _chunk_ids(b, bb, interner)
    blocks = []
    for t, i1, i2, j1, j2 in SequenceMatcher(a=ca, b=cb, autojunk=False).get_opcodes():
        if t != "equal":
            continue
        ti1, ti2, tj1, tj2 = ba[i1], ba[i2], bb[j1], bb[j2]
        if ti2 - ti1 != tj2 - tj1:
            return None
        blocks.append([ti1, ti2, tj1, tj2])
    if _CDC_EXTEND and blocks:
        for n, blk in enumerate(blocks):
            lo_a = blocks[n - 1][1] if n else 0
            lo_b = blocks[n - 1][3] if n else 0
            while blk[0] > lo_a and blk[2] > lo_b and a[blk[0] - 1] == b[blk[2] - 1]:
                blk[0] -= 1
                blk[2] -= 1
        for n in range(len(blocks) - 1, -1, -1):
            blk = blocks[n]
            hi_a = blocks[n + 1][0] if n + 1 < len(blocks) else len(a)
            hi_b = blocks[n + 1][2] if n + 1 < len(blocks) else len(b)
            while blk[1] < hi_a and blk[3] < hi_b and a[blk[1]] == b[blk[3]]:
                blk[1] += 1
                blk[3] += 1
    ops, pi, pj = [], 0, 0
    for ti1, ti2, tj1, tj2 in blocks:
        da, db = ti1 - pi, tj1 - pj
        if da and db:
            ops.append(("replace", pi, ti1, pj, tj1))
        elif da:
            ops.append(("delete", pi, ti1, pj, tj1))
        elif db:
            ops.append(("insert", pi, ti1, pj, tj1))
        ops.append(("equal", ti1, ti2, tj1, tj2))
        pi, pj = ti2, tj2
    da, db = len(a) - pi, len(b) - pj
    if da and db:
        ops.append(("replace", pi, len(a), pj, len(b)))
    elif da:
        ops.append(("delete", pi, len(a), pj, len(b)))
    elif db:
        ops.append(("insert", pi, len(a), pj, len(b)))
    return ops


def _diff_opcodes(a, b):
    """Token-level opcodes between the previous and the new prompt, verified."""
    if _FASTDIFF and min(len(a), len(b)) >= _FASTDIFF_MIN:
        try:
            ops = _fast_opcodes(a, b)
        except Exception as e:
            _log(f"fastdiff failed ({e}); using difflib")
            ops = None
        if ops is not None and _verify_ops(a, b, ops):
            _STATS["fastdiff_ok"] = _STATS.get("fastdiff_ok", 0) + 1
            return ops
        _STATS["fastdiff_reject"] = _STATS.get("fastdiff_reject", 0) + 1
        if _STATS["fastdiff_reject"] % 50 == 1:
            _log(f"fastdiff rejected by verifier "
                 f"[n={_STATS['fastdiff_reject']}]; falling back to difflib")
    return SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()


def _quick_ratio_shared(a, bcount, lb):
    """difflib quick_ratio with Counter(b) computed once per request."""
    avail = {}
    availhas = avail.__contains__
    matches = 0
    for elt in a:
        numb = avail[elt] if availhas(elt) else bcount.get(elt, 0)
        avail[elt] = numb - 1
        if numb > 0:
            matches += 1
    la = len(a)
    return 2.0 * matches / (la + lb) if (la + lb) else 1.0


# ---------------------------------------------------------------------------
# Session matching
# ---------------------------------------------------------------------------
def _find_session(sessions, new_ids, key=None):
    """Return the id of the tracked conversation this prompt continues, or None.

    Only conversations with the same cache key (`cache_salt`) are candidates, so
    requests that SGLang's radix cache keeps apart never share reused state.
    """
    from collections import Counter
    best_sid, best = None, 0
    best_n = 0
    best_unslotted = 0.0
    _dbg_slotted = _dbg_weak = _dbg_short = 0
    head = new_ids[:_MATCH_WINDOW]
    bcount, lb = Counter(new_ids), len(new_ids)
    for sid, sess in sessions.items():
        if sess.get("key") != key:
            continue
        old = sess.get("token_ids")
        if not old:
            continue
        slotted = sess.get("side_slot") is not None
        n = 0
        for x, y in zip(old[:_MATCH_WINDOW], head):
            if x != y:
                break
            n += 1
        if slotted:
            _dbg_slotted += 1
        if n < _MATCH_MIN_TOKENS:
            _dbg_short += 1
            continue
        r_pre = n / max(1, len(old))
        r_all = _quick_ratio_shared(old, bcount, lb)
        if r_pre < _MATCH_MIN_PREFIX_FRAC and r_all < _MATCH_MIN_OVERLAP:
            _STATS["match_weak"] = _STATS.get("match_weak", 0) + 1
            _dbg_weak += 1
            continue
        r = r_all
        if not slotted:
            best_unslotted = max(best_unslotted, r)
            continue
        if n > best_n or (n == best_n and r > best):
            best_n, best, best_sid = n, r, sid
    if _DEBUG:
        _log(f"match: cands={len(sessions)} slotted={_dbg_slotted} "
             f"best={best:.3f} best_unslotted={best_unslotted:.3f} "
             f"weak={_dbg_weak} short={_dbg_short} floor={_MATCH_MIN_RATIO}")
    if best < _MATCH_MIN_RATIO:
        _STATS["match_reject"] = _STATS.get("match_reject", 0) + 1
        return None
    if best_unslotted > best + _MATCH_AMBIG_MARGIN:
        _STATS["match_ambiguous"] = _STATS.get("match_ambiguous", 0) + 1
        return None
    return best_sid


_sid_counter = [0]


# ---------------------------------------------------------------------------
# Linear-attention (GatedDeltaNet / Mamba) recurrent state
#
# Relocated tokens are not forwarded, so the recurrent state never advances over
# them. KVREUSE_SSM_MODE selects what the request starts from:
#   fork    restore the state saved at the end of the previous prompt (default)
#   strict  relocate only when the saved state covers exactly the reused span
#   none    no state restore
# ---------------------------------------------------------------------------
_SSM_ON = os.environ.get("KVREUSE_SSM", "1") == "1"
_SSM_MODE = os.environ.get("KVREUSE_SSM_MODE", "fork").lower()
# 1 serves survivors from the side buffer on the single-phase path even when the
# radix match already covers them.
_FORCE = os.environ.get("KVREUSE_FORCE", "0") == "1"
# Single-phase path: relocate only when survivors form the leading run.
_SUFFIX_GATE = os.environ.get("KVREUSE_SUFFIX_GATE", "1") == "1"


def _ssm_init(mr):
    """Allocate a private copy of the recurrent state, one row per side slot."""
    _RT["ssm"] = None
    if not _SSM_ON:
        return
    rtp = getattr(mr, "req_to_token_pool", None)
    pool = getattr(rtp, "mamba_pool", None)
    if pool is None:
        return
    try:
        import torch
        cache = pool.mamba_cache
        n = _SIDE_SESSIONS
        conv = [torch.empty((c.shape[0], n) + tuple(c.shape[2:]),
                            dtype=c.dtype, device=c.device) for c in cache.conv]
        t = cache.temporal
        temporal = torch.empty((t.shape[0], n) + tuple(t.shape[2:]),
                               dtype=t.dtype, device=t.device)
        nbytes = sum(x.numel() * x.element_size() for x in conv) + \
            temporal.numel() * temporal.element_size()
        _RT["ssm"] = {"cache": cache, "conv": conv, "temporal": temporal, "n": n}
        if not getattr(rtp, "_kvreuse_alloc_wrapped", False):
            _inner = rtp.alloc

            def _wrapped_alloc(reqs, _inner=_inner):
                out = _inner(reqs)
                if out is not None:
                    for _r in reqs:
                        if getattr(_r, "_kvreuse_ssm_shadow", None) is not None:
                            _ssm_restore(_r)
                return out

            rtp.alloc = _wrapped_alloc
            rtp._kvreuse_alloc_wrapped = True
            print(f"[kvreuse] wrapped {type(rtp).__name__}.alloc for state restore",
                  flush=True)
        print(f"[kvreuse] ssm shadow: {n} sessions, "
              f"{nbytes / 2**30:.2f} GiB private (outside the mamba pool)",
              flush=True)
    except Exception as e:
        print(f"[kvreuse] ssm shadow init failed ({e}); freezing will be gated off",
              flush=True)
        _RT["ssm"] = None


def _ssm_save(sess, req, n_tokens):
    """Snapshot the recurrent state for the sequence just processed."""
    sess["ssm_tokens"] = -1
    s = _RT.get("ssm")
    slot = sess.get("side_slot")
    if not s or slot is None or slot >= s["n"]:
        return
    idx = getattr(req, "mamba_pool_idx", None)
    if idx is None:
        return
    try:
        i = int(idx)
        for dst, src in zip(s["conv"], s["cache"].conv):
            dst[:, int(slot)] = src[:, i]
        s["temporal"][:, int(slot)] = s["cache"].temporal[:, i]
        sess["ssm_tokens"] = int(n_tokens)
        req._kvreuse_ssm_saved = True
        _tr("ssm_save", rid=getattr(req, "rid", "?"), sid=sess.get("sid"),
            slot=int(slot), midx=i, n=int(n_tokens))
        _STATS["ssm_saved"] = _STATS.get("ssm_saved", 0) + 1
    except Exception as e:
        _log(f"ssm save failed: {e}")


def _ssm_mode_for(req):
    return _ks(req, "ssm", _SSM_MODE) if req is not None else _SSM_MODE


def _ssm_ready(plan, req=None):
    """Decide whether, and with which recurrent state, a plan may relocate."""
    r = plan["rows"]
    n = len(r)
    if n <= 0:
        return False
    plan["ssm_restore"] = False
    _w = _RT.get("swa_window") or 0
    if _w:
        n = n - _w
        if n <= 0:
            _STATS["swa_too_short"] = _STATS.get("swa_too_short", 0) + 1
            return False
        plan["swa_capped"] = True
    if _RT.get("ssm") is None:
        if _RT.get("ssm_na", False):
            plan["freeze_n"] = n
            return True
        return False
    _mode = _ssm_mode_for(req)
    if _mode == "none":
        plan["freeze_n"] = n
        return True
    have = (plan.get("side_slot") is not None
            and plan["side_slot"] < _RT["ssm"]["n"]
            and plan.get("ssm_tokens", -1) > 0)
    if not have:
        return False
    if _mode == "strict":
        t = plan["ssm_tokens"]
        if not (0 < t <= n):
            return False
        plan["freeze_n"] = t
        plan["ssm_restore"] = True
        return True
    plan["freeze_n"] = n
    plan["ssm_restore"] = True
    return True


def _ssm_restore(req):
    """Write the saved recurrent state into the request's live state slot."""
    s = _RT.get("ssm")
    slot = getattr(req, "_kvreuse_ssm_shadow", None)
    if not s or slot is None:
        _STATS["ssm_r_noslot"] = _STATS.get("ssm_r_noslot", 0) + 1
        if _DEBUG and _STATS["ssm_r_noslot"] % 50 == 1:
            _log(f"ssm restore skipped: ssm_obj={s is not None} slot={slot}")
        return
    idx = getattr(req, "mamba_pool_idx", None)
    if idx is None:
        _STATS["ssm_r_noidx"] = _STATS.get("ssm_r_noidx", 0) + 1
        if _DEBUG and _STATS["ssm_r_noidx"] % 50 == 1:
            _log(f"ssm restore deferred: no mamba_pool_idx yet (rid={getattr(req,'rid','?')})")
        return
    req._kvreuse_ssm_shadow = None
    try:
        i = int(idx)
        _before = float(s["cache"].temporal[:, i].float().abs().sum().item())
        _shadow = float(s["temporal"][:, int(slot)].float().abs().sum().item())
        for src, dst in zip(s["conv"], s["cache"].conv):
            dst[:, i] = src[:, int(slot)]
        s["cache"].temporal[:, i] = s["temporal"][:, int(slot)]
        _after = float(s["cache"].temporal[:, i].float().abs().sum().item())
        _tr("ssm_restore", rid=getattr(req, "rid", "?"), slot=int(slot), midx=i,
            sid=(getattr(req, "_kvreuse_session", None) or {}).get("sid"),
            before=round(_before, 3), shadow=round(_shadow, 3), after=round(_after, 3))
        _STATS["ssm_restored"] = _STATS.get("ssm_restored", 0) + 1
        if abs(_after - _shadow) > 1e-3:
            _STATS["ssm_write_lost"] = _STATS.get("ssm_write_lost", 0) + 1
        if abs(_before - _after) < 1e-6:
            _STATS["ssm_noop"] = _STATS.get("ssm_noop", 0) + 1
        if _STATS["ssm_restored"] % 10 == 1:
            _log(f"ssm verify: before={_before:.3f} shadow={_shadow:.3f} "
                 f"after={_after:.3f} noop={_STATS.get('ssm_noop',0)} "
                 f"lost={_STATS.get('ssm_write_lost',0)}")
    except Exception as e:
        _log(f"ssm restore failed: {e}")


# ---------------------------------------------------------------------------
# Planning (runs when a request enters the waiting queue)
# ---------------------------------------------------------------------------
def _kvreuse_intercept(scheduler, req):
    """Match the request to its conversation, diff, and attach a relocation plan."""
    if not hasattr(scheduler, "_kvreuse_sessions"):
        import collections
        scheduler._kvreuse_sessions = collections.OrderedDict()

    mr = scheduler.tp_worker.model_runner
    if _RT["alloc"] is None:
        _RT["alloc"] = mr.token_to_kv_pool_allocator
        _RT["pool"] = mr.token_to_kv_pool_allocator._kvcache
        _RT["layers"] = _full_layer_ids(_RT["pool"], getattr(mr, "model_config", None))
        _RT["swa_window"] = 0
        try:
            _mc = getattr(mr, "model_config", None)
            for _cfg in (getattr(_mc, "hf_text_config", None),
                         getattr(_mc, "hf_config", None), _mc):
                _lt = getattr(_cfg, "layer_types", None)
                _sw = getattr(_cfg, "sliding_window", None)
                if _lt and _sw and any("sliding" in str(t) for t in _lt):
                    _RT["swa_window"] = int(_sw)
                    break
        except Exception:
            pass
        if _RT["swa_window"]:
            print(f"[kvreuse] sliding-window model: reserving last "
                  f"{_RT['swa_window']} tokens for re-prefill", flush=True)
        _RT["rotary"] = _find_rotary(mr.model)
        _RT["sessions"] = scheduler._kvreuse_sessions
        _RT["scheduler"] = scheduler
        _RT["tree_cache"] = getattr(scheduler, "tree_cache", None)
        try:
            total = int(_RT["alloc"].size)
        except Exception:
            total = int(getattr(_RT["alloc"], "available_size", lambda: 0)())
        _RT["budget"] = int(total * _BUDGET_FRAC) if total else None
        _RT["evict_floor"] = int(total * _EVICT_FLOOR_FRAC) if total else 0
        _RT["ssm_na"] = getattr(
            getattr(mr, "req_to_token_pool", None), "mamba_pool", None) is None
        try:
            _ssm_init(mr)
            _RT["ssm_na"] = getattr(
                getattr(mr, "req_to_token_pool", None), "mamba_pool", None) is None
        except Exception as e:
            print(f"[kvreuse] ssm init error: {e}", flush=True)
        try:
            _side_init(_RT["pool"])
        except Exception as e:
            _log(f"side buffer init FAILED ({e}); kv-reuse disabled")
            _RT["side_k"] = None
        _log(f"init: full_attn_layers={len(_RT['layers'])} "
             f"rotary={_RT['rotary'] is not None} layers={_RT['layers'][:8]} "
             f"pool_slots={total} snapshot_budget={_RT['budget']} "
             f"evict_floor={_RT['evict_floor']}")

    if _req_switch(req) == "off":
        _STATS["ref_requests"] = _STATS.get("ref_requests", 0) + 1
        if _DEBUG:
            _log(f"kv6 rid={getattr(req,'rid','?')} REFERENCE arm (switch=off): no intercept")
        return
    sessions = scheduler._kvreuse_sessions
    new_ids = list(req.origin_input_ids)

    _t0 = _tick()
    cache_key = getattr(req, "extra_key", None)
    sid = _find_session(sessions, new_ids, cache_key)
    _tock("find_session", _t0)
    if sid is None:
        _sid_counter[0] += 1
        sid = f"s{_sid_counter[0]:06d}"
        sessions[sid] = {"sid": sid, "key": cache_key, "token_ids": [], "positions": [],
                         "side_slot": None, "n_side": 0}
    else:
        sessions.move_to_end(sid)
    sess = sessions[sid]
    if _TRACE:
        _tag = ""
        try:
            _tk = _RT.get("tok")
            if _tk is None:
                _tk = getattr(scheduler, "tokenizer", None)
                _RT["tok"] = _tk
            if _tk is not None:
                _tag = _tk.decode(new_ids[:48])[:90].replace("\n", " ")
        except Exception:
            _tag = "?"
        _tr("req", rid=getattr(req, "rid", "?"), sid=sid, slot=sess.get("side_slot"),
            n_new=len(new_ids), n_old=len(sess.get("token_ids") or []),
            nsess=len(sessions), tag=repr(_tag))
    _evict_sessions(scheduler)

    req._kvreuse_session = sess
    sess["inflight"] = sess.get("inflight", 0) + 1
    old_ids = sess["token_ids"]

    if not old_ids or sess.get("side_slot") is None:
        sess["token_ids"] = new_ids
        sess["positions"] = list(range(len(new_ids)))
        _STATS["cold_turns"] += 1
        if _DEBUG:
            _log(f"{sid} COLD (no private KV) total={len(new_ids)} "
                 f"n_sessions={len(sessions)}")
        return

    _t0 = _tick()
    ops = _diff_opcodes(old_ids, new_ids)
    _tock("get_opcodes", _t0)
    if _TIMING:
        _STATS["opc_tokens"] = _STATS.get("opc_tokens", 0) + len(old_ids) + len(new_ids)
        _STATS["opc_uniq_last"] = len(set(old_ids))
    kept = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t == "equal")
    deleted = sum(i2 - i1 for t, i1, i2, j1, j2 in ops if t in ("delete", "replace"))
    inserted = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t in ("insert", "replace"))

    if inserted == 0 or kept == 0 or deleted == 0:
        sess["token_ids"] = new_ids
        sess["positions"] = list(range(len(new_ids)))
        _STATS["passthrough_turns"] += 1
        if _DEBUG:
            _log(f"{sid} passthrough kept={kept} del={deleted} ins={inserted}")
        return

    new_positions = []
    for t, i1, i2, j1, j2 in ops:
        if t in ("equal", "insert", "replace"):
            base = len(new_positions)
            new_positions.extend(range(base, base + (j2 - j1)))

    rows, old_pos, new_pos = [], [], []
    positions = sess["positions"]
    n_side = int(sess.get("n_side") or 0)
    for t, i1, i2, j1, j2 in ops:
        if t != "equal":
            continue
        for k in range(i2 - i1):
            si = i1 + k
            if si < n_side and si < len(positions):
                rows.append(si)
                old_pos.append(positions[si])
                new_pos.append(new_positions[j1 + k])
    tail = None
    _mintail = int(_ks(req, "mintail", _V6_MIN_TAIL) or _V6_MIN_TAIL)
    if _kb(req, "v6", _V6) and not _kb(req, "pin", _PIN_ONLY) and ops and ops[0][0] == "equal" and ops[0][1] == 0 and ops[0][3] == 0:
        a_new_end = ops[0][4]
        _sids = sess.get("side_ids") or []
        _cands = []
        for kC, o in enumerate(ops):
            if kC == 0 or o[0] != "equal":
                continue
            _t, ci1, ci2, cj1, cj2 = o
            n_c = ci2 - ci1
            if not (cj1 >= a_new_end and n_c >= _mintail and ci2 <= n_side and ci2 <= len(positions)):
                continue
            _cands.append((n_c, kC, ci1, ci2, cj1, cj2))
        _cands.sort(reverse=True)
        _kext = max(1, int(_ks(req, "minext", _V6_MIN_EXTEND) or _V6_MIN_EXTEND))
        _maxb = max(1, int(_ks(req, "maxblocks", _MAX_BLOCKS) or _MAX_BLOCKS))
        _acc = []
        for n_c, kC, ci1, ci2, cj1, cj2 in _cands:
            if len(_acc) >= _maxb:
                break
            if n_c - _kext < _mintail:
                continue
            _aligned = len(_sids) >= ci2 and list(_sids[ci1:ci2 - _kext]) == list(old_ids[ci1:ci2 - _kext]) == list(new_ids[cj1:cj2 - _kext])
            if not _aligned:
                _STATS["v6_tail_misaligned"] = _STATS.get("v6_tail_misaligned", 0) + 1
                _log(f"kv6 rid={getattr(req,'rid','?')} TAIL ROWS MISALIGNED: side rows {ci1}:{ci2-_kext} != C tokens (side_ids={len(_sids)} n_side={n_side}) -> next candidate")
                continue
            if len(new_ids) - (cj2 - _kext) < _kext:
                continue
            _acc.append({"rows": list(range(ci1, ci2 - _kext)), "old_pos": positions[ci1:ci2 - _kext],
                         "bprime_end": cj1, "n_c": n_c, "cj2": cj2})
            if len(_acc) == 1:
                tail = {"rows": _acc[0]["rows"], "old_pos": _acc[0]["old_pos"],
                        "bprime_end": cj1, "lead_new_end": a_new_end, "n_c": n_c, "kext": _kext}
                _STATS["v6_tail_plans"] = _STATS.get("v6_tail_plans", 0) + 1
                if _DEBUG:
                    _log(f"kv6 rid={getattr(req,'rid','?')} v6.10 tail: C=ops[{kC}] n_c={n_c} new=[{cj1},{cj2}) "
                         f"D_ops={[o[0][:3] for o in ops[kC + 1:]][:6]} kext={_kext} reloc={n_c - _kext} "
                         f"extend_from={cj2 - _kext} live={len(new_ids)} cands={len(_cands)}")
            if _maxb <= 1:
                break
        if len(_acc) > 1:
            _sched = sorted(_acc, key=lambda b: b["bprime_end"])
            if _MB_MIN_GAP > 0:
                _keep, _end_prev = [_sched[0]], _sched[0]["cj2"] - _kext
                for _b in _sched[1:]:
                    if _b["bprime_end"] - _end_prev < _MB_MIN_GAP:
                        _STATS["v6_mb_gap_dropped"] = _STATS.get("v6_mb_gap_dropped", 0) + 1
                        continue
                    _keep.append(_b)
                    _end_prev = _b["cj2"] - _kext
                _sched = _keep
            if len(_sched) > 1:
                _first = _sched[0]
                tail["rows"], tail["old_pos"] = _first["rows"], _first["old_pos"]
                tail["bprime_end"], tail["n_c"] = _first["bprime_end"], _first["n_c"]
                tail["more"] = [{"rows": b["rows"], "old_pos": b["old_pos"],
                                 "bprime_end": b["bprime_end"], "n_c": b["n_c"]} for b in _sched[1:]]
                _STATS["v6_mb_plans"] = _STATS.get("v6_mb_plans", 0) + 1
                _STATS["v6_mb_extra_blocks"] = _STATS.get("v6_mb_extra_blocks", 0) + len(_sched) - 1
                _log(f"kv6 rid={getattr(req,'rid','?')} multiblock plan: K={len(_sched)} "
                     f"stops={[b['bprime_end'] for b in _sched]} "
                     f"reloc={[b['n_c'] - _kext for b in _sched]} "
                     f"kext={_kext} maxb={_maxb} cands={len(_cands)} live={len(new_ids)}")
    if _DEBUG and tail is None and _kb(req, "v6", _V6):
        _log(f"kv6 rid={getattr(req,'rid','?')} no tail plan: ops={[(o[0][:3], o[2]-o[1], o[4]-o[3]) for o in ops][:8]} n_side={n_side} mintail={_mintail} old={len(old_ids)} new={len(new_ids)}")
    if rows and sess.get("side_slot") is not None:
        req._kvreuse_plan = {"rows": rows, "old_pos": old_pos, "new_pos": new_pos, "tail": tail,
                             "new_ids": new_ids, "side_slot": sess["side_slot"],
                             "kept": kept, "n_side": n_side,
                             "ssm_tokens": sess.get("ssm_tokens", -1),
                             "suffix_only": new_pos == list(range(len(rows))),
                             "lead_run": next((i for i, p in enumerate(new_pos)
                                               if p != i), len(new_pos)),
                             "len_pos": len(positions), "old_len": len(old_ids)}

    sess["token_ids"] = new_ids
    sess["positions"] = new_positions
    if _DEBUG:
        _log(f"{sid} kept={kept} del={deleted} ins={inserted} "
             f"live={len(new_ids)} plan_rows={len(rows)}")


# ---------------------------------------------------------------------------
# Relocation (runs inside the scheduler when the plan executes)
# ---------------------------------------------------------------------------
def _evict_headroom_ok(scheduler):
    """True while enough reclaimable full-attention KV remains."""
    floor = _RT.get("evict_floor") or 0
    if floor <= 0:
        return True
    try:
        tc = scheduler.tree_cache
        eviction = None
        for name in ("full_evictable_size", "evictable_size"):
            fn = getattr(tc, name, None)
            if callable(fn):
                eviction = int(fn())
                break
        if eviction is None:
            return True
        avail = int(_RT["alloc"].available_size())
        return (eviction + avail) >= floor
    except Exception:
        return True


def _find_rotary(model):
    """Locate the model's rotary embedding (its cos/sin cache is used to re-rotate keys)."""
    roots = []
    if hasattr(model, "model"):
        roots.append(model.model)
        lm = getattr(model.model, "language_model", None)
        if lm is not None and hasattr(lm, "model"):
            roots.append(lm.model)
    for root in roots:
        if hasattr(root, "rotary_emb"):
            return root.rotary_emb
        for layer in getattr(root, "layers", []) or []:
            if hasattr(layer, "rotary_emb"):
                return layer.rotary_emb
            sa = getattr(layer, "self_attn", None)
            if sa is not None and hasattr(sa, "rotary_emb"):
                return sa.rotary_emb
    return None


def _evict_then_retry(alloc, tree_cache, n):
    """Evict the shortfall from the radix tree and retry the allocation once."""
    try:
        avail = int(alloc.available_size())
        ev = None
        for name in ("full_evictable_size", "evictable_size"):
            fn = getattr(tree_cache, name, None)
            if callable(fn):
                try:
                    ev = int(fn())
                except (NotImplementedError, TypeError):
                    continue
                break
        if ev is None:
            _log("evict-before-fail: no evictable_size API; skipping")
            return None
        floor = _RT.get("evict_floor") or 0
        if ev + avail < n + floor:
            _log(f"evict-before-fail: declined (avail={avail} evictable={ev} "
                 f"need={n} floor={floor})")
            return None
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        need = n - avail + _EVICT_MARGIN
        res = tree_cache.evict(EvictParams(num_tokens=need))
        got = getattr(res, "num_tokens_evicted", None)
        out = alloc.alloc(n)
        _STATS["evict_retry"] = _STATS.get("evict_retry", 0) + 1
        if out is not None:
            _STATS["evict_retry_ok"] = _STATS.get("evict_retry_ok", 0) + 1
        _log(f"evict-before-fail: requested={need} evicted={got} "
             f"retry={'OK' if out is not None else 'FAIL'} "
             f"[{_STATS.get('evict_retry_ok', 0)}/{_STATS['evict_retry']}]")
        return out
    except Exception as e:
        _log(f"evict-before-fail error: {e}")
        return None


def _rotate_keys(slots_t, old_t, new_t, pool, rotary, torch, no_rerot=None):
    """Re-rotate post-RoPE keys in slots_t from positions old_t to new_t."""
    if (_NO_REROT if no_rerot is None else no_rerot) or torch.equal(old_t, new_t):
        return
    cache = rotary.cos_sin_cache
    rd = cache.shape[-1] // 2
    cn, sn = cache[new_t, :rd].float(), cache[new_t, rd:].float()
    co, so = cache[old_t, :rd].float(), cache[old_t, rd:].float()
    cos2 = torch.cat([(cn * co + sn * so).unsqueeze(1)] * 2, dim=-1)
    sin2 = torch.cat([(sn * co - cn * so).unsqueeze(1)] * 2, dim=-1)

    def _rh(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    for lid in _RT["layers"]:
        kb = pool.get_key_buffer(lid)
        k = kb[slots_t].float()
        kr, kp = k[..., : rd * 2], k[..., rd * 2:]
        kb[slots_t] = torch.cat([kr * cos2 + _rh(kr) * sin2, kp], dim=-1).to(kb.dtype)


def _v6_end(req):
    """End index of the extend the adder admitted."""
    r = getattr(req, "extend_range", None)
    if r is not None:
        return int(r.end)
    return len(req.prefix_indices) + int(getattr(req, "extend_input_len", 0) or 0)


def _v6_full_len(req):
    """Full prompt length of the request."""
    f = getattr(req, "full_untruncated_fill_ids", None)
    return len(f) if f is not None else len(req.fill_ids)


def _v6_abort_phase1(req, why, restore_lengths=True):
    """Drop the two-phase plan for this request and continue as a normal prefill."""
    v6 = getattr(req, "_kvreuse_v6", None)
    if not v6:
        return
    if restore_lengths:
        from array import array as _arr6
        if hasattr(req, "full_untruncated_fill_ids"):
            req.full_untruncated_fill_ids = _arr6("q", v6["new_ids"])
            if hasattr(req, "set_extend_range"):
                req.set_extend_range(len(req.prefix_indices), len(req.full_untruncated_fill_ids))
        else:
            req.fill_ids = _arr6("q", v6["new_ids"])
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    req._kvreuse_v6 = None
    _v6_fallback_ssm(req, v6)
    _STATS["v6_abort"] = _STATS.get("v6_abort", 0) + 1
    _log(f"v6 phase1 aborted ({why}); normal prefill of the tail [aborts={_STATS['v6_abort']}]")


def _v6_fallback_ssm(req, v6):
    """Recurrent state for a request that falls back to normal prefill of the tail.

    When A came from the radix cache and no block of the plan was spliced yet, the
    request's own state already covers A and B', so it is kept and the tail is
    prefilled as usual. Otherwise the plan's saved state is restored as for a splice.
    """
    if v6.get("radix_a") and int(v6.get("bi", 1) or 1) <= 1:
        req._kvreuse_ssm_shadow = None
        _STATS["v6_fallback_own_state"] = _STATS.get("v6_fallback_own_state", 0) + 1
    elif v6.get("ssm_restore"):
        req._kvreuse_ssm_shadow = v6.get("side_slot")


def _v6_splice(req, v6):
    """Splice one relocated block (re-rotated to its new position) onto the request's
    prefix, then arm the next block of the plan if there is one.
    """
    import torch
    pool, rotary, alloc = _RT["pool"], _RT["rotary"], _RT["alloc"]
    pl = len(req.prefix_indices)
    if pl != v6["bprime_end"]:
        _STATS["v6_phase2_mismatch"] = _STATS.get("v6_phase2_mismatch", 0) + 1
        _log(f"v6 phase2: prefix {pl} != |A|+|B'| {v6['bprime_end']} -> normal prefill of the tail")
        if _v6_full_len(req) - pl < 1:
            req.prefix_indices = req.prefix_indices[:-1]
            req.extend_input_len = _v6_full_len(req) - len(req.prefix_indices)
        _v6_fallback_ssm(req, v6)
        return False
    n = len(v6["rows"])
    if n <= 0 or rotary is None or pool is None or _RT.get("side_k") is None:
        return False
    if _v6_full_len(req) - pl - n < 1:
        _STATS["v6_phase2_short"] = _STATS.get("v6_phase2_short", 0) + 1
        return False
    slots_t = alloc.alloc(n)
    if slots_t is None and _EVICT_ON_FAIL:
        slots_t = _evict_then_retry(alloc, _RT.get("tree_cache"), n)
    if slots_t is None:
        _STATS["v6_alloc_fail"] = _STATS.get("v6_alloc_fail", 0) + 1
        _log(f"v6 phase2: alloc({n}) failed -> normal prefill of the tail")
        _v6_fallback_ssm(req, v6)
        return False
    _sess6 = getattr(req, "_kvreuse_session", None)
    _own = _RT.get("side_owner", {}).get(v6["side_slot"], "?")
    _sid6 = (_sess6 or {}).get("sid")
    _sids6 = (_sess6 or {}).get("side_ids") or []
    _r0, _r1 = v6["rows"][0], v6["rows"][-1] + 1
    _want = list(v6["new_ids"][pl:pl + n])
    _have = list(_sids6[_r0:_r1])
    if _own != _sid6 or _have != _want:
        _STATS["v6_slot_stolen"] = _STATS.get("v6_slot_stolen", 0) + 1
        _log(f"v6 SIDE SLOT STOLEN: rid={getattr(req,'rid','?')} slot={v6['side_slot']} owner={_own} sess={_sid6} "
             f"rows={_r0}:{_r1} ids_match={_have == _want} guard={_T6_GUARD} -> "
             f"{'normal prefill of the tail' if _T6_GUARD else 'WOULD DECLINE (guard off, v6.9 behaviour)'} "
             f"[stolen={_STATS['v6_slot_stolen']}]")
    if _T6_GUARD and (_own != _sid6 or _have != _want):
        try:
            alloc.free(slots_t)
        except Exception:
            pass
        if v6.get("ssm_restore"):
            req._kvreuse_ssm_shadow = None
        return False
    try:
        dev = slots_t.device
        _side_load(v6["side_slot"], torch.tensor(v6["rows"], dtype=torch.int64, device=dev), slots_t)
        old_t = torch.tensor(v6["old_pos"], dtype=torch.long, device=dev)
        new_t = torch.arange(pl, pl + n, dtype=torch.long, device=dev)
        _rotate_keys(slots_t, old_t, new_t, pool, rotary, torch, no_rerot=not _kb(req, "rerot", not _NO_REROT))
        req.prefix_indices = torch.cat([req.prefix_indices.to(dev), slots_t])
        req.extend_input_len = _v6_full_len(req) - len(req.prefix_indices)
        if hasattr(req, "set_extend_range"):
            req.set_extend_range(len(req.prefix_indices), _v6_full_len(req))
        req.cached_tokens = int(getattr(req, "cached_tokens", 0) or 0) + n
        req._kvreuse_prefix_set = True
        if v6.get("ssm_restore"):
            req._kvreuse_ssm_shadow = v6.get("side_slot")
        if _DEBUG:
            _log(f"v6 ssm arm: restore={v6.get('ssm_restore')} slot={v6.get('side_slot')} "
                 f"shadow={getattr(req, '_kvreuse_ssm_shadow', None)} mamba_idx={getattr(req, 'mamba_pool_idx', None)}")
        _STATS["v6_spliced"] = _STATS.get("v6_spliced", 0) + 1
        _STATS["v6_spliced_tokens"] = _STATS.get("v6_spliced_tokens", 0) + n
        _STATS["tokens_frozen"] += n
        _STATS["tokens_prefilled"] -= n
        _log(f"kv6 rid={getattr(req,'rid','?')} phase2 prefilled_chunk1={v6.get('bprime_end',0)-(v6.get('radix_n0') if v6.get('radix_n0') is not None else pl - 0)} reused_tail={n} extend2={req.extend_input_len} total={_v6_full_len(req)} rot_delta={int(pl)-int(v6['old_pos'][0]) if v6.get('old_pos') else 0}")
        _bi, _bn = int(v6.get("bi", 1) or 1), int(v6.get("bn", 1) or 1)
        _log(f"v6 splice OK: rid={getattr(req,'rid','?')} sid={_sid6} slot={v6['side_slot']} "
             f"A+B'={pl} C'={n} extend={req.extend_input_len} live={_v6_full_len(req)} "
             f"[spliced={_STATS['v6_spliced']} tokens={_STATS['v6_spliced_tokens']}] blocks={_bi}/{_bn}")
        _more = list(v6.get("more") or [])
        if _more:
            _nxt = _more[0]
            _end2 = int(_nxt["bprime_end"])
            _pl2 = len(req.prefix_indices)
            if _end2 > _pl2 and len(_nxt["rows"]) > 0:
                from array import array as _arr6k
                _cut2 = _arr6k("q", v6["new_ids"][:_end2])
                if hasattr(req, "full_untruncated_fill_ids"):
                    req.full_untruncated_fill_ids = _cut2
                    if hasattr(req, "set_extend_range"):
                        req.set_extend_range(_pl2, _end2)
                else:
                    req.fill_ids = _cut2
                req.extend_input_len = _end2 - _pl2
                req._kvreuse_v6 = {"phase": 1, "rows": _nxt["rows"], "old_pos": _nxt["old_pos"],
                                   "bprime_end": _end2, "new_ids": v6["new_ids"],
                                   "side_slot": v6.get("side_slot"),
                                   "ssm_restore": False,
                                   "n_c": _nxt["n_c"], "radix_a": v6.get("radix_a"),
                                   "radix_n0": v6.get("radix_n0"),
                                   "more": _more[1:], "bi": _bi + 1, "bn": _bn}
                _STATS["v6_mb_next_armed"] = _STATS.get("v6_mb_next_armed", 0) + 1
                _log(f"kv6 rid={getattr(req,'rid','?')} multiblock next: block {_bi + 1}/{_bn} "
                     f"extend=[{_pl2},{_end2}) then splice {len(_nxt['rows'])} rows")
            else:
                _STATS["v6_mb_dropped"] = _STATS.get("v6_mb_dropped", 0) + 1
                _log(f"kv6 rid={getattr(req,'rid','?')} multiblock DROP: block {_bi + 1}/{_bn} stop={_end2} "
                     f"<= prefix={_pl2} (or empty); {len(_more)} block(s) abandoned, tail re-prefilled")
        return True
    except Exception as e:
        _STATS["v6_splice_err"] = _STATS.get("v6_splice_err", 0) + 1
        _log(f"v6 splice failed ({e}); freeing {n} slots")
        try:
            alloc.free(slots_t)
        except Exception as e2:
            _log(f"CRITICAL: could not free after v6 failure: {e2}")
        return False


def _materialise(req, plan, tree_cache):
    """Single-phase path: load the surviving leading run into fresh slots and re-rotate."""
    import torch
    from array import array as _arr
    pool, rotary, alloc = _RT["pool"], _RT["rotary"], _RT["alloc"]
    if rotary is None or pool is None or _RT.get("side_k") is None:
        _log(f"materialise: preconditions unmet "
             f"(rotary={rotary is not None} pool={pool is not None} "
             f"side={_RT.get('side_k') is not None})")
        return False
    rows, old_pos, new_pos = plan["rows"], plan["old_pos"], plan["new_pos"]
    _fn = plan.get("freeze_n")
    if _fn is not None and _fn < len(rows):
        rows, old_pos, new_pos = rows[:_fn], old_pos[:_fn], new_pos[:_fn]
    dev = pool.get_key_buffer(_RT["layers"][0]).device
    slots_t = alloc.alloc(len(rows))
    if slots_t is None and _EVICT_ON_FAIL:
        slots_t = _evict_then_retry(alloc, tree_cache, len(rows))
    if slots_t is not None:
        _STATS["freeze_alloc_n"] = _STATS.get("freeze_alloc_n", 0) + 1
        _STATS["freeze_alloc_slots"] = _STATS.get("freeze_alloc_slots", 0) + len(rows)
    if slots_t is None:
        _STATS["alloc_fail"] += 1
        try:
            av = alloc.available_size()
        except Exception:
            av = -1
        _log(f"materialise: alloc({len(rows)}) failed (available={av}) "
             f"[alloc_fail={_STATS['alloc_fail']}]")
        return False
    try:
        _side_load(plan["side_slot"], torch.tensor(rows, dtype=torch.int64, device=dev), slots_t)
    except Exception as e:
        _log(f"side load failed: {e}")
        try:
            alloc.free(slots_t)
        except Exception:
            pass
        return False

    try:
        return _materialise_tail(req, tree_cache, plan, slots_t, dev,
                                 old_pos, new_pos, pool, rotary, torch)
    except Exception as e:
        _STATS["materialise_err"] = _STATS.get("materialise_err", 0) + 1
        _log(f"materialise failed ({e}); freeing {len(slots_t)} slots "
             f"[errs={_STATS['materialise_err']}]")
        try:
            alloc.free(slots_t)
        except Exception as e2:
            _log(f"CRITICAL: could not free after failure: {e2}")
        return False


def _materialise_tail(req, tree_cache, plan, slots_t, dev,
                      old_pos, new_pos, pool, rotary, torch):
    """Re-rotate the loaded rows and point the request at them."""
    from array import array as _arr
    _tr0 = _tick()
    old_t = torch.tensor(old_pos, dtype=torch.long, device=dev)
    new_t = torch.tensor(new_pos, dtype=torch.long, device=dev)
    if not _NO_REROT and not torch.equal(old_t, new_t):
        cache = rotary.cos_sin_cache
        rd = cache.shape[-1] // 2
        cn, sn = cache[new_t, :rd].float(), cache[new_t, rd:].float()
        co, so = cache[old_t, :rd].float(), cache[old_t, rd:].float()
        cos2 = torch.cat([(cn * co + sn * so).unsqueeze(1)] * 2, dim=-1)
        sin2 = torch.cat([(sn * co - cn * so).unsqueeze(1)] * 2, dim=-1)

        def _rh(x):
            h = x.shape[-1] // 2
            return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

        for lid in _RT["layers"]:
            kb = pool.get_key_buffer(lid)
            k = kb[slots_t].float()
            kr, kp = k[..., : rd * 2], k[..., rd * 2:]
            kb[slots_t] = torch.cat([kr * cos2 + _rh(kr) * sin2, kp], dim=-1).to(kb.dtype)
    _tock("rotate", _tr0)

    _tf0 = _tick()
    fill = _arr("q", plan["new_ids"])
    req.prefix_indices = slots_t
    req.extend_input_len = len(plan["new_ids"]) - len(slots_t)
    req.fill_ids = fill
    req.full_untruncated_fill_ids = fill
    req.cached_tokens = len(slots_t)
    req.cache_protected_len = 0
    _tcname = type(tree_cache).__name__ if tree_cache is not None else ""
    if tree_cache is not None and "Radix" in _tcname and "Unified" not in _tcname:
        req.last_node = tree_cache.root_node
    req._kvreuse_prefix_set = True
    _tock("mat_fill", _tf0)
    _STATS["materialised"] = _STATS.get("materialised", 0) + 1
    if _STATS["materialised"] % 25 == 1:
        _log(f"materialise OK: prefix={len(slots_t)} extend={req.extend_input_len} "
             f"[total={_STATS['materialised']}]")
    return True

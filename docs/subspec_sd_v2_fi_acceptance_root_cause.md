# SubSpec `v2_fi` Post-Verify Root Cause and Current Status

Updated: 2026-03-02

## Scope

This note keeps only current context for `subspec_sd_v2` vs `subspec_sd_v2_fi` acceptance/throughput behavior.

## Accepted Policy

- Exact step-by-step parity between `v2` and `v2_fi` is not required.
- `v2_fi` may diverge due to overlap timing, as long as acceptance/throughput targets are met.
- Canonical runtime controls remain:
  - `generator_kwargs.disable_post_verify`
  - `generator_kwargs.step_trace`
  - `generator_kwargs.step_trace_debug_verify`

## Root Cause

`v2_fi` overlap postspec can advance draft carry-over tokens on partially updated target KV state. When `_post_verify` consumes those carry-over tokens directly, acceptance drops versus `v2`.

## Canonical Fix Implemented

Patch file:
- `specdecodes/models/generators/subspec_sd_v2_fi.py`

Patch site:
- carry-over path immediately before `_post_verify`

Behavior:
- force one deterministic commit-seed postspec step from the committed `[prefix + tree]` boundary:
  - sync request-cache metadata to boundary
  - `init_postspec(rebuild_frontier=True)`
  - one `postspec()` step
  - `update_tree_after_post()`
  - re-sync request-cache metadata

Intent:
- preserve overlap for throughput
- improve carry-over coherency before post-verify
- keep one canonical state machine (no mode matrix)

## Cleanup Completed

- Removed temporary/nonessential step-trace payload noise from decode loop in `subspec_sd_v2_fi`.
- Kept stable trace schema and stable verify-debug helpers only.
- Removed unused dead helper `_create_pre_draft_hook` from:
  - `specdecodes/helpers/offloaders/prefetch_offloader_postspec.py`
- Fixed stale method-config header in:
  - `configs/methods/subspec_sd_v2_fi.yaml`

## Focused Tests

Passing focused suite:
- `tests/test_subspec_v2_postverify.py`
- `tests/test_subspec_v2_index_remap.py`
- `tests/test_subspec_fi_draft_headroom.py`
- `tests/test_fi_request_cache_reuse.py`
- `tests/test_find_first_step_divergence.py`
- `tests/test_step_trace_helpers.py`

## Latest 5-Sample Benchmarks (`--compile-mode null`)

`v2` baseline:
- run: `experiments/20260302-004212/run_benchmark/mt-bench`
- `tput_mean`: `14.578`
- `mean_verify_accept_len_nonterminal`: `20.435`
- `post_verify_rate`: `0.405`

Patched `v2_fi` (`disable_post_verify=false`):
- run: `experiments/20260302-004839/run_benchmark/mt-bench`
- `tput_mean`: `24.741`
- `mean_verify_accept_len_nonterminal`: `16.109`
- `post_verify_rate`: `0.333`
- acceptance ratio vs `v2`: `0.788`

Guardrail `v2_fi` (`disable_post_verify=true`):
- run: `experiments/20260302-005259/run_benchmark/mt-bench`
- `tput_mean`: `35.314`
- `mean_verify_accept_len_nonterminal`: `28.505`
- `post_verify_rate`: `0.000`

## Target Status

Targets:
- `mean_verify_accept_len_nonterminal(v2_fi, post_verify=on) >= 0.90 * v2`
- `tput_mean(v2_fi, post_verify=on) > 35`
- guardrail `tput_mean(v2_fi, post_verify=off) > 35`

Current status:
- post-verify-on acceptance: fail (`0.788 < 0.90`)
- post-verify-on throughput: fail (`24.741 < 35`)
- post-verify-off guardrail: pass (`35.314 > 35`)

## Default Decision

`configs/methods/subspec_sd_v2_fi.yaml` keeps:
- `generator_kwargs.disable_post_verify: true`

Reason:
- post-verify-on targets are not met yet.

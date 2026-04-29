# Step 04 - Generation Cache Alignment

## Goal

Align the generation cache preparation flow with the upstream `v5.1.0` direction while preserving current MindOne model compatibility.

## Why Now

After shared loader and pipeline paths were stabilized, `generation/utils.py` still had a high-impact runtime delta around assisted generation and cache initialization.

## Runnable-Path Blocker

- minimal `generate()`
- assisted/contrastive generation cache selection

## Scope Classification

- classification: medium refactor
- runbook required: no

## Files

- `mindone/transformers/generation/utils.py`

## Changes

- split static cache setup into `_prepare_static_cache(...)`
- kept `_get_cache(...)` as a compatibility wrapper for model code that still calls it directly
- aligned assisted/contrastive generation to force `cache_implementation="dynamic_full"`
- kept explicit MindOne errors for unsupported `offloaded` and `quantized` cache implementations
- preserved current default `cache_implementation=None` behavior so legacy tuple-cache models such as GPT-2 continue to generate correctly

## Unsupported Paths Intentionally Skipped

- full default `DynamicCache` adoption for all models, because current MindOne GPT-2 still expects legacy tuple-style cache values in its forward path
- offloaded dynamic cache and quantized cache, which remain unsupported in MindSpore in this codebase

## Upstream Alignment Check

- followed upstream `v5.1.0` cache selection structure for static cache naming and assisted/contrastive `dynamic_full`
- intentionally retained MindOne-specific default behavior after runtime validation showed broad default `DynamicCache` breaks GPT-2

## Verification

- local `python -m py_compile mindone/transformers/generation/utils.py` passed
- remote `python -m py_compile mindone/transformers/generation/utils.py` passed
- remote local GPT-2 generation smoke passed:
  `GPT2LMHeadModel(GPT2Config(...)).generate(input_ids, max_new_tokens=2, do_sample=False)` returned shape `(1, 5)`
- final remote touched-file compile plus `vitpose_backbone`, local `BertModel` reload, and local GPT-2 generation smoke passed in one validation pass

## Threshold Touched

- touched: no
- rationale: n/a

## Current Conclusion

Generation cache setup now follows the target-version behavior where MindOne supports it, while avoiding a confirmed regression in legacy tuple-cache model implementations.

## Next Step

No code-blocking next step remains. Re-run hub-backed `from_pretrained` only after the remote mirror/network timeout is resolved.

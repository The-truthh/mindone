# Step 07 - Phimoe Partial Alignment

## Goal

Address the Phimoe model-level drift exposed by the broader model UT sweep against the installed upstream `transformers==5.1.0` baseline.

## Runnable-Path Blocker

- `tests/transformers_tests/models/phimoe/test_modeling_phimoe.py`

## Scope Classification

- classification: medium model patch
- runbook required: no

## Files

- `mindone/transformers/models/phimoe/modeling_phimoe.py`

## Changes

- aligned decoder block normalization with upstream `v5.1.0` by replacing Phimoe RMSNorm usage with LayerNorm semantics carrying `weight` and `bias`
- aligned `PhimoeTopKRouter` with upstream by passing `top_k=config.num_experts_per_tok` to `sparsemixer(...)`
- removed the post-router scatter so the router output shape matches the expert dispatch path used upstream

## Verification

- local `py_compile` passed
- remote `py_compile` passed
- remote Phimoe fp32 and fp16 parameterized tests passed
- original fp32 mismatch improved from `0.08733629440913437 > 0.0005` to passing

## Remaining Gap

- remote Phimoe bf16 case still fails narrowly: `0.00606036006018824 > 0.005`
- the remaining gap is now a precision residual, not the original structural LayerNorm/router mismatch
- `rotary_emb.inv_freq` remains a MindOne-only non-loaded parameter warning; current forward recomputes rope frequencies, so this warning has not yet been proven to cause the bf16 residual

## Threshold Touched

- touched: no
- rationale: threshold changes are intentionally deferred until the residual is localized further or accepted as bf16 tolerance drift

## Current Conclusion

The stable fp32 blocker is fixed. A small bf16 residual remains and should be diagnosed separately before changing thresholds or adding broader precision workarounds.

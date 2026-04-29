# Step 08 - Zamba2 Layer Construction

## Goal

Align MindOne Zamba2 layer construction with upstream `v5.1.0` so current upstream checkpoint keys load into the MindSpore model layout.

## Runnable-Path Blocker

- `tests/transformers_tests/models/zamba2/test_modeling_zamba2.py`

## Scope Classification

- classification: medium model patch
- runbook required: no

## Files

- `mindone/transformers/models/zamba2/modeling_zamba2.py`

## Changes

- replaced the old prebuilt/cycled shared transformer block construction with upstream-style per-layer construction in `get_layers()`
- recorded tied shared-transformer prefixes in `_tied_weights_keys`
- constructed each hybrid layer with its own `shared_transformer`, linear projection, and matching mamba layer
- preserved MindOne `nn.CellList` usage and MindSpore layer types

## Verification

- local `py_compile` passed
- remote `py_compile` passed
- remote `tests/transformers_tests/models/zamba2/test_modeling_zamba2.py` passed

## Threshold Touched

- touched: no
- rationale: structural mismatch fixed without changing numerical thresholds

## Current Conclusion

Zamba2 now matches the upstream `v5.1.0` layer/key layout closely enough for the targeted model test to pass.

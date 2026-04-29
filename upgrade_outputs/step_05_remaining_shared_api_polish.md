# Step 05 - Remaining Shared API Polish

## Goal

Land the remaining low-risk `v5.1.0` shared API deltas that affect current MindOne files without pulling in unsupported PyTorch-only or absent-model mappings.

## Runnable-Path Blocker

- processor fallback coverage
- image annotation API cleanup
- bidirectional sliding-window mask availability

## Scope Classification

- classification: small patch
- runbook required: no

## Files

- `mindone/transformers/models/auto/processing_auto.py`
- `mindone/transformers/image_utils.py`
- `mindone/transformers/masking_utils.py`

## Changes

- added `AutoVideoProcessor` to the `AutoProcessor` fallback chain
- changed the fallback chain to iterate over tokenizer, image processor, video processor, and feature extractor consistently
- removed the misspelled `AnnotionFormat` compatibility alias removed upstream
- added bidirectional sliding-window mask helpers and `create_bidirectional_sliding_window_mask(...)`
- updated SDPA bidirectional skip logic so local sliding-window attention does not incorrectly skip mask creation

## Deferred or Not Followed

- new auto mappings for models absent from MindOne were not added, because they would create broken lazy imports without the corresponding MindSpore model packages
- PyTorch-only integration changes and large batch model file changes were not applied directly
- `core_model_loading.py` upstream refactor was not copied because MindOne keeps a local minimal MindSpore loading layer and the active loader path was already adapted through `modeling_utils.py`

## Verification

- local `py_compile` passed for the three touched files
- remote `py_compile` passed for the three touched files
- remote `create_bidirectional_sliding_window_mask(...)` smoke passed and returned shape `(1, 1, 4, 4)`

## Threshold Touched

- touched: no
- rationale: n/a

## Current Conclusion

The remaining low-risk shared API deltas are now aligned where MindOne has active corresponding functionality.

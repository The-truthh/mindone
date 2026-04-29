# Step 01 - Shared API Surface Compatibility

## Goal

Patch the low-risk shared-component gaps required by the `v5.0.0 -> v5.1.0` upgrade without broadening into model-specific sync.

## Why Now

These changes unblock top-level exports, processor chat-template compatibility, and backbone entrypoints before touching the heavier loader stack.

## Runnable-Path Blocker

- `from_pretrained()` / auto entry import surface
- backbone public exports used by auto/backbone models

## Scope Classification

- classification: small patch
- runbook required: no

## Files

- `mindone/transformers/__init__.py`
- `mindone/transformers/backbone_utils.py`
- `mindone/transformers/processing_utils.py`
- `mindone/transformers/video_utils.py`
- `mindone/transformers/utils/generic.py`
- `mindone/transformers/utils/loading_report.py`

## Changes

- bumped `mindone.transformers.__version__` to `5.1.0`
- added top-level `backbone_utils.py` to mirror the upstream public import path
- exported `BackboneConfigMixin` and `BackboneMixin` from package root
- aligned processor chat-template kwarg routing with upstream `v5.1.0` behavior
- fixed batched-video normalization to preserve already-batched inputs
- tightened `to_py_obj()` list/tuple numeric conversion behavior
- added a local `utils.loading_report` helper for later loader-path compatibility work

## Unsupported Paths Intentionally Skipped

- torch-only quantization / accelerate / distributed loader branches
- full `core_model_loading.py` and `modeling_utils.py` API reshaping
- model-specific migration for target-version-added models

## Upstream Alignment Check

- aligned against upstream diffs for `__init__.py`, `processing_utils.py`, `video_utils.py`, `utils/generic.py`, and the new `utils/loading_report.py`
- used MindOne-specific adaptation for backbone exports by reusing existing `utils/backbone_utils.py`

## Verification

- local `python -m py_compile` passed for all touched files
- remote `python -m py_compile` passed in `/home/junyuan/DevPy/transformers-upgrade/mindone`
- remote `pytest -q tests/transformers_tests/models/vitpose_backbone/test_modeling_vitpose_backbone.py -k test_named_modules`
  passed
- remote selected `auto` test invocation did not expose shared import regressions; one selected case was skipped by test markers
- remote real `AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")` could not complete because the server timed out reaching `https://hf-mirror.com`, so loader-path validation remains network-blocked

## Threshold Touched

- touched: no
- rationale: n/a

## Current Conclusion

The low-risk shared API surface required by `v5.1.0` is in place and validated structurally. Remaining work is concentrated in broader shared-component completeness and loader-path parity, not in the patched surface above.

## Next Step

Decide whether to continue with `core_model_loading.py` / `modeling_utils.py` compatibility tightening now, or stop at the currently validated shared-surface upgrade slice.

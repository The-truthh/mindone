# Step 03 - Runtime Polish

## Goal

Close the remaining small shared-runtime gap that still affects real inference paths after the loader and public API surface were stabilized.

## Why Now

The remaining high-signal runtime delta was the `image_text_to_text` pipeline chat path not explicitly casting processor outputs to the pipeline dtype.

## Runnable-Path Blocker

- multimodal pipeline inference dtype consistency

## Scope Classification

- classification: small patch
- runbook required: no

## Files

- `mindone/transformers/pipelines/image_text_to_text.py`

## Changes

- aligned chat-template pipeline input preparation with upstream `v5.1.0` by applying `.to(dtype=self.dtype)` to processor outputs before dispatch

## Unsupported Paths Intentionally Skipped

- broader pipeline refactors outside the touched chat-input path

## Upstream Alignment Check

- matched the upstream `v5.1.0` pipeline change for `image_text_to_text.py`

## Verification

- local `python -m py_compile mindone/transformers/pipelines/image_text_to_text.py` passed
- remote `python -m py_compile mindone/transformers/pipelines/image_text_to_text.py` passed

## Threshold Touched

- touched: no
- rationale: n/a

## Current Conclusion

The remaining small pipeline runtime delta has been closed.

## Next Step

At this point, remaining untouched upstream shared diffs are predominantly torch-only branches, documentation churn, or low-value compatibility cleanup rather than current MindOne runtime blockers.

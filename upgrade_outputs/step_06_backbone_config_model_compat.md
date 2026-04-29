# Step 06 - Backbone Config Model Compatibility

## Goal

Fix model-level compatibility gaps exposed by the broader `tests/transformers_tests/models` sweep after the shared `v5.1.0` upgrade.

## Runnable-Path Blocker

- model initialization with upstream `v5.1.0` configs that no longer expose legacy backbone kwargs as direct attributes

## Scope Classification

- classification: small patch
- runbook required: no

## Files

- `mindone/transformers/models/dpt/modeling_dpt.py`
- `mindone/transformers/models/grounding_dino/modeling_grounding_dino.py`
- `mindone/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py`

## Changes

- updated `DPTForDepthEstimation` to stop reading removed `config.backbone`
- aligned DPT hidden-size helper with upstream by checking for `backbone_config.hidden_size`
- updated `GroundingDinoConvEncoder` to stop reading removed `config.use_timm_backbone` and `config.backbone`
- simplified GroundingDINO backbone loading to the MindOne-supported `load_backbone(config)` path
- made GroundingDINO feature extraction use backbone `return_dict=True` and `feature_maps`
- applied the same backbone config compatibility update to `MMGroundingDinoConvEncoder`

## Deferred or Not Followed

- timm-backed backbone loading remains unsupported in MindOne and was not reintroduced
- full upstream configuration refactors were not copied wholesale because MindOne uses upstream config classes from the installed `transformers` package in these tests

## Verification

- local `py_compile` passed for all touched model files
- remote `py_compile` passed for all touched model files
- remote `tests/transformers_tests/models/dpt/test_modeling_dpt.py` passed
- remote `tests/transformers_tests/models/grounding_dino/test_modeling_grounding_dino.py` passed
- remote `tests/transformers_tests/models/mm_grounding_dino/test_modeling_mm_grounding_dino.py` passed
- broader model UT sweep progressed past both previous blockers

## Threshold Touched

- touched: no
- rationale: n/a

## Current Conclusion

The first stable model-level blockers caused by upstream backbone config API changes are fixed.

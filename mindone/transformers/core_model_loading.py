# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, MindONE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Core helpers for loading model checkpoints (MindSpore version).

This module provides v5.0.0 compatible APIs for model weight loading,
adapted for MindSpore backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import mindspore as ms
from mindspore import nn


logger = logging.getLogger(__name__)


@dataclass
class WeightRenaming:
    """
    Configuration for weight renaming during checkpoint loading.

    Args:
        source_patterns: List of source patterns to match (can use * as wildcard)
        target_patterns: List of target patterns to replace with
    """
    source_patterns: list[str]
    target_patterns: list[str]

    def __repr__(self):
        return f"WeightRenaming({self.source_patterns} -> {self.target_patterns})"


@dataclass
class WeightConverter:
    """
    Weight converter with optional transformation operation.

    Args:
        renaming: WeightRenaming configuration
        op: Optional conversion operation (not implemented in MindONE minimal version)
    """
    renaming: WeightRenaming
    op: Any | None = None

    def __repr__(self):
        return f"WeightConverter({self.renaming}, op={self.op})"


@dataclass
class LoadStateDictConfig:
    """
    Config for loading weights. This allows bundling arguments that are just
    passed around.

    Adapted from transformers v5.0.0 for MindSpore compatibility.
    """
    pretrained_model_name_or_path: str | None = None
    use_safetensors: bool = True
    ignore_mismatched_sizes: bool = False
    sharded_metadata: dict | None = None
    device_map: dict | None = None
    disk_offload_folder: str | None = None
    offload_buffers: bool = False
    dtype: ms.Type | None = None
    weights_only: bool = True
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None

    @property
    def is_quantized(self) -> bool:
        """Check if quantization is enabled (MindONE: always False for minimal version)."""
        return False


def _apply_weight_mapping(
    state_dict: dict[str, ms.Tensor],
    weight_mapping: list[WeightConverter | WeightRenaming] | None,
) -> dict[str, ms.Tensor]:
    """
    Apply weight mapping to state_dict.

    Args:
        state_dict: Original state dict
        weight_mapping: List of WeightConverter or WeightRenaming to apply

    Returns:
        Modified state dict with renamed keys
    """
    if weight_mapping is None or len(weight_mapping) == 0:
        return state_dict

    new_state_dict = {}
    mapping_applied = {}

    for key, value in state_dict.items():
        new_key = key
        for mapping in weight_mapping:
            if isinstance(mapping, WeightConverter):
                renaming = mapping.renaming
            elif isinstance(mapping, WeightRenaming):
                renaming = mapping
            else:
                continue

            # Simple wildcard matching (replace * with .* for regex-like behavior)
            for src_pattern, tgt_pattern in zip(renaming.source_patterns, renaming.target_patterns):
                # Convert simple glob pattern to key matching
                if _match_pattern(key, src_pattern):
                    new_key = _apply_pattern(key, src_pattern, tgt_pattern)
                    break

            if new_key != key:
                break

        new_state_dict[new_key] = value
        mapping_applied[key] = new_key

    # Log applied mappings
    changed = {k: v for k, v in mapping_applied.items() if k != v}
    if changed:
        logger.info(f"Applied weight mapping to {len(changed)} keys")
        for old, new in list(changed.items())[:5]:  # Log first 5
            logger.info(f"  {old} -> {new}")
        if len(changed) > 5:
            logger.info(f"  ... and {len(changed) - 5} more")

    return new_state_dict


def _match_pattern(key: str, pattern: str) -> bool:
    """Match a key against a pattern (supports * as wildcard)."""
    import fnmatch
    return fnmatch.fnmatch(key, pattern)


def _apply_pattern(key: str, source_pattern: str, target_pattern: str) -> str:
    """
    Apply pattern transformation from source to target.

    This is a simplified implementation that handles basic wildcard replacement.
    """
    import re

    # Handle simple prefix/suffix replacement with *
    if source_pattern == target_pattern:
        return key

    # Convert glob pattern to regex for more complex matching
    # Replace * with (.*) to capture variable parts
    source_regex = "^" + source_pattern.replace(".", r"\.").replace("*", "(.+)") + "$"
    match = re.match(source_regex, key)

    if match:
        # Apply captured groups to target pattern
        result = target_pattern
        for i, group in enumerate(match.groups(), 1):
            result = result.replace(f"{{{i-1}}}", group)  # {0}, {1}, etc.
            # Also support * as placeholder for the first capture
            if i == 1 and "*" in result:
                result = result.replace("*", group, 1)
        return result

    return key


def convert_and_load_state_dict_in_model(
    model: nn.Cell,
    state_dict: dict[str, ms.Tensor],
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None,
    ignore_mismatched_sizes: bool = False,
    start_prefix: str = "",
    is_sharded: bool = False,
    **kwargs,
) -> tuple[set[str], set[str], list[tuple[str, tuple]], dict | None, set[str]]:
    """
    Load state_dict into model with optional weight mapping.

    This is the v5.0.0 compatible entry point for loading weights into models.
    It serves as a unified loading interface and delegates to MindONE's
    existing loading logic.

    Args:
        model: The MindSpore model to load weights into
        state_dict: The state dictionary containing weights
        weight_mapping: Optional list of weight mappings to apply
        ignore_mismatched_sizes: If True, ignore size mismatches
        start_prefix: Prefix to add to parameter names
        is_sharded: Whether the checkpoint is sharded
        **kwargs: Additional arguments (for API compatibility, may be ignored)

    Returns:
        Tuple of:
        - missing_keys: Set of keys expected by model but not in state_dict
        - unexpected_keys: Set of keys in state_dict but not expected by model
        - mismatched_keys: List of (key, checkpoint_shape, model_shape) tuples
        - disk_offload_index: Always None for MindONE minimal version
        - conversion_errors: Set of error messages during conversion
    """
    conversion_errors = set()
    disk_offload_index = None  # Not supported in MindONE minimal version

    # Track original keys before any transformation
    original_keys = set(state_dict.keys())

    # Step 1: Apply weight mapping if provided
    if weight_mapping is not None:
        try:
            state_dict = _apply_weight_mapping(state_dict, weight_mapping)
        except Exception as e:
            logger.error(f"Error applying weight mapping: {e}")
            conversion_errors.add(f"weight_mapping_error: {str(e)}")

    # Step 2: Get expected keys from model
    expected_keys = set(model.state_dict().keys())

    # Step 3: Handle mismatched sizes if requested
    mismatched_keys = []
    if ignore_mismatched_sizes:
        for key in list(state_dict.keys()):
            if key in expected_keys:
                model_shape = model.state_dict()[key].shape
                checkpoint_shape = state_dict[key].shape
                if checkpoint_shape != model_shape:
                    mismatched_keys.append((key, tuple(checkpoint_shape), tuple(model_shape)))
                    del state_dict[key]
                    logger.warning(
                        f"Skipping '{key}' due to size mismatch: "
                        f"checkpoint {checkpoint_shape} vs model {model_shape}"
                    )

    # Step 4: Load state dict into model (delegate to MindONE logic)
    # Import here to avoid circular dependency
    from .modeling_utils import _load_state_dict_into_model as _mindone_load

    try:
        _mindone_load(model, state_dict, start_prefix, is_sharded)
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        conversion_errors.add(f"load_error: {str(e)}")
        raise

    # Step 5: Calculate missing and unexpected keys
    loaded_keys = set(state_dict.keys())
    missing_keys = expected_keys - loaded_keys
    unexpected_keys = loaded_keys - expected_keys

    # Remove non-persistent buffers from unexpected keys
    model_buffers = {n for n, _ in model.named_buffers()}
    unexpected_keys = unexpected_keys - model_buffers

    # Log results
    if missing_keys:
        logger.warning(f"Missing keys: {sorted(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {sorted(unexpected_keys)}")
    if mismatched_keys:
        logger.warning(f"Mismatched keys: {len(mismatched_keys)}")

    return (
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        disk_offload_index,
        conversion_errors,
    )


def revert_weight_conversion(
    model: nn.Cell,
    state_dict: dict[str, ms.Tensor],
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None,
) -> dict[str, ms.Tensor]:
    """
    Revert weight conversion before saving.

    This is a placeholder for v5.0.0 API compatibility.
    In the minimal MindONE implementation, this is a no-op.

    Args:
        model: The MindSpore model
        state_dict: State dictionary to revert
        weight_mapping: Optional weight mapping to reverse

    Returns:
        The state dictionary (unchanged in minimal implementation)
    """
    # TODO: Implement reverse weight conversion if needed
    # For now, return as-is since MindONE doesn't have complex conversion ops
    if weight_mapping is not None:
        logger.debug("revert_weight_conversion called but not fully implemented")
    return state_dict


def load_state_dict_and_config(
    checkpoint_file: str,
    config: LoadStateDictConfig | None = None,
) -> tuple[dict[str, ms.Tensor], LoadStateDictConfig]:
    """
    Load state dict from checkpoint file with config.

    Args:
        checkpoint_file: Path to checkpoint file
        config: Optional LoadStateDictConfig (created if None)

    Returns:
        Tuple of (state_dict, config)
    """
    if config is None:
        config = LoadStateDictConfig()

    # Import here to avoid circular dependency
    from .modeling_utils import load_state_dict as _mindone_load_state_dict

    state_dict = _mindone_load_state_dict(checkpoint_file)
    return state_dict, config

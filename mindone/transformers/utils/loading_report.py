# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""State-dict loading diagnostics used by v5.1-compatible loaders."""

from __future__ import annotations

import logging
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")
_ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")

PALETTE = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "orange": "\033[38;5;208m",
    "purple": "\033[35m",
    "italic": "\033[3m",
}


def _pattern_of(key: str) -> str:
    return _DIGIT_RX.sub("*", key)


def _fmt_indices(values: list[int], cutoff: int = 10) -> str:
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    if len(values) > cutoff:
        return f"{values[0]}...{values[-1]}"
    return ", ".join(map(str, values))


def update_key_name(mapping: dict[str, Any] | set[str]) -> OrderedDict:
    not_mapping = not isinstance(mapping, dict)
    if not_mapping:
        mapping = {k: k for k in mapping}

    bucket: dict[tuple[str, Any], list[set[int]]] = defaultdict(list)
    for key, value in mapping.items():
        digits = _DIGIT_RX.findall(key)
        pattern = _pattern_of(key)
        entry = bucket[(pattern, value)]
        while len(entry) < len(digits):
            entry.append(set())
        for idx, digit in enumerate(digits):
            entry[idx].add(int(digit))

    out = OrderedDict()
    for (pattern, value), sets in bucket.items():
        parts = pattern.split("*")
        merged = parts[0]
        for idx, suffix in enumerate(parts[1:]):
            if idx < len(sets) and sets[idx]:
                insert = _fmt_indices(sorted(sets[idx]))
                merged += "{" + insert + "}" if len(sets[idx]) > 1 else insert
            else:
                merged += "*"
            merged += suffix
        out[merged] = value

    return OrderedDict((k, k) for k in out) if not_mapping else out


def _strip_ansi(value: str) -> str:
    return _ANSI_RX.sub("", str(value))


def _pad(text: str, width: int) -> str:
    pad = max(0, width - len(_strip_ansi(text)))
    return f"{text}{' ' * pad}"


def _make_table(rows: list[list[str]], headers: list[str]) -> str:
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(_strip_ansi(cell)) for cell in col) for col in cols]
    header_line = " | ".join(_pad(h, w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    body = [" | ".join(_pad(cell, w) for cell, w in zip(row, widths)) for row in rows]
    return "\n".join([header_line, sep_line] + body)


def _color(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{PALETTE[color]}{text}{PALETTE['reset']}"
    return text


def _get_terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


@dataclass
class LoadStateDictInfo:
    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int, ...], tuple[int, ...]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]

    def missing_and_mismatched(self) -> set[str]:
        return self.missing_keys | {name for name, _, _ in self.mismatched_keys}

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "mismatched_keys": self.mismatched_keys,
            "error_msgs": self.error_msgs,
        }

    def create_loading_report(self) -> str | None:
        rows: list[list[str]] = []
        tips = ""

        if self.unexpected_keys:
            tips += (
                f"\n- {_color('UNEXPECTED', 'orange') + PALETTE['italic']}\t:can be ignored when loading from a different "
                "task or architecture."
            )
            for key in update_key_name(self.unexpected_keys):
                rows.append([key, _color("UNEXPECTED", "orange"), ""])

        if self.missing_keys:
            tips += (
                f"\n- {_color('MISSING', 'red') + PALETTE['italic']}\t:params were newly initialized because they were "
                "missing from the checkpoint."
            )
            for key in update_key_name(self.missing_keys):
                rows.append([key, _color("MISSING", "red"), ""])

        if self.mismatched_keys:
            tips += (
                f"\n- {_color('MISMATCH', 'yellow') + PALETTE['italic']}\t:checkpoint weights were skipped because their "
                "shapes did not match the model."
            )
            iterator = {name: (shape_ckpt, shape_model) for name, shape_ckpt, shape_model in self.mismatched_keys}
            for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
                rows.append(
                    [
                        key,
                        _color("MISMATCH", "yellow"),
                        f"Reinit due to size mismatch - ckpt: {shape_ckpt} vs model: {shape_model}",
                    ]
                )

        if self.conversion_errors:
            tips += f"\n- {_color('CONVERSION', 'purple') + PALETTE['italic']}\t:errors emitted by weight conversion."
            for key, details in update_key_name(self.conversion_errors).items():
                rows.append([key, _color("CONVERSION", "purple"), f"\n{details}\n"])

        if not rows:
            return None

        headers = ["Key", "Status", "Details"] if _get_terminal_width() > 200 else ["Key", "Status", ""]
        return _make_table(rows, headers) + f"\n\n{PALETTE['italic']}Notes:{tips}{PALETTE['reset']}"


def log_state_dict_report(
    model,
    pretrained_model_name_or_path: str,
    ignore_mismatched_sizes: bool,
    loading_info: LoadStateDictInfo,
    logger: logging.Logger | None = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    if loading_info.error_msgs:
        error_msg = "\n\t".join(loading_info.error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    report = loading_info.create_loading_report()
    if report is not None:
        logger.warning(
            "State-dict loading report for %s from %s\n%s",
            model.__class__.__name__,
            pretrained_model_name_or_path,
            report,
        )
    elif ignore_mismatched_sizes:
        logger.info("No remaining loading issues for %s after mismatch filtering.", model.__class__.__name__)

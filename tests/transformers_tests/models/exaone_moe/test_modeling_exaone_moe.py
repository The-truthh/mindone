"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/exaone_moe/test_modeling_exaone_moe.py."""

# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
import inspect

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.exaone_moe")
from transformers import ExaoneMoeConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

from ...causal_lm_tester import CausalLMModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class ExaoneMoeModelTester(CausalLMModelTester):
    config_class = ExaoneMoeConfig

    def __init__(self, parent=None):
        super().__init__(parent)


model_tester = ExaoneMoeModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "ExaoneMoeModel",
        "transformers.ExaoneMoeModel",
        "mindone.transformers.ExaoneMoeModel",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "ExaoneMoeForCausalLM",
        "transformers.ExaoneMoeForCausalLM",
        "mindone.transformers.ExaoneMoeForCausalLM",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "logits": "logits",
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in TEST_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    threshold = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < threshold).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {threshold}"
    )

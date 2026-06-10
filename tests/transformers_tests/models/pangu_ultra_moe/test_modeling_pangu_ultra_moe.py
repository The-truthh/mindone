# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Testing suite for the MindSpore PanguUltraMoE model."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import mindspore as ms

from mindone.transformers import PanguUltraMoEConfig
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy


DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]
REFERENCE_PACKAGE = "openpangu_ultra_moe_reference"


def _find_source_repo():
    env_path = os.environ.get("OPENPANGU_SOURCE_PATH")
    if env_path:
        return Path(env_path).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "openPangu-Ultra-MoE-718B-V1.1"
        if candidate.exists():
            return candidate
        candidate = parent.parent / "openPangu-Ultra-MoE-718B-V1.1"
        if candidate.exists():
            return candidate
    return None


def _load_reference_module(source_repo):
    package = types.ModuleType(REFERENCE_PACKAGE)
    package.__path__ = [str(source_repo)]
    package.__spec__ = importlib.util.spec_from_loader(REFERENCE_PACKAGE, loader=None, is_package=True)
    sys.modules[REFERENCE_PACKAGE] = package

    for module_name in ("configuration_openpangu_moe", "modeling_openpangu_moe"):
        full_name = f"{REFERENCE_PACKAGE}.{module_name}"
        if full_name in sys.modules:
            continue
        module_path = source_repo / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    _patch_reference_initialization(sys.modules[f"{REFERENCE_PACKAGE}.modeling_openpangu_moe"])


def _patch_reference_initialization(modeling_module):
    original_init_weights = modeling_module.PanguUltraMoEPreTrainedModel._init_weights

    def _init_weights(self, module):
        original_init_weights(self, module)
        if isinstance(module, modeling_module.MoEGate):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, modeling_module.PanguUltraMoERMSNorm):
            module.weight.data.fill_(1.0)

    modeling_module.PanguUltraMoEPreTrainedModel._init_weights = _init_weights

    if not hasattr(modeling_module.DynamicCache, "get_usable_length"):

        def get_usable_length(self, new_seq_length=None, layer_idx=0):
            return self.get_seq_length(layer_idx)

        modeling_module.DynamicCache.get_usable_length = get_usable_length


SOURCE_REPO = _find_source_repo()
pytestmark = pytest.mark.skipif(
    SOURCE_REPO is None,
    reason="openPangu-Ultra-MoE-718B-V1.1 source repo is required as PyTorch reference",
)
if SOURCE_REPO is not None:
    _load_reference_module(SOURCE_REPO)


class PanguUltraMoEModelTester:
    def __init__(
        self,
        batch_size=3,
        seq_length=2,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_shared_experts=1,
        num_routed_experts=4,
        routed_scaling_factor=1.0,
        attention_kv_lora_dim=4,
        attention_q_lora_dim=4,
        attention_qk_rope_dim=4,
        attention_v_dim=8,
        attention_qk_dim=8,
        num_experts_per_tok=1,
        num_dense_layers=1,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.attention_kv_lora_dim = attention_kv_lora_dim
        self.attention_q_lora_dim = attention_q_lora_dim
        self.attention_qk_rope_dim = attention_qk_rope_dim
        self.attention_v_dim = attention_v_dim
        self.attention_qk_dim = attention_qk_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_dense_layers = num_dense_layers
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def get_config(self):
        return PanguUltraMoEConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            attention_kv_lora_dim=self.attention_kv_lora_dim,
            attention_q_lora_dim=self.attention_q_lora_dim,
            attention_qk_rope_dim=self.attention_qk_rope_dim,
            attention_v_dim=self.attention_v_dim,
            attention_qk_dim=self.attention_qk_dim,
            num_experts_per_tok=self.num_experts_per_tok,
            num_dense_layers=self.num_dense_layers,
            norm_topk_prob=self.norm_topk_prob,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=False,
            sandwich_norm=True,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = np.ones_like(input_ids)
        return self.get_config(), input_ids, input_mask


model_tester = PanguUltraMoEModelTester()
config, input_ids, input_mask = model_tester.prepare_config_and_inputs_for_common()

PANGU_ULTRA_MOE_CASES = [
    [
        "PanguUltraMoEModel",
        f"{REFERENCE_PACKAGE}.modeling_openpangu_moe.PanguUltraMoEModel",
        "mindone.transformers.PanguUltraMoEModel",
        (config,),
        {},
        (input_ids,),
        {"attention_mask": input_mask, "use_cache": False},
        {"last_hidden_state": 0},
    ],
    [
        "PanguUltraMoEForCausalLM",
        f"{REFERENCE_PACKAGE}.modeling_openpangu_moe.PanguUltraMoEForCausalLM",
        "mindone.transformers.PanguUltraMoEForCausalLM",
        (config,),
        {},
        (input_ids,),
        {"attention_mask": input_mask, "use_cache": False},
        {"logits": "logits"},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case + [dtype] + [mode]
        for case in PANGU_ULTRA_MOE_CASES
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

    pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    pt_outputs_n = []
    ms_outputs_n = []
    for pt_key, ms_idx in outputs_map.items():
        pt_outputs_n.append(getattr(pt_outputs, pt_key))
        ms_outputs_n.append(getattr(ms_outputs, ms_idx) if isinstance(ms_idx, str) else ms_outputs[ms_idx])
    diffs = compute_diffs(pt_outputs_n, ms_outputs_n)

    threshold = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < threshold).all(), (
        f"{name} ms_dtype: {ms_dtype}, pt_type: {pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {threshold}"
    )


@pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
def test_causal_lm_cache_prefill_decode(dtype):
    ms.set_context(mode=1)

    pt_model, ms_model, pt_dtype, ms_dtype = get_modules(
        f"{REFERENCE_PACKAGE}.modeling_openpangu_moe.PanguUltraMoEForCausalLM",
        "mindone.transformers.PanguUltraMoEForCausalLM",
        dtype,
        config,
    )
    input_ids_pair = input_ids[:, :2]
    prefill_ids = input_ids_pair[:, :1]
    decode_ids = input_ids_pair[:, 1:]
    prefill_mask = np.ones_like(prefill_ids)
    decode_mask = np.ones_like(input_ids_pair)

    with torch.no_grad():
        pt_prefill = pt_model(
            input_ids=torch.tensor(prefill_ids),
            attention_mask=torch.tensor(prefill_mask),
            use_cache=True,
        )
        pt_decode = pt_model(
            input_ids=torch.tensor(decode_ids),
            attention_mask=torch.tensor(decode_mask),
            past_key_values=pt_prefill.past_key_values,
            use_cache=True,
        )

    ms_prefill = ms_model(
        input_ids=ms.Tensor(prefill_ids, ms.int64),
        attention_mask=ms.Tensor(prefill_mask, ms.int64),
        use_cache=True,
    )
    ms_decode = ms_model(
        input_ids=ms.Tensor(decode_ids, ms.int64),
        attention_mask=ms.Tensor(decode_mask, ms.int64),
        past_key_values=ms_prefill.past_key_values,
        use_cache=True,
    )

    diffs = compute_diffs([pt_decode.logits], [ms_decode.logits])
    threshold = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < threshold).all(), (
        f"PanguUltraMoEForCausalLM cache ms_dtype: {ms_dtype}, pt_type: {pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {threshold}"
    )

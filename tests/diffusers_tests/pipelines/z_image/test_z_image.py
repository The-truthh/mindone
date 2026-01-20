# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import Qwen3Config

import mindspore as ms
from mindspore import mint

from mindone.diffusers import ZImagePipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class ZImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel",
            dict(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=32,
                n_layers=2,
                n_refiner_layers=1,
                n_heads=2,
                n_kv_heads=2,
                norm_eps=1e-5,
                qk_norm=True,
                cap_feat_dim=16,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[8, 4, 4],
                axes_lens=[256, 32, 32],
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[32, 64],
                layers_per_block=1,
                latent_channels=16,
                norm_num_groups=32,
                sample_size=32,
                scaling_factor=0.3611,
                shift_factor=0.1159,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.qwen3.modeling_qwen3.Qwen3Model",
            "mindone.transformers.models.qwen3.modeling_qwen3.Qwen3Model",
            dict(
                config=Qwen3Config(
                    hidden_size=16,
                    intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    vocab_size=151936,
                    max_position_embeddings=512,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
            ]
        }
        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        # `x_pad_token` and `cap_pad_token` are initialized with `torch.empty`.
        # This can cause NaN data values in our testing environment. Fixating them
        # helps prevent that issue.
        with torch.no_grad():
            pt_components["transformer"].x_pad_token.copy_(
                torch.ones_like(pt_components["transformer"].x_pad_token.data)
            )
            pt_components["transformer"].cap_pad_token.copy_(
                torch.ones_like(pt_components["transformer"].cap_pad_token.data)
            )

        with ms._no_grad():
            ms_components["transformer"].x_pad_token.copy_(
                mint.ones_like(ms_components["transformer"].x_pad_token.data)
            )
            ms_components["transformer"].cap_pad_token.copy_(
                mint.ones_like(ms_components["transformer"].cap_pad_token.data)
            )

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.z_image.ZImagePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.z_image.ZImagePipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_image_slice = pt_frame.images[0, -3:, -3:, -1]
        ms_image_slice = ms_frame[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class ZImagePipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("FP16 runs black results on both torch and mindspore")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", mindspore_dtype=ms_dtype)

        # Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in MindSpore.
        # (1) Use flash attention 2
        # pipe.transformer.set_attention_backend("flash")
        # (2) Use flash attention 3
        # pipe.transformer.set_attention_backend("_flash_3")

        prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        torch.manual_seed(0)
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
        )[
            0
        ][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"z_image_t2i_{dtype}.npy",
            subfolder="z_image",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

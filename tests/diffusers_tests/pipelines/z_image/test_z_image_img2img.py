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
import torch
from ddt import data, ddt, unpack
from transformers import Qwen3Config

import mindspore as ms

from mindone.diffusers import ZImageImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class ZImageImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        import random

        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        ms_image = ms.tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "image": pt_image,
            "strength": 0.6,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "image": ms_image,
            "strength": 0.6,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.z_image.ZImageImg2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.z_image.ZImageImg2ImgPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16

        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class ZImageImg2ImgPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", mindspore_dtype=ms_dtype)

        init_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "sketch-mountains-input.png",
            subfolder="stable_diffusion_img2img",
        ).resize((1024, 1024))

        prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"
        torch.manual_seed(0)
        image = pipe(
            prompt,
            image=init_image,
            strength=0.6,
            num_inference_steps=9,
            guidance_scale=0.0,
        )[
            0
        ][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"z_image_i2i_{dtype}.npy",
            subfolder="z_image",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

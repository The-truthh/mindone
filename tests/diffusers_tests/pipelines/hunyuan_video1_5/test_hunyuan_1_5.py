# Copyright 2025 The HuggingFace Team.
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
from PIL import Image
import pytest
from transformers import Qwen2_5_VLTextConfig

import mindspore as ms

from mindone.diffusers import HunyuanVideo15Pipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


# due to text_encoder has precision issue in bf16, improve the threshold.
THRESHOLD_PIXEL = 30.0


@ddt
class HunyuanVideo15PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_hunyuan_video15.HunyuanVideo15Transformer3DModel",
            "mindone.diffusers.models.transformers.transformer_hunyuan_video15.HunyuanVideo15Transformer3DModel",
            dict(
                in_channels=9,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=8,
                num_layers=1,
                num_refiner_layers=1,
                mlp_ratio=2.0,
                patch_size=1,
                patch_size_t=1,
                text_embed_dim=16,
                text_embed_2_dim=32,
                image_embed_dim=12,
                rope_axes_dim=(2, 2, 4),
                target_size=16,
                task_type="t2v",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15.AutoencoderKLHunyuanVideo15",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15.AutoencoderKLHunyuanVideo15",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                block_out_channels=(16, 16),
                layers_per_block=1,
                spatial_compression_ratio=4,
                temporal_compression_ratio=2,
                downsample_match_channel=False,
                upsample_match_channel=False,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(shift=7.0),
        ],
        [
            "text_encoder",
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel",
            "mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel",
            dict(
                config=Qwen2_5_VLTextConfig(
                    **{
                        "hidden_size": 16,
                        "intermediate_size": 16,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "rope_scaling": {
                            "mrope_section": [1, 1, 2],
                            "rope_type": "default",
                            "type": "default",
                        },
                        "rope_theta": 1000000.0,
                    }
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
        [
            "text_encoder_2",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.byt5.tokenization_byt5.ByT5Tokenizer",
            "transformers.models.byt5.tokenization_byt5.ByT5Tokenizer",
            dict(),
        ],
        [
            "guider",
            "diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance",
            "mindone.diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance",
            dict(guidance_scale=1.0),
        ],
    ]

    def get_dummy_components(self, num_layers: int = 1):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "guider",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["transformer", "vae", "text_encoder", "text_encoder_2"]
        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "monkey",
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("HunyuanVideo15Pipeline does not support float16 on CPU in PyTorch")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.hunyuan_video1_5.HunyuanVideo15Pipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.hunyuan_video1_5.HunyuanVideo15Pipeline")

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

        pt_image_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_image_slice = ms_frame.frames[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class HunyuanVideo15PipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline will OOM in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
        pipe = HunyuanVideo15Pipeline.from_pretrained(model_id, mindspore_dtype=ms_dtype)
        pipe.vae.enable_tiling()

        torch.manual_seed(0)
        image = pipe(
            prompt="A cat walks on the grass, realistic",
            num_inference_steps=50,
        ).frames[
            0
        ][1]
        image = Image.fromarray((image * 255).astype("uint8"))

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"hunyuan_video1_5_t2v_{dtype}.npy",
            subfolder="hunyuan_video1_5",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

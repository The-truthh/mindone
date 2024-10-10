import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers.models.attention_processor import AttnAddedKVProcessor as PTAttnAddedKVProcessor
from mindone.diffusers.models.attention_processor import AttnAddedKVProcessor
from mindone.diffusers.pipelines.deepfloyd_if import IFWatermarker

# WARN: the hf-internal-testing/tiny-random-t5 text encoder has some non-determinism in the `save_load` tests.

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    get_module,
    get_pipeline_components,
    randn_tensor,
    floats_tensor,
)


class IFPipelineTesterMixin:
    pipeline_config = [
        [
            "text_encoder",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                sample_size=32,
                layers_per_block=1,
                block_out_channels=[32, 64],
                down_block_types=[
                    "ResnetDownsampleBlock2D",
                    "SimpleCrossAttnDownBlock2D",
                ],
                mid_block_type="UNetMidBlock2DSimpleCrossAttn",
                up_block_types=["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
                in_channels=3,
                out_channels=6,
                cross_attention_dim=32,
                encoder_hid_dim=32,
                attention_head_dim=8,
                addition_embed_type="text",
                addition_embed_type_num_heads=2,
                cross_attention_norm="group_norm",
                resnet_time_scale_shift="scale_shift",
                act_fn="gelu",
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                beta_start=0.0001,
                beta_end=0.02,
                thresholding=True,
                dynamic_thresholding_ratio=0.95,
                sample_max_value=1.0,
                prediction_type="epsilon",
                variance_type="learned_range",
            ),
        ],
        [
            "watermarker",
            "diffusers.pipelines.deepfloyd_if.IFWatermarker",
            "mindone.diffusers.pipelines.deepfloyd_if.IFWatermarker",
            dict(),
        ],
    ]

    pipeline_superresolution_config = [
        [
            "text_encoder",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                sample_size=32,
                layers_per_block=[1, 2],
                block_out_channels=[32, 64],
                down_block_types=[
                    "ResnetDownsampleBlock2D",
                    "SimpleCrossAttnDownBlock2D",
                ],
                mid_block_type="UNetMidBlock2DSimpleCrossAttn",
                up_block_types=["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
                in_channels=6,
                out_channels=6,
                cross_attention_dim=32,
                encoder_hid_dim=32,
                attention_head_dim=8,
                addition_embed_type="text",
                addition_embed_type_num_heads=2,
                cross_attention_norm="group_norm",
                resnet_time_scale_shift="scale_shift",
                act_fn="gelu",
                class_embed_type="timestep",
                mid_block_scale_factor=1.414,
                time_embedding_act_fn="gelu",
                time_embedding_dim=32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                beta_start=0.0001,
                beta_end=0.02,
                thresholding=True,
                dynamic_thresholding_ratio=0.95,
                sample_max_value=1.0,
                prediction_type="epsilon",
                variance_type="learned_range",
            ),
        ],
        [
            "image_noising_scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                beta_start=0.0001,
                beta_end=0.02,
            ),
        ],
        [
            "watermarker",
            "diffusers.pipelines.deepfloyd_if.IFWatermarker",
            "mindone.diffusers.pipelines.deepfloyd_if.IFWatermarker",
            dict(),
        ],
    ]

    def _get_dummy_components(self):
        components = {
            key: None
            for key in [
                "text_encoder",
                "tokenizer",
                "unet",
                "scheduler",
                "watermarker",
                "safety_checker",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        # For reproducibility tests
        pt_components["unet"] = pt_components["unet"].set_attn_processor(PTAttnAddedKVProcessor())
        ms_components["unet"] = ms_components["unet"].set_attn_processor(AttnAddedKVProcessor())

        return pt_components, ms_components

    def _get_superresolution_dummy_components(self):
        components = {
            key: None
            for key in [
                "text_encoder",
                "tokenizer",
                "unet",
                "scheduler",
                "image_noising_scheduler"
                "watermarker",
                "safety_checker",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_superresolution_config)
        # For reproducibility tests
        pt_components["unet"] = pt_components["unet"].set_attn_processor(PTAttnAddedKVProcessor())
        ms_components["unet"] = ms_components["unet"].set_attn_processor(AttnAddedKVProcessor())

        return pt_components, ms_components

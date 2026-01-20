import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import OvisImagePipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@slow
@ddt
class OvisImagePipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = OvisImagePipeline.from_pretrained("AIDC-AI/Ovis-Image-7B", mindspore_dtype=ms_dtype)
        prompt = 'A creative 3D artistic render where the text "OVIS-IMAGE" is written in a bold, expressive handwritten brush style using thick, wet oil paint. The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste or impasto art. You can see the ridges of the brush bristles and the glossy, wet texture of the paint. The background is a clean artist\'s canvas. Dynamic lighting creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile texture, 4k detail.'  # noqa: E501

        torch.manual_seed(0)
        image = pipe(prompt, negative_prompt="", num_inference_steps=50, guidance_scale=5.0)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"ovis_image_t2i_{dtype}.npy",
            subfolder="ovis_image",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

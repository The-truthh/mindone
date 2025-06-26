from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

import mindspore as ms

from mindone.diffusers.utils import BaseOutput


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`ms.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: ms.Tensor


@dataclass
class HunyuanVideoFramepackPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`. Or, a list of torch tensors where each tensor
            corresponds to a latent that decodes to multiple frames.
    """

    frames: Union[ms.Tensor, np.ndarray, List[List[PIL.Image.Image]], List[ms.Tensor]]

"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers/scheduling_amused.py."""

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.mindspore_utils import dtype_to_max
from .scheduling_utils import SchedulerMixin


def gumbel_noise(t: ms.Tensor, generator: Optional[ms.Generator] = None) -> ms.Tensor:
    """
    Generate Gumbel noise for sampling.

    Args:
        t (`ms.Tensor`):
            Input tensor to match the shape and dtype of the output noise.
        generator (`ms.Generator`, *optional*):
            A random number generator for reproducible sampling.

    Returns:
        `ms.Tensor`:
            Gumbel-distributed noise with the same shape, dtype, and device as the input tensor.
    """
    noise = mint.zeros_like(t).uniform_(0, 1, generator=generator)
    return -mint.log((-mint.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(
    mask_len: ms.Tensor,
    probs: ms.Tensor,
    temperature: float = 1.0,
    generator: Optional[ms.Generator] = None,
) -> ms.Tensor:
    """
    Mask tokens by selecting the top-k lowest confidence scores with temperature-based randomness.

    Args:
        mask_len (`ms.Tensor`):
            Number of tokens to mask per sample in the batch.
        probs (`ms.Tensor`):
            Probability scores for each token.
        temperature (`float`, *optional*, defaults to 1.0):
            Temperature parameter for controlling randomness in the masking process.
        generator (`ms.Generator`, *optional*):
            A random number generator for reproducible sampling.

    Returns:
        `ms.Tensor`:
            Boolean mask indicating which tokens should be masked.
    """
    confidence = mint.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = mint.sort(confidence, dim=-1)[0]
    cut_off = mint.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


@dataclass
class AmusedSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`ms.Tensor` of shape `(batch_size, height, width)` or `(batch_size, sequence_length)`):
            Computed sample `(x_{t-1})` of previous timestep with token IDs. `prev_sample` should be used as next model
            input in the denoising loop.
        pred_original_sample (`ms.Tensor` of shape `(batch_size, height, width)` or `(batch_size, sequence_length)`, *optional*):
            The predicted fully denoised sample `(x_{0})` with token IDs based on the model output from the current
            timestep. `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: ms.Tensor
    pred_original_sample: ms.Tensor = None


class AmusedScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler for masked token generation as used in [`AmusedPipeline`].

    This scheduler iteratively unmasks tokens based on their confidence scores, following either a cosine or linear
    schedule. Unlike traditional diffusion schedulers that work with continuous pixel values, this scheduler operates
    on discrete token IDs, making it suitable for autoregressive and non-autoregressive masked token generation models.

    This scheduler inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods the library implements for all schedulers such as loading and saving.

    Args:
        mask_token_id (`int`):
            The token ID used to represent masked tokens in the sequence.
        masking_schedule (`Literal["cosine", "linear"]`, *optional*, defaults to `"cosine"`):
            The schedule type for determining the mask ratio at each timestep. Can be either `"cosine"` or `"linear"`.
    """

    order = 1

    temperatures: Optional[ms.Tensor]
    timesteps: Optional[ms.Tensor]

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: Literal["cosine", "linear"] = "cosine",
    ):
        self.temperatures = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[float, Tuple[float, float], List[float]] = (2, 0),
    ) -> None:
        """
        Set the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            temperature (`Union[float, Tuple[float, float], List[float]]`, *optional*, defaults to `(2, 0)`):
                Temperature parameter(s) for controlling the randomness of sampling. If a tuple or list is provided,
                temperatures will be linearly interpolated between the first and second values across all timesteps. If
                a single value is provided, temperatures will be linearly interpolated from that value to 0.01.
        """
        self.timesteps = mint.arange(num_inference_steps).flip((0,))

        if isinstance(temperature, (tuple, list)):
            self.temperatures = mint.linspace(temperature[0], temperature[1], num_inference_steps)
        else:
            self.temperatures = mint.linspace(temperature, 0.01, num_inference_steps)

    def step(
        self,
        model_output: ms.Tensor,
        timestep: int,
        sample: ms.Tensor,
        starting_mask_ratio: float = 1.0,
        generator: Optional[ms.Generator] = None,
        return_dict: bool = False,
    ) -> Union[AmusedSchedulerOutput, Tuple[ms.Tensor, ms.Tensor]]:
        """
        Predict the sample at the previous timestep by masking tokens based on confidence scores.

        Args:
            model_output (`ms.Tensor`):
                The direct output from the learned diffusion model. Typically of shape `(batch_size, num_tokens,
                codebook_size)` or `(batch_size, codebook_size, height, width)` for 2D inputs.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process. Contains token IDs, with masked
                positions indicated by `mask_token_id`.
            starting_mask_ratio (`float`, *optional*, defaults to 1.0):
                A multiplier applied to the mask ratio schedule. Values less than 1.0 will result in fewer tokens being
                masked at each step.
            generator (`ms.Generator`, *optional*):
                A random number generator for reproducible sampling.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] or a plain tuple.

        Returns:
            [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] or `tuple`:
                If `return_dict` is `True`, [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor (`prev_sample`) and the
                second element is the predicted original sample tensor (`pred_original_sample`).
        """
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

        unknown_map = sample == self.config.mask_token_id

        probs = model_output.softmax(axis=-1)

        probs_ = probs
        probs_ = probs_.reshape(-1, probs.shape[-1])
        pred_original_sample = mint.multinomial(probs_, 1, generator=generator)
        pred_original_sample = pred_original_sample[:, 0].view(probs.shape[:-1])
        pred_original_sample = mint.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = mint.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = mint.min(unknown_map.sum(axis=-1, keepdims=True) - 1, mask_len)
            # mask at least one
            mask_len = mint.max(ms.tensor([1]), mask_len)

            selected_probs = mint.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = mint.where(unknown_map, selected_probs, dtype_to_max(selected_probs.dtype))

            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)

            # Masks tokens with lower confidence.
            prev_sample = mint.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return AmusedSchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(
        self,
        sample: ms.Tensor,
        timesteps: int,
        generator: Optional[ms.Generator] = None,
    ) -> ms.Tensor:
        """
        Add noise to a sample by randomly masking tokens according to the masking schedule.

        Args:
            sample (`ms.Tensor`):
                The input sample containing token IDs to be partially masked.
            timesteps (`int`):
                The timestep that determines how much masking to apply. Higher timesteps result in more masking.
            generator (`ms.Generator`, *optional*):
                A random number generator for reproducible masking.

        Returns:
            `ms.Tensor`:
                The sample with some tokens replaced by `mask_token_id` according to the masking schedule.
        """
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = mint.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        mask_indices = mint.rand(sample.shape, generator=generator) < mask_ratio

        masked_sample = sample.copy()

        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample

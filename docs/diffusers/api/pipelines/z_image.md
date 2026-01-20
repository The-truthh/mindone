<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Z-Image

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Z-Image](https://huggingface.co/papers/2511.22699) is a powerful and highly efficient image generation model with 6B parameters. Currently there's only one model with two more to be released:

| Model         | Hugging Face                                     |
|---------------|--------------------------------------------------|
| Z-Image-Turbo | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo  |

## Z-Image-Turbo

Z-Image-Turbo is a distilled version of Z-Image that matches or exceeds leading competitors with only 8 NFEs (Number of Function Evaluations). It excels in photorealistic image generation, bilingual text rendering (English & Chinese), and robust instruction adherence.

## Image-to-image

Use [`ZImageImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/z_image/#mindone.diffusers.ZImageImg2ImgPipeline) to transform an existing image based on a text prompt.

```python
import mindspore as ms
import numpy as np
from mindone.diffusers import ZImageImg2ImgPipeline
from mindone.diffusers.utils import load_image

pipe = ZImageImg2ImgPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", mindspore_dtype=ms.bfloat16)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

prompt = "A fantasy landscape with mountains and a river, detailed, vibrant colors"
image = pipe(
    prompt,
    image=init_image,
    strength=0.6,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=np.random.Generator(np.random.PCG64(42)),
).images[0]
image.save("zimage_img2img.png")
```

::: mindone.diffusers.ZImagePipeline

::: mindone.diffusers.ZImageImg2ImgPipeline

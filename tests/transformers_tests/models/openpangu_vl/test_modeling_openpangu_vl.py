import numpy as np

import mindspore as ms

from mindone.transformers import OpenPanguVL, OpenPanguVLConfig


class OpenPanguVLModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=4,
        vocab_size=99,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        image_token_id=4,
        video_token_id=5,
        vision_start_token_id=3,
        vision_end_token_id=6,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    def get_config(self):
        rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]}
        text_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "use_cache": False,
            "bias": True,
            "rope_scaling": rope_scaling,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
        }
        vision_config = {
            "depth": 1,
            "num_heads": self.num_attention_heads,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "out_hidden_size": self.hidden_size,
            "in_chans": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "window_size": 4,
            "fullatt_block_indexes": [0],
            "temporal_patch_size": 2,
            "mm_unit_vision_select_layer": [-1],
        }
        return OpenPanguVLConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            use_cache=False,
            bias=True,
            rope_scaling=rope_scaling,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            text_config=text_config,
            vision_config=vision_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = np.arange(7, 7 + self.seq_length, dtype=np.int64).reshape(self.batch_size, self.seq_length)
        attention_mask = np.ones_like(input_ids)
        return config, input_ids, attention_mask


def test_openpangu_vl_forward_text_only():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    model_tester = OpenPanguVLModelTester()
    config, input_ids, attention_mask = model_tester.prepare_config_and_inputs()

    model = OpenPanguVL(config).set_train(False)
    outputs = model(
        input_ids=ms.Tensor(input_ids, ms.int64),
        attention_mask=ms.Tensor(attention_mask, ms.int64),
        use_cache=False,
    )

    assert outputs.logits.shape == (model_tester.batch_size, model_tester.seq_length, model_tester.vocab_size)
    assert np.isfinite(outputs.logits.asnumpy()).all()

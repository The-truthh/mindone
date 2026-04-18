import mindspore as ms
from mindone.transformers import OpenPanguVL, OpenPanguVLProcessor
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info

model_path = "JohnsonWythe/openPangu-VL-7B"

print(f"LOAD MODEL FROM: {model_path}")

key_mapping = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}

model = OpenPanguVL.from_pretrained(
    model_path,
    trust_remote_code=False,
    dtype=ms.bfloat16,
    key_mapping=key_mapping,
).set_train(False)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": '你是华为公司开发的多模态大模型，名字是openPangu-VL-7B。你能够处理文本和视觉模态的输入，并给出文本输出。'},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": '你好，你是谁？'},
        ],
    },
]

processor = OpenPanguVLProcessor.from_pretrained(model_path, trust_remote_code=False)
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(conversation)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=False,
    return_tensors="ms",
)
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
res = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(f"OUTPUT: {res}")

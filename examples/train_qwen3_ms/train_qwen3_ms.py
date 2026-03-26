from datasets import load_dataset
from transformers import AutoTokenizer
from mindone.transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import mindspore as ms

model_name = "Qwen/Qwen3-0.6B"

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize, batched=True)
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(model_name, mindspore_dtype=ms.bfloat16, num_labels=5)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="yelp_review_classifier",
    per_device_train_batch_size=1,
    eval_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

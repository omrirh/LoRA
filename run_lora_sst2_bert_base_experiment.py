import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate

MODEL_NAME: str = "bert-base-uncased"  # TODO: scale the experiment to RoBERTa with sufficient GPU resources
GLUE_TASK_NAME: str = "sst2"


def load_glue_data(task_name: str = GLUE_TASK_NAME) -> Any:
    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_datasets


def initialize_lora_model() -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05
    )

    peft_model = get_peft_model(model, lora_config)

    return peft_model


def train_model(
    model: torch.nn.Module,
    tokenized_datasets: Any
) -> Trainer:
    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=500,
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        seed=42,
        save_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


def main() -> None:
    tokenized_datasets = load_glue_data(GLUE_TASK_NAME)
    model = initialize_lora_model()

    trainer = train_model(model, tokenized_datasets)
    eval_results = trainer.evaluate()

    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()

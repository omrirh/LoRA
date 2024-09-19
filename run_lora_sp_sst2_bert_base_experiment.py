import torch
import numpy as np
from typing import Dict, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    IntervalStrategy
)
from datasets import load_dataset
from peft import (
    PeftModel,
    LoraConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model
)
import evaluate


MODEL_NAME: str = "bert-base-uncased"  # You can scale to RoBERTa if needed
GLUE_TASK_NAME: str = "sst2"


def load_glue_data(task_name: str = GLUE_TASK_NAME) -> Any:
    """
    Load the GLUE dataset for SST2 task and tokenize it.

    Args:
        task_name (str): The GLUE task name, default is SST2.

    Returns:
        tokenized_datasets (Dataset): Tokenized dataset with input_ids, attention_mask, and label.
    """
    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_datasets


def initialize_lora_p_tuning_v2_model() -> PeftModel:
    """
    Initializes the BERT model with both LoRA and P-tuning v2 applied for sequence classification (SST2).

    Returns:
        model (PeftModel): The model with both LoRA and P-tuning v2 applied.
    """
    # Load the base BERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, num_labels=2)

    # Step 1: LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05
    )

    # Apply LoRA to the model
    model = get_peft_model(model=model, peft_config=lora_config)

    # Step 2: P-tuning v2 configuration
    prompt_encoder_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_CLS,  # SST2 task is a sequence classification
        num_virtual_tokens=20,  # Number of continuous prompt tokens
        token_dim=768,  # Hidden size matching the BERT model
        encoder_reparameterization_type="MLP",
        encoder_hidden_size=768,
    )

    # Apply P-tuning v2 on top of the same model
    # TODO: check if this runs ok
    model = get_peft_model(model=model, peft_config=prompt_encoder_config)

    return model


def train_model(
    model: torch.nn.Module,
    tokenized_datasets: Any
) -> Trainer:
    """
    Train the model with SST2 task using both LoRA and P-tuning v2.

    Args:
        model (torch.nn.Module): The model with LoRA and P-tuning v2 applied.
        tokenized_datasets (Any): Tokenized GLUE SST2 dataset.

    Returns:
        trainer (Trainer): Hugging Face trainer after training.
    """
    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        """
        Compute accuracy metrics for evaluation.

        Args:
            eval_pred (Any): Predictions from evaluation.

        Returns:
            result (Dict[str, float]): Accuracy metric result.
        """
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        result = metric.compute(predictions=predictions, references=labels)
        print(f"Accuracy: {result['accuracy']}")
        return result

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        do_eval=True,
        logging_dir='./logs',
        logging_steps=500,
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        seed=42,
        save_steps=500,
        evaluation_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS
    )

    # Initialize the Trainer
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
    """
    Main function to load the dataset, initialize the model, and train it using LoRA and P-tuning v2.
    """
    tokenized_datasets = load_glue_data(task_name=GLUE_TASK_NAME)
    model = initialize_lora_p_tuning_v2_model()

    # Train the model
    trainer = train_model(model=model, tokenized_datasets=tokenized_datasets)

    # Evaluate and print final results
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")


if __name__ == "__main__":
    main()

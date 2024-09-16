import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


# Step 1: Load GLUE dataset (SST-2 task)
def load_glue_data(task_name="sst2"):
    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "label"])

    return tokenized_datasets


# Step 2: Initialize model and apply LoRA
def initialize_lora_model():
    # Load a pre-trained BERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # LoRA configuration with rank=16, lora_alpha=32, and dropout=0.05
    lora_config = LoraConfig(
        r=16,  # Low-rank dimension
        lora_alpha=32,  # Scaling factor for LoRA
        target_modules=["query", "value"],  # Modules to apply LoRA on
        lora_dropout=0.05
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)

    return peft_model


# Step 3: Train the model
def train_model(model, tokenized_datasets):
    # Define training arguments with updated hyperparameters
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-4,  # Updated learning rate
        per_device_train_batch_size=32,  # Updated batch size
        per_device_eval_batch_size=32,
        num_train_epochs=3,  # Updated epochs
        weight_decay=0.01,  # Weight decay
        seed=42
    )

    # Trainer to handle training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


# Main function to load the data, initialize the model with LoRA, and train it
def main():
    # Load GLUE data
    tokenized_datasets = load_glue_data("sst2")

    # Initialize model with LoRA
    model = initialize_lora_model()

    # Train the model
    train_model(model, tokenized_datasets)


# TODO: re-produce best accuracy results as present in huggingface.io (92.43%)
if __name__ == "__main__":
    main()

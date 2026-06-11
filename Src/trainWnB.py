import wandb
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault('WANDB_API_KEY', os.getenv('WANDB_API_KEY'))

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from preprocess import preprocess_data


MODEL_NAME = os.getenv("MODEL_NAME")
NUM_LABELS = int(os.getenv("NUM_LABELS"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")


def main():

    # Initialize W&B
    wandb.init(
        project="imdb-sentiment",
        name="distilbert-v1",
        config={
            "model": MODEL_NAME,
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 2e-5,
        },
    )

    print("Loading and preprocessing data...")

    train_dataset, test_dataset = preprocess_data()

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=1,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        learning_rate=2e-5,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=100,

        load_best_model_at_end=True,

        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Starting training...")

    trainer.train()

    print("Evaluating model...")

    eval_results = trainer.evaluate()

    print("\nEvaluation Results")
    print("-" * 30)

    for metric, value in eval_results.items():
        print(f"{metric}: {value}")

    print("\nSaving model...")

    trainer.save_model(MODEL_SAVE_PATH)

    # Save tokenizer too
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Log final metrics manually
    wandb.log(eval_results)

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    wandb.finish()


if __name__ == "__main__":
    main()
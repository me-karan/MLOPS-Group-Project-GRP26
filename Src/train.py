from preprocess import preprocess_data

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def main():

    run_name = "distilbert_v0_best_model"
    train_dataset, test_dataset = preprocess_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f"./Results/Trainer/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    trainer.save_model(f"Model/Trainer/{run_name}/best_model")


if __name__ == "__main__":
    main()

import os

from preprocess import preprocess_data
from dotenv import load_dotenv

import argparse
import yaml
import wandb

load_dotenv()
os.environ.setdefault('WANDB_API_KEY', os.getenv('WANDB_API_KEY'))
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def main():

    #Read config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    #Load YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    #Extract parameters
    run_name = config["run_name"]
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])

    #Initialize W&B
    wandb.init(
        project="imdb-sentiment",
        name=run_name,
        config=config
    )

    train_dataset, test_dataset = preprocess_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./Results/Trainer",
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    
    eval_results = trainer.evaluate()

    print("\nEvaluation Results")
    print(eval_results)

    trainer.save_model(
        "Model/Trainer/"+run_name+"_best_model"
    )

    
    wandb.log(eval_results)

    
    wandb.finish()


if __name__ == "__main__":
    main()
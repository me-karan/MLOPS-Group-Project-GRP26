import json
from pathlib import Path

import numpy as np
import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from preprocess import preprocess_data


MODEL_PATH = "Model/Trainer/best_model"

REPORTS_DIR = Path("Reports/Trainer")


def compute_metrics(labels, predictions):
    """
    Compute evaluation metrics.
    """

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_reports(labels, predictions, metrics):
    """
    Save evaluation outputs to disk.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save metrics
    with open(
        REPORTS_DIR / "evaluation_metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # Save confusion matrix
    cm = confusion_matrix(
        labels,
        predictions,
    )

    cm_df = pd.DataFrame(
        cm,
        index=["negative", "positive"],
        columns=["negative", "positive"],
    )

    cm_df.to_csv(
        REPORTS_DIR / "confusion_matrix.csv"
    )

    # Save classification report
    report = classification_report(
        labels,
        predictions,
        target_names=[
            "negative",
            "positive",
        ],
        output_dict=True,
    )

    with open(
        REPORTS_DIR / "classification_report.json",
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    print(f"Reports saved to: {REPORTS_DIR}")


def main():

    print("Loading test dataset...")

    _, test_dataset = preprocess_data()

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    trainer = Trainer(model=model)

    print("Running predictions...")

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions

    preds = np.argmax(
        logits,
        axis=-1,
    )

    labels = predictions.label_ids

    metrics = compute_metrics(
        labels,
        preds,
    )

    print("\nEvaluation Metrics")
    print("-" * 30)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    save_reports(
        labels,
        preds,
        metrics,
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
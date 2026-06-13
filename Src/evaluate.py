import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained sentiment classification model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )

    parser.add_argument(
        "--report_dir",
        type=str,
        required=True,
        help="Directory where evaluation reports will be saved",
    )

    return parser.parse_args()


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


def save_reports(
    labels,
    predictions,
    metrics,
    reports_dir,
):
    """
    Save evaluation outputs to disk.
    """

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save metrics
    with open(
        reports_dir / "evaluation_metrics.json",
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
        reports_dir / "confusion_matrix.csv",
        index=True,
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
        reports_dir / "classification_report.json",
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    print(f"\nReports saved to: {reports_dir}")


def main():

    args = parse_args()

    model_path = args.model_path
    reports_dir = Path(args.report_dir)

    print("Loading test dataset...")

    _, test_dataset = preprocess_data()

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
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
        print(f"{metric:<10}: {value:.4f}")

    save_reports(
        labels,
        preds,
        metrics,
        reports_dir,
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
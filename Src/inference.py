import os
import wandb
from transformers import pipeline

MODEL_NAME = os.getenv("HF_MODEL_NAME", "g25Ait2048/imdb-sentiment-analysis")

PROJECT_NAME = "mlops-distilbert"


def get_text():
    text = os.getenv("INPUT_TEXT")
    if not text:
        raise EnvironmentError("Missing INPUT_TEXT environment variable.")
    return text


def load_classifier(token):
    kwargs = {
        "task": "text-classification",
        "model": MODEL_NAME,
    }
    if token:
        kwargs["use_auth_token"] = token
    return pipeline(**kwargs)


def display_result(text, prediction):
    print("\n" + "-" * 60)
    print("Inference Result -")
    print("-" * 60)
    print(f"Input Text      : {text}")
    print(f"Predicted Class : {prediction['label']}")
    print(f"Confidence Score: {prediction['score']:.4f}")
    print("-" * 60 + "\n")


def main():
    text = get_text()
    hf_token = os.getenv("HF_TOKEN")

    run = wandb.init(
        project=PROJECT_NAME,
        job_type="inference",
    )

    classifier = load_classifier(hf_token)

    prediction = classifier(text)[0]

    display_result(text, prediction)

    run.log(
        {
            "input_text": text,
            "predicted_label": prediction["label"],
            "confidence": prediction["score"],
        }
    )

    run.finish()


if __name__ == "__main__":
    main()
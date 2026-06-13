from fastapi import FastAPI
from pydantic import BaseModel

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_PATH = "Model/Trainer/distilbert_v1_best_model"

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    version="1.0.0",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1,
    ).item()

    confidence = probabilities.max().item()

    label_map = {
        0: "negative",
        1: "positive",
    }

    return {
        "prediction": label_map[prediction],
        "confidence": round(confidence, 4),
    }
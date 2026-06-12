# ML-Group-Project-GRP-26
This Repo is for MLops Group 26

## Itroduction
Sentiment analysis is a Natural Language Processing (NLP) task that aims to automatically determine the emotional tone or opinion expressed in textual data. With the rapid growth of user-generated content on online platforms, sentiment analysis has become an important tool for understanding customer feedback, product reviews, and public opinion. Movie review classification is one of the most widely studied sentiment analysis problems, where reviews are categorized as either positive or negative based on their content.


## Model and dataset

* Model
https://huggingface.co/distilbert/distilbert-base-uncased
  
* Dataset
https://huggingface.co/datasets/stanfordnlp/imdb

## Links 

* kaggle
https://www.kaggle.com/code/karang25ait2048/mlops-group-project-grp-26

* Eda Notebook
https://colab.research.google.com/drive/1hVZSukB_KQbzJSlJWhkPx0UVXoAOjorf#scrollTo=riIgzp0R8Qhb

* W&B Dashboard
https://wandb.ai/g25ait2048-iit-jodhpur/imdb-sentiment

* Hugging Face Repository
https://huggingface.co/g25Ait2048/imdb-sentiment-analysis/tree/main

* Full Report
https://github.com/Karan-IITJ-G25AIT2048/ML-Group-Project-GRP-26/blob/main/MLOPS-GRP-26-Report.pdf

## Results

### Experiment Results

| Version | Learning Rate | Batch Size | Epochs | Max Length | Accuracy | Precision | Recall | F1 Score |
|----------|-------------|------------|---------|------------|----------|-----------|--------|----------|
| DistilBERT V1 | 2e-5 | 16 | 1 | 256 | 0.9108 | 0.9073 | 0.9157 | 0.9115 |
| DistilBERT V2 | 2e-5 | 16 | 2 | 256 | 0.9124 | 0.9156 | 0.9092 | 0.9124 |
| DistilBERT V3 | 3e-5 | 16 | 2 | 256 | 0.9143 | 0.9068 | 0.9240 | 0.9154 |
| DistilBERT V4 | 5e-5 | 16 | 2 | 256 | 0.9142 | 0.9025 | 0.9293 | **0.9157** |

### Best Performing Model: DistilBERT V4

| Metric | Value |
|----------|----------|
| Learning Rate | 5e-5 |
| Batch Size | 16 |
| Epochs | 2 |
| Max Length | 256 |
| Accuracy | 0.9142 |
| Precision | 0.9025 |
| Recall | 0.9293 |
| F1 Score | 0.9157 |

### Observations

- Increasing the number of epochs from 1 to 2 improved performance from V1 to V2.
- Increasing the learning rate to 3e-5 (V3) improved both Accuracy and F1 Score.
- V4 achieved the highest F1 Score (0.9157) and Recall (0.9293).
- V3 achieved the highest Accuracy (0.9143), though only marginally higher than V4.
- Based on F1 Score, which balances Precision and Recall, V4 was selected as the final model.


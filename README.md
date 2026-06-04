---
title: Sentiment Analysis Group 15
emoji: 🔍
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: true
---

# Sentiment Analysis on Product Reviews

Binary sentiment classification using SVM + TF-IDF and Bidirectional LSTM.

## Results

| Metric | SVM | BiLSTM |
|---|---|---|
| Accuracy | 95.24% | 93.65% |
| Weighted F1 | 0.9511 | 0.9392 |
| ROC-AUC | 0.9751 | 0.9691 |
| **Negative Recall** | 73.94% | **84.53%** |

**Key finding:** SVM wins on aggregate. BiLSTM wins on negative recall
(+10.59pp) — the metric that matters most for complaint detection.

## Dataset
Women's E-Commerce Clothing Review Reviews — Kaggle

## References
See full report in notebooks/

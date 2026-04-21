# Sentiment Classifier (NLP + Transformer)

A sentiment analysis system built using PyTorch and Transformer architecture to classify movie reviews into positive or negative sentiments.

---

## 📌 Overview

This project implements a deep learning-based sentiment classifier using a Transformer encoder. It processes raw text, builds a vocabulary, and trains a model to perform binary classification on movie reviews.

The system focuses on:
- Efficient text preprocessing
- Sequence modeling with positional encoding
- Transformer-based representation learning
- End-to-end training and evaluation pipeline

---

## 🧠 Features

- Text preprocessing with NLTK (tokenization, stopword removal)
- Custom vocabulary construction with frequency filtering
- Transformer-based model with positional encoding
- Efficient batching using PyTorch DataLoader
- Binary classification with BCE loss
- Model checkpointing (best validation accuracy)
- Evaluation using accuracy metric

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NLTK
- NumPy / Pandas
- scikit-learn

---

## 🏗️ Model Architecture

- Token Embedding Layer
- Positional Encoding
- Transformer Encoder (multi-head attention)
- Classification head (Linear layer)

---

## 📂 Project Structure
```
├── data/
│   └── movie_reviews_labelled.csv   # dataset (not included)
├── sentiment_classifier.py         # main training & model file
├── best_model.pth                  # saved model (generated after training)
├── README.md
```

---

## ⚙️ Installation

```bash
pip install torch nltk pandas numpy scikit-learn tqdm


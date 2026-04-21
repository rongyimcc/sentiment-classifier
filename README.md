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

## ⚙️ Installation and run

```bash
pip install torch nltk pandas numpy scikit-learn tqdm

```
---
```

python sentiment_classifier.py

```
---

## Results

- Validation Accuracy: ~82%  
- Test Accuracy: ~85%  
- Model trained using Transformer encoder with positional encoding


---
##Key Highlights

- Built a custom NLP pipeline from raw text to model training
- Implemented Transformer encoder with positional encoding from scratch
- Designed efficient batching using PyTorch DataLoader
- Applied validation-based checkpointing to prevent overfitting
- Achieved strong performance on sentiment classification task
---













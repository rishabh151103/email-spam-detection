# Email Spam Detection

## Overview

This project implements a supervised machine-learning workflow for **binary text classification**. The objective is to classify a message as **ham (legitimate)** or **spam (unwanted/fraudulent)** from its text content.

The project covers the complete workflow: data cleaning, text feature extraction, model comparison, evaluation, cross-validation, confusion-matrix analysis and prediction on new messages.

## Objectives

- Clean and inspect the SMS dataset.
- Convert text messages into numerical features.
- Compare multiple classification algorithms under the same protocol.
- Evaluate performance using more than accuracy alone.
- Use stratified cross-validation as a robustness check.
- Select a final classifier and demonstrate predictions on unseen messages.

## Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The cleaned dataset documented in the accompanying training report contains:

- 5,169 messages
- 4,516 ham messages
- 653 spam messages
- Approximately 87% ham and 13% spam

The class imbalance is preserved during splitting using stratification.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Machine Learning Models

The following classifiers are benchmarked:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine
4. K-Nearest Neighbours
5. Decision Tree

## Preprocessing

Message text is represented using `CountVectorizer` with English stop-word removal.

The data is split using an **80/20 stratified train/test split** with `random_state=42`.

The vectorizer and classifier are combined in a scikit-learn `Pipeline`. This keeps text preprocessing inside the modelling workflow and avoids fitting the vocabulary globally before cross-validation.

## Evaluation

The project reports:

- Test accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrix
- 5-fold stratified cross-validation accuracy

Macro-averaged metrics are useful here because the spam class is smaller than the ham class.

## Reported Results

The accompanying training report records these test-set results:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 98.45% | 0.9739 | 0.9552 | 0.9643 |
| Linear SVM | 98.36% | 0.9833 | 0.9416 | 0.9612 |
| Logistic Regression | 97.39% | 0.9813 | 0.9002 | 0.9357 |
| Decision Tree | 96.91% | 0.9402 | 0.9170 | 0.9282 |
| KNN | 90.43% | 0.9506 | 0.6221 | 0.6703 |

Multinomial Naive Bayes was selected as the final model under the report's stated test-accuracy criterion.

## Workflow

```text
SMS Dataset
    ↓
Data Cleaning
    ↓
Label Encoding
    ↓
Stratified 80/20 Split
    ↓
CountVectorizer
    ↓
Train Multiple Classifiers
    ↓
Test-set Evaluation
    ↓
5-Fold Stratified CV
    ↓
Model Selection
    ↓
Confusion Matrix
    ↓
Sample Predictions
```

## Project Structure

```text
email-spam-detection/
├── email_spam_detection.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python email_spam_detection.py
```

The script downloads the UCI SMS Spam Collection automatically when the dataset is not already present.

## Output

The program prints the model-comparison table and classification report and creates:

```text
confusion_matrix.png
```

## Example Use

The script includes representative messages containing promotional language, normal conversation and urgent account-verification language. It prints the predicted class for each example.

## Limitations

- The dataset is a public SMS corpus and may not represent every real-world messaging domain.
- Spam vocabulary and behaviour can change over time.
- Performance on a different message population may require retraining.
- This is an academic machine-learning implementation, not a production spam-filtering service.

## Reproducibility

The experiment uses `random_state=42`, stratified splitting and 5-fold stratified cross-validation. Pipeline-based preprocessing keeps feature extraction within the modelling workflow.

## Author

**Rishabh Gaur**  
B.Tech — Artificial Intelligence & Machine Learning

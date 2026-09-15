# Email Spam Detection

Binary text classification of messages as **ham** or **spam** using the SMS Spam Collection dataset.

## Models
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- KNN
- Decision Tree

## Workflow
Cleaning → stratified 80/20 split → CountVectorizer → model comparison → macro precision/recall/F1 → 5-fold stratified CV → confusion matrix → sample predictions.

## Run
```bash
pip install -r requirements.txt
python email_spam_detection.py
```

The script downloads the UCI SMS Spam Collection automatically on first run.

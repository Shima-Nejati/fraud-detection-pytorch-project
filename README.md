# Fraud Detection with PyTorch

A small, reusable end-to-end binary-classification project for identifying potentially fraudulent financial transactions. It uses a fully connected PyTorch network and compares it with a scikit-learn logistic-regression baseline.

## Why Fraud Detection?

Fraud detection is an imbalanced-classification problem: legitimate transactions greatly outnumber fraudulent ones. A model can appear highly accurate by predicting every transaction as legitimate while failing to catch fraud. This project therefore emphasizes precision, recall, F1, and especially PR-AUC, alongside ROC-AUC and confusion matrices.

## Project Structure

```text
fraud-detection-pytorch/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
└── artifacts/
```

Preprocessing writes the split arrays and fitted transformer to `data/processed/`. Training writes the best model checkpoint to `artifacts/best_model.pt`.

## Dataset Setup

This project is designed for the public [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Download its CSV and place it at:

```text
data/raw/creditcard.csv
```

The expected target column is `Class`, with `0` for legitimate transactions and `1` for fraud. Every other column must be numeric. The commonly used dataset includes `Time`, `Amount`, and anonymized `V1` through `V28` feature columns. If your CSV uses a different filename or target name, update `data_path` or `target_column` in `config.yaml`.

The preprocessing script reports missing values, imputes missing feature values using training-set medians, and standardizes numeric features. It fits the imputer and scaler on the training split only, then applies them to validation and test data. Stratified splitting preserves the fraud/legitimate class proportions as much as possible in each split. This matters because a random unstratified split can leave too few rare fraud examples for meaningful validation or testing.

## Installation

From the repository root, create and activate a virtual environment, then install the listed dependencies. For PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Pipeline

Run commands from the repository root.

Preprocess the raw CSV:

```powershell
python -m src.preprocessing
```

Train the neural network and save the best validation PR-AUC checkpoint:

```powershell
python -m src.train
```

Evaluate the MLP and logistic-regression baseline on the held-out test set:

```powershell
python -m src.evaluate
```

Run the focused tests:

```powershell
pytest
```

The default settings are in `config.yaml`: random seed, split sizes, batch size, learning rate, number of epochs, hidden dimensions, dropout, and classification threshold. Preprocessing, training, and evaluation accept `--config path/to/config.yaml` to use another settings file.

## Prediction

Create a CSV containing exactly one transaction row. Include every feature column used in training, with the same names. Omit the target column; if `Class` is present, prediction ignores it. Then run:

```powershell
python -m src.predict --transaction-csv data/raw/example_transaction.csv
```

The prediction function loads the saved fitted transformer, applies its stored feature order and transformations, and returns a fraud probability and predicted class. The CLI prints both values. The class threshold comes from `classification_threshold` in `config.yaml`.

## Evaluation Metrics

- **Precision** measures how many transactions flagged as fraud are actually fraud. Higher precision means fewer false alerts.
- **Recall** measures how many actual fraud cases are detected. Higher recall means fewer missed fraud cases.
- **F1** is the harmonic mean of precision and recall, useful when both types of error matter.
- **PR-AUC** summarizes the precision-recall tradeoff across thresholds and is especially informative when the positive class is rare. The scripts calculate it with scikit-learn's average precision score.
- **ROC-AUC** measures ranking quality across thresholds, but can look strong on heavily imbalanced data even when positive-class performance is weak.

Accuracy alone is misleading here because the majority class dominates the score: predicting every transaction as legitimate can yield high accuracy while catching no fraud. Always read the minority-class metrics and confusion matrix in context. The default classification threshold is configurable and is not necessarily optimal for a real financial institution's costs or alert capacity.

## Main ML Concepts

- Missing-value inspection and train-only imputation/scaling
- Stratified train, validation, and test splits
- Custom PyTorch `Dataset` and `DataLoader`
- A configurable fully connected network trained with `BCEWithLogitsLoss`
- Positive-class weighting for imbalance
- Validation metrics and best-checkpoint selection by PR-AUC
- Test-set metrics and comparison with logistic regression
- Reusable single-transaction inference with the saved preprocessing pipeline
- Configuration, random seeds, and focused pytest coverage

## Why this project is relevant to Applied ML Engineering

This project demonstrates:

- **Messy real-world data handling:** inspecting missing values, validating columns, preserving class proportions, and preventing preprocessing leakage.
- **Reusable training and inference code:** separate modules for preprocessing, dataset loading, training, evaluation, and prediction.
- **PyTorch:** a custom dataset, dataloaders, a small neural network, loss weighting, and an explicit training loop.
- **Model evaluation:** validation-based checkpointing, held-out test metrics, and an interpretable baseline comparison.
- **Hyperparameter configuration:** central YAML settings for training and the classification threshold.
- **Testing:** focused tests for dataset behavior and model output shape.
- **Reproducibility:** seeded Python, NumPy, PyTorch, and stratified data splitting.
- **Baseline comparison:** logistic regression provides a simple reference point for the MLP.

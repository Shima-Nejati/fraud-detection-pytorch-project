"""Compare the trained fraud classifier with a logistic-regression baseline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.dataset import FraudDataset
from src.model import FraudClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project settings from a YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    return config


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate thresholded classification metrics and ranking metrics."""
    predicted_labels = (probabilities >= threshold).astype(np.int32)
    metrics = {
        "Precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "Recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "F1": f1_score(true_labels, predicted_labels, zero_division=0),
        "ROC-AUC": roc_auc_score(true_labels, probabilities),
        "PR-AUC": average_precision_score(true_labels, probabilities),
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    return metrics, matrix


def evaluate_models(config: dict[str, Any]) -> None:
    """Evaluate the best MLP and logistic-regression baseline on the test set."""
    set_random_seeds(int(config["random_seed"]))
    processed_directory = PROJECT_ROOT / "data" / "processed"
    train_features = np.load(processed_directory / "X_train.npy")
    train_labels = np.load(processed_directory / "y_train.npy")
    test_features = np.load(processed_directory / "X_test.npy")
    test_labels = np.load(processed_directory / "y_test.npy").astype(np.int32)
    threshold = float(config["classification_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run training first.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = FraudClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=[int(size) for size in checkpoint["hidden_dims"]],
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        FraudDataset(test_features, test_labels),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    mlp_probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            logits = model(batch_features.to(device))
            mlp_probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    mlp_probabilities = np.concatenate(mlp_probability_batches)
    mlp_metrics, mlp_confusion = calculate_metrics(test_labels, mlp_probabilities, threshold)

    # Balanced class weights give the baseline a comparable response to rare fraud labels.
    baseline = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(config["random_seed"]))
    baseline.fit(train_features, train_labels)
    baseline_probabilities = baseline.predict_proba(test_features)[:, 1]
    baseline_metrics, baseline_confusion = calculate_metrics(
        test_labels,
        baseline_probabilities,
        threshold,
    )

    print(f"Best MLP checkpoint: epoch {checkpoint['epoch']}")
    print(f"Validation PR-AUC: {float(checkpoint['validation_pr_auc']):.4f}")
    print(f"\nTest-set comparison (classification threshold: {threshold:.2f})")
    print(f"{'Metric':<15}{'Logistic Regression':>24}{'PyTorch MLP':>16}")
    for metric_name in ("Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"):
        print(
            f"{metric_name:<15}"
            f"{baseline_metrics[metric_name]:>24.4f}"
            f"{mlp_metrics[metric_name]:>16.4f}"
        )

    # With rare fraud labels, predicting every transaction as legitimate can still
    # produce high accuracy, so precision, recall, F1, and PR-AUC are more informative.
    print("\nConfusion matrix (rows: actual [0, 1], columns: predicted [0, 1])")
    print("Logistic Regression:")
    print(baseline_confusion)
    print("PyTorch MLP:")
    print(mlp_confusion)


def main() -> None:
    """Run evaluation using the project's configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    arguments = parser.parse_args()
    evaluate_models(load_config(arguments.config))


if __name__ == "__main__":
    main()

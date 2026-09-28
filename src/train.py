"""Train the fraud classifier and save the best validation checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
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
    """Seed Python, NumPy, and PyTorch for repeatable runs where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(config: dict[str, Any]) -> None:
    """Train the model, report validation metrics, and save the best epoch."""
    set_random_seeds(int(config["random_seed"]))
    processed_directory = PROJECT_ROOT / "data" / "processed"
    train_features = np.load(processed_directory / "X_train.npy")
    train_labels = np.load(processed_directory / "y_train.npy")
    validation_features = np.load(processed_directory / "X_validation.npy")
    validation_labels = np.load(processed_directory / "y_validation.npy")

    train_dataset = FraudDataset(train_features, train_labels)
    validation_dataset = FraudDataset(validation_features, validation_labels)
    loader_generator = torch.Generator().manual_seed(int(config["random_seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [int(size) for size in config["hidden_dims"]]
    dropout = float(config["dropout"])
    model = FraudClassifier(
        input_dim=train_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    positive_count = float(np.count_nonzero(train_labels == 1))
    negative_count = float(np.count_nonzero(train_labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain examples from both classes.")
    positive_weight = torch.tensor([negative_count / positive_count], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    threshold = float(config["classification_threshold"])

    artifacts_directory = PROJECT_ROOT / "artifacts"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_directory / "best_model.pt"
    best_validation_pr_auc = float("-inf")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        training_loss_total = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            training_loss_total += loss.item() * len(batch_labels)
        training_loss = training_loss_total / len(train_dataset)

        model.eval()
        validation_loss_total = 0.0
        validation_probabilities: list[np.ndarray] = []
        validation_targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch_features, batch_labels in validation_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_features)
                loss = loss_function(logits, batch_labels)
                validation_loss_total += loss.item() * len(batch_labels)
                validation_probabilities.append(torch.sigmoid(logits).cpu().numpy())
                validation_targets.append(batch_labels.cpu().numpy())

        validation_loss = validation_loss_total / len(validation_dataset)
        probabilities = np.concatenate(validation_probabilities)
        targets = np.concatenate(validation_targets)
        predictions = (probabilities >= threshold).astype(np.int32)
        validation_precision = precision_score(targets, predictions, zero_division=0)
        validation_recall = recall_score(targets, predictions, zero_division=0)
        validation_f1 = f1_score(targets, predictions, zero_division=0)
        validation_pr_auc = average_precision_score(targets, probabilities)

        print(
            f"Epoch {epoch:02d}/{int(config['epochs'])} | "
            f"train loss: {training_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"precision: {validation_precision:.4f} | "
            f"recall: {validation_recall:.4f} | "
            f"F1: {validation_f1:.4f} | "
            f"PR-AUC: {validation_pr_auc:.4f}"
        )

        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = validation_pr_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_features.shape[1],
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "epoch": epoch,
                    "validation_pr_auc": validation_pr_auc,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint to {checkpoint_path}.")


def main() -> None:
    """Run training using the project's configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    arguments = parser.parse_args()
    train_model(load_config(arguments.config))


if __name__ == "__main__":
    main()

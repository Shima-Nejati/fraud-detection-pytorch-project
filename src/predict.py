"""Run inference for one transaction using the saved preprocessing and model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.model import FraudClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project settings from a YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    return config


def predict_transaction(
    model: nn.Module,
    transaction_features: Mapping[str, Any] | pd.DataFrame | np.ndarray | Sequence[float],
) -> tuple[float, int]:
    """Return fraud probability and predicted class for one raw transaction.

    Named inputs must use the feature columns saved during preprocessing. Array
    inputs must follow that same saved feature order.
    """
    processed_directory = PROJECT_ROOT / "data" / "processed"
    preprocessing_path = processed_directory / "preprocessor.joblib"
    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {preprocessing_path}. Run preprocessing first."
        )

    preprocessing_bundle = joblib.load(preprocessing_path)
    feature_columns = preprocessing_bundle["feature_columns"]
    target_column = preprocessing_bundle["target_column"]
    preprocessor = preprocessing_bundle["preprocessor"]

    if isinstance(transaction_features, pd.DataFrame):
        transaction_frame = transaction_features.copy()
    elif isinstance(transaction_features, Mapping):
        transaction_frame = pd.DataFrame([transaction_features])
    else:
        feature_array = np.asarray(transaction_features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        if feature_array.ndim != 2 or feature_array.shape[0] != 1:
            raise ValueError("Provide exactly one transaction as a 1D array or a one-row 2D array.")
        if feature_array.shape[1] != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} features, got {feature_array.shape[1]}.")
        transaction_frame = pd.DataFrame(feature_array, columns=feature_columns)

    if len(transaction_frame) != 1:
        raise ValueError("Provide exactly one transaction for prediction.")
    if target_column in transaction_frame.columns:
        transaction_frame = transaction_frame.drop(columns=[target_column])

    missing_columns = [column for column in feature_columns if column not in transaction_frame.columns]
    unexpected_columns = [column for column in transaction_frame.columns if column not in feature_columns]
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"Transaction columns do not match training features. "
            f"Missing: {missing_columns}; unexpected: {unexpected_columns}."
        )

    ordered_features = transaction_frame.loc[:, feature_columns]
    transformed_features = preprocessor.transform(ordered_features).astype(np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.as_tensor(transformed_features, dtype=torch.float32, device=device)
        fraud_probability = float(torch.sigmoid(model(feature_tensor)).item())

    threshold = float(load_config(PROJECT_ROOT / "config.yaml")["classification_threshold"])
    predicted_class = int(fraud_probability >= threshold)
    return fraud_probability, predicted_class


def load_trained_model() -> nn.Module:
    """Load the best saved checkpoint and rebuild its model architecture."""
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
    return model


def main() -> None:
    """Predict fraud probability for a CSV file containing one transaction row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transaction-csv",
        type=Path,
        required=True,
        help="CSV containing exactly one transaction row with the training feature columns.",
    )
    arguments = parser.parse_args()
    transaction_frame = pd.read_csv(arguments.transaction_csv)
    model = load_trained_model()
    probability, predicted_class = predict_transaction(model, transaction_frame)
    print(f"Fraud probability: {probability:.4f}")
    print(f"Predicted class: {predicted_class} ({'fraud' if predicted_class else 'legitimate'})")


if __name__ == "__main__":
    main()

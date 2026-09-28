"""Load, split, and scale the fraud dataset for model training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load project settings from a YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    return config


def print_class_distribution(labels: pd.Series, split_name: str) -> None:
    """Print counts and percentages for each class in a dataset split."""
    counts = labels.value_counts().sort_index()
    percentages = labels.value_counts(normalize=True).sort_index() * 100
    print(f"{split_name} class distribution:")
    for label, count in counts.items():
        print(f"  Class {label}: {count:,} ({percentages[label]:.2f}%)")


def preprocess_data(config: dict[str, Any]) -> None:
    """Create stratified splits and save train-fitted preprocessing outputs."""
    data_path = PROJECT_ROOT / config["data_path"]
    target_column = config["target_column"]
    test_size = float(config["test_size"])
    validation_size = float(config["validation_size"])
    random_seed = int(config["random_seed"])

    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must both be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place the CSV there or update data_path in config.yaml."
        )

    dataframe = pd.read_csv(data_path)
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. Available columns: {list(dataframe.columns)}"
        )

    missing_counts = dataframe.isna().sum()
    columns_with_missing_values = missing_counts[missing_counts > 0]
    if columns_with_missing_values.empty:
        print("Missing values: none")
    else:
        print("Missing values by column:")
        print(columns_with_missing_values.to_string())

    if dataframe[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    feature_columns = [column for column in dataframe.columns if column != target_column]
    non_numeric_columns = dataframe[feature_columns].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "All feature columns must be numeric for this simple pipeline. "
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    labels = dataframe[target_column]
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Target column '{target_column}' must contain only 0 and 1 labels.")

    # Fraud datasets are highly imbalanced. Stratification keeps the rare fraud
    # examples represented in each split instead of leaving a split nearly empty.
    label_counts = labels.value_counts()
    if label_counts.min() < 3:
        raise ValueError("Each class needs at least three examples for stratified train/validation/test splits.")

    features = dataframe[feature_columns]
    test_count = math.ceil(len(dataframe) * test_size)
    validation_count = math.ceil(len(dataframe) * validation_size)
    if test_count + validation_count >= len(dataframe):
        raise ValueError("The configured split sizes leave no rows for training.")

    train_validation_features, test_features, train_validation_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_count,
        random_state=random_seed,
        stratify=labels,
    )
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_validation_features,
        train_validation_labels,
        test_size=validation_count,
        random_state=random_seed,
        stratify=train_validation_labels,
    )

    # Fit imputation and scaling only on training rows to avoid leaking
    # validation or test information into the model's training process.
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    train_features_array = preprocessor.fit_transform(train_features).astype(np.float32)
    validation_features_array = preprocessor.transform(validation_features).astype(np.float32)
    test_features_array = preprocessor.transform(test_features).astype(np.float32)

    processed_directory = PROJECT_ROOT / "data" / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)
    for split_name, split_features, split_labels in (
        ("train", train_features_array, train_labels),
        ("validation", validation_features_array, validation_labels),
        ("test", test_features_array, test_labels),
    ):
        np.save(processed_directory / f"X_{split_name}.npy", split_features)
        np.save(
            processed_directory / f"y_{split_name}.npy",
            split_labels.to_numpy(dtype=np.float32),
        )
        print_class_distribution(split_labels, split_name)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "target_column": target_column,
        },
        processed_directory / "preprocessor.joblib",
    )
    print(f"Processed arrays and fitted preprocessor saved in {processed_directory}.")


def main() -> None:
    """Run preprocessing using the project's configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    arguments = parser.parse_args()
    preprocess_data(load_config(arguments.config))


if __name__ == "__main__":
    main()

"""PyTorch dataset for preprocessed transaction features and labels."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """Store transaction features and binary labels as PyTorch tensors."""

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must contain the same number of examples.")

    def __len__(self) -> int:
        """Return the number of transactions in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one feature vector and its label."""
        return self.features[index], self.labels[index]
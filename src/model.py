"""A small fully connected neural network for fraud classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FraudClassifier(nn.Module):
    """Predict one fraud logit for each transaction in a batch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive).")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits; apply sigmoid only when probabilities are needed."""
        return self.network(features).squeeze(-1)

"""PyTorch model architectures for LSTM and GRU temperature prediction."""

import torch
import torch.nn as nn

from src.config import DROPOUT, HIDDEN_DIM, NUM_LAYERS


class RecurrentTemperatureModel(nn.Module):
    """Stacked recurrent neural network with fully connected projection head."""

    def __init__(
        self,
        cell_type: str = "LSTM",
        input_dim: int = 4,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.cell_type = cell_type.upper()

        rnn_class = nn.LSTM if self.cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through recurrent encoder and linear regression head."""
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)

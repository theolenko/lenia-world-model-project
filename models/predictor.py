import torch
import torch.nn as nn


class LeniaPredictor(nn.Module):
    """MLP that predicts the next frame's embedding from the current one.

    Used in the JEPA setup to predict target encoder embeddings.

    Args:
        embed_dim: Input and output dimensionality. Default: 128.
        hidden_dim: Hidden layer width. Default: 256.

    Input:
        z: Tensor of shape (batch, embed_dim).

    Output:
        Tensor of shape (batch, embed_dim) — predicted next-frame embedding.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

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


class PatchPredictor(nn.Module):
    """Per-patch MLP predictor for patch-level JEPA.

    Applies the same MLP independently to each patch embedding.
    nn.Linear supports arbitrary batch dims, so (B, N, D) works directly.

    Args:
        embed_dim: Patch embedding dimensionality.
        hidden_dim: Hidden layer width.

    Input:  (batch, N_patches, embed_dim)
    Output: (batch, N_patches, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

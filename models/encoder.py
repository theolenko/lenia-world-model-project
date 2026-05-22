import torch
import torch.nn as nn


class LeniaEncoder(nn.Module):
    """CNN encoder mapping Lenia frames to a fixed-size embedding.

    Args:
        embed_dim: Output embedding dimensionality. Default: 128.

    Input:
        x: Tensor of shape (batch, 1, 64, 64) with values in [0, 1].

    Output:
        Tensor of shape (batch, embed_dim).
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Four stride-2 conv blocks: 64 -> 32 -> 16 -> 8 -> 4 spatial dims
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x

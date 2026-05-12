import torch
import torch.nn as nn


class LeniaDecoder(nn.Module):
    """CNN decoder that reconstructs a Lenia frame from an embedding.

    Mirrors the LeniaEncoder: four transposed conv blocks upsample a
    (256, 4, 4) feature map back to (1, 64, 64).

    Args:
        embed_dim: Input embedding dimensionality. Must match encoder. Default: 128.

    Input:
        z: Tensor of shape (batch, embed_dim).

    Output:
        Tensor of shape (batch, 1, 64, 64) with values in [0, 1].
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.projection = nn.Linear(embed_dim, 256 * 4 * 4)

        # Four stride-2 transposed conv blocks: 4 -> 8 -> 16 -> 32 -> 64 spatial dims
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.projection(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.deconv_blocks(x)
        x = self.output_activation(x)
        return x

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


class SpatialLeniaDecoder(nn.Module):
    """Decoder that reconstructs a Lenia frame from a spatial feature map.

    Takes the pre-GAP (B, 256, 4, 4) output of LeniaEncoder directly,
    preserving spatial structure that AdaptiveAvgPool would discard.
    Use with encoder.forward(x, return_spatial=True).

    Input:
        features: Tensor of shape (batch, 256, 4, 4).

    Output:
        Tensor of shape (batch, 1, 64, 64) with values in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.deconv_blocks(features))


class PatchDecoder(nn.Module):
    """Decode patch embeddings back to a Lenia frame.

    Takes (B, N_patches, embed_dim), reshapes to a spatial grid,
    then upsamples to the full frame resolution via transposed convolutions.

    Assumes square image with square patches:
        img_size=64, patch_size=8 → N_patches=64, grid=8×8

    Args:
        embed_dim: Patch embedding dimensionality. Default: 256.
        patch_size: Side length of each patch in pixels. Default: 8.
        img_size: Spatial side-length of input frames. Default: 64.

    Input:  (batch, N_patches, embed_dim)
    Output: (batch, 1, img_size, img_size) with values in [0, 1].
    """

    def __init__(self, embed_dim: int = 256, patch_size: int = 8, img_size: int = 64) -> None:
        super().__init__()
        self.n_side = img_size // patch_size  # 8 — number of patches per side

        # Project each patch embedding to 256 channels for deconvolution
        self.patch_proj = nn.Linear(embed_dim, 256)

        # Three stride-2 deconv blocks: 8→16→32→64
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_proj(x)                          # (B, N, 256)
        x = x.permute(0, 2, 1)                          # (B, 256, N)
        x = x.reshape(B, 256, self.n_side, self.n_side) # (B, 256, 8, 8)
        x = self.deconv_blocks(x)                        # (B, 1, 64, 64)
        return self.output_activation(x)

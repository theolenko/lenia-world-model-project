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

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        x = self.conv_blocks(x)          # (B, 256, 4, 4)
        if return_spatial:
            return x                     # spatial feature map — no GAP, no projection
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x
class SimpleCNNEncoder(nn.Module):
    """Lightweight CNN encoder — shallower and simpler than LeniaEncoder.

    Three conv blocks with 3x3 kernels and MaxPool instead of stride-2
    convolutions. No BatchNorm. Useful as a low-capacity baseline.

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

        # Three conv+pool blocks: spatial dims 64 -> 32 -> 16 -> 8
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for Lenia frames.

    No CLS token.  The full set of patch representations is returned by default,
    preserving spatial structure for a downstream temporal world model.

    When ``pool=True`` the N patch vectors are averaged (Global Average Pooling)
    into a single (B, embed_dim) vector.  This makes the encoder a drop-in
    replacement for LeniaEncoder inside JEPAWorldModel / PixelWorldModel.

    Pipeline
    --------
    (B, C, H, W)
        ↓  Conv2d(kernel=stride=patch_size)  →  patch embedding
    (B, embed_dim, H//P, W//P)
        ↓  flatten spatial, transpose
    (B, N_patches, embed_dim)
        ↓  + learnable positional embedding
        ↓  Transformer encoder (L layers of MHA + MLP + residuals + LayerNorm)
        ↓  LayerNorm
    (B, N_patches, embed_dim)          pool=False  (default)
        or
    (B, embed_dim)                     pool=True   (JEPA / PixelWorldModel)

    Args:
        embed_dim:    Token / model dimensionality. Default: 256.
        patch_size:   Side length of each square patch in pixels. Default: 8.
                      Must divide img_size evenly.
                      img_size=64, patch_size=8  →  N_patches = 64.
        in_channels:  Input channels (1 = grayscale, 3 = RGB). Default: 1.
        num_heads:    Self-attention heads. embed_dim % num_heads == 0. Default: 8.
        num_layers:   Transformer encoder layers. Default: 6.
        mlp_ratio:    FFN hidden-dim expansion factor. Default: 4.0.
        dropout:      Dropout in attention and FFN. Default: 0.0.
        img_size:     Spatial side-length of input frames. Default: 64.
        pool:         If True, apply Global Average Pooling over the patch
                      dimension after the Transformer and return (B, embed_dim).
                      If False, return the full token map (B, N_patches, embed_dim).
                      Default: False.

    Input:
        x : Tensor (batch, in_channels, img_size, img_size), values in [0, 1].

    Output:
        pool=False → Tensor (batch, N_patches, embed_dim)
        pool=True  → Tensor (batch, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 8,
        in_channels: int = 1,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        img_size: int = 64,
        pool: bool = False,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pool = pool

        # ── 1. Patch Embedding ────────────────────────────────────────────────
        # Conv2d with kernel_size = stride = patch_size is mathematically
        # identical to: split into P×P patches → flatten → shared Linear.
        # Advantage: one fused GPU kernel, no intermediate reshape buffer.
        # Output: (B, embed_dim, H//P, W//P)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # ── 2. Positional Embedding ───────────────────────────────────────────
        # Self-attention is permutation-invariant — without positional info every
        # patch looks identical regardless of its location in the frame.
        # Shape (1, N_patches, embed_dim): broadcast over the batch dimension.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # ── 3. Transformer Encoder Blocks ─────────────────────────────────────
        # Each layer: Multi-Head Self-Attention → residual → LayerNorm → MLP → residual
        # norm_first=True  →  Pre-LN layout (more stable gradients than Post-LN)
        # batch_first=True →  layout (B, seq, dim) throughout, matching our tensors
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # incompatible with norm_first; no effect on output
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 1. Patch embedding ────────────────────────────────────────────────
        # (B, C, H, W) → (B, embed_dim, H//P, W//P) → (B, N_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # ── 2. Positional embedding ───────────────────────────────────────────
        x = x + self.pos_embed

        # ── 3. Transformer + LayerNorm ────────────────────────────────────────
        x = self.norm(self.transformer(x))

        # ── 4. Optional pooling ───────────────────────────────────────────────
        # Mean over the patch dimension: each of the N_patches vectors
        # contributes equally to the final representation.
        # (B, N_patches, embed_dim) → (B, embed_dim)
        if self.pool:
            x = x.mean(dim=1)

        return x

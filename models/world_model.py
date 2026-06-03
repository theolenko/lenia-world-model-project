import copy
import torch
import torch.nn as nn

from models.encoder import LeniaEncoder, SimpleCNNEncoder, ViTEncoder
from models.decoder import LeniaDecoder
from models.predictor import LeniaPredictor

_ENCODER_REGISTRY = {
    "cnn": LeniaEncoder,
    "simple_cnn": SimpleCNNEncoder,
    # pool=True collapses (B, N_patches, embed_dim) → (B, embed_dim) via
    # Global Average Pooling, making ViTEncoder compatible with JEPAWorldModel
    # and PixelWorldModel, which both expect a flat embedding vector.
    "vit": lambda embed_dim: ViTEncoder(embed_dim=embed_dim, pool=True),
}


def _build_encoder(encoder_type: str, embed_dim: int) -> nn.Module:
    if encoder_type not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: {list(_ENCODER_REGISTRY)}"
        )
    return _ENCODER_REGISTRY[encoder_type](embed_dim=embed_dim)

class PixelWorldModel(nn.Module):
    """World model for Setup A: predicts next Lenia frame in pixel space.

    The encoder embeds frame_t, the decoder reconstructs from that embedding.
    Both components share the same embed_dim.

    Args:
        embed_dim: Embedding dimensionality. Default: 128.
    """

    def __init__(self, embed_dim: int = 128, encoder_type: str ="cnn") -> None:
        super().__init__()
        self.encoder = _build_encoder(encoder_type,embed_dim)
        self.decoder = LeniaDecoder(embed_dim=embed_dim)

    def forward(self, frame_t: torch.Tensor) -> torch.Tensor:
        """Encode frame_t and decode to a predicted pixel-space output.

        Args:
            frame_t: Tensor of shape (batch, 1, 64, 64).

        Returns:
            predicted_frame: Tensor of shape (batch, 1, 64, 64).
        """
        z = self.encoder(frame_t)
        return self.decoder(z)

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Return the embedding for a frame (used in probing/evaluation).

        Args:
            frame: Tensor of shape (batch, 1, 64, 64).

        Returns:
            Tensor of shape (batch, embed_dim).
        """
        return self.encoder(frame)


class JEPAWorldModel(nn.Module):
    """World model for Setup B: JEPA — predicts next frame's embedding.

    Uses an online encoder + predictor trained against a target encoder updated
    via EMA. The target encoder's parameters are frozen (stop-gradient).

    Args:
        embed_dim: Embedding dimensionality. Default: 128.
        ema_momentum: EMA decay for target encoder updates. Default: 0.99.
        predictor_hidden_dim: Hidden width of the MLP predictor. Default: 256.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        ema_momentum: float = 0.99,
        predictor_hidden_dim: int = 256,
        encoder_type: str = "cnn",
    ) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum

        self.online_encoder = _build_encoder(encoder_type, embed_dim)
        self.target_encoder = _build_encoder(encoder_type, embed_dim)
        self.predictor = LeniaPredictor(embed_dim=embed_dim, hidden_dim=predictor_hidden_dim)

        # Initialise target encoder as a copy of the online encoder
        self.target_encoder.load_state_dict(
            copy.deepcopy(self.online_encoder.state_dict())
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "JEPAWorldModel":
        super().train(mode)
        self.target_encoder.eval()  # target encoder always in eval — stable BN running stats
        return self

    def forward(
        self, frame_t: torch.Tensor, frame_t_plus_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predicted and target embeddings for a consecutive frame pair.

        Args:
            frame_t: Current frame, shape (batch, 1, 64, 64).
            frame_t_plus_1: Next frame, shape (batch, 1, 64, 64).

        Returns:
            (z_pred, z_target, z_context): All of shape (batch, embed_dim).
            z_pred and z_context have gradients; z_target is detached (stop-gradient).
        """
        z_context = self.online_encoder(frame_t)
        with torch.no_grad():
            z_target = self.target_encoder(frame_t_plus_1)
        z_pred = self.predictor(z_context)
        return z_pred, z_target, z_context

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target encoder from online encoder weights."""
        for online_p, target_p in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(self.ema_momentum).add_(
                online_p.data, alpha=1.0 - self.ema_momentum
            )

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Return the online encoder embedding (used in probing/evaluation).

        Args:
            frame: Tensor of shape (batch, 1, 64, 64).

        Returns:
            Tensor of shape (batch, embed_dim).
        """
        return self.online_encoder(frame)

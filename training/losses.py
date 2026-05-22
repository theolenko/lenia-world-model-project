import torch
import torch.nn.functional as F


def pixel_prediction_loss(
    predicted_frame: torch.Tensor, true_frame: torch.Tensor
) -> torch.Tensor:
    """MSE loss between predicted and true Lenia frames (Setup A).

    Args:
        predicted_frame: Tensor of shape (batch, 1, 64, 64).
        true_frame: Tensor of shape (batch, 1, 64, 64).

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(predicted_frame, true_frame)


def jepa_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """MSE loss between predicted and target embeddings (Setup B / JEPA).

    Args:
        z_pred: Predicted embedding, shape (batch, embed_dim).
        z_target: Target encoder embedding, shape (batch, embed_dim).

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(z_pred, z_target)


def variance_regularization(
    embeddings: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """VICReg-style variance term that penalizes low per-dimension variance.

    Encourages each embedding dimension to have standard deviation >= 1
    across the batch. Useful as an anti-collapse regulariser for JEPA.

    Args:
        embeddings: Tensor of shape (batch, embed_dim).
        eps: Small constant for numerical stability in sqrt. Default: 1e-4.

    Returns:
        Scalar loss tensor (mean over dimensions of max(0, 1 - std_d)).
    """
    std = torch.sqrt(embeddings.var(dim=0) + eps)  # (embed_dim,)
    return torch.mean(F.relu(1.0 - std))


def covariance_regularization(embeddings: torch.Tensor) -> torch.Tensor:
    """VICReg-style covariance term that penalises off-diagonal covariance.

    Encourages different embedding dimensions to carry independent information,
    reducing redundancy. Useful as an anti-collapse regulariser for JEPA.

    Args:
        embeddings: Tensor of shape (batch, embed_dim).

    Returns:
        Scalar loss tensor (sum of squared off-diagonal cov entries / embed_dim).
    """
    batch_size, embed_dim = embeddings.shape
    z = embeddings - embeddings.mean(dim=0)
    cov = (z.T @ z) / (batch_size - 1)  # (embed_dim, embed_dim)

    # Zero out the diagonal, sum squares of off-diagonal entries
    off_diag_mask = ~torch.eye(embed_dim, dtype=torch.bool, device=embeddings.device)
    return cov.pow(2)[off_diag_mask].sum() / embed_dim

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm


def _load_inception(device: torch.device) -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_inception_features(
    dataloader: DataLoader,
    device: torch.device,
    inception: nn.Module | None = None,
) -> np.ndarray:
    """Extract InceptionV3 pool3 features (2048-d) from a DataLoader.

    Images must be in [-1, 1] range (as produced by get_dataloaders).
    Pass a pre-loaded `inception` model to avoid reloading it between calls.
    Returns array of shape (N, 2048).
    """
    if inception is None:
        inception = _load_inception(device)
    inception = inception.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    features = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        imgs = batch.to(device)
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] → [0, 1]
        imgs = (imgs - mean) / std  # ImageNet normalisation
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        feats = inception(imgs)
        features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Fréchet Inception Distance between two sets of pool3 features."""
    mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_f, sigma_f = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))


def make_fake_dataloader(images: torch.Tensor, batch_size: int = 64) -> DataLoader:
    """Wrap a tensor of generated images in a DataLoader for feature extraction.

    Images must be in [-1, 1] range (matching the training normalisation).
    """
    return DataLoader(TensorDataset(images), batch_size=batch_size)

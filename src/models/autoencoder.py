"""
Shallow fully-connected autoencoder for return-abuse anomaly scoring.

Design contract (brief §8.2)
─────────────────────────────
  Training population
    The autoencoder is trained on the FULL training split — which is ~90%
    normal/impulse customers with mild contamination (~10% serial/fraud).
    This is the production-defensible setup: in deployment, labels are
    unavailable, so we can only assume the bulk of customers are legitimate.

    Alternative: train strictly on known-"normal" customers from the
    latent_type column.  This gives a cleaner reconstruction manifold but
    silently consumes label knowledge not available in production.
    If you switch to this variant, document it explicitly.

  Anomaly score
    Per-sample MSE of the reconstruction on SCALED features.
    High error → customer's feature vector sits far from the normal manifold
    the encoder learned to compress → flagged as anomalous.

  Leakage
    The model and its inner val split are fitted on training data only.
    The StandardScaler used to produce the input must also be fitted on
    training data only (managed in run_phase2.py).

Architecture
────────────
  input_dim → 16 → 8 (bottleneck) → 16 → input_dim

  Shallow depth is intentional: a very deep AE can memorise any distribution,
  defeating anomaly detection.  Two encoder layers compress while retaining
  enough capacity to reconstruct the ~27-dimensional normal manifold faithfully.

  No BatchNorm / Dropout — the dataset fits comfortably in memory and the
  model is small enough that regularisation is not needed; BatchNorm would
  also complicate the reconstruction-error interpretation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ── Model ──────────────────────────────────────────────────────────────────────

class ReturnAbuseAutoencoder(nn.Module):
    """
    Fully-connected autoencoder: input_dim → 16 → 8 → 16 → input_dim.

    Parameters
    ----------
    input_dim : int
        Number of input features (equals len(ALL_FEATURES) from build_features).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            # No output activation — inputs are StandardScaler-transformed
            # (mean 0, unit variance), so unbounded linear output is correct.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Training ───────────────────────────────────────────────────────────────────

def train_autoencoder(
    X_train_scaled: np.ndarray,
    *,
    n_epochs: int = 150,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.10,
    patience: int = 15,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[ReturnAbuseAutoencoder, list[float]]:
    """
    Train the autoencoder on scaled training features.

    An inner 90/10 split drives early stopping.  This inner val set is a
    slice of the training split — it never influences the StandardScaler
    (already fitted before this function is called).

    Parameters
    ----------
    X_train_scaled : np.ndarray, shape (n_train, n_features)
        Output of scaler.transform(X_train).  Must be the TRAINING set only.
    n_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size for the inner DataLoader.
    lr : float
        Adam learning rate.
    val_fraction : float
        Fraction of X_train_scaled held out as the early-stopping val set.
    patience : int
        Early stopping patience (epochs without val improvement).
    seed : int
        RNG seed for the inner val split and torch.
    device : str or None
        'cpu', 'cuda', or None (auto-detect: CUDA if available, else CPU).
    verbose : bool
        Print epoch-level progress every 25 epochs and early-stop event.

    Returns
    -------
    model : ReturnAbuseAutoencoder
        Trained model loaded with best-val-loss weights.
    val_history : list[float]
        Validation MSE at each epoch (useful for diagnosing convergence).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = len(X_train_scaled)
    n_val = max(32, int(n * val_fraction))
    val_idx   = rng.choice(n, n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    X_tr  = torch.tensor(X_train_scaled[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train_scaled[val_idx],   dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, X_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = ReturnAbuseAutoencoder(input_dim=X_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss   = float("inf")
    best_state_dict: dict | None = None
    patience_ctr    = 0
    val_history: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        for xb, _ in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(
                        f"    Early stop @ epoch {epoch + 1}  "
                        f"(best val MSE = {best_val_loss:.6f})"
                    )
                break

        if verbose and (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch + 1:>4d} / {n_epochs}  val MSE = {val_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, val_history


# ── Scoring ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_error(
    model: ReturnAbuseAutoencoder,
    X_scaled: np.ndarray,
    device: str | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Compute per-sample MSE reconstruction error (the anomaly score).

    Higher error → the customer's feature vector deviates more from the
    normal manifold the encoder learned → more anomalous.

    Parameters
    ----------
    model : ReturnAbuseAutoencoder
        A trained autoencoder (will be moved to `device`).
    X_scaled : np.ndarray, shape (n, n_features)
        Scaled features (same scaler as used during training).
    device : str or None
        Device for inference.  Defaults to CPU.
    batch_size : int
        Inference batch size (avoids OOM for large datasets).

    Returns
    -------
    errors : np.ndarray, shape (n,)
        Per-sample mean-squared reconstruction error.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    errors: list[np.ndarray] = []

    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device)
        recon = model(batch)
        err = ((batch - recon) ** 2).mean(dim=1)
        errors.append(err.cpu().numpy())

    return np.concatenate(errors)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_autoencoder(model: ReturnAbuseAutoencoder, path: Path) -> None:
    """
    Save model architecture metadata and state dict together.
    input_dim is saved so the checkpoint is self-describing (no config required
    to reload it).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim":  model.encoder[0].in_features,
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"  Autoencoder saved → {path}")


def load_autoencoder(path: Path) -> ReturnAbuseAutoencoder:
    """Load a previously saved autoencoder and return it in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ReturnAbuseAutoencoder(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

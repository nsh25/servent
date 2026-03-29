from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def build_mixture_centers(
    x0: np.ndarray,
    directions: np.ndarray,
    alphas: np.ndarray,
    center_scaling: float = 1.0,
) -> np.ndarray:
    """Create mixture centers mu_m = x0 - c * alpha_m * d_m."""
    x0 = np.asarray(x0, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    return x0[None, :] - center_scaling * alphas[:, None] * directions


def choose_mixture_weights(alphas: np.ndarray, mode: str = "uniform", beta: float = 1.0) -> np.ndarray:
    """Choose mixture probabilities by uniform or distance-based rule."""
    a2 = np.asarray(alphas, dtype=np.float64) ** 2
    m = len(a2)
    if m == 0:
        raise ValueError("No active mixture components")
    if mode == "uniform":
        p = np.ones(m, dtype=np.float64) / m
    elif mode == "distance_based":
        logits = -beta * a2
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
    else:
        raise ValueError(f"Unknown mixture weight mode: {mode}")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("Mixture probabilities do not sum to one")
    return p


def sample_from_mixture(centers: np.ndarray, probs: np.ndarray, sigma: float, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample y from Gaussian mixture with isotropic variance sigma^2."""
    comps = rng.choice(len(probs), size=n_samples, p=probs)
    noise = rng.normal(0.0, sigma, size=(n_samples, centers.shape[1]))
    y = centers[comps] + noise
    return y, comps


def log_gaussian_pdf_diag(y: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Log pdf of isotropic Gaussian N(mu, sigma^2 I) for batch y."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    d = y.shape[-1]
    quad = np.sum((y - mu) ** 2, axis=-1) / (sigma * sigma)
    return -0.5 * (d * np.log(2.0 * np.pi * sigma * sigma) + quad)


def log_target_density(y: np.ndarray, x0: np.ndarray, sigma: float) -> np.ndarray:
    """Target log-density p(y)=N(x0,sigma^2I)."""
    return log_gaussian_pdf_diag(y, x0, sigma)


def log_proposal_density(y: np.ndarray, centers: np.ndarray, probs: np.ndarray, sigma: float) -> np.ndarray:
    """Proposal log-density q(y)=sum pi_m N(mu_m,sigma^2I), evaluated stably."""
    y = np.asarray(y, dtype=np.float64)
    terms = []
    for m in range(len(probs)):
        terms.append(np.log(probs[m]) + log_gaussian_pdf_diag(y, centers[m], sigma))
    return logsumexp(np.vstack(terms), axis=0)


def importance_weight(y: np.ndarray, x0: np.ndarray, centers: np.ndarray, probs: np.ndarray, sigma: float) -> np.ndarray:
    """Compute importance weight p(y)/q(y) from log densities."""
    lw = log_target_density(y, x0, sigma) - log_proposal_density(y, centers, probs, sigma)
    return np.exp(np.clip(lw, -745.0, 700.0))

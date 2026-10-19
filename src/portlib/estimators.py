"""Moment estimators: sample, Ledoit-Wolf constant-correlation shrinkage, Bayes-Stein means."""
from __future__ import annotations
import numpy as np


def sample_moments(R: np.ndarray):
    """R: T x N matrix of returns. Returns (mu, Sigma) with the unbiased covariance."""
    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False, ddof=1)
    return mu, np.atleast_2d(Sigma)


def ledoit_wolf_cc(R: np.ndarray):
    """Ledoit & Wolf (2004, JPM) 'Honey, I shrunk the sample covariance matrix': shrink the sample
    covariance toward the constant-correlation target with the analytically optimal intensity.
    Returns (Sigma_shrunk, delta)."""
    T, N = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / T
    var = np.diag(S)
    sd = np.sqrt(var)
    corr = S / np.outer(sd, sd)
    rbar = (corr.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    # pi-hat: sum of asymptotic variances of the sample covariance entries
    X2 = X ** 2
    piMat = X2.T @ X2 / T - S ** 2
    pihat = piMat.sum()
    # rho-hat: covariance between sample covariances and the target's off-diagonal entries
    term1 = (X ** 3).T @ X / T
    help_ = X.T @ X / T
    helpDiag = np.diag(help_)
    term2 = helpDiag[:, None] * S
    term3 = help_ * var[:, None]
    term4 = var[:, None] * S
    thetaMat = term1 - term2 - term3 + term4
    np.fill_diagonal(thetaMat, 0.0)
    rhohat = np.trace(piMat) + rbar * ((1.0 / sd)[:, None] * sd[None, :] * thetaMat).sum()
    gammahat = np.linalg.norm(S - F, "fro") ** 2
    kappa = (pihat - rhohat) / gammahat if gammahat > 0 else 0.0
    delta = float(np.clip(kappa / T, 0.0, 1.0))
    Sigma = delta * F + (1 - delta) * S
    # rescale to the unbiased (ddof=1) convention used elsewhere
    Sigma *= T / (T - 1)
    return Sigma, delta


def bayes_stein_mean(R: np.ndarray, Sigma: np.ndarray):
    """Jorion (1986) Bayes-Stein shrinkage of the mean vector toward the grand mean implied by the
    global-minimum-variance portfolio. Returns (mu_bs, shrink_weight)."""
    T, N = R.shape
    mu = R.mean(axis=0)
    ones = np.ones(N)
    Sinv = np.linalg.pinv(Sigma)
    mu_g = float(ones @ Sinv @ mu / (ones @ Sinv @ ones))
    d = mu - mu_g * ones
    lam = (N + 2) / ((N + 2) + T * float(d @ Sinv @ d))
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1 - lam) * mu + lam * mu_g * ones, lam


def nearest_psd(Sigma: np.ndarray, eps: float = 1e-10):
    w, V = np.linalg.eigh((Sigma + Sigma.T) / 2)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

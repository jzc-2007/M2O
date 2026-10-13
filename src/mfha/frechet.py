"""Small dependency-free (apart from NumPy/SciPy) Fréchet distance helper."""

import numpy as np
import scipy.linalg


def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
  mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
  sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
  if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
    raise ValueError("FID statistics have incompatible shapes")
  diff = mu1 - mu2
  covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      raise ValueError(f"FID covariance square root has imaginary part {np.max(np.abs(covmean.imag))}")
    covmean = covmean.real
  return float(
      diff.dot(diff)
      + np.trace(sigma1)
      + np.trace(sigma2)
      - 2 * np.trace(covmean)
  )

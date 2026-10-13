"""Streaming VAE decode and FID helpers shared by teacher/student evaluation."""

from __future__ import annotations

import numpy as np


def save_fid_features(path, features, dtype="float16"):
  """Export features without changing the array used for the reported FID."""
  if dtype not in ("float16", "float32"):
    raise ValueError("feature export dtype must be float16 or float32")
  np.save(path, features.astype(dtype))


class StreamingFIDEvaluator:
  def __init__(self, config):
    # Lazy imports keep trajectory/unit tests independent of torch/diffusers.
    from .legacy_utils import fid_util
    from .legacy_utils.vae_util import LatentManager
    from .latent_codec import IMFLatentDecoder

    self.fid_util = fid_util
    self.batch_size = int(config["device_batch_size"])
    self.latent_codec = config.get("latent_codec", "ldm")
    if self.latent_codec == "imeanflow":
      self.latent_manager = IMFLatentDecoder(
          self.batch_size, 32, config["vae_path"]
      )
    else:
      self.latent_manager = LatentManager(
          "mse",
          self.batch_size,
          32,
          vae_path=config["vae_path"],
          enable_decoder=True,
      )
    self.inception = fid_util.build_jax_inception(
        batch_size=self.batch_size,
        checkpoint_path=config["inception_checkpoint"],
        backend=config.get("inception_backend"),
    )
    self.reference = fid_util.get_reference(config["fid_reference"], self.inception)

  def _decode_fixed_batch(self, latent_bhwc):
    import jax
    import jax.numpy as jnp
    import torch

    actual = latent_bhwc.shape[0]
    if actual > self.batch_size:
      raise ValueError("decode batch exceeds configured device batch")
    if actual < self.batch_size:
      pad = jnp.zeros((self.batch_size - actual,) + latent_bhwc.shape[1:], latent_bhwc.dtype)
      latent_bhwc = jnp.concatenate((latent_bhwc, pad), axis=0)
    latent_bchw = latent_bhwc.transpose(0, 3, 1, 2)
    decoded = self.latent_manager.decode(latent_bchw)
    decoded = decoded.transpose(0, 2, 3, 1)
    padded_images = jnp.clip(127.5 * decoded + 128.0, 0, 255).astype(jnp.uint8)

    tensor = torch.from_numpy(np.asarray(jax.device_get(padded_images))).float().permute(0, 3, 1, 2)
    resized = self.fid_util.resize.forward(tensor)
    resized = resized.numpy().transpose(0, 2, 3, 1)
    features = self.inception["fn"](
        self.inception["params"], jax.lax.stop_gradient(resized)
    )
    features = features.squeeze(axis=1).squeeze(axis=1)
    return (
        np.asarray(jax.device_get(padded_images[:actual])),
        np.asarray(jax.device_get(features[:actual])),
    )

  def decode_and_features(self, latent_bhwc):
    images, features = [], []
    for start in range(0, latent_bhwc.shape[0], self.batch_size):
      image_batch, feature_batch = self._decode_fixed_batch(
          latent_bhwc[start : start + self.batch_size]
      )
      images.append(image_batch)
      features.append(feature_batch)
    return np.concatenate(images), np.concatenate(features)

  def fid(self, features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return float(
        self.fid_util.compute_fid(
            mu, self.reference["mu"], sigma, self.reference["sigma"]
        )
    )


def save_image_grid(images, path, rows, columns, row_labels=None):
  from PIL import Image, ImageDraw

  if len(images) != rows * columns:
    raise ValueError("grid dimensions do not match image count")
  height, width = images.shape[1:3]
  canvas = Image.new("RGB", (columns * width, rows * height), color="white")
  for idx, image in enumerate(images):
    canvas.paste(Image.fromarray(image), ((idx % columns) * width, (idx // columns) * height))
  draw = ImageDraw.Draw(canvas)
  if row_labels is None:
    row_labels = [f"t={1-row/(rows-1):.3f}" for row in range(rows)]
  for row, label in enumerate(row_labels):
    draw.text((4, row * height + 4), label, fill=(255, 0, 0))
  canvas.save(path)

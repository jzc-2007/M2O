"""Latent-coordinate conversions for model-specific VAE decoding."""

from __future__ import annotations

import jax.numpy as jnp


IMF_LATENT_MEAN = (0.86488, -0.27787343, 0.21616915, 0.3738409)
IMF_LATENT_STD = (4.85503674, 5.31922414, 3.93725398, 3.9870003)


def normalized_latents_to_vae(latents, codec="ldm", channel_axis=-1):
  """Convert normalized model latents to the SD VAE coordinate system."""
  if codec in (None, "ldm", "ldm_scale"):
    return latents / jnp.asarray(0.18215, dtype=latents.dtype)
  if codec != "imeanflow":
    raise ValueError(f"unknown latent codec: {codec}")

  axis = channel_axis if channel_axis >= 0 else latents.ndim + channel_axis
  if not 0 <= axis < latents.ndim:
    raise ValueError(f"invalid channel axis {channel_axis} for rank {latents.ndim}")
  if latents.shape[axis] != 4:
    raise ValueError("iMeanFlow latents must have four channels")
  shape = [1] * latents.ndim
  shape[axis] = 4
  mean = jnp.asarray(IMF_LATENT_MEAN, dtype=latents.dtype).reshape(shape)
  std = jnp.asarray(IMF_LATENT_STD, dtype=latents.dtype).reshape(shape)
  return latents * std + mean


def vae_latents_to_normalized(latents, codec="ldm", channel_axis=-1):
  """Convert VAE-coordinate latents into a model's normalized coordinates."""
  if codec in (None, "ldm", "ldm_scale"):
    return latents * jnp.asarray(0.18215, dtype=latents.dtype)
  if codec != "imeanflow":
    raise ValueError(f"unknown latent codec: {codec}")
  axis = channel_axis if channel_axis >= 0 else latents.ndim + channel_axis
  if not 0 <= axis < latents.ndim:
    raise ValueError(f"invalid channel axis {channel_axis} for rank {latents.ndim}")
  if latents.shape[axis] != 4:
    raise ValueError("iMeanFlow latents must have four channels")
  shape = [1] * latents.ndim
  shape[axis] = 4
  mean = jnp.asarray(IMF_LATENT_MEAN, dtype=latents.dtype).reshape(shape)
  std = jnp.asarray(IMF_LATENT_STD, dtype=latents.dtype).reshape(shape)
  return (latents - mean) / std


def convert_normalized_latents(latents, source, target, channel_axis=-1):
  """Convert between normalized LDM and iMeanFlow latent coordinates."""
  if source == target:
    return latents
  vae_latents = normalized_latents_to_vae(
      latents, codec=source, channel_axis=channel_axis
  )
  return vae_latents_to_normalized(
      vae_latents, codec=target, channel_axis=channel_axis
  )


class IMFLatentDecoder:
  """Compiled VAE decoder for iMeanFlow's per-channel latent coordinates."""

  def __init__(self, decode_batch_size, input_size, vae_path):
    # Import lazily so codec unit tests do not require diffusers.
    from .legacy_utils.vae_util import LatentManager

    self._manager = LatentManager(
        "mse",
        decode_batch_size,
        input_size,
        vae_path=vae_path,
        enable_decoder=True,
    )

  def decode(self, latents_bchw):
    vae_latents = normalized_latents_to_vae(
        latents_bchw, codec="imeanflow", channel_axis=1
    )
    return self._manager.decode_fn(vae_latents)["sample"]

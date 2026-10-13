"""Generated-only ConvNeXt distribution loss for paired self-teacher batches."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fid_perceptual import _paired_offdiagonal_kid, _unbiased_kid
from .latent_codec import normalized_latents_to_vae


def build_convnext_kid_feature_loss(config):
  """Match student and frozen teacher marginal features with paired KID.

  Only synthetic student/teacher endpoints enter this function.  The frozen
  VAE and ConvNeXt weights supply a feature map, not a real-data target.
  """
  from diffusers.models import FlaxAutoencoderKL
  from .convnext_biflow import load_convnext_jax_model

  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      config["evaluation"]["vae_path"]
  )
  feature_config = config["training"]["feature_loss"]
  convnext, convnext_params = load_convnext_jax_model(
      model_name=feature_config.get("model_size", "tiny"),
      use_bf16=bool(feature_config.get("bfloat16", True)),
      pretrained_path=feature_config["pretrained_path"],
  )
  feature_key = feature_config.get("feature", "global_mean")
  cross_term = feature_config.get("cross_term", "paired_offdiagonal")
  if cross_term not in ("paired_offdiagonal", "all"):
    raise ValueError(f"unsupported ConvNeXt KID cross_term: {cross_term!r}")
  latent_codec = feature_config.get(
      "latent_codec", config["evaluation"].get("latent_codec", "ldm")
  )

  def embed(latent):
    vae_latent = normalized_latents_to_vae(
        latent.transpose((0, 3, 1, 2)),
        codec=latent_codec,
        channel_axis=1,
    )
    image = vae.apply(
        {"params": vae_params}, vae_latent, method=FlaxAutoencoderKL.decode
    ).sample
    image = image.transpose((0, 2, 3, 1)).astype(jnp.float32)
    image = jax.image.resize(
        image, (image.shape[0], 224, 224, 3), method="bicubic"
    )
    features = convnext.apply(
        convnext_params, image, method=convnext.get_activations
    )[feature_key]
    return features.astype(jnp.float32).reshape((features.shape[0], -1))

  def feature_loss(predicted_latent, teacher_latent):
    predicted = embed(predicted_latent)
    target = jax.lax.stop_gradient(embed(teacher_latent))
    if cross_term == "all":
      return _unbiased_kid(predicted, target)
    return _paired_offdiagonal_kid(predicted, target)

  return feature_loss

"""Frozen decoded-image feature losses used by BiFlow-style distillation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .latent_codec import normalized_latents_to_vae


def _rms_normalize(value):
  value = value.astype(jnp.float32).reshape((value.shape[0], -1))
  return value / (jnp.sqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True)) + 1e-8)


def relational_feature_losses(predicted, target):
  """Return pointwise, pairwise-geometry, and batch-moment feature losses."""
  predicted = _rms_normalize(predicted)
  target = jax.lax.stop_gradient(_rms_normalize(target))
  pointwise = jnp.mean(jnp.square(predicted - target))

  feature_dim = predicted.shape[-1]
  predicted_gram = predicted @ predicted.T / feature_dim
  target_gram = target @ target.T / feature_dim
  relational = jnp.mean(jnp.square(predicted_gram - target_gram))

  predicted_mean = jnp.mean(predicted, axis=0)
  target_mean = jnp.mean(target, axis=0)
  predicted_std = jnp.sqrt(
      jnp.mean(jnp.square(predicted - predicted_mean), axis=0) + 1e-6
  )
  target_std = jnp.sqrt(
      jnp.mean(jnp.square(target - target_mean), axis=0) + 1e-6
  )
  statistics = (
      jnp.mean(jnp.square(predicted_mean - target_mean))
      + jnp.mean(jnp.square(predicted_std - target_std))
  )
  return pointwise, relational, statistics


def combine_relational_feature_losses(
    pointwise,
    relational,
    statistics,
    *,
    pointwise_weight=1.0,
    relational_weight=0.0,
    statistics_weight=0.0,
    balance_auxiliary=True,
):
  """Combine ConvNeXt losses, optionally without paired supervision.

  Setting ``pointwise_weight=0`` is important for unpaired real/generated
  batches: their row ordering does not define a valid feature target. The
  relational and statistics terms remain distribution-level objectives.
  """
  if balance_auxiliary:
    relational = relational * jax.lax.stop_gradient(
        pointwise / (relational + 1e-8)
    )
    statistics = statistics * jax.lax.stop_gradient(
        pointwise / (statistics + 1e-8)
    )
  return (
      float(pointwise_weight) * pointwise
      + float(relational_weight) * relational
      + float(statistics_weight) * statistics
  )


def build_convnext_feature_loss(config):
  """Return a differentiable BiFlow ConvNeXt feature-distance function.

  Both endpoints are decoded with the frozen SD VAE. ConvNeXt stage features
  are independently RMS-normalized per example before their MSEs are summed,
  matching the class-head loss used in BiFlow.
  """
  from diffusers.models import FlaxAutoencoderKL
  from .convnext_biflow import load_convnext_jax_model

  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      config["evaluation"]["vae_path"]
  )
  feature_config = config["training"]["feature_loss"]
  convnext, convnext_params = load_convnext_jax_model(
      model_name=feature_config.get("model_size", "base"),
      use_bf16=bool(feature_config.get("bfloat16", True)),
      pretrained_path=feature_config.get("pretrained_path"),
  )
  selected = tuple(feature_config.get("features", ()))
  pointwise_weight = float(feature_config.get("pointwise_weight", 1.0))
  relational_weight = float(feature_config.get("relational_weight", 0.0))
  statistics_weight = float(feature_config.get("statistics_weight", 0.0))
  balance_auxiliary = bool(
      feature_config.get("balance_auxiliary_losses", True)
  )
  latent_codec = feature_config.get(
      "latent_codec", config["evaluation"].get("latent_codec", "ldm")
  )

  def decode(latent):
    latent = normalized_latents_to_vae(
        latent.transpose((0, 3, 1, 2)),
        codec=latent_codec,
        channel_axis=1,
    )
    image = vae.apply(
        {"params": vae_params}, latent, method=FlaxAutoencoderKL.decode
    ).sample
    image = image.transpose((0, 2, 3, 1)).astype(jnp.float32)
    return jax.image.resize(
        image, (image.shape[0], 224, 224, 3), method="bicubic"
    )

  def feature_loss(predicted_latent, target_latent):
    predicted_image = decode(predicted_latent)
    target_image = jax.lax.stop_gradient(decode(target_latent))
    predicted = convnext.apply(
        convnext_params, predicted_image, method=convnext.get_activations
    )
    target = convnext.apply(
        convnext_params, target_image, method=convnext.get_activations
    )
    keys = selected or tuple(sorted(predicted))
    losses = []
    for key in keys:
      pointwise, relational, statistics = relational_feature_losses(
          predicted[key], target[key]
      )
      losses.append(combine_relational_feature_losses(
          pointwise,
          relational,
          statistics,
          pointwise_weight=pointwise_weight,
          relational_weight=relational_weight,
          statistics_weight=statistics_weight,
          balance_auxiliary=balance_auxiliary,
      ))
    return jnp.sum(jnp.stack(losses))

  return feature_loss

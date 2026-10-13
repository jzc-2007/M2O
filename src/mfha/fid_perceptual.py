"""Differentiable training-time losses in the official FID feature space."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .latent_codec import normalized_latents_to_vae


def _off_diagonal_mean(matrix):
  """Return the mean over all non-diagonal entries of a square matrix."""
  count = matrix.shape[0]
  if count < 2:
    return jnp.asarray(0.0, matrix.dtype)
  return (jnp.sum(matrix) - jnp.trace(matrix)) / (count * (count - 1))


def _unbiased_kid(predicted, target):
  """Unbiased cubic-polynomial MMD in normalized Inception feature space."""
  feature_dim = predicted.shape[-1]
  target_scale = jax.lax.stop_gradient(
      jnp.sqrt(jnp.mean(jnp.square(target)) + 1e-6)
  )
  predicted = predicted / target_scale
  target = target / target_scale

  def kernel(left, right):
    return (left @ right.T / feature_dim + 1.0) ** 3

  return (
      _off_diagonal_mean(kernel(predicted, predicted))
      + _off_diagonal_mean(kernel(target, target))
      - 2.0 * jnp.mean(kernel(predicted, target))
  )


def _paired_offdiagonal_kid(predicted, target):
  """Estimate marginal KID when student/teacher samples share noise IDs.

  Excluding the paired cross diagonal makes cross terms independent draws
  from the two marginals. In particular, identical paired feature batches
  have exactly zero loss, unlike the ordinary independent-batch estimator.
  """
  feature_dim = predicted.shape[-1]
  target_scale = jax.lax.stop_gradient(
      jnp.sqrt(jnp.mean(jnp.square(target)) + 1e-6)
  )
  predicted = predicted / target_scale
  target = target / target_scale

  def kernel(left, right):
    return (left @ right.T / feature_dim + 1.0) ** 3

  return (
      _off_diagonal_mean(kernel(predicted, predicted))
      + _off_diagonal_mean(kernel(target, target))
      - 2.0 * _off_diagonal_mean(kernel(predicted, target))
  )


def inception_feature_losses(predicted, target, covariance_mode="gram"):
  """Return scale-normalized paired, mean, and covariance feature losses.

  ``gram`` preserves the original paired-batch behavior. ``diagonal`` matches
  per-feature variances and is permutation invariant, so it is appropriate
  when generated and real examples are independent draws rather than pairs.
  """
  predicted = predicted.astype(jnp.float32).reshape((predicted.shape[0], -1))
  target = jax.lax.stop_gradient(
      target.astype(jnp.float32).reshape((target.shape[0], -1))
  )
  feature_scale = jax.lax.stop_gradient(jnp.mean(jnp.square(target)) + 1e-6)
  pointwise = jnp.mean(jnp.square(predicted - target)) / feature_scale

  predicted_mean = jnp.mean(predicted, axis=0)
  target_mean = jnp.mean(target, axis=0)
  mean_scale = jax.lax.stop_gradient(
      jnp.mean(jnp.square(target_mean)) + 1e-6
  )
  mean_loss = jnp.mean(jnp.square(predicted_mean - target_mean)) / mean_scale

  predicted_centered = predicted - predicted_mean
  target_centered = target - target_mean
  if covariance_mode == "gram":
    denominator = max(predicted.shape[0] - 1, 1)
    # This low-rank Gram comparison is useful for genuinely paired batches,
    # but is not invariant to independently permuting one of the batches.
    predicted_covariance = (
        predicted_centered @ predicted_centered.T / denominator
    )
    target_covariance = target_centered @ target_centered.T / denominator
  elif covariance_mode == "diagonal":
    # FID covariance is a distributional statistic. Matching its diagonal is
    # a stable, permutation-invariant proxy when minibatches are unpaired.
    predicted_covariance = jnp.mean(jnp.square(predicted_centered), axis=0)
    target_covariance = jnp.mean(jnp.square(target_centered), axis=0)
  elif covariance_mode == "kid":
    # KID is an unbiased, permutation-invariant two-sample objective. Unlike
    # direct minibatch mean matching, it has no variance-collapse bias for
    # independent generated and real examples.
    return pointwise, mean_loss, _unbiased_kid(predicted, target)
  elif covariance_mode == "kid_paired_offdiagonal":
    if predicted.shape[0] != target.shape[0]:
      raise ValueError("paired KID requires equal student/teacher batch sizes")
    return pointwise, mean_loss, _paired_offdiagonal_kid(
        predicted, target
    )
  else:
    raise ValueError(f"unsupported covariance_mode: {covariance_mode!r}")
  covariance_scale = jax.lax.stop_gradient(
      jnp.mean(jnp.square(target_covariance)) + 1e-6
  )
  covariance_loss = jnp.mean(jnp.square(
      predicted_covariance - target_covariance
  )) / covariance_scale
  return pointwise, mean_loss, covariance_loss


def build_inception_embedder(config):
  """Build the frozen differentiable image-to-FID-feature pipeline."""
  from diffusers.models import FlaxAutoencoderKL
  from .legacy_utils.jax_fid import inception

  evaluation = config["evaluation"]
  feature_config = config["training"].get(
      "feature_loss", config["training"].get("feature_adversarial", {})
  )
  vae, vae_params = FlaxAutoencoderKL.from_pretrained(evaluation["vae_path"])
  inception_model = inception.InceptionV3(
      pretrained=True,
      ckpt_path=evaluation["inception_checkpoint"],
  )
  # The tiny 16x16 shape is used only to materialize pretrained variables.
  # cuDNN on H200 can reject one of the corresponding initialization
  # convolutions even though the real 299x299 forward is valid.  Initialize
  # this frozen network on CPU; the enclosing jitted training step then places
  # the variables and real forward/backward on the selected GPU.
  with jax.default_device(jax.devices("cpu")[0]):
    inception_variables = inception_model.init(
        jax.random.PRNGKey(0), jnp.ones((1, 16, 16, 3), jnp.float32)
    )
  latent_codec = feature_config.get(
      "latent_codec", evaluation.get("latent_codec", "ldm")
  )
  straight_through_quantize = bool(
      feature_config.get("straight_through_quantize", True)
  )

  def decode_and_embed(latent):
    latent = normalized_latents_to_vae(
        latent.transpose((0, 3, 1, 2)),
        codec=latent_codec,
        channel_axis=1,
    )
    image = vae.apply(
        {"params": vae_params}, latent, method=FlaxAutoencoderKL.decode
    ).sample
    image = image.transpose((0, 2, 3, 1)).astype(jnp.float32)
    pixels = jnp.clip(127.5 * image + 128.0, 0.0, 255.0)
    if straight_through_quantize:
      pixels = pixels + jax.lax.stop_gradient(jnp.round(pixels) - pixels)
    resized = jax.image.resize(
        pixels,
        (pixels.shape[0], 299, 299, 3),
        method="linear",
        antialias=False,
    )
    normalized = (resized - 128.0) / 128.0
    features = inception_model.apply(
        inception_variables, normalized, train=False
    )
    return features.reshape((features.shape[0], -1))

  return decode_and_embed


def build_inception_feature_loss(config):
  """Build a frozen, differentiable approximation of the exact FID pipeline."""
  feature_config = config["training"]["feature_loss"]
  pointwise_weight = float(feature_config.get("pointwise_weight", 1.0))
  mean_weight = float(feature_config.get("mean_weight", 0.0))
  covariance_weight = float(feature_config.get("covariance_weight", 0.0))
  covariance_mode = feature_config.get("covariance_mode", "gram")
  decode_and_embed = build_inception_embedder(config)

  def feature_loss(predicted_latent, target_latent):
    predicted_features = decode_and_embed(predicted_latent)
    target_features = jax.lax.stop_gradient(decode_and_embed(target_latent))
    pointwise, mean_loss, covariance_loss = inception_feature_losses(
        predicted_features, target_features, covariance_mode=covariance_mode
    )
    return (
        pointwise_weight * pointwise
        + mean_weight * mean_loss
        + covariance_weight * covariance_loss
    )

  return feature_loss

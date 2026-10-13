"""Trajectory normalization, losses, and diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def state_rms(states, eps: float = 1e-8):
  """One scalar RMS per trajectory time, reducing batch and latent axes."""
  axes = tuple(range(1, states.ndim))
  return jnp.sqrt(jnp.mean(jnp.square(states.astype(jnp.float32)), axis=axes) + eps)


def update_running_rms_sq(running_rms_sq, states, decay: float, step):
  batch_rms_sq = jnp.mean(
      jnp.square(states.astype(jnp.float32)), axis=tuple(range(1, states.ndim))
  )
  # The first batch initializes the statistic instead of mixing it with ones.
  return jnp.where(step == 0, batch_rms_sq, decay * running_rms_sq + (1 - decay) * batch_rms_sq)


def student_prediction_scales(running_rms_sq, mode):
  """Return the per-block factor mapping model outputs to raw latent states."""
  rms = jnp.sqrt(running_rms_sq[1:] + 1e-8)
  if mode == "endpoint_only":
    return jnp.ones_like(rms)
  if mode == "hidden_alignment":
    return rms
  if mode == "hybrid_alignment":
    # Intermediate heads predict normalized states, while the endpoint stays in
    # raw latent space so its loss matches the endpoint-only control exactly.
    return rms.at[-1].set(1.0)
  raise ValueError(f"unknown training mode: {mode}")


def raw_student_predictions(predicted, running_rms_sq, mode):
  scales = student_prediction_scales(running_rms_sq, mode)
  return predicted * scales[:, None, None, None, None]


def endpoint_reference_mse(
    predicted, reference_endpoint, running_rms_sq, mode
):
  """Penalize drift from a frozen reference generator in raw latent space."""
  raw_endpoint = raw_student_predictions(predicted, running_rms_sq, mode)[-1]
  reference_endpoint = jax.lax.stop_gradient(
      reference_endpoint.astype(jnp.float32)
  )
  return jnp.mean(
      jnp.square(raw_endpoint.astype(jnp.float32) - reference_endpoint)
  )


def trajectory_loss(
    predicted,
    teacher_states,
    running_rms_sq,
    mode,
    final_weight,
    intermediate_weight=1.0,
):
  """Return the requested loss and both normalized/raw per-block diagnostics."""
  teacher_states = jax.lax.stop_gradient(teacher_states.astype(jnp.float32))
  rms = jax.lax.stop_gradient(jnp.sqrt(running_rms_sq + 1e-8))
  raw_targets = teacher_states[1:]
  normalized_targets = raw_targets / rms[1:, None, None, None, None]

  raw_predictions = raw_student_predictions(predicted, running_rms_sq, mode)
  normalized_predictions = raw_predictions / rms[1:, None, None, None, None]
  raw_mse = jnp.mean(jnp.square(raw_predictions - raw_targets), axis=(1, 2, 3, 4))
  normalized_mse = jnp.mean(
      jnp.square(normalized_predictions - normalized_targets), axis=(1, 2, 3, 4)
  )

  if mode == "endpoint_only":
    loss = final_weight * raw_mse[-1]
  elif mode == "hidden_alignment":
    weights = jnp.ones((predicted.shape[0],), dtype=jnp.float32)
    weights = weights.at[-1].set(final_weight)
    loss = jnp.sum(weights * normalized_mse)
  elif mode == "hybrid_alignment":
    loss = intermediate_weight * jnp.sum(normalized_mse[:-1])
    loss = loss + final_weight * raw_mse[-1]
  else:
    raise ValueError(f"unknown training mode: {mode}")
  return loss, {
      "loss": loss,
      "raw_mse": raw_mse,
      "normalized_mse": normalized_mse,
      "rms": rms,
  }


def denormalize_student_endpoint(endpoint, running_rms_sq, mode):
  if mode == "hidden_alignment":
    return endpoint * jnp.sqrt(running_rms_sq[-1] + 1e-8)
  if mode in (
      "endpoint_only",
      "hybrid_alignment",
      "endpoint_plus_convnext",
      "endpoint_plus_convnext_anchor",
      "endpoint_plus_convnext_cfg_distillation",
      "endpoint_plus_convnext_cfg_distillation_hidden_alignment",
  ):
    return endpoint
  raise ValueError(f"unknown training mode: {mode}")


def trajectory_diagnostics(states):
  """Compute norms, step lengths, velocity cosines, and endpoint deltas."""
  states = states.astype(jnp.float32)
  displacements = states[1:] - states[:-1]
  displacement_rms = state_rms(displacements)
  flat_velocity = displacements.reshape(displacements.shape[0], displacements.shape[1], -1)
  velocity_norm = jnp.linalg.norm(flat_velocity, axis=-1)
  cosine = jnp.sum(flat_velocity[1:] * flat_velocity[:-1], axis=-1) / (
      velocity_norm[1:] * velocity_norm[:-1] + 1e-8
  )
  return {
      "state_rms": state_rms(states),
      "displacement_rms": displacement_rms,
      "adjacent_velocity_cosine": jnp.mean(cosine, axis=1),
      "trajectory_curvature": jnp.mean(1.0 - cosine, axis=1),
  }


def endpoint_difference(one_step_endpoint, eight_step_endpoint):
  delta = one_step_endpoint.astype(jnp.float32) - eight_step_endpoint.astype(jnp.float32)
  flat_a = one_step_endpoint.reshape(one_step_endpoint.shape[0], -1).astype(jnp.float32)
  flat_b = eight_step_endpoint.reshape(eight_step_endpoint.shape[0], -1).astype(jnp.float32)
  cosine = jnp.sum(flat_a * flat_b, axis=-1) / (
      jnp.linalg.norm(flat_a, axis=-1) * jnp.linalg.norm(flat_b, axis=-1) + 1e-8
  )
  return {
      "rms": jnp.sqrt(jnp.mean(jnp.square(delta))),
      "mae": jnp.mean(jnp.abs(delta)),
      "cosine": jnp.mean(cosine),
  }

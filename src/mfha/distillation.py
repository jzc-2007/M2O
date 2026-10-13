"""Losses for compressing a multi-step MeanFlow endpoint into one step."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def endpoint_distillation_loss(
    student_endpoint,
    teacher_endpoint,
    reference_endpoint,
    *,
    teacher_weight: float = 1.0,
    reference_weight: float = 0.0,
    teacher_loss_kind: str = "mse",
    teacher_huber_delta: float = 1.0,
):
  """Match a frozen multi-step endpoint with an optional one-step anchor."""
  student_endpoint = student_endpoint.astype(jnp.float32)
  teacher_endpoint = jax.lax.stop_gradient(teacher_endpoint.astype(jnp.float32))
  reference_endpoint = jax.lax.stop_gradient(
      reference_endpoint.astype(jnp.float32)
  )
  teacher_difference = student_endpoint - teacher_endpoint
  teacher_mse = jnp.mean(jnp.square(teacher_difference))
  if teacher_loss_kind == "mse":
    teacher_loss = teacher_mse
  elif teacher_loss_kind == "huber":
    if teacher_huber_delta <= 0.0:
      raise ValueError("teacher_huber_delta must be positive")
    absolute_difference = jnp.abs(teacher_difference)
    delta = jnp.asarray(teacher_huber_delta, jnp.float32)
    # Scale the conventional Huber loss by two so its quadratic region is
    # exactly MSE. This keeps teacher_weight comparable to the MSE control.
    teacher_loss = jnp.mean(jnp.where(
        absolute_difference <= delta,
        jnp.square(teacher_difference),
        2.0 * delta * absolute_difference - jnp.square(delta),
    ))
  else:
    raise ValueError(f"unsupported teacher_loss_kind: {teacher_loss_kind!r}")
  reference_mse = jnp.mean(jnp.square(student_endpoint - reference_endpoint))
  loss = teacher_weight * teacher_loss + reference_weight * reference_mse
  return loss, {
      "loss": loss,
      "teacher_endpoint_loss": teacher_loss,
      "teacher_endpoint_mse": teacher_mse,
      "reference_endpoint_mse": reference_mse,
  }


def intermediate_alignment_loss(
    predicted_states,
    teacher_states,
    running_rms_sq,
    state_indices=None,
):
  """Align only intermediate projected latents with frozen teacher states.

  ``predicted_states`` includes the final endpoint while ``teacher_states``
  includes the input noise.  The final endpoint is intentionally excluded so
  callers can keep their endpoint/perceptual objective exactly unchanged.
  """
  if predicted_states.shape[0] + 1 != teacher_states.shape[0]:
    raise ValueError(
        "predicted readouts must correspond to teacher_states[1:]"
    )
  if state_indices is None:
    state_indices = tuple(range(1, teacher_states.shape[0] - 1))
  else:
    state_indices = tuple(int(index) for index in state_indices)
  if not state_indices:
    raise ValueError("state_indices must contain at least one intermediate state")
  if any(
      index < 1 or index >= teacher_states.shape[0] - 1
      for index in state_indices
  ):
    raise ValueError("state_indices must select intermediate teacher states")
  predicted_indices = tuple(index - 1 for index in state_indices)
  predicted = predicted_states[jnp.asarray(predicted_indices)].astype(jnp.float32)
  targets = jax.lax.stop_gradient(
      teacher_states[jnp.asarray(state_indices)].astype(jnp.float32)
  )
  rms = jax.lax.stop_gradient(
      jnp.sqrt(running_rms_sq[jnp.asarray(state_indices)].astype(jnp.float32) + 1e-8)
  )
  raw_mse = jnp.mean(
      jnp.square(predicted - targets), axis=(1, 2, 3, 4)
  )
  normalized_mse = jnp.mean(
      jnp.square(
          predicted / rms[:, None, None, None, None]
          - targets / rms[:, None, None, None, None]
      ),
      axis=(1, 2, 3, 4),
  )
  loss = jnp.mean(normalized_mse)
  return loss, {
      "hidden_loss": loss,
      "hidden_raw_mse": raw_mse,
      "hidden_normalized_mse": normalized_mse,
      "hidden_state_indices": jnp.asarray(state_indices, dtype=jnp.int32),
      "trajectory_rms": jnp.sqrt(running_rms_sq + 1e-8),
  }

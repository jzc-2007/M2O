"""Adapters for the official improved MeanFlow (iMF) checkpoints.

The upstream project is kept as a read-only source checkout.  This module
provides the experiment-specific pieces that are absent upstream: a uniform
multi-step rollout, deterministic checkpoint restoration, and a one-NFE
warm-start student with configurable latent projection readouts.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import importlib
import sys
from typing import Any

import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp


OFFICIAL_B2_READOUT_LAYERS = (2, 5, 8, 11)


@dataclass(frozen=True)
class IMFModelSpec:
  """Static architecture metadata for an official inference checkpoint."""

  config_name: str
  constructor_name: str
  display_name: str
  depth: int
  hidden_size: int
  num_heads: int
  aux_head_depth: int = 8


IMF_MODEL_SPECS = {
    "iMF-XL-2": IMFModelSpec("iMF-XL-2", "imfDiT_XL_2", "iMF-XL/2", 48, 1024, 16),
    "iMF-B-2": IMFModelSpec(
        "iMF-B-2", "imfDiT_B_2", "iMF-B/2", 12, 768, 12
    ),
    "iMF-M-2": IMFModelSpec(
        "iMF-M-2", "imfDiT_M_2", "iMF-M/2", 24, 768, 12
    ),
    "iMF-L-2": IMFModelSpec(
        "iMF-L-2", "imfDiT_L_2", "iMF-L/2", 32, 1024, 16
    ),
}
_IMF_MODEL_ALIASES = {
    "imfDiT_XL_2": "iMF-XL-2",
    "iMF-XL/2": "iMF-XL-2",
    "imfDiT_B_2": "iMF-B-2",
    "imfDiT_M_2": "iMF-M-2",
    "imfDiT_L_2": "iMF-L-2",
    "iMF-B/2": "iMF-B-2",
    "iMF-M/2": "iMF-M-2",
    "iMF-L/2": "iMF-L-2",
}


def get_imf_model_spec(model_name: str) -> IMFModelSpec:
  """Resolve a config, constructor, or display name to official metadata."""
  canonical = _IMF_MODEL_ALIASES.get(model_name, model_name)
  try:
    return IMF_MODEL_SPECS[canonical]
  except KeyError as error:
    supported = ", ".join(IMF_MODEL_SPECS)
    raise ValueError(
        f"unsupported iMF model {model_name!r}; expected one of {supported}"
    ) from error


def load_upstream(upstream_root: str | Path):
  """Import the pinned iMeanFlow checkout without modifying it."""
  root = str(Path(upstream_root).expanduser().resolve())
  if root not in sys.path:
    sys.path.insert(0, root)
  imf = importlib.import_module("imf")
  imf_dit = importlib.import_module("models.imfDiT")
  return imf, imf_dit


def build_imf_teacher(upstream_root: str | Path, model_name: str):
  """Build an official evaluation-only iMF module."""
  spec = get_imf_model_spec(model_name)
  imf, _ = load_upstream(upstream_root)
  return imf.iMeanFlow(model_str=spec.constructor_name, eval=True)


def build_imf_b2_teacher(upstream_root: str | Path):
  """Backward-compatible B/2 teacher constructor."""
  return build_imf_teacher(upstream_root, "iMF-B-2")


def _init_teacher_params(model, key):
  dummy = jnp.zeros((1, 32, 32, 4), jnp.float32)
  scalar = jnp.ones((1,), jnp.float32)
  labels = jnp.zeros((1,), jnp.int32)
  return model.init(
      {"params": key},
      dummy,
      scalar,
      scalar,
      scalar,
      scalar,
      scalar,
      labels,
      method=model.u_fn,
  )["params"]


def restore_imf_teacher(model, checkpoint: str | Path, key):
  """Restore EMA parameters from an official OCDBT inference checkpoint.

  The release intentionally omits the unused v head.  Restoring into an
  initialized target retains deterministic zero-initialized v-final leaves,
  matching the upstream evaluation path while loading every released u leaf.
  Orbax-checkpoint 0.6.4 or newer is required for this OCDBT layout.
  """
  initial = _init_teacher_params(model, key)
  restored = checkpoints.restore_checkpoint(
      str(Path(checkpoint).expanduser().resolve()),
      {"ema_params": initial, "step": 0},
  )
  if int(restored["step"]) <= 0:
    raise ValueError("iMF checkpoint did not restore a positive training step")
  return flax.core.freeze(restored["ema_params"]), int(restored["step"])


def restore_imf_b2_teacher(model, checkpoint: str | Path, key):
  """Backward-compatible B/2 inference-checkpoint restore."""
  return restore_imf_teacher(model, checkpoint, key)


def restore_imf_training_params(
    checkpoint: str | Path,
    *,
    params_source: str = "raw",
):
  """Load model parameters from one of this project's training checkpoints."""
  if params_source not in ("raw", "ema"):
    raise ValueError("params_source must be 'raw' or 'ema'")
  restored = checkpoints.restore_checkpoint(
      str(Path(checkpoint).expanduser().resolve()), target=None
  )
  key = "params" if params_source == "raw" else "ema_params"
  if key not in restored:
    raise ValueError(f"training checkpoint has no {key!r} parameters")
  return flax.core.freeze(restored[key]), int(restored.get("step", 0))


def imf_velocity(
    model,
    params,
    state,
    labels,
    *,
    t,
    h,
    omega: float = 8.0,
    t_min: float = 0.4,
    t_max: float = 0.65,
):
  """Evaluate the official single-branch interval-average velocity."""
  batch = state.shape[0]
  return model.apply(
      {"params": params},
      state,
      jnp.broadcast_to(jnp.asarray(t, jnp.float32), (batch,)),
      jnp.broadcast_to(jnp.asarray(h, jnp.float32), (batch,)),
      jnp.full((batch,), omega, jnp.float32),
      jnp.full((batch,), t_min, jnp.float32),
      jnp.full((batch,), t_max, jnp.float32),
      labels,
      method=model.u_fn,
  )[0]


def imf_rollout(
    model,
    params,
    noise,
    labels,
    *,
    num_steps: int = 4,
    omega: float = 8.0,
    omega_schedule: tuple[float, ...] | None = None,
    t_min: float = 0.4,
    t_max: float = 0.65,
):
  """Return ``[N+1,B,32,32,4]`` on the uniform 1-to-0 iMF grid."""
  if num_steps < 1:
    raise ValueError("num_steps must be positive")
  if omega_schedule is not None and len(omega_schedule) != num_steps:
    raise ValueError(
        "omega_schedule length must equal num_steps: "
        f"{len(omega_schedule)} != {num_steps}"
    )
  state = noise
  states = [state]
  interval = 1.0 / num_steps
  for step in range(num_steps):
    t = 1.0 - step * interval
    step_omega = omega if omega_schedule is None else omega_schedule[step]
    velocity = imf_velocity(
        model,
        params,
        state,
        labels,
        t=t,
        h=interval,
        omega=step_omega,
        t_min=t_min,
        t_max=t_max,
    )
    state = state - interval * velocity
    states.append(state)
  return jnp.stack(states, axis=0)


def imf_rollout_scan(
    model,
    params,
    noise,
    labels,
    *,
    num_steps: int = 4,
    omega: float = 8.0,
    omega_schedule: tuple[float, ...] | None = None,
    t_min: float = 0.4,
    t_max: float = 0.65,
):
  """Uniform rollout with one scanned body instead of a statically unrolled graph."""
  if num_steps < 1:
    raise ValueError("num_steps must be positive")
  if omega_schedule is not None and len(omega_schedule) != num_steps:
    raise ValueError(
        "omega_schedule length must equal num_steps: "
        f"{len(omega_schedule)} != {num_steps}"
    )
  interval = jnp.asarray(1.0 / num_steps, jnp.float32)
  if omega_schedule is None:
    omegas = jnp.full((num_steps,), omega, jnp.float32)
  else:
    omegas = jnp.asarray(omega_schedule, jnp.float32)
  step_indices = jnp.arange(num_steps, dtype=jnp.float32)

  def body(state, step_and_omega):
    step, step_omega = step_and_omega
    t = jnp.asarray(1.0, jnp.float32) - step * interval
    velocity = imf_velocity(
        model,
        params,
        state,
        labels,
        t=t,
        h=interval,
        omega=step_omega,
        t_min=t_min,
        t_max=t_max,
    )
    state = state - interval * velocity
    return state, state

  _, states = jax.lax.scan(body, noise, (step_indices, omegas))
  return jnp.concatenate((noise[None], states), axis=0)


def imf_inverse_rollout(
    model,
    params,
    data,
    labels,
    *,
    num_steps: int = 8,
    omega: float = 8.0,
    omega_schedule: tuple[float, ...] | None = None,
    t_min: float = 0.4,
    t_max: float = 0.65,
):
  """Encode data on a uniform 0-to-1 grid using signed iMF intervals.

  This deliberately probes ``h=-1/N``, which is outside the released model's
  positive-interval training support.  Callers must validate cycle error and
  encoded-prior statistics before treating the result as a usable pairing.
  """
  if num_steps < 1:
    raise ValueError("num_steps must be positive")
  if omega_schedule is not None and len(omega_schedule) != num_steps:
    raise ValueError(
        "omega_schedule length must equal num_steps: "
        f"{len(omega_schedule)} != {num_steps}"
    )
  state = data
  states = [state]
  interval = 1.0 / num_steps
  for step in range(num_steps):
    t = step * interval
    step_omega = omega if omega_schedule is None else omega_schedule[step]
    velocity = imf_velocity(
        model,
        params,
        state,
        labels,
        t=t,
        h=-interval,
        omega=step_omega,
        t_min=t_min,
        t_max=t_max,
    )
    state = state + interval * velocity
    states.append(state)
  return jnp.stack(states, axis=0)


def imf_inverse_endpoint_loop(
    model,
    params,
    data,
    labels,
    *,
    num_steps: int = 32,
    omega: float = 1.0,
    t_min: float = 0.4,
    t_max: float = 0.65,
):
  """Encode only the final state with a compiled loop body.

  This is numerically equivalent to ``imf_inverse_rollout(...)[-1]`` but does
  not materialize intermediate states or statically replicate the model graph
  once per interval.  It is intended for offline real-pair precomputation.
  """
  if num_steps < 1:
    raise ValueError("num_steps must be positive")
  interval = jnp.asarray(1.0 / num_steps, jnp.float32)

  def body(step, state):
    t = jnp.asarray(step, jnp.float32) * interval
    velocity = imf_velocity(
        model,
        params,
        state,
        labels,
        t=t,
        h=-interval,
        omega=omega,
        t_min=t_min,
        t_max=t_max,
    )
    return state + interval * velocity

  return jax.lax.fori_loop(0, num_steps, body, data)


def make_imf_endpoint_student_class(
    upstream_root: str | Path,
    model_name: str,
):
  """Create an exact-size one-NFE endpoint student for B/M/L inference trees.

  Unlike the historical B/2 trajectory student below, this class adds no
  projection heads. It is the minimal deployment-preserving module used by
  distribution-only scale experiments.
  """
  spec = get_imf_model_spec(model_name)
  _, imf_dit = load_upstream(upstream_root)
  constructor = getattr(imf_dit, spec.constructor_name)

  class iMFEndpointStudent(nn.Module):
    omega: float = 8.0
    t_min: float = 0.4
    t_max: float = 0.65

    def setup(self):
      self.net = constructor(name="net", eval=True)

    def _velocity(self, state, labels, h):
      batch = state.shape[0]
      h = jnp.broadcast_to(jnp.asarray(h, jnp.float32), (batch,))
      sequence = self.net._build_sequence(
          state,
          h,
          jnp.full((batch,), self.omega, jnp.float32),
          jnp.full((batch,), self.t_min, jnp.float32),
          jnp.full((batch,), self.t_max, jnp.float32),
          labels,
      )
      global_layer = 0
      for block in self.net.shared_blocks:
        sequence = block(sequence, self.net.rope_freqs)
        global_layer += 1
      for block in self.net.u_heads:
        sequence = block(sequence, self.net.rope_freqs)
        global_layer += 1
      if global_layer != spec.depth:
        raise AssertionError(
            f"unexpected {spec.display_name} depth: {global_layer}"
        )
      tokens = sequence[:, self.net.prefix_tokens :]
      return self.net.unpatchify(
          self.net.u_final_layer(tokens)
      ).astype(jnp.float32)

    def endpoint(self, noise, labels):
      velocity = self._velocity(noise, labels, jnp.asarray(1.0, jnp.float32))
      return noise.astype(jnp.float32) - velocity

    def interval_velocity(self, state, labels, t, h):
      del t  # The released network conditions on interval length h.
      return self._velocity(state, labels, h)

    def __call__(self, noise, labels):
      return self.endpoint(noise, labels)

  return iMFEndpointStudent


def restore_imf_endpoint_student(
    student,
    checkpoint: str | Path,
    key,
):
  """Restore an exact-size endpoint student from an official inference tree."""
  dummy = jnp.zeros((1, 32, 32, 4), jnp.float32)
  labels = jnp.zeros((1,), jnp.int32)
  initial = student.init(key, dummy, labels)["params"]
  official_target = {"ema_params": {"net": initial["net"]}, "step": 0}
  restored = checkpoints.restore_checkpoint(
      str(Path(checkpoint).expanduser().resolve()), official_target
  )
  if int(restored["step"]) <= 0:
    raise ValueError("iMF checkpoint did not restore a positive training step")
  mutable = flax.core.unfreeze(initial)
  mutable["net"] = restored["ema_params"]["net"]
  return flax.core.freeze(mutable), int(restored["step"])


def make_imf_b2_student_class(upstream_root: str | Path):
  """Create the B/2 one-NFE student class using official upstream layers.

  The official u path has four shared blocks followed by eight u-head blocks,
  i.e. twelve transformer layers.  Every configured intermediate readout uses
  an independent latent projection head.  Layer 11 must remain the endpoint;
  it uses the released final u projection and exact one-step residual
  ``noise - u``.  Consequently a restored student reproduces the released
  one-step endpoint before tuning.
  """
  _, imf_dit = load_upstream(upstream_root)

  class iMFB2WarmStartStudent(nn.Module):
    omega: float = 8.0
    t_min: float = 0.4
    t_max: float = 0.65
    readout_layers: tuple[int, ...] = OFFICIAL_B2_READOUT_LAYERS

    def setup(self):
      layers = tuple(self.readout_layers)
      if not layers or layers[-1] != 11:
        raise ValueError(
            "iMF-B/2 readouts must be non-empty and end at layer 11"
        )
      if layers != tuple(sorted(set(layers))) or layers[0] < 0:
        raise ValueError("iMF-B/2 readouts must be unique increasing layers")
      self.net = imf_dit.imfDiT_B_2(name="net", eval=True)
      self.trajectory_heads = [
          imf_dit.FinalLayer(
              hidden_size=768,
              patch_size=2,
              out_channels=4,
              name=f"projection_head_{idx}",
          )
          for idx in range(len(layers) - 1)
      ]

    def _conditioning(self, batch):
      return (
          jnp.ones((batch,), jnp.float32),
          jnp.ones((batch,), jnp.float32),
          jnp.full((batch,), self.omega, jnp.float32),
          jnp.full((batch,), self.t_min, jnp.float32),
          jnp.full((batch,), self.t_max, jnp.float32),
      )

    def _forward_tokens(self, noise, labels, collect_readouts):
      batch = noise.shape[0]
      _, h, omega, t_min, t_max = self._conditioning(batch)
      sequence = self.net._build_sequence(
          noise, h, omega, t_min, t_max, labels
      )
      projected = []
      readout_index = 0
      global_layer = 0
      for block in self.net.shared_blocks:
        sequence = block(sequence, self.net.rope_freqs)
        if collect_readouts and global_layer in self.readout_layers[:-1]:
          tokens = sequence[:, self.net.prefix_tokens :]
          projected.append(self.net.unpatchify(
              self.trajectory_heads[readout_index](tokens)
          ).astype(jnp.float32))
          readout_index += 1
        global_layer += 1
      for block in self.net.u_heads:
        sequence = block(sequence, self.net.rope_freqs)
        if collect_readouts and global_layer in self.readout_layers[:-1]:
          tokens = sequence[:, self.net.prefix_tokens :]
          projected.append(self.net.unpatchify(
              self.trajectory_heads[readout_index](tokens)
          ).astype(jnp.float32))
          readout_index += 1
        global_layer += 1
      if global_layer != 12:
        raise AssertionError(f"unexpected iMF-B/2 depth: {global_layer}")
      return sequence, projected

    def endpoint(self, noise, labels):
      sequence, _ = self._forward_tokens(noise, labels, False)
      tokens = sequence[:, self.net.prefix_tokens :]
      velocity = self.net.unpatchify(self.net.u_final_layer(tokens))
      return noise.astype(jnp.float32) - velocity.astype(jnp.float32)

    def interval_velocity(self, state, labels, t, h):
      """Evaluate the same single-branch u network on an arbitrary interval.

      This method is used only by composition-consistency training.  Endpoint
      inference continues to call ``endpoint`` once at ``t=h=1`` and therefore
      has unchanged model size, branch count, and NFE.
      """
      batch = state.shape[0]
      del t  # The released iMF DiT conditions on interval length h, not t.
      h = jnp.broadcast_to(jnp.asarray(h, jnp.float32), (batch,))
      sequence = self.net._build_sequence(
          state,
          h,
          jnp.full((batch,), self.omega, jnp.float32),
          jnp.full((batch,), self.t_min, jnp.float32),
          jnp.full((batch,), self.t_max, jnp.float32),
          labels,
      )
      for block in self.net.shared_blocks:
        sequence = block(sequence, self.net.rope_freqs)
      for block in self.net.u_heads:
        sequence = block(sequence, self.net.rope_freqs)
      tokens = sequence[:, self.net.prefix_tokens :]
      velocity = self.net.unpatchify(self.net.u_final_layer(tokens))
      return velocity.astype(jnp.float32)

    def __call__(self, noise, labels):
      sequence, projected = self._forward_tokens(noise, labels, True)
      expected = len(self.readout_layers) - 1
      if len(projected) != expected:
        raise AssertionError(
            f"expected {expected} intermediate heads, got {len(projected)}"
        )
      tokens = sequence[:, self.net.prefix_tokens :]
      velocity = self.net.unpatchify(self.net.u_final_layer(tokens))
      endpoint = noise.astype(jnp.float32) - velocity.astype(jnp.float32)
      return jnp.stack((*projected, endpoint), axis=0)

  return iMFB2WarmStartStudent


def restore_imf_b2_student(
    student,
    checkpoint: str | Path,
    key,
):
  """Initialize projection heads and restore the official u trunk exactly."""
  dummy = jnp.zeros((1, 32, 32, 4), jnp.float32)
  labels = jnp.zeros((1,), jnp.int32)
  initial = student.init(key, dummy, labels)["params"]
  official_target = {"ema_params": {"net": initial["net"]}, "step": 0}
  restored = checkpoints.restore_checkpoint(
      str(Path(checkpoint).expanduser().resolve()), official_target
  )
  mutable = flax.core.unfreeze(initial)
  mutable["net"] = restored["ema_params"]["net"]
  return flax.core.freeze(mutable), int(restored["step"])


def count_parameters(params: Any) -> int:
  return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))

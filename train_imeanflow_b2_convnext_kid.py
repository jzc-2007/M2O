#!/usr/bin/env python3
"""Distill a multi-step official iMF-B/2 rollout into one evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import flax
from flax.training import checkpoints, train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mfha.config import load_yaml
from mfha.distillation import (
    endpoint_distillation_loss,
    intermediate_alignment_loss,
)
from mfha.imeanflow import (
    OFFICIAL_B2_READOUT_LAYERS,
    build_imf_b2_teacher,
    build_imf_teacher,
    count_parameters,
    get_imf_model_spec,
    imf_rollout,
    imf_rollout_scan,
    make_imf_b2_student_class,
    make_imf_endpoint_student_class,
    restore_imf_b2_student,
    restore_imf_b2_teacher,
    restore_imf_endpoint_student,
    restore_imf_teacher,
)
from mfha.meanflow import initial_conditions_from_ids
from mfha.optimization import build_optimizer
from mfha.trajectory import update_running_rms_sq


class iMFDistillState(train_state.TrainState):
  ema_params: flax.struct.PyTreeNode
  rms_sq: jax.Array


FEATURE_TARGETS = ("teacher_endpoint", "reference_endpoint")
# Separate XLA graphs for the same bf16 iMF endpoint are not bitwise identical
# on H200.  A direct same-parameter diagnostic gives MSE 3.2e-6; keep the
# guard above that measured compiler floor and far below the K=1 teacher-path
# discrepancy (8.9e-2 on the same batch).
REFERENCE_IDENTITY_TOLERANCE = 1e-5


def validate_feature_target_config(config):
  """Validate and return the endpoint used by the feature distribution loss.

  The default preserves every historical config: its feature loss is matched to
  the final teacher rollout state.  ``reference_endpoint`` instead freezes the
  released checkpoint at process start and uses its one-step endpoint on the
  same student input.  An additional initialization would change what
  "reference" means, so it is rejected for that explicit protocol.
  """
  training = config.get("training", {})
  feature_target = training.get("feature_target", "teacher_endpoint")
  if feature_target not in FEATURE_TARGETS:
    raise ValueError(
        "training.feature_target must be one of "
        f"{FEATURE_TARGETS}, got {feature_target!r}"
    )
  feature_weight = float(training.get("feature_weight", 0.1))
  if feature_weight < 0.0:
    raise ValueError("training.feature_weight must be non-negative")
  if feature_target == "reference_endpoint":
    if feature_weight <= 0.0 or not isinstance(
        training.get("feature_loss"), dict
    ):
      raise ValueError(
          "reference_endpoint requires a positive feature_weight and a "
          "feature_loss mapping"
      )
    if config.get("student", {}).get("initialization"):
      raise ValueError(
          "reference_endpoint requires direct released-checkpoint "
          "initialization; student.initialization is not allowed"
      )
  return feature_target


def select_feature_target(teacher_states, reference_endpoint, feature_target):
  """Select the frozen latent endpoint used by the distribution loss."""
  if feature_target == "teacher_endpoint":
    return teacher_states[-1]
  if feature_target == "reference_endpoint":
    return reference_endpoint
  raise ValueError(f"unsupported feature target: {feature_target!r}")


def composition_interval_pairs(num_steps):
  """Return proper multi-interval spans, excluding the full endpoint span.

  Each pair ``(i, j)`` refers to teacher states on the uniform N-step grid.
  Adjacent pairs are omitted because they contain no composition, and
  ``(0, N)`` is omitted because the base endpoint loss already supervises it.
  """
  if num_steps < 3:
    raise ValueError("composition consistency requires at least three steps")
  return tuple(
      (start, stop)
      for start in range(num_steps)
      for stop in range(start + 2, num_steps + 1)
      if (start, stop) != (0, num_steps)
  )


def restore_latest_state(workdir, state, resume_policy):
  """Restore a training state according to the explicit resume policy."""
  latest_checkpoint = checkpoints.latest_checkpoint(str(workdir))
  if resume_policy == "required" and latest_checkpoint is None:
    raise ValueError(
        f"--resume=required but no checkpoint exists in {workdir}"
    )
  if latest_checkpoint is None or resume_policy == "never":
    return state, 0, latest_checkpoint
  restored = checkpoints.restore_checkpoint(str(workdir), target=state)
  return restored, int(jax.device_get(restored.step)), latest_checkpoint


def restore_models(config):
  seed = int(config["training"]["seed"])
  key = jax.random.PRNGKey(seed)
  teacher_key, student_key = jax.random.split(key)
  upstream = config["student"]["upstream_root"]
  checkpoint = config["teacher"]["checkpoint"]
  model_name = config["teacher"].get("model", "iMF-B-2")
  architecture = config["student"].get("architecture", "imeanflow_b2")

  student_config = config["student"]
  if architecture == "imeanflow_b2":
    # Preserve the historical B/2 construction and parameter tree exactly.
    if get_imf_model_spec(model_name).config_name != "iMF-B-2":
      raise ValueError("imeanflow_b2 architecture requires iMF-B-2")
    teacher = build_imf_b2_teacher(upstream)
    teacher_params, teacher_step = restore_imf_b2_teacher(
        teacher, checkpoint, teacher_key
    )
    Student = make_imf_b2_student_class(upstream)
    student = Student(
        omega=float(student_config.get("omega", 8.0)),
        t_min=float(student_config.get("t_min", 0.4)),
        t_max=float(student_config.get("t_max", 0.65)),
        readout_layers=tuple(student_config.get(
            "block_end_layers", OFFICIAL_B2_READOUT_LAYERS
        )),
    )
    student_params, student_step = restore_imf_b2_student(
        student, checkpoint, student_key
    )
  elif architecture == "imeanflow_endpoint":
    teacher = build_imf_teacher(upstream, model_name)
    teacher_params, teacher_step = restore_imf_teacher(
        teacher, checkpoint, teacher_key
    )
    Student = make_imf_endpoint_student_class(upstream, model_name)
    student = Student(
        omega=float(student_config.get("omega", 8.0)),
        t_min=float(student_config.get("t_min", 0.4)),
        t_max=float(student_config.get("t_max", 0.65)),
    )
    student_params, student_step = restore_imf_endpoint_student(
        student, checkpoint, student_key
    )
  else:
    raise ValueError(f"unsupported iMeanFlow architecture: {architecture}")
  initialization = student_config.get("initialization")
  if initialization:
    restored = checkpoints.restore_checkpoint(
        str(Path(initialization["checkpoint_dir"])), target=None
    )
    source = initialization.get("params_source", "raw")
    tuned_params = restored[
        "ema_params" if source == "ema" else "params"
    ]
    alpha = jnp.asarray(
        float(initialization.get("blend_alpha", 1.0)), jnp.float32
    )
    student_params = flax.core.freeze(jax.tree.map(
        lambda base_value, tuned_value: (
            base_value + alpha * (tuned_value - base_value)
        ),
        flax.core.unfreeze(student_params),
        tuned_params,
    ))
  if teacher_step != student_step:
    raise ValueError(
        f"teacher/student restore steps differ: {teacher_step} vs {student_step}"
    )
  return teacher, teacher_params, student, student_params, teacher_step


def make_target_step(teacher, teacher_params, student, reference_params, config):
  training = config["training"]
  teacher_config = config["teacher"]
  batch_size = int(training["batch_size"])
  num_classes = int(config["data"]["num_classes"])
  teacher_steps = int(training.get("teacher_steps", 4))
  sample_seed = int(training.get("sample_seed", training["seed"]))
  sample_key = jax.random.PRNGKey(sample_seed)
  omega = float(teacher_config.get("omega", 8.0))
  omega_schedule = teacher_config.get("omega_schedule")
  if omega_schedule is not None:
    omega_schedule = tuple(float(value) for value in omega_schedule)
    if len(omega_schedule) != teacher_steps:
      raise ValueError(
          "teacher omega_schedule length must equal teacher_steps"
      )
  t_min = float(teacher_config.get("t_min", 0.4))
  t_max = float(teacher_config.get("t_max", 0.65))
  student_input_noise_scale = float(
      training.get("student_input_noise_scale", 1.0)
  )
  student_output_scale = float(training.get("student_output_scale", 1.0))
  rollout_impl = training.get("teacher_rollout_impl", "unrolled")
  if rollout_impl not in ("unrolled", "scan"):
    raise ValueError("teacher_rollout_impl must be 'unrolled' or 'scan'")
  rollout_fn = imf_rollout_scan if rollout_impl == "scan" else imf_rollout
  if student_input_noise_scale <= 0.0 or student_output_scale <= 0.0:
    raise ValueError("student input/output scales must be positive")

  @jax.jit
  def target_step(step):
    start = step.astype(jnp.uint32) * jnp.uint32(batch_size)
    ids = start + jnp.arange(batch_size, dtype=jnp.uint32)
    noise, labels = initial_conditions_from_ids(
        sample_key, ids, (32, 32, 4), jnp.float32, num_classes
    )
    teacher_states = rollout_fn(
        teacher,
        teacher_params,
        noise,
        labels,
        num_steps=teacher_steps,
        omega=omega,
        omega_schedule=omega_schedule,
        t_min=t_min,
        t_max=t_max,
    )
    student_input = noise * student_input_noise_scale
    reference_endpoint = student.apply(
        {"params": reference_params},
        student_input,
        labels,
        method=student.endpoint,
    ) * student_output_scale
    return (
        student_input,
        labels,
        jax.lax.stop_gradient(teacher_states),
        jax.lax.stop_gradient(reference_endpoint),
    )

  return target_step


def make_train_step(student, config, feature_loss_fn=None):
  training = config["training"]
  feature_target = validate_feature_target_config(config)
  teacher_weight = float(training.get("teacher_weight", 1.0))
  reference_weight = float(training.get("reference_weight", 1.0))
  teacher_loss_kind = training.get("teacher_loss_kind", "mse")
  teacher_huber_delta = float(training.get("teacher_huber_delta", 1.0))
  feature_weight = float(training.get("feature_weight", 0.1))
  hidden_weight = float(training.get("hidden_weight", 0.0))
  composition_weight = float(training.get(
      "composition_consistency_weight", 0.0
  ))
  if composition_weight < 0.0:
    raise ValueError("composition_consistency_weight must be non-negative")
  teacher_steps = int(training.get("teacher_steps", 4))
  if composition_weight:
    interval_pairs = composition_interval_pairs(teacher_steps)
    interval_starts = jnp.asarray(
        [pair[0] for pair in interval_pairs], jnp.int32
    )
    interval_stops = jnp.asarray(
        [pair[1] for pair in interval_pairs], jnp.int32
    )
  else:
    interval_pairs = ()
    interval_starts = interval_stops = None
  hidden_state_indices = training.get("hidden_state_indices")
  if hidden_state_indices is not None:
    hidden_state_indices = tuple(int(index) for index in hidden_state_indices)
  rms_decay = float(training.get("rms_decay", 0.999))
  ema_decay = float(training["ema_decay"])
  student_output_scale = float(training.get("student_output_scale", 1.0))

  @jax.jit
  def train_step(state, noise, labels, teacher_states, reference_endpoint):
    teacher_states = jax.lax.stop_gradient(teacher_states)
    reference_endpoint = jax.lax.stop_gradient(reference_endpoint)
    selected_feature_target = jax.lax.stop_gradient(select_feature_target(
        teacher_states, reference_endpoint, feature_target
    ))
    new_rms_sq = update_running_rms_sq(
        state.rms_sq, teacher_states, rms_decay, state.step
    )

    def loss_fn(params):
      if hidden_weight:
        predicted_states = student.apply({"params": params}, noise, labels)
        endpoint = predicted_states[-1] * student_output_scale
      else:
        predicted_states = None
        endpoint = student.apply(
            {"params": params}, noise, labels, method=student.endpoint
        ) * student_output_scale
      loss, metrics = endpoint_distillation_loss(
          endpoint,
          teacher_states[-1],
          reference_endpoint,
          teacher_weight=teacher_weight,
          reference_weight=reference_weight,
          teacher_loss_kind=teacher_loss_kind,
          teacher_huber_delta=teacher_huber_delta,
      )
      metrics = {
          **metrics,
          "feature_target_is_reference_endpoint": jnp.asarray(
              feature_target == "reference_endpoint", jnp.int32
          ),
          "feature_target_vs_reference_endpoint_mse": jnp.mean(jnp.square(
              selected_feature_target - reference_endpoint
          )),
          "feature_target_vs_teacher_endpoint_mse": jnp.mean(jnp.square(
              selected_feature_target - teacher_states[-1]
          )),
      }
      if feature_loss_fn is not None and feature_weight:
        feature_loss = feature_loss_fn(endpoint, selected_feature_target)
        loss = loss + feature_weight * feature_loss
        metrics = {**metrics, "feature_loss": feature_loss}
      if hidden_weight:
        hidden_loss, hidden_metrics = intermediate_alignment_loss(
            predicted_states,
            teacher_states,
            new_rms_sq,
            state_indices=hidden_state_indices,
        )
        loss = loss + hidden_weight * hidden_loss
        metrics = {**metrics, **hidden_metrics}
      if composition_weight:
        pair_index = jnp.mod(state.step, len(interval_pairs))
        start_index = interval_starts[pair_index]
        stop_index = interval_stops[pair_index]
        start_state = jax.lax.dynamic_index_in_dim(
            teacher_states, start_index, axis=0, keepdims=False
        )
        target_state = jax.lax.dynamic_index_in_dim(
            teacher_states, stop_index, axis=0, keepdims=False
        )
        interval = jnp.asarray(1.0 / teacher_steps, jnp.float32)
        t = 1.0 - start_index.astype(jnp.float32) * interval
        h = (stop_index - start_index).astype(jnp.float32) * interval
        velocity = student.apply(
            {"params": params},
            start_state,
            labels,
            t,
            h,
            method=student.interval_velocity,
        )
        predicted_state = start_state - h * velocity
        composition_mse = jnp.mean(jnp.square(
            predicted_state - target_state
        ))
        loss = loss + composition_weight * composition_mse
        metrics = {
            **metrics,
            "composition_mse": composition_mse,
            "composition_start_index": start_index,
            "composition_stop_index": stop_index,
        }
      return loss, metrics

    (loss, metrics), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=gradients)
    ema_params = jax.tree.map(
        lambda ema, value: ema_decay * ema + (1.0 - ema_decay) * value,
        state.ema_params,
        state.params,
    )
    state = state.replace(ema_params=ema_params, rms_sq=new_rms_sq)
    return state, {
        **metrics,
        "grad_norm": optax.global_norm(gradients),
        "loss": loss,
    }

  return train_step


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--workdir", type=Path, required=True)
  parser.add_argument("--max-steps", type=int)
  parser.add_argument(
      "--resume",
      choices=("auto", "never", "required"),
      default="auto",
      help=(
          "Checkpoint resume policy. 'auto' resumes the latest workdir "
          "checkpoint when present; 'required' fails if none exists."
      ),
  )
  args = parser.parse_args()
  config = load_yaml(args.config)
  feature_target = validate_feature_target_config(config)
  args.workdir = args.workdir.resolve()
  args.workdir.mkdir(parents=True, exist_ok=True)
  saved_config = args.workdir / "config.yml"
  source_config = Path(args.config).resolve()
  if (
      saved_config.exists()
      and saved_config.read_bytes() != source_config.read_bytes()
  ):
    raise ValueError(
        "refusing to reuse a workdir with a different config: "
        f"{saved_config}"
    )
  if not saved_config.exists():
    shutil.copy2(source_config, saved_config)

  teacher, teacher_params, student, params, restored_step = restore_models(config)
  reference_params = params
  optimizer, partition = build_optimizer(params, config)
  state = iMFDistillState.create(
      apply_fn=student.apply,
      params=params,
      ema_params=params,
      rms_sq=jnp.ones(
          (int(config["training"].get("teacher_steps", 4)) + 1,),
          jnp.float32,
      ),
      tx=optimizer,
  )
  state, resumed_from_step, latest_checkpoint = restore_latest_state(
      args.workdir, state, args.resume
  )
  if resumed_from_step:
    print(json.dumps({
        "resume_checkpoint": latest_checkpoint,
        "resume_step": resumed_from_step,
    }), flush=True)

  target_step = make_target_step(
      teacher, teacher_params, student, reference_params, config
  )
  preflight = target_step(jnp.asarray(0, jnp.int32))
  teacher_config = config["teacher"]
  student_config = config["student"]

  @jax.jit
  def check_warmstart(noise, labels):
    rollout_fn = (
        imf_rollout_scan
        if config["training"].get("teacher_rollout_impl", "unrolled") == "scan"
        else imf_rollout
    )
    official_endpoint = rollout_fn(
        teacher,
        teacher_params,
        noise,
        labels,
        num_steps=1,
        omega=float(student_config.get("omega", 8.0)),
        t_min=float(student_config.get("t_min", 0.4)),
        t_max=float(student_config.get("t_max", 0.65)),
    )[-1]
    student_endpoint = student.apply(
        {"params": params}, noise, labels, method=student.endpoint
    )
    return jnp.mean(jnp.square(student_endpoint - official_endpoint))

  warmstart_mse = float(jax.device_get(
      check_warmstart(preflight[0], preflight[1])
  ))
  @jax.jit
  def compiled_student_endpoint(noise, labels):
    return student.apply(
        {"params": params}, noise, labels, method=student.endpoint
    ) * float(config["training"].get("student_output_scale", 1.0))

  initial_student_endpoint = compiled_student_endpoint(
      preflight[0], preflight[1]
  )
  selected_feature_target = select_feature_target(
      preflight[2], preflight[3], feature_target
  )
  initial_reference_mse = float(jax.device_get(jnp.mean(jnp.square(
      initial_student_endpoint - preflight[3]
  ))))
  feature_reference_mse = float(jax.device_get(jnp.mean(jnp.square(
      selected_feature_target - preflight[3]
  ))))
  feature_teacher_mse = float(jax.device_get(jnp.mean(jnp.square(
      selected_feature_target - preflight[2][-1]
  ))))
  identity_tolerance = REFERENCE_IDENTITY_TOLERANCE
  warmstart_tolerance = (
      identity_tolerance
      if student_config.get("architecture") == "imeanflow_endpoint"
      else 1e-8
  )
  preflight_report = {
      "checkpoint_step": restored_step,
      "student_parameters": count_parameters(params),
      "teacher_trainable_leaves": 0,
      "feature_target": feature_target,
      "feature_target_source": (
          "frozen_released_reference_endpoint"
          if feature_target == "reference_endpoint"
          else "frozen_teacher_rollout_endpoint"
      ),
      "initial_student_vs_official_one_step_mse": warmstart_mse,
      "warmstart_identity_tolerance": warmstart_tolerance,
      "step0_initial_student_vs_reference_endpoint_mse": (
          initial_reference_mse
      ),
      "step0_feature_target_vs_reference_endpoint_mse": (
          feature_reference_mse
      ),
      "step0_feature_target_vs_teacher_endpoint_mse": feature_teacher_mse,
      "step0_reference_identity_tolerance": identity_tolerance,
      "step0_reference_identity_verified": (
          initial_reference_mse <= identity_tolerance
          and (
              feature_target != "reference_endpoint"
              or feature_reference_mse <= identity_tolerance
          )
      ),
      "teacher_one_vs_target_mse": float(jax.device_get(jnp.mean(
          (preflight[3] - preflight[2][-1]) ** 2
      ))),
  }
  print(json.dumps(preflight_report), flush=True)
  if (not student_config.get("initialization")
      and warmstart_mse > warmstart_tolerance):
    raise ValueError(
        "iMF warm-start failed to reproduce the official one-step endpoint: "
        f"MSE={warmstart_mse}"
    )
  if not preflight_report["step0_reference_identity_verified"]:
    raise ValueError(
        "step-0 frozen reference identity check failed: "
        f"student/reference MSE={initial_reference_mse}, "
        f"feature/reference MSE={feature_reference_mse}"
    )

  feature_config = config["training"].get("feature_loss")
  feature_loss_fn = None
  if feature_config and float(config["training"].get("feature_weight", 0.0)):
    feature_kind = feature_config.get("kind", "convnext")
    if feature_kind == "convnext":
      from mfha.perceptual import build_convnext_feature_loss

      feature_loss_fn = build_convnext_feature_loss(config)
    elif feature_kind == "convnext_kid":
      from mfha.perceptual_convnext_kid import build_convnext_kid_feature_loss

      feature_loss_fn = build_convnext_kid_feature_loss(config)
    elif feature_kind == "latent_kid":
      # Same paired-off-diagonal kernel as the ConvNeXt arm, applied directly
      # to normalized iMF endpoints without a pretrained image encoder.
      from mfha.fid_perceptual import _paired_offdiagonal_kid

      def feature_loss_fn(predicted_latent, target_latent):
        predicted = predicted_latent.astype(jnp.float32).reshape(
            (predicted_latent.shape[0], -1)
        )
        target = jax.lax.stop_gradient(
            target_latent.astype(jnp.float32).reshape(
                (target_latent.shape[0], -1)
            )
        )
        return _paired_offdiagonal_kid(predicted, target)
    elif feature_kind == "inception":
      from mfha.fid_perceptual import build_inception_feature_loss

      feature_loss_fn = build_inception_feature_loss(config)
    else:
      raise ValueError(f"unsupported feature loss: {feature_kind}")
  train_step = make_train_step(student, config, feature_loss_fn)

  total_steps = args.max_steps or int(config["training"]["max_steps"])
  log_every = int(config["training"]["log_every"])
  checkpoint_every = int(config["training"]["checkpoint_every"])
  keep = int(config["training"].get("keep_checkpoints", 5))
  metrics_file = (args.workdir / "metrics.jsonl").open(
      "a" if resumed_from_step else "w", buffering=1
  )
  started = time.perf_counter()
  window_started = started
  for step in range(resumed_from_step, total_steps):
    batch = target_step(state.step)
    state, metrics = train_step(state, *batch)
    if step == resumed_from_step or (step + 1) % log_every == 0:
      jax.block_until_ready(metrics["loss"])
      record = {
          "step": step + 1,
          "steps_per_second": log_every / max(
              time.perf_counter() - window_started, 1e-9
          ),
          **jax.tree.map(
              lambda value: np.asarray(jax.device_get(value)).tolist(), metrics
          ),
      }
      print(json.dumps(record), flush=True)
      metrics_file.write(json.dumps(record) + "\n")
      if (
          step == 0
          and feature_target == "reference_endpoint"
          and float(record["reference_endpoint_mse"]) > identity_tolerance
      ):
        raise ValueError(
            "step-0 training-graph frozen reference identity check failed: "
            f"student/reference MSE={record['reference_endpoint_mse']}, "
            f"tolerance={identity_tolerance}"
        )
      window_started = time.perf_counter()
    if (step + 1) % checkpoint_every == 0 or step + 1 == total_steps:
      checkpoints.save_checkpoint(
          str(args.workdir),
          jax.device_get(state),
          step + 1,
          keep=keep,
          overwrite=True,
      )
  metrics_file.close()

  batch_size = int(config["training"]["batch_size"])
  report = {
      **preflight_report,
      "steps": total_steps,
      "resumed_from_step": resumed_from_step,
      "seconds": time.perf_counter() - started,
      "batch_size": batch_size,
      "training_pairs": total_steps * batch_size,
      "pair_id_start": 0,
      "pair_id_stop_exclusive": total_steps * batch_size,
      "sample_seed": int(config["training"].get(
          "sample_seed", config["training"]["seed"]
      )),
      "teacher_steps": int(config["training"].get("teacher_steps", 4)),
      "model": get_imf_model_spec(
          config["teacher"].get("model", "iMF-B-2")
      ).display_name,
      "teacher_rollout_impl": config["training"].get(
          "teacher_rollout_impl", "unrolled"
      ),
      "omega": float(config["teacher"].get("omega", 8.0)),
      "omega_schedule": config["teacher"].get("omega_schedule"),
      "t_min": float(config["teacher"].get("t_min", 0.4)),
      "t_max": float(config["teacher"].get("t_max", 0.65)),
      "student_omega": float(config["student"].get("omega", 8.0)),
      "student_t_min": float(config["student"].get("t_min", 0.4)),
      "student_t_max": float(config["student"].get("t_max", 0.65)),
      "student_input_noise_scale": float(config["training"].get(
          "student_input_noise_scale", 1.0
      )),
      "student_output_scale": float(config["training"].get(
          "student_output_scale", 1.0
      )),
      "hidden_weight": float(config["training"].get("hidden_weight", 0.0)),
      "composition_consistency_weight": float(
          config["training"].get("composition_consistency_weight", 0.0)
      ),
      "composition_interval_pairs": (
          composition_interval_pairs(int(config["training"].get(
              "teacher_steps", 4
          )))
          if float(config["training"].get(
              "composition_consistency_weight", 0.0
          ))
          else []
      ),
      "hidden_state_indices": config["training"].get("hidden_state_indices"),
      **partition,
  }
  with (args.workdir / "training_report.json").open("w") as handle:
    json.dump(report, handle, indent=2)


if __name__ == "__main__":
  main()

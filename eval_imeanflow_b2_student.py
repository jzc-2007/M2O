#!/usr/bin/env python3
"""Matched FID evaluation for a single-branch, one-NFE iMF-B/2 student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np

from mfha.config import load_yaml
from mfha.evaluation import StreamingFIDEvaluator, save_fid_features
from mfha.imeanflow import (
    OFFICIAL_B2_READOUT_LAYERS,
    count_parameters,
    get_imf_model_spec,
    make_imf_b2_student_class,
    make_imf_endpoint_student_class,
    restore_imf_b2_student,
    restore_imf_endpoint_student,
)
from mfha.meanflow import initial_conditions_from_ids


def blend_param_deltas(
    base,
    tuned_params,
    alpha: float,
    path_regex: str | None = None,
    weights=None,
):
  """Apply a weighted soup of checkpoint deltas to the base parameters."""
  base_flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(base))
  tuned_flats = [
      flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
      for params in tuned_params
  ]
  if not tuned_flats:
    raise ValueError("at least one tuned parameter tree is required")
  for tuned_flat in tuned_flats:
    if base_flat.keys() != tuned_flat.keys():
      missing = sorted(
          "/".join(path) for path in base_flat.keys() - tuned_flat.keys()
      )
      extra = sorted(
          "/".join(path) for path in tuned_flat.keys() - base_flat.keys()
      )
      raise ValueError(
          f"parameter trees differ: missing={missing[:5]}, extra={extra[:5]}"
      )
  if weights is None:
    normalized_weights = np.full(
        len(tuned_flats), 1.0 / len(tuned_flats), dtype=np.float64
    )
  else:
    normalized_weights = np.asarray(weights, dtype=np.float64)
    if normalized_weights.shape != (len(tuned_flats),):
      raise ValueError(
          "checkpoint weights must have one value per tuned parameter tree"
      )
    if np.any(normalized_weights < 0.0):
      raise ValueError("checkpoint weights must be non-negative")
    weight_sum = float(normalized_weights.sum())
    if weight_sum <= 0.0:
      raise ValueError("checkpoint weights must have a positive sum")
    normalized_weights = normalized_weights / weight_sum
  pattern = re.compile(path_regex) if path_regex else None
  blended = {}
  matched_paths = []
  matched_parameters = 0
  total_parameters = 0
  alpha_array = jnp.asarray(alpha, jnp.float32)
  for path, base_value in base_flat.items():
    path_string = "/".join(str(segment) for segment in path)
    selected = pattern is None or pattern.search(path_string) is not None
    total_parameters += int(base_value.size)
    if selected:
      matched_paths.append(path_string)
      matched_parameters += int(base_value.size)
      average_delta = sum(
          jnp.asarray(weight, jnp.float32) * (tuned_flat[path] - base_value)
          for weight, tuned_flat in zip(normalized_weights, tuned_flats)
      )
      blended[path] = base_value + alpha_array * average_delta
    else:
      blended[path] = base_value
  if not matched_paths:
    raise ValueError(
        f"--warmstart-blend-path-regex matched no parameters: {path_regex!r}"
    )
  result = flax.traverse_util.unflatten_dict(blended)
  if isinstance(base, flax.core.FrozenDict):
    result = flax.core.freeze(result)
  return result, {
      "path_regex": path_regex,
      "matched_leaves": len(matched_paths),
      "matched_parameters": matched_parameters,
      "total_parameters": total_parameters,
      "matched_path_examples": matched_paths[:20],
      "checkpoint_count": len(tuned_flats),
      "normalized_checkpoint_weights": normalized_weights.tolist(),
  }


def blend_params(base, tuned, alpha: float, path_regex: str | None = None):
  """Backward-compatible single-checkpoint parameter interpolation."""
  return blend_param_deltas(base, [tuned], alpha, path_regex)


def overlay_net_checkpoint(base, tuned):
  """Expand a net-only continuation checkpoint to a student parameter tree.

  Bidirectional iMF continuation uses the official module directly and may
  retain an unused v head, while the endpoint student has projection heads and
  no v head.  Only matching net leaves are deployment parameters; student-only
  leaves stay at initialization and continuation-only leaves are ignored.
  """
  if set(tuned.keys()) != {"net"}:
    raise ValueError("--checkpoint-net-only requires a params tree containing only 'net'")
  base_flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(base))
  tuned_flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(tuned))
  matching = base_flat.keys() & tuned_flat.keys()
  required_net = {path for path in base_flat if path and path[0] == "net"}
  missing = required_net - matching
  if missing:
    examples = ["/".join(path) for path in sorted(missing)[:5]]
    raise ValueError(f"net-only checkpoint is missing deployment leaves: {examples}")
  expanded = {
      path: tuned_flat[path] if path in matching else value
      for path, value in base_flat.items()
  }
  result = flax.traverse_util.unflatten_dict(expanded)
  if isinstance(base, flax.core.FrozenDict):
    result = flax.core.freeze(result)
  ignored = tuned_flat.keys() - matching
  return result, {
      "matching_leaves": len(matching),
      "ignored_checkpoint_leaves": len(ignored),
      "ignored_path_examples": ["/".join(path) for path in sorted(ignored)[:20]],
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--checkpoint-dir", type=Path)
  parser.add_argument(
      "--delta-base-checkpoint-dir",
      type=Path,
      help=(
          "Optional checkpoint used as the parameter-delta origin. With this "
          "set, alpha evaluates base + alpha * (checkpoint - base), enabling "
          "a continuation-delta extrapolation without changing NFE."
      ),
  )
  parser.add_argument(
      "--soup-checkpoint-dir",
      type=Path,
      action="append",
      default=[],
      help=(
          "Additional checkpoint directory. Repeat to average several "
          "distillation deltas into one single-model, one-NFE student."
      ),
  )
  parser.add_argument(
      "--checkpoint-weight",
      type=float,
      action="append",
      default=None,
      help=(
          "Optional weight for each checkpoint, including --checkpoint-dir. "
          "Repeat once per checkpoint; weights are normalized to sum to one."
      ),
  )
  parser.add_argument("--warmstart-initial", action="store_true")
  parser.add_argument(
      "--checkpoint-net-only",
      action="store_true",
      help=(
          "Treat each training checkpoint as an official-module net tree. "
          "Student projection heads remain initialized and unused at endpoint inference."
      ),
  )
  parser.add_argument("--evaluation-config", required=True)
  parser.add_argument("--workdir", type=Path, required=True)
  parser.add_argument("--params-source", choices=("raw", "ema"), default="raw")
  parser.add_argument(
      "--warmstart-blend-alpha",
      type=float,
      default=1.0,
      help=(
          "Interpolate checkpoint parameters from the released one-step "
          "warm-start.  This remains a single model and one NFE."
      ),
  )
  parser.add_argument(
      "--allow-blend-extrapolation",
      action="store_true",
      help=(
          "Allow --warmstart-blend-alpha above 1.0 for an explicit "
          "checkpoint-delta extrapolation experiment."
      ),
  )
  parser.add_argument(
      "--warmstart-blend-path-regex",
      default=None,
      help=(
          "Optional regex over slash-separated parameter paths. Only matched "
          "leaves receive the checkpoint delta; unmatched leaves stay at the "
          "released one-step warm-start."
      ),
  )
  parser.add_argument(
      "--student-omega",
      type=float,
      default=None,
      help="Override the single-branch iMF omega conditioning at evaluation.",
  )
  parser.add_argument(
      "--student-t-min",
      type=float,
      default=None,
      help="Override the single-branch iMF lower guidance interval endpoint.",
  )
  parser.add_argument(
      "--student-t-max",
      type=float,
      default=None,
      help="Override the single-branch iMF upper guidance interval endpoint.",
  )
  # Use 5K as the matched proxy throughout checkpoint selection and ablations.
  # Larger evaluations are reserved for a final code sanity/reproduction run.
  parser.add_argument("--samples", type=int, default=5000)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--seed", type=int, default=20260831)
  parser.add_argument("--id-offset", type=int, default=500000)
  parser.add_argument(
      "--noise-scale",
      type=float,
      default=1.0,
      help="Scale the deterministic Gaussian input before the one-NFE model.",
  )
  parser.add_argument(
      "--output-scale",
      type=float,
      default=1.0,
      help=(
          "Globally rescale the predicted normalized latent before decoding. "
          "This affine can be folded into the existing output projection."
      ),
  )
  parser.add_argument(
      "--output-channel-scale",
      type=float,
      nargs=4,
      default=None,
      metavar=("S0", "S1", "S2", "S3"),
      help="Optional per-channel scale applied after --output-scale.",
  )
  parser.add_argument(
      "--output-channel-bias",
      type=float,
      nargs=4,
      default=None,
      metavar=("B0", "B1", "B2", "B3"),
      help="Optional per-channel bias in normalized-latent coordinates.",
  )
  parser.add_argument("--inception-backend", choices=("cpu", "gpu"), default="cpu")
  parser.add_argument(
      "--feature-export-dtype", choices=("float16", "float32"), default="float16",
      help="Saved feature precision only; does not change the reported FID.",
  )
  args = parser.parse_args()
  if args.warmstart_initial == bool(args.checkpoint_dir):
    raise ValueError(
        "provide exactly one of --checkpoint-dir or --warmstart-initial"
    )
  if args.warmstart_initial and args.soup_checkpoint_dir:
    raise ValueError("checkpoint soup requires --checkpoint-dir")
  if args.warmstart_initial and args.checkpoint_weight is not None:
    raise ValueError("checkpoint weights require --checkpoint-dir")
  if args.warmstart_initial and args.checkpoint_net_only:
    raise ValueError("--checkpoint-net-only requires --checkpoint-dir")
  if (
      args.warmstart_blend_alpha < 0.0
      and args.delta_base_checkpoint_dir is None
  ):
    raise ValueError(
        "negative --warmstart-blend-alpha requires "
        "--delta-base-checkpoint-dir"
    )
  if (
      args.warmstart_blend_alpha > 1.0
      and not args.allow_blend_extrapolation
      and args.delta_base_checkpoint_dir is None
  ):
    raise ValueError(
        "--warmstart-blend-alpha must be in [0, 1] unless "
        "--allow-blend-extrapolation is set"
    )
  if args.warmstart_initial and args.warmstart_blend_alpha != 1.0:
    raise ValueError("parameter blending requires --checkpoint-dir")
  if args.warmstart_initial and args.delta_base_checkpoint_dir is not None:
    raise ValueError("continuation-delta blending requires --checkpoint-dir")
  if args.warmstart_initial and args.warmstart_blend_path_regex is not None:
    raise ValueError("parameter path filtering requires --checkpoint-dir")
  args.workdir.mkdir(parents=True, exist_ok=True)
  args.workdir = args.workdir.resolve()
  if args.checkpoint_dir is not None:
    args.checkpoint_dir = args.checkpoint_dir.resolve()
  if args.delta_base_checkpoint_dir is not None:
    args.delta_base_checkpoint_dir = args.delta_base_checkpoint_dir.resolve()
  args.soup_checkpoint_dir = [
      checkpoint_dir.resolve()
      for checkpoint_dir in args.soup_checkpoint_dir
  ]

  config = load_yaml(args.config)
  evaluation = load_yaml(args.evaluation_config)["evaluation"]
  evaluation["device_batch_size"] = args.batch_size
  evaluation["inception_backend"] = args.inception_backend
  student_config = config["student"]
  architecture = student_config.get("architecture", "imeanflow_b2")
  model_spec = get_imf_model_spec(
      config["teacher"].get("model", "iMF-B-2")
  )
  student_omega = (
      float(args.student_omega)
      if args.student_omega is not None
      else float(student_config.get("omega", 8.0))
  )
  student_t_min = (
      float(args.student_t_min)
      if args.student_t_min is not None
      else float(student_config.get("t_min", 0.4))
  )
  student_t_max = (
      float(args.student_t_max)
      if args.student_t_max is not None
      else float(student_config.get("t_max", 0.65))
  )
  if not student_t_min <= student_t_max:
    raise ValueError("student t_min must be <= student t_max")
  if args.noise_scale <= 0.0:
    raise ValueError("--noise-scale must be positive")
  if args.output_scale <= 0.0:
    raise ValueError("--output-scale must be positive")
  if (
      args.output_channel_scale is not None
      and any(scale <= 0.0 for scale in args.output_channel_scale)
  ):
    raise ValueError("--output-channel-scale entries must be positive")
  if architecture == "imeanflow_b2":
    if model_spec.config_name != "iMF-B-2":
      raise ValueError("imeanflow_b2 architecture requires iMF-B-2")
    Student = make_imf_b2_student_class(student_config["upstream_root"])
    student = Student(
        omega=student_omega,
        t_min=student_t_min,
        t_max=student_t_max,
        readout_layers=tuple(student_config.get(
            "block_end_layers", OFFICIAL_B2_READOUT_LAYERS
        )),
    )
    restore_student = restore_imf_b2_student
  elif architecture == "imeanflow_endpoint":
    Student = make_imf_endpoint_student_class(
        student_config["upstream_root"], model_spec.config_name
    )
    student = Student(
        omega=student_omega,
        t_min=student_t_min,
        t_max=student_t_max,
    )
    restore_student = restore_imf_endpoint_student
  else:
    raise ValueError(f"unsupported iMeanFlow architecture: {architecture}")
  init_key = jax.random.PRNGKey(int(config["training"]["seed"]))
  initial, _ = restore_student(
      student, config["student"]["warmstart_checkpoint"], init_key
  )
  if args.warmstart_initial:
    params = initial
    blend_scope = None
  else:
    checkpoint_dirs = [args.checkpoint_dir, *args.soup_checkpoint_dir]
    if (
        args.checkpoint_weight is not None
        and len(args.checkpoint_weight) != len(checkpoint_dirs)
    ):
      raise ValueError(
          "--checkpoint-weight must be repeated once per checkpoint"
      )
    restored_params = []
    net_overlay_reports = []
    for checkpoint_dir in checkpoint_dirs:
      restored = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
      restored_param = restored[
          "ema_params" if args.params_source == "ema" else "params"
      ]
      if args.checkpoint_net_only:
        restored_param, overlay_report = overlay_net_checkpoint(
            initial, restored_param
        )
        net_overlay_reports.append(overlay_report)
      restored_params.append(restored_param)
    blend_base = initial
    delta_base_overlay_report = None
    if args.delta_base_checkpoint_dir is not None:
      restored_base = checkpoints.restore_checkpoint(
          args.delta_base_checkpoint_dir, target=None
      )
      blend_base = restored_base[
          "ema_params" if args.params_source == "ema" else "params"
      ]
      if args.checkpoint_net_only:
        blend_base, delta_base_overlay_report = overlay_net_checkpoint(
            initial, blend_base
        )
    params, blend_scope = blend_param_deltas(
        blend_base,
        restored_params,
        args.warmstart_blend_alpha,
        args.warmstart_blend_path_regex,
        args.checkpoint_weight,
    )
    blend_scope["checkpoint_dirs"] = [str(path) for path in checkpoint_dirs]
    blend_scope["delta_base_checkpoint_dir"] = (
        None
        if args.delta_base_checkpoint_dir is None
        else str(args.delta_base_checkpoint_dir)
    )
    if args.checkpoint_net_only:
      blend_scope["net_checkpoint_overlays"] = net_overlay_reports
      blend_scope["delta_base_checkpoint_overlay"] = delta_base_overlay_report
    print(json.dumps({"blend_scope": blend_scope}, indent=2), flush=True)

  apply_endpoint = jax.jit(lambda noise, labels: student.apply(
      {"params": params}, noise, labels, method=student.endpoint
  ))
  evaluator = StreamingFIDEvaluator(evaluation)
  key = jax.random.PRNGKey(args.seed)
  features = []
  latent_sum = np.zeros((4,), dtype=np.float64)
  latent_sum_sq = np.zeros((4,), dtype=np.float64)
  latent_count = 0
  channel_scale = jnp.asarray(
      args.output_channel_scale or [1.0] * 4, jnp.float32
  ).reshape((1, 1, 1, 4))
  channel_bias = jnp.asarray(
      args.output_channel_bias or [0.0] * 4, jnp.float32
  ).reshape((1, 1, 1, 4))
  started = time.perf_counter()
  for start in range(0, args.samples, args.batch_size):
    stop = min(start + args.batch_size, args.samples)
    ids = jnp.arange(
        args.id_offset + start, args.id_offset + stop, dtype=jnp.uint32
    )
    noise, labels = initial_conditions_from_ids(
        key,
        ids,
        (32, 32, 4),
        jnp.float32,
        int(config["data"]["num_classes"]),
        balanced_labels=True,
    )
    noise = noise * jnp.asarray(args.noise_scale, jnp.float32)
    endpoint = apply_endpoint(noise, labels)
    endpoint = (
        endpoint * jnp.asarray(args.output_scale, jnp.float32) * channel_scale
        + channel_bias
    )
    endpoint_np = np.asarray(endpoint, dtype=np.float64)
    latent_sum += endpoint_np.sum(axis=(0, 1, 2))
    latent_sum_sq += np.square(endpoint_np).sum(axis=(0, 1, 2))
    latent_count += int(np.prod(endpoint_np.shape[:-1]))
    _, batch_features = evaluator.decode_and_features(endpoint)
    features.append(batch_features)
    if start == 0 or stop % 256 == 0 or stop == args.samples:
      print(json.dumps({
          "evaluated": stop,
          "samples": args.samples,
          "elapsed_seconds": time.perf_counter() - started,
      }), flush=True)
  features = np.concatenate(features, axis=0)[:args.samples]
  latent_mean = latent_sum / latent_count
  latent_variance = np.maximum(
      latent_sum_sq / latent_count - np.square(latent_mean), 0.0
  )
  report = {
      "model": model_spec.display_name,
      "student_parameters": count_parameters(params),
      "feature_export_dtype": args.feature_export_dtype,
      "params_source": args.params_source,
      "warmstart_blend_alpha": args.warmstart_blend_alpha,
      "warmstart_blend_scope": blend_scope,
      "delta_base_checkpoint_dir": (
          None
          if args.delta_base_checkpoint_dir is None
          else str(args.delta_base_checkpoint_dir)
      ),
      "warmstart_initial": args.warmstart_initial,
      "samples": args.samples,
      "id_offset": args.id_offset,
      "seed": args.seed,
      "batch_size": args.batch_size,
      "inception_backend": args.inception_backend,
      "latent_codec": evaluation.get("latent_codec", "ldm"),
      "label_sampling": "class_balanced_round_robin",
      "noise_scale": args.noise_scale,
      "output_scale": args.output_scale,
      "output_channel_scale": (
          args.output_channel_scale or [1.0] * 4
      ),
      "output_channel_bias": (
          args.output_channel_bias or [0.0] * 4
      ),
      "output_latent_channel_mean": latent_mean.tolist(),
      "output_latent_channel_std": np.sqrt(latent_variance).tolist(),
      "nfe": 1,
      "conditional_branches": 1,
      "omega_baked_into_single_branch": student_omega,
      "student_t_min": student_t_min,
      "student_t_max": student_t_max,
      "fid": evaluator.fid(features),
      "seconds_excluding_evaluator_initialization": time.perf_counter() - started,
  }
  save_fid_features(args.workdir / "features.npy", features, args.feature_export_dtype)
  with (args.workdir / "fid_report.json").open("w") as handle:
    json.dump(report, handle, indent=2)
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()

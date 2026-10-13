#!/usr/bin/env python3
"""Evaluate official iMF-B/2 at 1/2/4/8 steps and inspect trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mfha.config import load_yaml
from mfha.evaluation import StreamingFIDEvaluator, save_image_grid, save_fid_features
from mfha.imeanflow import (
    build_imf_b2_teacher,
    build_imf_teacher,
    count_parameters,
    get_imf_model_spec,
    imf_rollout,
    imf_rollout_scan,
    restore_imf_b2_teacher,
    restore_imf_teacher,
)
from mfha.meanflow import initial_conditions_from_ids
from mfha.trajectory import endpoint_difference, trajectory_diagnostics


def json_ready(tree):
  return jax.tree.map(
      lambda value: np.asarray(jax.device_get(value)).tolist(), tree
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--workdir", type=Path, required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--upstream-root")
  parser.add_argument("--omega", type=float)
  parser.add_argument(
      "--omega-schedule",
      type=float,
      nargs="+",
      help="Optional one-omega-per-interval schedule (requires one step count).",
  )
  parser.add_argument("--t-min", type=float)
  parser.add_argument("--t-max", type=float)
  parser.add_argument("--fid-samples", type=int)
  parser.add_argument("--device-batch-size", type=int)
  parser.add_argument("--inception-backend", choices=("cpu", "gpu"))
  parser.add_argument("--diagnostic-batch-size", type=int)
  parser.add_argument("--progress-every", type=int, default=500)
  parser.add_argument("--step-counts", type=int, nargs="+")
  parser.add_argument("--seed", type=int)
  parser.add_argument("--id-offset", type=int, default=0)
  parser.add_argument("--diagnostics-only", action="store_true")
  parser.add_argument(
      "--save-latents",
      action="store_true",
      help=(
          "Persist deterministic input noise and generated endpoint latents "
          "as float16 NPY arrays for downstream pairing experiments."
      ),
  )
  parser.add_argument(
      "--feature-export-dtype", choices=("float16", "float32"), default="float16",
      help="Saved feature precision only; does not change the reported FID.",
  )
  args = parser.parse_args()

  config = load_yaml(args.config)
  if args.checkpoint:
    config["teacher"]["checkpoint"] = args.checkpoint
  if args.upstream_root:
    config["student"]["upstream_root"] = args.upstream_root
  if args.omega is not None:
    config["teacher"]["omega"] = args.omega
  if args.omega_schedule is not None:
    config["teacher"]["omega_schedule"] = args.omega_schedule
  if args.t_min is not None:
    config["teacher"]["t_min"] = args.t_min
  if args.t_max is not None:
    config["teacher"]["t_max"] = args.t_max
  evaluation = config["evaluation"]
  if args.fid_samples is not None:
    evaluation["fid_samples"] = args.fid_samples
  if args.device_batch_size is not None:
    evaluation["device_batch_size"] = args.device_batch_size
  if args.inception_backend is not None:
    evaluation["inception_backend"] = args.inception_backend
  if args.diagnostic_batch_size is not None:
    evaluation["diagnostic_batch_size"] = args.diagnostic_batch_size
  if args.step_counts is not None:
    evaluation["step_counts"] = args.step_counts
  if args.omega_schedule is not None and len(evaluation["step_counts"]) != 1:
    raise ValueError("--omega-schedule requires exactly one --step-counts value")
  if args.seed is not None:
    evaluation["seed"] = args.seed
  args.workdir.mkdir(parents=True, exist_ok=True)

  teacher_config = config["teacher"]
  model_spec = get_imf_model_spec(
      teacher_config.get("model", "iMF-B-2")
  )
  if model_spec.config_name == "iMF-B-2":
    # Keep the historical B/2 path as the default.
    teacher = build_imf_b2_teacher(config["student"]["upstream_root"])
    restore_teacher = restore_imf_b2_teacher
  else:
    teacher = build_imf_teacher(
        config["student"]["upstream_root"], model_spec.config_name
    )
    restore_teacher = restore_imf_teacher
  key = jax.random.PRNGKey(int(evaluation["seed"]))
  params, checkpoint_step = restore_teacher(
      teacher, teacher_config["checkpoint"], key
  )
  omega = float(teacher_config.get("omega", 8.0))
  omega_schedule = teacher_config.get("omega_schedule")
  if omega_schedule is not None:
    omega_schedule = tuple(float(value) for value in omega_schedule)
  t_min = float(teacher_config.get("t_min", 0.4))
  t_max = float(teacher_config.get("t_max", 0.65))

  rollout_impl = evaluation.get("teacher_rollout_impl", "unrolled")
  if rollout_impl not in ("unrolled", "scan"):
    raise ValueError("teacher_rollout_impl must be 'unrolled' or 'scan'")
  rollout_fn = imf_rollout_scan if rollout_impl == "scan" else imf_rollout
  rollout_functions = {
      steps: jax.jit(lambda noise, labels, steps=steps: rollout_fn(
          teacher,
          params,
          noise,
          labels,
          num_steps=steps,
          omega=omega,
          omega_schedule=omega_schedule,
          t_min=t_min,
          t_max=t_max,
      ))
      for steps in evaluation["step_counts"]
  }
  diagnostic_ids = jnp.arange(
      int(evaluation["diagnostic_batch_size"]), dtype=jnp.uint32
  )
  diagnostic_noise, diagnostic_labels = initial_conditions_from_ids(
      key,
      diagnostic_ids,
      (32, 32, 4),
      jnp.float32,
      int(config["data"]["num_classes"]),
  )
  diagnostic_steps = max(rollout_functions)
  states = rollout_functions[diagnostic_steps](
      diagnostic_noise, diagnostic_labels
  )
  one_steps = 1 if 1 in rollout_functions else min(rollout_functions)
  one_endpoint = rollout_functions[one_steps](
      diagnostic_noise, diagnostic_labels
  )[-1]
  jax.block_until_ready(states)
  report = {
      "model": model_spec.display_name,
      "parameters": count_parameters(params),
      "checkpoint_step": checkpoint_step,
      "teacher_rollout_impl": rollout_impl,
      "feature_export_dtype": args.feature_export_dtype,
      "samples": int(evaluation["fid_samples"]),
      "omega": omega,
      "omega_schedule": omega_schedule,
      "t_min": t_min,
      "t_max": t_max,
      "schedule": np.linspace(
          1.0, 0.0, diagnostic_steps + 1
      ).tolist(),
      "trajectory": json_ready(trajectory_diagnostics(states)),
      "one_vs_max_step_endpoint": json_ready(
          endpoint_difference(one_endpoint, states[-1])
      ),
      "fid": {},
      "seconds": {},
  }
  report_path = args.workdir / "teacher_report.json"
  report_path.write_text(json.dumps(report, indent=2))
  if args.diagnostics_only:
    print(json.dumps(report, indent=2), flush=True)
    return

  evaluator = StreamingFIDEvaluator(evaluation)
  grid_columns = min(
      int(evaluation["grid_samples"]),
      int(evaluation["diagnostic_batch_size"]),
  )
  grid_latents = states[:, :grid_columns].reshape(-1, 32, 32, 4)
  grid_images, _ = evaluator.decode_and_features(grid_latents)
  save_image_grid(
      grid_images,
      args.workdir / (
          f"{model_spec.config_name.lower()}_"
          f"{diagnostic_steps}step_trajectory.png"
      ),
      rows=diagnostic_steps + 1,
      columns=grid_columns,
  )

  total_samples = int(evaluation["fid_samples"])
  batch_size = int(evaluation["device_batch_size"])
  saved_noise = None
  if args.save_latents:
    np.save(
        args.workdir / "sample_ids.npy",
        np.arange(
            args.id_offset,
            args.id_offset + total_samples,
            dtype=np.uint32,
        ),
    )
    np.save(
        args.workdir / "labels.npy",
        np.arange(
            args.id_offset,
            args.id_offset + total_samples,
            dtype=np.uint32,
        ).astype(np.int64)
        % int(config["data"]["num_classes"]),
    )
    saved_noise = np.lib.format.open_memmap(
        args.workdir / "noise.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_samples, 32, 32, 4),
    )
  for steps, rollout in rollout_functions.items():
    started = time.perf_counter()
    features = []
    saved_endpoints = None
    if args.save_latents:
      saved_endpoints = np.lib.format.open_memmap(
          args.workdir / f"latents_{steps}step.npy",
          mode="w+",
          dtype=np.float16,
          shape=(total_samples, 32, 32, 4),
      )
    next_progress = int(args.progress_every)
    for start in range(0, total_samples, batch_size):
      stop = min(start + batch_size, total_samples)
      sample_ids = jnp.arange(
          args.id_offset + start, args.id_offset + stop, dtype=jnp.uint32
      )
      noise, labels = initial_conditions_from_ids(
          key,
          sample_ids,
          (32, 32, 4),
          jnp.float32,
          int(config["data"]["num_classes"]),
          balanced_labels=True,
      )
      endpoint = rollout(noise, labels)[-1]
      if saved_endpoints is not None:
        saved_endpoints[start:stop] = np.asarray(
            jax.device_get(endpoint), dtype=np.float16
        )
        if steps == evaluation["step_counts"][0]:
          saved_noise[start:stop] = np.asarray(
              jax.device_get(noise), dtype=np.float16
          )
      _, feature_batch = evaluator.decode_and_features(endpoint)
      features.append(feature_batch)
      if stop == total_samples or stop >= next_progress:
        elapsed = time.perf_counter() - started
        print(
            f"{steps}-step progress: {stop}/{total_samples} "
            f"({stop / elapsed:.2f} samples/s)",
            flush=True,
        )
        while next_progress <= stop:
          next_progress += int(args.progress_every)
    features = np.concatenate(features, axis=0)[:total_samples]
    report["fid"][str(steps)] = evaluator.fid(features)
    report["seconds"][str(steps)] = time.perf_counter() - started
    save_fid_features(
        args.workdir / f"features_{steps}step.npy",
        features, args.feature_export_dtype,
    )
    if saved_endpoints is not None:
      saved_endpoints.flush()
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"{steps}-step {model_spec.display_name} FID-{total_samples}: "
        f"{report['fid'][str(steps)]:.4f}",
        flush=True,
    )

  if model_spec.config_name == "iMF-B-2":
    report["baseline_check"] = {
        "official_one_step_fid50k_approx": 3.37,
        "note": "small-sample FID is only a restore/sampling sanity check",
    }
  else:
    report["baseline_check"] = {
        "note": (
            "Use this run's one-step result as the scale-specific baseline; "
            "the B/2 approximate FID-50K constant does not apply."
        ),
    }
  report["seed"] = int(evaluation["seed"])
  report["id_offset"] = int(args.id_offset)
  report["latent_codec"] = evaluation.get("latent_codec", "ldm")
  report["label_sampling"] = "class_balanced_round_robin"
  report["saved_latents"] = bool(args.save_latents)
  if saved_noise is not None:
    saved_noise.flush()
  report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()

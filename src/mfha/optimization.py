"""Optimizer construction for matched trajectory-student controls."""

import re

import flax
import jax
import optax


def optimizer_partition(params, config):
  """Label parameters for an optional late-backbone fine-tuning control."""
  start_layer = config["training"].get("trainable_backbone_from_layer")
  imf_u_head_from = config["training"].get(
      "trainable_imeanflow_u_head_from"
  )
  if start_layer is not None and imf_u_head_from is not None:
    raise ValueError(
        "trainable_backbone_from_layer and "
        "trainable_imeanflow_u_head_from are mutually exclusive"
    )
  if start_layer is None and imf_u_head_from is None:
    return jax.tree.map(lambda _: "train", params)
  if imf_u_head_from is not None:
    if config["student"].get("architecture") not in (
        "imeanflow_b2",
        "imeanflow_endpoint",
    ):
      raise ValueError(
          "trainable_imeanflow_u_head_from requires an iMeanFlow student"
      )
    imf_u_head_from = int(imf_u_head_from)
    # Eight u-head blocks follow the four shared blocks.  A value of 8 is the
    # useful final-head-only control: no transformer block is updated.
    if not 0 <= imf_u_head_from <= 8:
      raise ValueError(
          "trainable_imeanflow_u_head_from must be in [0, 8]"
      )

    flat_params = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    labels = {}
    for path in flat_params:
      # Projection heads are training-only and remain trainable.  Within the
      # released iMF net, keep the final velocity projection plus the selected
      # suffix of u-head transformer blocks trainable.  Embeddings, shared
      # blocks, and the earlier u-head blocks are frozen exactly.
      if path[0] != "net":
        labels[path] = "train"
        continue
      if len(path) > 1 and path[1] == "u_final_layer":
        labels[path] = "train"
        continue
      head = next(
          (
              int(match.group(1))
              for segment in path
              if (match := re.fullmatch(r"u_heads_(\d+)", str(segment)))
          ),
          None,
      )
      labels[path] = (
          "train"
          if head is not None and head >= imf_u_head_from
          else "freeze"
      )
    partition = flax.traverse_util.unflatten_dict(labels)
    if isinstance(params, flax.core.FrozenDict):
      partition = flax.core.freeze(partition)
    return partition
  if config["student"].get("architecture") != "meanflow_warmstart":
    raise ValueError(
        "trainable_backbone_from_layer requires meanflow_warmstart"
    )
  start_layer = int(start_layer)
  depth = int(config["student"]["depth"])
  if not 0 <= start_layer < depth:
    raise ValueError(
        f"trainable_backbone_from_layer={start_layer} is outside [0, {depth})"
    )

  flat_params = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
  labels = {}
  for path in flat_params:
    # New trajectory heads stay trainable. In the restored MeanFlow net, only
    # the requested suffix of transformer blocks and the final velocity head
    # are updated; embeddings and earlier blocks remain exactly frozen.
    if path[0] != "net":
      labels[path] = "train"
      continue
    if len(path) > 1 and path[1] == "final_layer":
      labels[path] = "train"
      continue
    layer = next(
        (
            int(match.group(1))
            for segment in path
            if (match := re.fullmatch(r"layers_(\d+)", str(segment)))
        ),
        None,
    )
    labels[path] = (
        "train" if layer is not None and layer >= start_layer else "freeze"
    )
  partition = flax.traverse_util.unflatten_dict(labels)
  if isinstance(params, flax.core.FrozenDict):
    partition = flax.core.freeze(partition)
  return partition


def build_optimizer(params, config):
  """Create AdamW plus a zero-update transform for frozen parameters."""
  training = config["training"]
  train_transform = optax.adamw(
      learning_rate=float(training["learning_rate"]),
      b2=float(training["adam_b2"]),
      weight_decay=float(training["weight_decay"]),
  )
  partition = optimizer_partition(params, config)
  flat_params = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
  flat_partition = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(partition)
  )
  counts = {"train": 0, "freeze": 0}
  for path, value in flat_params.items():
    counts[flat_partition[path]] += int(value.size)
  if counts["freeze"]:
    optimizer = optax.multi_transform(
        {"train": train_transform, "freeze": optax.set_to_zero()}, partition
    )
  else:
    optimizer = train_transform
  return optimizer, {
      "trainable_parameters": counts["train"],
      "frozen_parameters": counts["freeze"],
  }

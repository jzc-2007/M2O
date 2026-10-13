from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path):
  with Path(path).open() as handle:
    return yaml.safe_load(handle)


def dtype_from_name(name):
  import jax.numpy as jnp

  table = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
  try:
    return table[name]
  except KeyError as exc:
    raise ValueError(f"unsupported dtype {name!r}") from exc

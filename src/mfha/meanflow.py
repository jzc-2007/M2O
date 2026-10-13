"""Deterministic generated noise and labels for M2O."""
import jax
import jax.numpy as jnp

def initial_conditions_from_ids(rng, sample_ids, image_shape, dtype, num_classes,
                                class_idx=None, balanced_labels=False):
  """Create topology-invariant noise and labels from global sample IDs."""
  sample_ids = jnp.asarray(sample_ids, dtype=jnp.uint32)

  def one_row(sample_id):
    row_key = jax.random.fold_in(rng, sample_id)
    noise_key, label_key = jax.random.split(row_key)
    noise = jax.random.normal(noise_key, image_shape, dtype=dtype)
    if class_idx is not None:
      label = jnp.asarray(class_idx, dtype=jnp.int32)
    elif balanced_labels:
      label = jnp.asarray(sample_id % num_classes, dtype=jnp.int32)
    else:
      label = jax.random.randint(
          label_key, (), 0, num_classes, dtype=jnp.int32
      )
    return noise, label

  return jax.vmap(one_row)(sample_ids)


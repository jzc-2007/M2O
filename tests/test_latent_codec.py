import jax.numpy as jnp
import numpy as np
import pytest

from mfha.latent_codec import (
    convert_normalized_latents,
    IMF_LATENT_MEAN,
    IMF_LATENT_STD,
    normalized_latents_to_vae,
    vae_latents_to_normalized,
)


def test_imeanflow_codec_bhwc():
  latent = jnp.ones((2, 3, 3, 4), dtype=jnp.float32)
  decoded = normalized_latents_to_vae(latent, "imeanflow", channel_axis=-1)
  expected = np.asarray(IMF_LATENT_MEAN) + np.asarray(IMF_LATENT_STD)
  np.testing.assert_allclose(decoded[0, 0, 0], expected, rtol=1e-6)


def test_imeanflow_codec_bchw():
  latent = jnp.zeros((2, 4, 3, 3), dtype=jnp.float32)
  decoded = normalized_latents_to_vae(latent, "imeanflow", channel_axis=1)
  np.testing.assert_allclose(decoded[0, :, 0, 0], IMF_LATENT_MEAN, rtol=1e-6)


def test_legacy_codec_is_unchanged():
  latent = jnp.asarray([0.18215], dtype=jnp.float32)
  np.testing.assert_allclose(normalized_latents_to_vae(latent), [1.0])


def test_imeanflow_codec_rejects_wrong_channels():
  with pytest.raises(ValueError, match="four channels"):
    normalized_latents_to_vae(jnp.zeros((1, 3, 2, 2)), "imeanflow", 1)


def test_latent_coordinate_conversions_round_trip():
  latent = jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4) / 10
  for codec in ("ldm", "imeanflow"):
    vae = normalized_latents_to_vae(latent, codec=codec)
    restored = vae_latents_to_normalized(vae, codec=codec)
    np.testing.assert_allclose(restored, latent, atol=1e-6)

  converted = convert_normalized_latents(latent, "ldm", "imeanflow")
  restored = convert_normalized_latents(converted, "imeanflow", "ldm")
  np.testing.assert_allclose(restored, latent, atol=1e-6)

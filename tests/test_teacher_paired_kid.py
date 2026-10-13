"""Teacher/student distribution loss for shared synthetic noise IDs."""

import jax
import jax.numpy as jnp
import numpy as np

from mfha.fid_perceptual import inception_feature_losses


def test_paired_kid_zero_for_identical_batches():
    key = jax.random.PRNGKey(7)
    features = jax.random.normal(key, (16, 32))
    _, _, loss = inception_feature_losses(
        features, features, covariance_mode="kid_paired_offdiagonal"
    )
    np.testing.assert_allclose(float(loss), 0.0, atol=2e-6)


def test_paired_kid_detects_shift_and_has_finite_gradient():
    key = jax.random.PRNGKey(8)
    target = jax.random.normal(key, (16, 32))
    predicted = target + 0.5

    def loss_fn(value):
        return inception_feature_losses(
            value, target, covariance_mode="kid_paired_offdiagonal"
        )[2]

    value, gradient = jax.value_and_grad(loss_fn)(predicted)
    assert float(value) > 0.0
    assert np.all(np.isfinite(np.asarray(gradient)))

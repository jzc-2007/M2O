import numpy as np
import pytest

from mfha.evaluation import save_fid_features


def test_float32_export_preserves_information_and_input(tmp_path):
    features = np.array([[1.0001234, 0.123456789], [-3.1415927, 2.7182817]], dtype=np.float32)
    original = features.copy()
    save_fid_features(tmp_path / 'full.npy', features, 'float32')
    save_fid_features(tmp_path / 'legacy.npy', features)
    full = np.load(tmp_path / 'full.npy')
    legacy = np.load(tmp_path / 'legacy.npy')
    assert full.dtype == np.float32
    assert legacy.dtype == np.float16
    np.testing.assert_array_equal(full, original)
    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(legacy, original.astype(np.float16))
    assert np.any(legacy.astype(np.float32) != full)


def test_invalid_export_dtype_does_not_create_file(tmp_path):
    path = tmp_path / 'features.npy'
    with pytest.raises(ValueError, match='float16 or float32'):
        save_fid_features(path, np.ones((2, 3)), 'int8')
    assert not path.exists()

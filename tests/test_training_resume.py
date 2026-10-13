"""A saved optimizer state must produce the same next update as uninterrupted training."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training import checkpoints
from train_imeanflow_b2_convnext_kid import iMFDistillState, restore_latest_state

def test_optimizer_resume_roundtrip(tmp_path):
 state=iMFDistillState.create(apply_fn=lambda:None,params={'weight':jnp.array([1.,2.])},
   ema_params={'weight':jnp.array([.9,1.8])},rms_sq=jnp.array([.3,.7]),tx=optax.adam(1e-3,b2=.95))
 grads={'weight':jnp.array([.1,-.2])};one=state.apply_gradients(grads=grads)
 checkpoints.save_checkpoint(str(tmp_path),jax.device_get(one),step=1)
 restored,step,path=restore_latest_state(tmp_path,state,'required')
 assert step==1 and path.endswith('checkpoint_1')
 expected=one.apply_gradients(grads=grads);actual=restored.apply_gradients(grads=grads)
 for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
 np.testing.assert_array_equal(restored.rms_sq,one.rms_sq)
 np.testing.assert_array_equal(restored.ema_params['weight'],one.ema_params['weight'])

def test_required_resume_rejects_missing_checkpoint(tmp_path):
 with pytest.raises(ValueError,match='no checkpoint'):
  restore_latest_state(tmp_path,None,'required')

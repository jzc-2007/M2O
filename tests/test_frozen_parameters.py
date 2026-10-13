import jax
import jax.numpy as jnp
import numpy as np
from mfha.optimization import build_optimizer

def test_endpoint_optimizer_freezes_backbone_and_embeddings():
 params={'net':{'shared_blocks_0':{'kernel':jnp.ones((2,2))},'class_tokens':jnp.ones((2,2)),
                'u_heads_0':{'kernel':jnp.ones((2,2))},'u_final_layer':{'kernel':jnp.ones((2,2))}}}
 cfg={'student':{'architecture':'imeanflow_endpoint'},'training':{'learning_rate':1e-6,'adam_b2':.95,'weight_decay':0.,'trainable_imeanflow_u_head_from':0}}
 optimizer,partition=build_optimizer(params,cfg)
 updates,_=optimizer.update(jax.tree.map(jnp.ones_like,params),optimizer.init(params),params)
 np.testing.assert_array_equal(updates['net']['shared_blocks_0']['kernel'],0.)
 np.testing.assert_array_equal(updates['net']['class_tokens'],0.)
 assert np.any(np.asarray(updates['net']['u_heads_0']['kernel'])!=0.)
 assert np.any(np.asarray(updates['net']['u_final_layer']['kernel'])!=0.)
 assert partition['trainable_parameters']==8 and partition['frozen_parameters']==8

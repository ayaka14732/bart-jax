from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.init_utils import initialise; initialise(default_device_name='tpu', n_devices=2)

import jax
import jax.numpy as jnp
import jax.random as rand

assert jax.process_count() == 1
assert jax.local_device_count() == 2

a = jnp.array([1., 2.])
assert repr(a.device()).startswith('TpuDevice')

key = rand.PRNGKey(42)
assert repr(key).startswith('PRNGKeyArray[rbg] {')
assert repr(key.unsafe_raw_array().device()).startswith('TpuDevice')

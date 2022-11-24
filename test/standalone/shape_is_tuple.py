from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.init_utils import initialise; initialise(default_device_name='cpu', n_devices=1)

import jax.numpy as jnp

a = jnp.array([[1, 2], [3, 4]])
assert isinstance(a.shape, tuple)

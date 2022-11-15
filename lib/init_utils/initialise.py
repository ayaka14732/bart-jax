import os
from typing import Literal

def _find_free_port() -> int:
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def _initialise_cpu(n_devices: int | None=None) -> None:
    if n_devices is None:
        n_devices = 1

    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=' + str(n_devices)

def _initialise_tpu(n_devices: int | None=None) -> None:
    from jax._src.cloud_tpu_init import get_metadata
    import logging

    if n_devices is None:
        n_devices = 8

    def init_one_chip():
        port = _find_free_port()
        os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
        os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
        # Different per process:
        os.environ['TPU_VISIBLE_DEVICES'] = '0'  # '0', '1', '2', '3'
        # Pick a unique port per process
        os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
        os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

    def init_two_chip() -> None:
        port = _find_free_port()
        os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,2,1'
        os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
        # Different per process:
        os.environ['TPU_VISIBLE_DEVICES'] = '0,1'  # '2,3'
        # Pick a unique port per process
        os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
        os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

    def init_four_chip() -> None:
        os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
        os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
        os.environ['TPU_VISIBLE_DEVICES'] = '0,1,2,3'

    os.environ['JAX_PLATFORMS'] = ''

    accelerator_type = get_metadata('accelerator-type')

    if accelerator_type == 'v4-16':
        match n_devices:
            case 1: init_one_chip()
            case 2: init_two_chip()
            case 4: init_four_chip()
            case 8: pass
            case _: raise ValueError(f'Invalid value `n_devices`: {n_devices}')
    else:
        logging.warn('Only the initialisation on Cloud TPU v4-16 is supported.')

def initialise(default_device_name: Literal['cpu', 'tpu'], n_devices: int | None=None) -> None:
    match default_device_name:
        case 'cpu': _initialise_cpu(n_devices)
        case 'tpu': _initialise_tpu(n_devices)

    # post-init flags
    import jax
    jax.config.update('jax_array', True)
    jax.config.update('jax_enable_custom_prng', True)
    jax.config.update('jax_default_prng_impl', 'rbg')
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

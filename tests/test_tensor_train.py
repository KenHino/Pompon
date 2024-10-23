import logging
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pompon.layers.tensor import BasisBatch, dot
from pompon.layers.tt import TensorTrain

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "shape",
    [
        ((3, 3)),
        ((4, 4)),
        ((4, 4, 4)),
        ((3, 3, 3, 3)),
    ],
)
def test_tensor_train(shape: Tuple[int, ...]):
    assert all(
        shape[0] == s for s in shape
    ), "Only square tensors are supported at the moment"
    key = jax.random.PRNGKey(0)
    # Generate a random tensor with shape (N, N, ..., N) and ndim = f
    tensor = jax.random.normal(key, shape, dtype=jnp.float64)
    print(tensor)
    logger.debug(f"Shape of tensor: {tensor.shape}")
    logger.debug(f"tensor: {tensor}")
    # Exact tensor train decomposition
    tt = TensorTrain.decompose(tensor)
    print(tt)
    print(*[core.shape for core in tt])
    if (ndim := len(shape)) == 2:
        print("tensor=", tensor)
        print("tt.norm=", tt.norm)
        print("tt[0].data[0, :, :] @ tt[1].data[:, :, 0] * tt.norm=")
        print(tt[0].data[0, :, :] @ tt[1].data[:, :, 0] * tt.norm)
        assert jnp.allclose(
            tensor, tt[0].data[0, :, :] @ tt[1].data[:, :, 0] * tt.norm
        )
    elif ndim == 3:
        assert jnp.allclose(
            tensor,
            jnp.einsum("abc,cde,efg->bdf", tt[0].data, tt[1].data, tt[2].data)
            * tt.norm,
        )

    # Confirm that the (0, ..., 0) element of the tensor is reproduced
    phis = np.zeros((1, tt.ndim, shape[0]))
    phis[0, :, 0] += 1
    phis = jnp.asarray(phis, dtype=jnp.float32)
    phis = [phis[:, i_dim, :] for i_dim in range(ndim)]
    print(tensor[(0,) * ndim])
    print(tt.forward(phis)[0, 0])
    assert jnp.allclose(tensor[(0,) * ndim], tt.forward(phis)[0, 0], atol=1e-5)

    tt.set_blocks_batch(phis)
    # Test for twodot core sweep
    for i_dim in range(ndim - 1):
        left_block = tt.left_blocks_batch[-1]
        right_block = tt.right_blocks_batch[-2]
        core_left = tt[tt.center]
        core_right = tt[tt.center + 1]
        phi_left = BasisBatch(
            phis[tt.center], leg_names=("D", f"i{tt.center+1}")
        )
        phi_right = BasisBatch(
            phis[tt.center + 1], leg_names=("D", f"i{tt.center+2}")
        )
        data = (
            dot(
                right_block,
                left_block,
                core_right,
                core_left,
                phi_right,
                phi_left,
            )
            * tt.norm
        )
        assert i_dim == tt.center
        assert jnp.allclose(
            tensor[(0,) * ndim], data.as_ndarray()[0], atol=1e-5
        )
        p = tt.shift_center(to_right=True, basis=phis, is_onedot_center=False)
        logger.debug(f"{p=}")
    for i_dim in range(ndim - 1, 0, -1):
        left_block = tt.left_blocks_batch[-2]
        right_block = tt.right_blocks_batch[-1]
        core_left = tt[tt.center - 1]
        core_right = tt[tt.center]
        phi_left = BasisBatch(
            phis[tt.center - 1], leg_names=("D", f"i{tt.center}")
        )
        phi_right = BasisBatch(
            phis[tt.center], leg_names=("D", f"i{tt.center+1}")
        )
        data = (
            dot(
                right_block,
                left_block,
                core_right,
                core_left,
                phi_right,
                phi_left,
            )
            * tt.norm
        )
        assert i_dim == tt.center
        assert jnp.allclose(
            tensor[(0,) * ndim], data.as_ndarray()[0], atol=1e-5
        )
        p = tt.shift_center(to_right=False, basis=phis, is_onedot_center=False)
        logger.debug(f"{p=}")

    # Test for onedot core sweep
    for i_dim in range(ndim):
        left_block = tt.left_blocks_batch[-1]
        right_block = tt.right_blocks_batch[-1]
        core = tt[tt.center]
        phi = BasisBatch(phis[tt.center], leg_names=("D", f"i{tt.center+1}"))
        data = (
            dot(
                right_block,
                left_block,
                core,
                phi,
            )
            * tt.norm
        )
        assert i_dim == tt.center
        assert jnp.allclose(
            tensor[(0,) * ndim], data.as_ndarray()[0], atol=1e-5
        )
        if i_dim < ndim - 1:
            p = tt.shift_center(
                to_right=True, basis=phis, is_onedot_center=True
            )
            logger.debug(f"{p=}")
    tt.set_blocks_batch(phis)
    for i_dim in range(ndim - 1, -1, -1):
        left_block = tt.left_blocks_batch[-1]
        right_block = tt.right_blocks_batch[-1]
        core = tt[tt.center]
        phi = BasisBatch(phis[i_dim], leg_names=("D", f"i{tt.center+1}"))
        data = dot(right_block, left_block, core, phi) * tt.norm
        assert i_dim == tt.center
        assert jnp.allclose(
            tensor[(0,) * ndim], data.as_ndarray()[0], atol=1e-5
        )
        if i_dim > 0:
            p = tt.shift_center(
                to_right=False, basis=phis, is_onedot_center=True
            )
            logger.debug(f"{p=}")


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.utils").setLevel(logging.DEBUG)
    logging.getLogger("test_tensor_train").setLevel(logging.DEBUG)
    # pytest.main([__file__])
    test_tensor_train((3, 3, 3, 3, 3))

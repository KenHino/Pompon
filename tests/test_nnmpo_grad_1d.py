import logging

import jax
import jax.numpy as jnp
import pytest

import pompon
from pompon import NNMPO


@pytest.mark.parametrize(
    "N, D, lr, activation, b_scale",
    [
        (30, 50, 1.0e-04, "tanh", 1.0),
        (3, 50, 1.0e-02, "polynomial", 0.0),
        (10, 50, 1.0e-04, "gaussian", 0.0),
        (10, 50, 0.02, "moderate", 0.5),
        (20, 50, 0.02, "erf", 1.0),
        (10, 50, 0.02, "moderate+silu", 0.0),
    ],
)  # erf fails with N=10
def test_nnmpo_grad_1d(N, D, lr, activation, b_scale):
    d = f = 1
    M = 1
    nnmpo = NNMPO(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
        b_scale=b_scale,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (D, 1))
    y = jnp.sum(x**2, axis=1) / 2.0
    y_pred = nnmpo.forward(x)
    loss_before = pompon.losses.mse(y, y_pred)
    params = nnmpo.grad(
        x,
        y,
        loss="mse",
        twodot_grad=False,
        basis_grad=True,
        coordinator_grad=True,
    )
    logger = logging.getLogger(__name__)
    logger.debug(f"params_with_grad = {params}")
    for param in params:
        logger.debug(f"param = {param}")
        logger.debug(f"param.data (before) = {param.data}")
        logger.debug(f"param.grad = {param.grad}")
        param.data -= lr * param.grad
        logger.debug(f"param.data (after) = {param.data}")
    y_pred = nnmpo.forward(x)
    loss_after = pompon.losses.mse(y, y_pred)
    assert loss_after < loss_before


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.tt").setLevel(logging.DEBUG)
    logging.getLogger("test_nnmpo_grad_1d").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.layers").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.coordinator").setLevel(logging.DEBUG)
    pytest.main([__file__])

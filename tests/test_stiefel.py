import logging

import jax
import jax.numpy as jnp
import pytest

import pompon
from pompon import NNMPO


def y_true(x):
    """
    y = x^2

    Args:
        x (jnp.ndarray): input with shape (D, f)

    Returns:
        jnp.ndarray: output with shape (D, 1)
    """
    return jnp.einsum("Di,Di->D", x, x)[:, None]


@pytest.mark.parametrize(
    "d, f, N, D, lr, activation, b_scale",
    [
        (2, 1, 30, 50, 1.0e-02, "silu", 1.0),
        (3, 2, 30, 50, 1.0e-02, "silu", 1.0),
        (4, 4, 30, 50, 1.0e-02, "silu", 1.0),
    ],
)  # erf fails with N=10
def test_stiefel(d, f, N, D, lr, activation, b_scale):
    M = 1
    nnmpo = NNMPO(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
        b_scale=b_scale,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (D, d))
    y = jnp.sum(x**2, axis=1) / 2.0
    x /= jnp.std(x)
    y /= jnp.std(y)
    y_pred = nnmpo.forward(x)
    loss_before = pompon.losses.mse(y, y_pred)
    params = nnmpo.grad(
        x,
        y,
        loss="mse",
        twodot_grad=False,
        basis_grad=False,
        coordinator_grad=True,
    )
    logger = logging.getLogger("pompon").getChild("optimizer")
    logger.debug(f"params_with_grad = {params}")
    for param in params:
        logger.debug(f"param = {param}")
        logger.debug(f"param.data (before) = {param.data}")
        logger.debug(f"param.grad = {param.grad}")
        assert (
            param.data.shape == param.grad.shape
        ), f"{param.data.shape} != {param.grad.shape}"
        param.data -= lr * param.grad
        logger.debug(f"param.data (after) = {param.data}")
    y_pred = nnmpo.forward(x)
    loss_after = pompon.losses.mse(y, y_pred)
    logger.debug(f"{loss_before=}, {loss_after=}")
    assert loss_after < loss_before
    optimizer = pompon.optimizer.Adam(lr=lr).setup(nnmpo, x, y)
    trace = optimizer.optimize(epochs=10)
    logger.info(f"{trace=}")
    loss_final = pompon.losses.mse(y, nnmpo.forward(x))
    assert loss_final < loss_after


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.tt").setLevel(logging.DEBUG)
    logging.getLogger("test_nnmpo_grad_1d").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.layers").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.coordinator").setLevel(logging.DEBUG)
    pytest.main([__file__])

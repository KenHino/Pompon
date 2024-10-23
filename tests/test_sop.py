import logging

import jax
import jax.numpy as jnp
import pytest

import pompon
import pompon.utils
from pompon import OneBody
from pompon.optimizer import LinearRegression


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
        (2, 1, 10, 100, 1.0e-02, "silu", 1.0),
        (3, 2, 10, 100, 1.0e-02, "silu", 1.0),
        (4, 4, 10, 100, 1.0e-02, "silu", 1.0),
    ],
)  # erf fails with N=10
def test_sop(d, f, N, D, lr, activation, b_scale):
    sop = OneBody(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        activation=activation,
        b_scale=b_scale,
    )
    logger = logging.getLogger("pompon").getChild("optimizer")
    logger.setLevel(logging.DEBUG)
    x = jax.random.normal(jax.random.PRNGKey(0), (D, d))
    y = jnp.sum(x**2, axis=1) / 2.0
    x_train, x_test, y_train, y_test = pompon.utils.train_test_split(
        x, y, test_size=0.5
    )
    logger.info(
        f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}"
    )
    x_scale = jnp.std(x_train)
    x_train /= x_scale
    x_test /= x_scale
    y_scale = jnp.std(y_train)
    y_train /= y_scale
    y_test /= y_scale
    y_pred = sop.forward(x_train)
    loss_0 = pompon.losses.mse(y_train, y_pred)
    params = sop.grad(
        x_train,
        y_train,
    )
    logger.debug(f"params_with_grad = {params}")
    for param in params:
        assert (
            param.data.shape == param.grad.shape
        ), f"{param.data.shape} != {param.grad.shape}"
        param.data -= lr * param.grad
    y_pred = sop.forward(x_train)
    loss_1 = pompon.losses.mse(y_train, y_pred)
    logger.debug(f"{loss_0=}, {loss_1=}")
    assert loss_1 < loss_0
    optimizer = pompon.optimizer.Adam(lr=lr).setup(
        sop, x_train, y_train, x_test=x_test, y_test=y_test
    )
    trace = optimizer.optimize(epochs=10)
    logger.info(f"{trace=}")
    loss_2 = pompon.losses.mse(y_train, sop.forward(x_train))
    assert loss_2 < loss_1
    linreg = LinearRegression(optimizer)
    linreg.regress(lam=1.0e-03)
    loss_3 = pompon.losses.mse(y_train, sop.forward(x_train))
    assert loss_3 < loss_2
    optimizer.lr = 1.0e-04
    trace = optimizer.optimize(epochs=10)
    logger.info(f"{trace[10:]=}")
    loss_4 = pompon.losses.mse(y_train, sop.forward(x_train))
    assert loss_4 < loss_3
    linreg.regress(lam=1.0e-05)
    loss_5 = pompon.losses.mse(y_train, sop.forward(x_train))
    assert loss_5 < loss_4
    nnmpo = sop.to_nnmpo()
    loss_converted = pompon.losses.mse(y_train, nnmpo.forward(x_train))
    assert jnp.allclose(loss_5, loss_converted)

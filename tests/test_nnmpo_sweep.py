import logging

import jax
import jax.numpy as jnp
import optax
import pytest

import pompon
from pompon import NNMPO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        (3, 3, 10, 50, 1.0e-02, "silu", 1.0),
        (5, 4, 10, 50, 1.0e-02, "tanh", 1.0),
        (5, 4, 4, 50, 1.0e-02, "polynomial", 1.0),
    ],
)  # erf fails with N=10
def test_nnmpo_sweep(d, f, N, D, lr, activation, b_scale):
    M = 2
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
    y = y[:, None]
    y_pred = nnmpo.forward(x)
    loss_before = pompon.losses.mse(y, y_pred)
    optimizer = pompon.optimizer.Adam(lr=lr).setup(nnmpo, x, y)
    sweeper = pompon.optimizer.Sweeper(optimizer)
    sweeper.sweep(
        nsweeps=1,
        cutoff=1.0e-01,
        opt_maxiter=10,
        opt_tol=1.0e-06,
        opt_batchsize=25,
        optax_solver=optax.adam(learning_rate=1.0e-04),
        onedot=False,
    )
    loss_twodot = pompon.losses.mse(y, nnmpo.forward(x))
    assert loss_twodot < loss_before
    loss_before = loss_twodot

    sweeper.sweep(
        nsweeps=2,
        cutoff=1.0e-01,
        opt_maxiter=10,
        opt_tol=1.0e-06,
        opt_batchsize=25,
        optax_solver=optax.adam(learning_rate=1.0e-04),
        onedot=False,
    )
    loss_twodot = pompon.losses.mse(y, nnmpo.forward(x))
    assert loss_twodot < loss_before
    loss_before = loss_twodot

    sweeper.sweep(
        nsweeps=3,
        cutoff=1.0e-01,
        opt_maxiter=10,
        opt_tol=1.0e-06,
        opt_batchsize=25,
        optax_solver=optax.adam(learning_rate=1.0e-04),
        onedot=False,
    )
    loss_twodot = pompon.losses.mse(y, nnmpo.forward(x))
    assert loss_twodot < loss_before
    loss_before = loss_twodot

    sweeper.sweep(
        nsweeps=4,
        opt_maxiter=10,
        opt_tol=1.0e-06,
        opt_batchsize=25,
        optax_solver=optax.adam(learning_rate=1.0e-04),
        onedot=True,
    )
    loss_onedot = pompon.losses.mse(y, nnmpo.forward(x))
    assert loss_onedot < loss_before
    loss_before = loss_onedot


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    pytest.main([__file__])

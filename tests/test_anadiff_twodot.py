import logging

import jax
import jax.numpy as jnp
import pytest

import pompon


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
    "D, N, M, f, activation, lr",
    [
        (2, 3, 2, 2, "moderate", 1.0e-09),
        (2, 4, 4, 3, "gauss", 1.0e-09),
        (10, 5, 3, 4, "exp", 1.0e-09),
        (20, 6, 10, 5, "tanh", 1.0e-12),
    ],
)
def test_anadiff_twodot(D, N, M, f, activation, lr):
    """
    Test the analytical differentiation of the two-dot integral of MSE

    Args:
        D (int): batch size
        N (int): number of basis functions
        M (int): bond dimension
        f (int): degree of freedom
        activation (str): activation function
    """
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (D, f))
    y = y_true(x)
    # Define a model
    nnmpo = pompon.NNMPO(
        input_size=f,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
    )
    assert y.shape == (D, 1)
    assert x.shape == (D, f)
    x0 = nnmpo.x0.data
    basis = nnmpo.basis.forward(
        nnmpo.coordinator.forward(x), nnmpo.coordinator.forward(x0)
    )
    nnmpo.update_blocks_batch(x=None, basis=basis)
    while True:
        assert nnmpo.tt.B.grad is None
        param_auto = nnmpo.grad(x, y, twodot_grad=True, use_auto_diff=True)
        B = param_auto[0]
        assert B.name == "B"
        B_grad_auto = B.grad.copy()
        B.clear_grad()
        param_anal = nnmpo.grad(x, y, twodot_grad=True, use_auto_diff=False)
        B = param_anal[0]
        assert B.name == "B"
        B_grad_anal = B.grad.copy()
        B.clear_grad()
        assert jnp.allclose(B_grad_auto, B_grad_anal)
        if nnmpo.tt.center == nnmpo.tt.ndim - 2:
            mse_before = pompon.losses.mse(y, nnmpo.forward(x))
            B.data -= lr * B_grad_anal
            nnmpo.tt.decompose_and_assign_center_twodot(to_right=True)
            mse_after = pompon.losses.mse(y, nnmpo.forward(x))
            assert mse_after < mse_before, f"{mse_after=} {mse_before=}"
            break
        else:
            nnmpo.tt.shift_center(to_right=True, basis=basis)


if __name__ == "__main__":
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.tt").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.layers").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.coordinator").setLevel(logging.DEBUG)
    pytest.main([__file__])

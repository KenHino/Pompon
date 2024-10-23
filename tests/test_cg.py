import jax
import jax.numpy as jnp
import pytest

import pompon
from pompon.optimizer.lin_reg import (
    conjugate_gradient_matrix,
    conjugate_gradient_onedot,
    conjugate_gradient_twodot,
)


def test_cg():
    D = 100
    basis = 10
    lam = 1.0e-05
    assert basis < D
    Phi = jax.random.normal(
        jax.random.PRNGKey(0), (D, basis), dtype=pompon.DTYPE
    )
    true_W = jnp.linspace(-1.0, 1.0, basis, dtype=pompon.DTYPE).reshape(
        basis, 1
    )
    y = jnp.dot(Phi, true_W)
    pred_W = conjugate_gradient_matrix(
        jnp.zeros((basis, 1), dtype=pompon.DTYPE), Phi, y, lam=lam, tol=1.0e-06
    )
    print(true_W)
    print(pred_W)
    assert jnp.allclose(true_W, pred_W, atol=1.0e-03)

    m = 3
    M = 4
    n = 5
    lam = 0.0
    maxiter = 60
    assert m * n * M < D
    phi = jax.random.normal(jax.random.PRNGKey(0), (D, n), dtype=pompon.DTYPE)
    L_block = jax.random.normal(
        jax.random.PRNGKey(0), (D, m), dtype=pompon.DTYPE
    )
    R_block = jax.random.normal(
        jax.random.PRNGKey(0), (D, M), dtype=pompon.DTYPE
    )
    true_C = jnp.arange(m * n * M, dtype=pompon.DTYPE).reshape(m, n, M)
    y = jnp.einsum("Dm,Dn,DM,mnM->D", L_block, phi, R_block, true_C)
    # lam = 1.0e-05
    Phi = jnp.einsum("Dm,Dn,DM->DmnM", L_block, phi, R_block).reshape(
        D, m * n * M
    )
    pred_C = conjugate_gradient_matrix(
        jnp.zeros((m * n * M, 1), dtype=pompon.DTYPE),
        Phi,
        y[:, jnp.newaxis],
        lam=lam,
        tol=1.0e-06,
        maxiter=maxiter,
    ).reshape(m, n, M)
    print(true_C)
    print(pred_C)
    assert jnp.allclose(true_C, pred_C, atol=1.0e-03, rtol=1.0e-03)

    pred_C = conjugate_gradient_onedot(
        jnp.zeros((m, n, M), dtype=pompon.DTYPE),
        L_block,
        R_block,
        phi,
        y,
        lam=lam,
        tol=1.0e-06,
        maxiter=maxiter,
    )
    print(true_C.shape)
    print(pred_C.shape)
    assert jnp.allclose(true_C, pred_C, atol=1.0e-03, rtol=1.0e-03)
    m = 2
    M = 3
    n = 4
    N = 5
    D = 200
    lam = 0.0
    maxiter = 10
    # lam = 1.0e-05
    assert m * n * N * M < D
    L_phi = jax.random.normal(jax.random.PRNGKey(0), (D, n), dtype=pompon.DTYPE)
    R_phi = jax.random.normal(jax.random.PRNGKey(0), (D, N), dtype=pompon.DTYPE)
    L_block = jax.random.normal(
        jax.random.PRNGKey(0), (D, m), dtype=pompon.DTYPE
    )
    R_block = jax.random.normal(
        jax.random.PRNGKey(0), (D, M), dtype=pompon.DTYPE
    )
    true_W = jnp.arange(m * n * N * M, dtype=pompon.DTYPE).reshape(m, n, N, M)
    y = jnp.einsum(
        "Dm,Dn,DN,DM,mnNM->D", L_block, L_phi, R_phi, R_block, true_W
    )
    Phi = jnp.einsum(
        "Dm,Dn,DN,DM->DmnNM", L_block, L_phi, R_phi, R_block
    ).reshape(D, m * n * N * M)
    pred_W1 = conjugate_gradient_matrix(
        jnp.zeros((m * n * N * M, 1), dtype=pompon.DTYPE),
        Phi,
        y[:, jnp.newaxis],
        lam=lam,
        tol=1.0e-06,
        maxiter=maxiter,
    ).reshape(m, n, N, M)
    print(true_W)
    print(pred_W1)
    # assert jnp.allclose(true_W, pred_W, atol=1.0e-03, rtol=1.0e-03)

    pred_W2 = conjugate_gradient_twodot(
        jnp.zeros((m, n, N, M), dtype=pompon.DTYPE),
        L_block,
        R_block,
        L_phi,
        R_phi,
        y,
        lam=lam,
        tol=1.0e-06,
        maxiter=maxiter,
    )
    print(true_W.shape)
    print(pred_W2.shape)
    assert jnp.allclose(pred_W1, pred_W2)
    # assert jnp.allclose(true_W, pred_W, rtol=1.0e-03, atol=1.0e-03)


if __name__ == "__main__":
    pytest.main([__file__])

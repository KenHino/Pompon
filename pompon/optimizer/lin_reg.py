import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import pompon

from .optimizer import Optimizer

logger = logging.getLogger("pompon").getChild("optimizer")


class LinearRegression:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        assert isinstance(optimizer.model, pompon.sop.SumOfProducts)
        self.sop: pompon.sop.SumOfProducts = optimizer.model
        self.linear = optimizer.model.linear

    def regress(self, lam: float = 1.0e-03, maxbatch: int = 5000):
        """
        solve the linear regression problem
            y = Φ A
        where Φ is the design matrix and A is the coefficient matrix
            A = (Φ^T Φ + λ I)^{-1} Φ^T y
        """
        x_train: Array = self.optimizer.x_train
        y_train: Array = self.optimizer.y_train
        x_test: Array | None = self.optimizer.x_test
        y_test: Array | None = self.optimizer.y_test
        mse_train = self.optimizer.model.mse(x_train, y_train)
        if x_test is not None and y_test is not None:
            mse_test = self.optimizer.model.mse(x_test, y_test)
            logger.info(f"Before: {mse_train=:.3e} {mse_test=:.3e}")
        else:
            logger.info(f"Before: {mse_train=:.3e}")

        if (n := len(x_train)) > maxbatch:
            indices = np.random.choice(n, maxbatch, replace=False)
            x = x_train[indices]
            y: np.ndarray = np.array(y_train)[indices]
        else:
            x = x_train
            y = np.array(y_train)
        q = self.sop.coordinator.forward(x)
        q0 = self.sop.coordinator.forward(self.sop.x0.data)
        Φ: np.ndarray = np.array(
            self.sop.flatten_basis(self.sop.basis.forward(q, q0))
        )

        A = np.linalg.inv(Φ.T @ Φ + lam * np.eye(Φ.shape[1])) @ Φ.T @ y
        self.linear.A.data = jnp.array(A, dtype=pompon.DTYPE).reshape(
            self.linear.A.data.shape
        )
        mse_train = self.optimizer.model.mse(x_train, y_train)
        if x_test is not None and y_test is not None:
            mse_test = self.optimizer.model.mse(x_test, y_test)
            logger.info(f"After: {mse_train=:.3e} {mse_test=:.3e}")
        else:
            logger.info(f"After: {mse_train=:.3e}")


def conjugate_gradient_matrix(
    W: Array,
    Phi: Array,
    y: Array,
    tol: float = 1e-10,
    maxiter: int = 1000,
    lam: float = 0.0,
) -> Array:
    r"""
    Conjugate gradient descent

    y = Φ @ W
    -> Φ^T @ y = (Φ^T @ Φ) @ W
    -> b = A @ x
    where A = Φ^T Φ and b = Φ^T y

    find x such that
        x = argmin ||Φ x - y||^2
    where Φ is the design matrix and y is the target.

    A = Φ^T Φ # Hessian
    b = Φ^T y # gradient
    r = b - A x # residual
    p = r # search direction

    Args:
        Phi (Array): design matrix with shape (D, N)
        y (Array): target with shape (D, 1)
        W0 (Array): initial weight with shape (N, 1)
        tol (float): tolerance
        maxiter (int): maximum number of iterations

    """
    assert W.ndim == 2
    if W.size > y.size:
        logger.warning(f"size of W = {W.size} > size of y = {y.size}")
    x = W  # initial guess (N, 1)
    A = jnp.dot(
        Phi.T, Phi
    )  # Hessian (N, N) <- when N is large, this part is expensive
    if lam > 0.0:
        A += lam * jnp.eye(A.shape[0])
    b = jnp.dot(Phi.T, y)  # gradient (N, 1)
    r = b - jnp.dot(A, x)  # residual (N, 1)
    p = r  # search direction (N, 1)
    res_old = jnp.dot(r.T, r)  # residual norm

    maxiter = min(maxiter, x.size)
    iteration = maxiter
    residual = float("inf")
    for i in range(maxiter):
        Ap = jnp.dot(A, p)  # (N, 1)
        alpha = res_old / jnp.dot(p.T, Ap)  # step size scalar
        x = x + alpha * p  # update weight (N, 1)
        r = r - alpha * Ap  # update residual (N, 1)
        res_new = jnp.dot(r.T, r)  # new residual norm (N, 1)

        if (residual := float(jnp.sqrt(res_new.squeeze()))) < tol:
            iteration = i
            break

        p = r + (res_new / res_old) * p
        res_old = res_new
    logger.debug(f"{iteration=}, {residual=:.3e}")
    return x


def conjugate_gradient_onedot(
    C: Array,
    L_block: Array,
    R_block: Array,
    phi: Array,
    y: Array,
    tol: float = 1e-10,
    maxiter: int = 1000,
    lam: float = 0.0,
) -> Array:
    r"""
    Conjugate gradient descent

    Φ_ijk = L_block_i ⊗ phi_j ⊗ R_block_k
    y = ∑_ijk Φ_ijk C_ijk

    y = Φ @ W
    -> Φ^T @ y = (Φ^T @ Φ) @ W
    -> b = A @ x
    where A = Φ^T Φ and b = Φ^T y

    find x such that
        x = argmin ||Φ x - y||^2
    where Φ is the design matrix and y is the target.

    A = Φ^T Φ # Hessian
    b = Φ^T y # gradient
    r = b - A x # residual
    p = r # search direction

    Args:
        L_block (Array): left block with shape (D, m)
        R_block (Array): right block with shape (D, M)
        phi (Array): basis with shape (D, N)
        y (Array): target with shape (D, 1)
        C (Array): initial weight with shape (m, N, M)
        tol (float): tolerance
        maxiter (int): maximum number of iterations

    """
    assert C.ndim == 3
    if C.size > y.size:
        logger.warning(f"size of C = {C.size} > size of y = {y.size}")
    x = C  # initial guess (m, N, M)
    y = y.squeeze()  # (D, 1) -> (D,)
    # A = jnp.dot(Phi.T, Phi)
    # A = jnp.einsum("Da,Db,Dc,Dm,DN,DM->abcmNM",
    #                 L_block, phi, R_block, L_block, phi, R_block)
    # b = jnp.dot(Phi.T, y)
    # gradient (m, N, M)
    b = jnp.einsum("Dm,DN,DM,D->mNM", L_block, phi, R_block, y)
    # r = b - jnp.dot(A+λI, x)  # residual (N, 1)
    r = b - jnp.einsum(
        "Da,Db,Dc,Dm,DN,DM,mNM->abc",
        L_block,
        phi,
        R_block,
        L_block,
        phi,
        R_block,
        x,
    )  # residual (m, N, M)
    if lam > 0.0:
        r -= lam * x
    p = r  # search direction (m, N, M)
    # res_old = jnp.dot(r.T, r)  # residual norm
    res_old = jnp.einsum("abc,abc->", r, r)  # residual norm

    maxiter = min(maxiter, x.size)
    iteration = maxiter
    residual = float("inf")

    @jax.jit
    def cond(args):
        i, x, p, res_new, r = args
        residual = jnp.sqrt(res_new).astype(float)
        return jnp.logical_and(residual > tol, i < maxiter)

    def body(args):
        i, x, p, res_old, r = args
        return (i + 1,) + _cg_onedot_step(
            L_block=L_block,
            R_block=R_block,
            phi=phi,
            x=x,
            p=p,
            res_old=res_old,
            r=r,
            lam=lam,
        )

    iteration, x, p, res_new, r = jax.lax.while_loop(
        cond, body, (0, x, p, res_old, r)
    )
    iteration = int(iteration)
    residual = float(jnp.sqrt(res_new))
    logger.debug(f"{iteration=}, {residual=:.3e}")
    return x


@partial(jax.jit, static_argnums=(7,))
def _cg_onedot_step(
    L_block: Array,
    R_block: Array,
    phi: Array,
    x: Array,
    p: Array,
    res_old: Array,
    r: Array,
    lam: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    Ap = jnp.einsum(
        "Da,Db,Dc,Dm,DN,DM,mNM->abc",
        L_block,
        phi,
        R_block,
        L_block,
        phi,
        R_block,
        p,
    )
    if lam > 0.0:
        Ap += lam * p
    pAp = jnp.einsum("abc,abc->", p, Ap)
    alpha = res_old / pAp
    x = x + alpha * p
    r = r - alpha * Ap
    res_new = jnp.einsum("abc,abc->", r, r)
    p = r + (res_new / res_old) * p
    return x, p, res_new, r


def conjugate_gradient_twodot(
    B: Array,
    L_block: Array,
    R_block: Array,
    L_phi: Array,
    R_phi: Array,
    y: Array,
    tol: float = 1e-10,
    maxiter: int = 1000,
    lam: float = 0.0,
) -> Array:
    r"""
    Conjugate gradient descent

    Φ_ijkl = L_block_i ⊗ L_phi_j ⊗ R_phi_k ⊗ R_block_l
    y = ∑_ijkl Φ_ijkl B_ijkl

    y = Φ @ W
    -> Φ^T @ y = (Φ^T @ Φ) @ W
    -> b = A @ x
    where A = Φ^T Φ, b = Φ^T y, and x = W

    find x such that
        x = argmin ||Φ x - y||^2
    where Φ is the design matrix and y is the target.

    A = Φ^T Φ # Hessian
    b = Φ^T y # gradient
    r = b - A x # residual
    p = r # search direction

    Args:
        L_block (Array): left block with shape (D, m)
        R_block (Array): right block with shape (D, M)
        L_phi (Array): basis with shape (D, n)
        R_phi (Array): basis with shape (D, N)
        y (Array): target with shape (D, 1)
        B (Array): initial weight with shape (m, n, N, M)
        tol (float): tolerance
        maxiter (int): maximum number of iterations

    """
    assert B.ndim == 4
    if B.size > y.size:
        logger.warning(f"size of B = {B.size} > size of y = {y.size}")

    x = B  # initial guess (m, n, N, M)
    y = y.squeeze()  # (D, 1) -> (D,)
    # b = jnp.dot(Phi.T, y)
    b = jnp.einsum("Dm,Dn,DN,DM,D->mnNM", L_block, L_phi, R_phi, R_block, y)
    # gradient (m, n, N, M)

    # A = jnp.dot(Phi.T, Phi)
    # Keeping Hessians in memory is expensive
    # A = jnp.einsum("Da,Db,Dc,Dd,Dm,Dn,DN,DM->abcdmnNM",
    #                L_block, L_phi, R_phi, R_block,
    #                L_block, L_phi, R_phi, R_block)
    # r = b - jnp.dot(A, x)  # residual (N, 1)
    # r = b - jnp.einsum("Da,Db,Dc,Dd,Dm,Dn,DN,DM,mnNM->abcd",
    #                    L_block, L_phi, R_phi, R_block,
    #                    L_block, L_phi, R_phi, R_block,
    #                    x) # residual (m, n, N, M)
    # r = b - jnp.einsum("abcdmnNM,mnNM->abcd", A, x)  # residual (m, n, N, M)
    r = b - jnp.einsum(
        "Da,Db,Dc,Dd,D->abcd",
        L_block,
        L_phi,
        R_phi,
        R_block,
        jnp.einsum("Dm,Dn,DN,DM,mnNM->D", L_block, L_phi, R_phi, R_block, x),
    )
    if lam > 0.0:
        r -= lam * x
    # residual (m, n, N, M)
    # p = r  # search direction (N, 1)
    p = r  # search direction (m, N, M)
    # res_old = jnp.dot(r.T, r)  # residual norm
    res_old = jnp.einsum("abcd,abcd->", r, r)  # residual norm
    maxiter = min(maxiter, x.size)
    residual = float("inf")
    iteration = maxiter

    @jax.jit
    def cond(args):
        i, x, p, res_new, r = args
        residual = jnp.sqrt(res_new).astype(float)
        return jnp.logical_and(residual > tol, i < maxiter)

    def body(args):
        i, x, p, res_old, r = args
        return (i + 1,) + _cg_twodot_step(
            L_block=L_block,
            R_block=R_block,
            L_phi=L_phi,
            R_phi=R_phi,
            x=x,
            p=p,
            res_old=res_old,
            r=r,
            lam=lam,
        )

    iteration, x, p, res_new, r = jax.lax.while_loop(
        cond, body, (0, x, p, res_old, r)
    )

    iteration = int(iteration)
    residual = float(jnp.sqrt(res_new))
    logger.debug(f"{iteration=}, {residual=:.3e}")
    return x


@partial(jax.jit, static_argnums=(8,))
def _cg_twodot_step(
    L_block: Array,
    R_block: Array,
    L_phi: Array,
    R_phi: Array,
    x: Array,
    p: Array,
    res_old: Array,
    r: Array,
    lam: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    Φp = jnp.einsum("Da,Db,Dc,Dd,abcd->D", L_block, L_phi, R_phi, R_block, p)
    Ap = jnp.einsum("Dm,Dn,DN,DM,D->mnNM", L_block, L_phi, R_phi, R_block, Φp)
    if lam > 0.0:
        Ap += lam * p
    pAp = jnp.einsum("abcd,abcd->", p, Ap)
    alpha = res_old / pAp
    x = x + alpha * p
    r = r - alpha * Ap
    res_new = jnp.einsum("abcd,abcd->", r, r)
    p = r + (res_new / res_old) * p
    return x, p, res_new, r

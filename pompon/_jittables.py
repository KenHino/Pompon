"""Jittable (JAX-compatible) functions"""

import logging
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

import pompon.losses

logger = logging.getLogger("pompon").getChild(__name__)
logger.setLevel(logging.DEBUG)


@jax.jit
def _forward_basis2y(basis: list[Array], W: list[Array], norm: Array) -> Array:
    r"""
    This function is separated from ``TensorTrain.forward`` to be jittable.

    In validation batch, the block cores must be constructed from scratch.

    """
    contracted_cores = jnp.einsum("anb,...n->...b", W[0], basis[0])
    for i_mode in range(1, len(W)):
        core = W[i_mode]
        phi = basis[i_mode]
        contracted_cores = jnp.einsum(
            "anb,...a,...n->...b", core, contracted_cores, phi
        )
    return contracted_cores * norm


@jax.jit
def _forward_x2q(x: Array, U: Array) -> Array:
    # x: (D, d) or (d,) or (N, d)
    # U: (d, f)
    # q: (D, f) or (f,) or (N, f)
    return x @ U


@partial(jax.jit, static_argnames=("activation",))
def _forward_q2phi(
    q: Array,
    q0: Array,
    activation: Callable[[Array], Array],
    w: Array,
    b: Array,
) -> Array:
    """
    q: (D,) or scalar
    q0: (N-1,)
    w: (N-1,)
    b: (N-1,)
    """
    logger.debug(f"{q.shape=}")
    if no_batch := (q.ndim == 0):
        q = q[jnp.newaxis]
    assert q.ndim == 1, f"q.ndim = {q.ndim}"
    _q = q[:, jnp.newaxis] - q0[jnp.newaxis, :]  # (D, N-1)
    logger.debug(f"{_q.shape=}, {w.shape=}, {b.shape=}")
    ones = jnp.ones((_q.shape[0], 1), dtype=pompon.DTYPE)
    phis = jnp.concatenate(
        (
            ones,
            activation(jnp.einsum("...n,n->...n", _q, w) + b),
        ),
        axis=-1,
    )
    if no_batch:
        return phis.squeeze(0)
    else:
        return phis


@partial(jax.jit, static_argnames=("activation",))
def _partial_q2phi(
    q: Array,
    q0: Array,
    activation: Callable[[Array], Array],
    w: Array,
    b: Array,
) -> Array:
    func = partial(_forward_q2phi, q0=q0, activation=activation, w=w, b=b)
    return jax.vmap(jax.jacfwd(func))(q)


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2basis(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
) -> list[Array]:
    q = _forward_x2q(x=x, U=U)
    q0 = _forward_x2q(x=x0, U=U)
    return [
        _forward_q2phi(q=_q, q0=_q0, activation=_activation, w=_w, b=_b)
        for _activation, _q, _q0, _w, _b in zip(
            activations, q.T, q0.T, w, b, strict=True
        )
    ]


@partial(jax.jit, static_argnames=("activations",))
def _forward_q2basis(
    activations: tuple[Callable[[Array], Array], ...],
    q: Array,
    q0: Array,
    w: list[Array],
    b: list[Array],
) -> list[Array]:
    return [
        _forward_q2phi(q=_q, q0=_q0, activation=_activation, w=_w, b=_b)
        for _activation, _q, _q0, _w, _b in zip(
            activations, q.T, q0.T, w, b, strict=True
        )
    ]


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2y(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
) -> Array:
    # jax cannot jit list[Callable], so use tuple[Callable, ...] instead
    q = _forward_x2q(x=x, U=U)
    q0 = _forward_x2q(x=x0, U=U)
    basis = _forward_q2basis(activations=activations, q=q, q0=q0, w=w, b=b)
    y = _forward_basis2y(basis=basis, W=W, norm=norm)
    return y


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2f(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
) -> Array:
    # DO NOT USE THIS FUNCTION EXCEPT FOR TESTING
    # USE _forward_q2f INSTEAD and then transform gradient to x

    if no_batch := (x.ndim == 1):
        x = x[jnp.newaxis, :]
    assert x.ndim == 2, f"x.ndim = {x.ndim}"
    q = _forward_x2q(x=x, U=U)
    q0 = _forward_x2q(x=x0, U=U)

    # def minus_squeeze(x, U, activations, w, b, W, norm):
    #    return -1.0 * _forward_x2y(x, U, activations, w, b, W, norm).squeeze()
    def minus_squeeze(q, q0, activations, w, b, W, norm):
        if q.ndim == 1:
            q = q[jnp.newaxis, :]
            return -1.0 * _forward_q2y(
                q, q0, activations, w, b, W, norm
            ).squeeze(0)
        else:
            return -1.0 * _forward_q2y(q, q0, activations, w, b, W, norm)

    func = partial(
        minus_squeeze, q0=q0, activations=activations, w=w, b=b, W=W, norm=norm
    )
    if no_batch:
        jacobian = (jax.vmap(jax.jacrev(func))(q) @ U.T).squeeze(0)
    else:
        jacobian = jax.vmap(jax.jacrev(func))(q) @ U.T
    return jacobian.squeeze(-2)


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2y(
    x: Array,
    x0: Array,
    y: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
) -> Array:
    y_pred = _forward_x2y(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
    )
    return pompon.losses.mse(y, y_pred)


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2f(
    x: Array,
    x0: Array,
    f: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
) -> Array:
    f_pred = _forward_x2f(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
    )  # D x d
    return pompon.losses.mse(f, f_pred)


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2yf(
    x: Array,
    x0: Array,
    y: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
    f: Array,
    wf: float,
) -> Array:
    y_pred = _forward_x2y(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
    )
    f_pred = _forward_x2f(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
    )
    return pompon.losses.mse(y, y_pred) + wf * pompon.losses.mse(f, f_pred)


@partial(jax.jit, static_argnames=("activations",))
def _mse_q2yf(
    q: Array,
    y: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
    x: Array,
    x0: Array,
    f: Array,
    U: Array,
    wf: float,
) -> Array:
    q0 = _forward_x2q(x=x0, U=U)
    y_pred = _forward_q2y(
        q=q, q0=q0, activations=activations, w=w, b=b, W=W, norm=norm
    )
    f_pred = _forward_x2f(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
    )
    return pompon.losses.mse(y, y_pred) + wf * pompon.losses.mse(f, f_pred)


@partial(jax.jit, static_argnames=("activations",))
def _total_loss_x2y(
    x: Array,
    x0: Array,
    y: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
    lambda1: float,
    mu1: float,
    mu2: float,
) -> Array:
    basis = _forward_x2basis(x=x, x0=x0, U=U, activations=activations, w=w, b=b)
    y_pred = _forward_basis2y(basis=basis, W=W, norm=norm)
    return pompon.losses.total_loss(y, y_pred, basis, lambda1, mu1, mu2)


@partial(jax.jit, static_argnames=("activations",))
def _total_loss_q2y(
    q: Array,
    y: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
    q0: Array,
    lambda1: float,
    mu1: float,
    mu2: float,
) -> Array:
    basis = _forward_q2basis(activations=activations, q=q, q0=q0, w=w, b=b)
    y_pred = _forward_basis2y(basis=basis, W=W, norm=norm)
    return pompon.losses.total_loss(y, y_pred, basis, lambda1, mu1, mu2)


@partial(jax.jit, static_argnames=("activations",))
def _forward_q2y(
    q: Array,
    q0: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
) -> Array:
    basis = _forward_q2basis(activations=activations, q=q, q0=q0, w=w, b=b)
    y = _forward_basis2y(basis=basis, W=W, norm=norm)
    return y


@partial(jax.jit, static_argnames=("activations",))
def _mse_q2y(
    q: Array,
    y: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    W: list[Array],
    norm: Array,
    q0: Array,
) -> Array:
    y_pred = _forward_q2y(
        q=q, q0=q0, activations=activations, w=w, b=b, W=W, norm=norm
    )
    return pompon.losses.mse(y, y_pred)


def _mse_basis2y(
    basis: list[Array],
    y: Array,
    W: list[Array],
    norm: Array,
) -> Array:
    y_pred = _forward_basis2y(basis=basis, W=W, norm=norm)
    return pompon.losses.mse(y, y_pred)


@jax.jit
def _forward_block_and_basis2y_twodot(
    center_twodot: Array,
    left_phi_batch: Array,
    right_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    return (
        jnp.einsum(
            "abcd,...a,...d,...b,...c->...",
            center_twodot,
            left_block_batch,
            right_block_batch,
            left_phi_batch,
            right_phi_batch,
        )
        * norm
    )[:, None]  # D x 1


@jax.jit
def _forward_block_and_basis2y_onedot(
    center_onedot: Array,
    center_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    return (
        jnp.einsum(
            "abc,...a,...c,...b->...",
            center_onedot,
            left_block_batch,
            right_block_batch,
            center_phi_batch,
        )
        * norm
    )[:, None]  # D x 1


@jax.jit
def _mse_block_and_basis2y_twodot(
    center_twodot: Array,
    y: Array,
    left_phi_batch: Array,
    right_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    y_pred = _forward_block_and_basis2y_twodot(
        center_twodot=center_twodot,
        left_phi_batch=left_phi_batch,
        right_phi_batch=right_phi_batch,
        left_block_batch=left_block_batch,
        right_block_batch=right_block_batch,
        norm=norm,
    )
    return pompon.losses.mse(y, y_pred)


@jax.jit
def _mse_block_and_basis2y_onedot(
    center_onedot: Array,
    y: Array,
    center_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    y_pred = _forward_block_and_basis2y_onedot(
        center_onedot=center_onedot,
        center_phi_batch=center_phi_batch,
        left_block_batch=left_block_batch,
        right_block_batch=right_block_batch,
        norm=norm,
    )
    return pompon.losses.mse(y, y_pred)


@jax.jit
def _grad_block_and_basis2y_twodot(
    center_twodot: Array,
    y: Array,
    left_phi_batch: Array,
    right_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    """
    Analytical gradient for the mean squared error loss function.
    """
    y_pred = _forward_block_and_basis2y_twodot(
        center_twodot=center_twodot,
        left_phi_batch=left_phi_batch,
        right_phi_batch=right_phi_batch,
        left_block_batch=left_block_batch,
        right_block_batch=right_block_batch,
        norm=norm,
    )
    assert (
        y_pred.shape == y.shape
    ), f"y_pred.shape = {y_pred.shape}, y.shape = {y.shape}"

    mean_deviation = pompon.losses._deviation_for_mse(y_train=y, y_pred=y_pred)

    grads = (
        jnp.einsum(
            "...,...a,...d,...b,...c->abcd",
            mean_deviation,
            left_block_batch,
            right_block_batch,
            left_phi_batch,
            right_phi_batch,
        )
        * norm
    )
    return grads


@jax.jit
def _grad_block_and_basis2y_onedot(
    center_onedot: Array,
    y: Array,
    center_phi_batch: Array,
    left_block_batch: Array,
    right_block_batch: Array,
    norm: Array,
) -> Array:
    """
    Analytical gradient for the mean squared error loss function.
    """
    y_pred = _forward_block_and_basis2y_onedot(
        center_onedot=center_onedot,
        center_phi_batch=center_phi_batch,
        left_block_batch=left_block_batch,
        right_block_batch=right_block_batch,
        norm=norm,
    )
    assert (
        y_pred.shape == y.shape
    ), f"y_pred.shape = {y_pred.shape}, y.shape = {y.shape}"

    mean_deviation = pompon.losses._deviation_for_mse(y_train=y, y_pred=y_pred)

    grads = (
        jnp.einsum(
            "...,...a,...c,...b->abc",
            mean_deviation,
            left_block_batch,
            right_block_batch,
            center_phi_batch,
        )
        * norm
    )
    return grads


@jax.jit
def _forward_linear(A, x):
    return x @ A


@jax.jit
def _flatten_basis2onebody(basis: list[Array]) -> Array:
    if basis[0].ndim == 1:
        return jnp.concatenate(
            [jnp.ones((1,), dtype=pompon.DTYPE)]
            + [phi[..., 1:] for phi in basis],
            axis=-1,
        )
    else:
        return jnp.concatenate(
            [jnp.ones((basis[0].shape[0], 1), dtype=pompon.DTYPE)]
            + [phi[..., 1:] for phi in basis],
            axis=-1,
        )


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2onebody2y(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
) -> Array:
    # jax cannot jit list[Callable], so use tuple[Callable, ...] instead
    q = _forward_x2q(x=x, U=U)
    q0 = _forward_x2q(x=x0, U=U)
    basis = _forward_q2basis(activations=activations, q=q, q0=q0, w=w, b=b)
    onebody = _flatten_basis2onebody(basis)
    y = _forward_linear(A, onebody)
    return y


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2onebody2y(
    x: Array,
    x0: Array,
    y: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
) -> Array:
    y_pred = _forward_x2onebody2y(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, A=A
    )
    return pompon.losses.mse(y, y_pred)


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2onebody2f(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
) -> Array:
    # jax cannot jit list[Callable], so use tuple[Callable, ...] instead

    def minus_squeeze(x, x0, U, activations, w, b, A):
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
            return (
                -1.0 * _forward_x2onebody2y(x, x0, U, activations, w, b, A)
            ).squeeze(0)
        else:
            return -1.0 * _forward_x2onebody2y(x, x0, U, activations, w, b, A)

    func = partial(
        minus_squeeze, x0=x0, U=U, activations=activations, w=w, b=b, A=A
    )
    return jax.vmap(jax.jacrev(func))(x)


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2onebody2f(
    x: Array,
    x0: Array,
    f: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
) -> Array:
    f_pred = _forward_x2onebody2f(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, A=A
    )
    return pompon.losses.mse(f, f_pred)


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2onebody2yf(
    x: Array,
    x0: Array,
    y: Array,
    f: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
    wf: float,
) -> Array:
    y_pred = _forward_x2onebody2y(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, A=A
    )
    f_pred = _forward_x2onebody2f(
        x=x, x0=x0, U=U, activations=activations, w=w, b=b, A=A
    )
    return pompon.losses.mse(y, y_pred) + wf * pompon.losses.mse(f, f_pred)


@jax.jit
def _flatten_basis2nn(basis_one: list[Array], basis_two: list[Array]) -> Array:
    raise NotImplementedError


@jax.jit
def _split_basis(basis: list[Array], n: int) -> tuple[list[Array], list[Array]]:
    raise NotImplementedError


@partial(jax.jit, static_argnames=("activations",))
def _forward_x2nn2y(
    x: Array,
    x0: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
    basis_twobody_size: int,
) -> Array:
    raise NotImplementedError


@partial(jax.jit, static_argnames=("activations",))
def _mse_x2nn2y(
    x: Array,
    x0: Array,
    y: Array,
    U: Array,
    activations: tuple[Callable[[Array], Array], ...],
    w: list[Array],
    b: list[Array],
    A: Array,
) -> Array:
    raise NotImplementedError

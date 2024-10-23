"""Coordinator layer"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pompon import DTYPE
from pompon._jittables import _forward_x2q
from pompon.layers.layers import Layer
from pompon.layers.parameters import Parameter

logger = logging.getLogger("pompon").getChild("layers")


class Stiefel(Parameter):
    r"""Stiefel manifold class

    `.data` is always orthogonal matrix, i.e. Stiefel manifold

    Attributes:
        data (Array): orthogonal matrix with shape (d, f) (d >= f)

    References:

    - https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/stiefel.py
    - https://github.com/geoopt/geoopt/blob/master/geoopt/optim/radam.py

    """

    def __init__(
        self,
        data: Array,
        name: str = "U",
        X_out: np.ndarray | None = None,
    ):
        if X_out is None:
            self.projector: Array | None = None
        else:
            logger.warning("X_out is experimental feature")
            X = jnp.array(X_out, dtype=DTYPE)
            self.projector = jnp.eye(data.shape[0], dtype=DTYPE) - X @ X.T
            assert jnp.allclose(
                self.projector @ self.projector, self.projector
            ), "P^2 != P, X_out is wrong"
        super().__init__(data, name)
        assert jnp.allclose(
            (data.T @ data)[: data.shape[1], : data.shape[1]],
            jnp.eye(data.shape[1], dtype=DTYPE),
        ), (
            "Stiefel manifold condition is not satisfied: "
            + f"{name}^T @ {name} = {data.T @ data} != I"
        )

    def __setattr__(self, key: str, value: Any):
        # when trying to set self.data,
        # it should be retracted to Stiefel manifold
        match (key, value is None):
            case ("data", False):
                if hasattr(self, key):
                    assert (
                        value.shape == self.data.shape
                    ), f"shape {value.shape} != {self.data.shape}"
                if self.projector is not None:
                    value = self.projector @ value
                # ChatGPT gave me this idea to avoid infinite recursion
                object.__setattr__(self, "data", self._qr(value))
            case ("grad", False):
                object.__setattr__(self, "grad", self.proju(self.data, value))
            case ("m", False):
                # Adam optimizer
                # Note that vector m should be transported to the new point,
                # therefore self.m should be set after new self.data is set
                object.__setattr__(self, "m", self.proju(self.data, value))
            case ("momentum", False):
                # Adam or Momentum optimizer
                # Note that momentum should be transported to the new point,
                # therefore self.momentum should be set
                # after new self.data is set
                object.__setattr__(
                    self, "momentum", self.proju(self.data, value)
                )
            case _:
                super().__setattr__(key, value)

    def _qr(self, matrix: Array) -> Array:
        # Retraction to Stiefel manifold using QR decomposition
        return _qr_stiefel(matrix)

    def project(self) -> None:
        self.data = self._qr(self.data)

    def egrad2rgrad(self, x: Array, u: Array) -> Array:
        """
        Transform Euclidean gradient to Riemannian gradient
        for the point $x$

        Args:
           x (Array): Point on the manifold
           u (Array): Euclidean gradient to pe projected

        Returns:
           Array: Gradient vector in the Riemannian manifold

        """
        return self.proju(x, u)

    @staticmethod
    def sym(x: Array) -> Array:
        return 0.5 * (x + x.T)

    def retr(self, x: Array, u: Array) -> Array:
        """
        Retraction from point $x$ with given direction $u$

        Args:
           x (Array): Point on the manifold
           u (Array): Tangent vector at point $x$

        Returns:
           Array: Retracted point on the manifold

        """
        return self._qr(x + u)

    def transp(
        self,
        x: Array,
        y: Array,
        v: Array,
    ) -> Array:
        r"""
        Vector transport $𝔗_{x \to y}(v)$

        Args:
           x (Array): Start point on the manifold
           y (Array): Target point on the manifold
           v (Array): Tangent vector at point $x$

        Returns:
           Array: Transported tangent vector at point $y$

        """
        return self.proju(y, v)

    def proju(self, x: Array, u: Array) -> Array:
        """
        Project vector $u$ on a tangent space for $x$,
        usually is the same as ``egrad2rgrad``.

        Args:
           x (Array): Point on the manifold
           u (Array): Vector to be projected

        Returns:
           Array: Projected vector

        """
        return _proju_stiefel(x, u)
        # return u - x @ self.sym(x.T @ u)

    def retr_transp(
        self,
        x: Array,
        u: Array,
        v: Array,
    ) -> tuple[Array, Array]:
        """
        Perform a retraction + vector transport at once.

        Args:
           x (Array): Point on the manifold
           u (Array): Tangent vector at point $x$
           v (Array): Tangent vector at point $x$

        """
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    def project_out(self, X: Array) -> Array:
        """
        Project out some orthogonal vectors from self.data

        Args:
           X (Array): shape (d, n)
                where n is the number of vectors to project out
                and d is the input dimension.

        For example,
        if you want to project out translational vectors, x, y, z from U,
        you need to set
        X = jnp.array(jnp.vstack([x, y, z]).T)
        where x, y, z have shape (n,).
        """
        self.data = _project_out(self.data, X)
        return self.data


class Coordinator(Layer):
    r"""Coordinator layer class

    This class has a matrix data, which
    transforms the input coordinates to the hidden coordinates.

    The data is optimized to be orthogonal, i.e. Stiefel manifold

    $$
       \mathrm{St}(f, d) =
       \{ U \in \mathbb{R}^{d \times f} \mid U^\top U = I_f \}
    $$

    Forward transformation is given by

    $$
       D @ \boldsymbol{q} = \left(D @ \boldsymbol{x}\right) U
    $$

    where row vector $\boldsymbol{q} \in \mathbb{R}^f$
    is the hidden coordinates
    and column vector $\boldsymbol{x} \in \mathbb{R}^d$
    is the input coordinates.

    Args:
        input_size (int): input dimension $d$
        hidden_size (int): hidden dimension $f$
        seed (int): random seed
        random (bool): if True,
            the data is initialized by random orthogonal matrix
            using QR decomposition. Otherwise,
            the data is initialized by identity matrix.
            Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        seed: int = 0,
        random: bool = False,
        X_out: np.ndarray | None = None,
        adjacency_blocks: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.X_out = X_out
        if random:
            matrix = jax.random.normal(
                jax.random.PRNGKey(seed),
                (input_size, hidden_size),
                dtype=DTYPE,
            )
            data, _ = jnp.linalg.qr(matrix)
        else:
            data = jnp.eye(input_size, hidden_size, dtype=DTYPE)  # d x f
        self.U = Stiefel(data, "U", X_out)
        if adjacency_blocks is None:
            self.adjacency_matrix = jnp.ones(
                (input_size, hidden_size), dtype=DTYPE
            )
        else:
            self.adjacency_matrix = jax.scipy.linalg.block_diag(
                *adjacency_blocks
            )
            self.adjacency_matrix = jnp.array(
                self.adjacency_matrix, dtype=DTYPE
            )
        assert self.adjacency_matrix.shape == (input_size, hidden_size)
        assert jnp.diag(self.adjacency_matrix).all() == 1.0
        assert self.adjacency_matrix.all() in (0.0, 1.0)
        self.U.data, _ = jnp.linalg.qr(self.adjacency_matrix * self.U.data)
        # Adjacency matrix is not used at the moment
        # but it will be used for the future implementation like
        # Q = X @ (A * U).
        # by doing this we can keep U as block-diagonal & orthogonal matrix.

    def forward(self, x: Array) -> Array:
        r"""Forward transformation

        Args:
            x (Array): input coordinates
                $D$ @ $\boldsymbol{x}$
                with shape ($D$, $d$)
                where $D$ is the size of the batch
                and $d$ is the input dimension.

        Returns:
            Array: hidden coordinates
                $\boldsymbol{q}$ with shape ($D$, $f$)
                where $D$ is the size of the batch
                and $f$ is the hidden dimension.

        """
        #  q = x @ U
        return _forward_x2q(x, self.U.data)

    def __call__(self, x: Array) -> Array:
        return _forward_x2q(x, self.U.data)

    def __repr__(self):
        return (
            f"Coordinator(input_size={self.input_size}, "
            + f"hidden_size={self.hidden_size})"
        )


@jax.jit
def _proju_stiefel(x: Array, u: Array) -> Array:
    return u - x @ (0.5 * (x.T @ u + u.T @ x))


@jax.jit
def _qr_stiefel(matrix: Array) -> Array:
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    unflip = jnp.sign(jnp.sign(jnp.diag(r)) + 0.5)
    q *= unflip[None, :]
    return q


@jax.jit
def _project_out(U: Array, X: Array) -> Array:
    I = jnp.eye(X.shape[1], dtype=DTYPE)  # noqa
    return (I - X @ X.T) @ U

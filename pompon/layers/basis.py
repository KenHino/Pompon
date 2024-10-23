"""Basis layer"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

import pompon
from pompon._jittables import _forward_q2basis, _forward_q2phi, _partial_q2phi
from pompon.layers.layers import Layer
from pompon.layers.parameters import Parameter

logger = logging.getLogger(__name__)


class Basis(Layer):
    r"""Basis layer class

    This class consisists of
    [`Phi`](layers.basis.Phi.qmd) layer of each mode as a list.

    Args:
        hidden_size (int): number of modes $f$
        basis_size (int): number of basis $N$
        activation (str): activation function
        key (Array, optional): random key. Defaults to None.
        w_dist (str): distribution of the weight.
            Available distributions are "uniform", "normal", "ones".
        w_scale (float): scale of the weight.
        b_dist (str): distribution of the bias.
            Available distributions are "uniform", "normal", "linspace".
        b_scale (float): scale of the bias.

    """

    def __init__(
        self,
        hidden_size: int,
        basis_size: int,
        activation: str,
        key: Array | None = None,
        w_dist: str = "uniform",
        w_scale: float = 1.0,
        b_dist: str = "linspace",
        b_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.basis_size = basis_size
        self.activation = activation
        if key is None:
            key = jax.random.PRNGKey(0)
        self.phis = []
        for i in range(hidden_size):
            assert isinstance(key, Array)
            key_i, key = jax.random.split(key)
            phi_i = Phi(
                basis_size,
                activation,
                key_i,
                w_dist,
                w_scale,
                b_dist,
                b_scale,
                imode=i,
            )
            setattr(self, f"phi{i}", phi_i)
            self.phis.append(phi_i)

    def forward(self, q: Array, q0: Array) -> list[Array]:
        """Forward transformation

        Args:
            q (Array): hidden coordinates with shape
                (D, f) where D is the size of the batch and
                f is the hidden dimension.
            q0 (Array): reference hidden coordinates with shape
                (N-1, f) where N is the basis size.

        Returns:
            list[Array]: basis with shape (D, N)
                where D is the size of the batch
                and N is the basis size.
        """
        w = [getattr(phi, f"w{phi.imode}").data for phi in self.phis]
        b = [getattr(phi, f"b{phi.imode}").data for phi in self.phis]
        return _forward_q2basis(
            activations=self.activations, q=q, q0=q0, w=w, b=b
        )

    def partial(self, q: Array, q0: Array) -> list[Array]:
        """
        Partial derivative of the basis with respect to the q-th hidden coordinate.

        Args:
            q (Array): hidden coordinates with shape
                (D, f) where D is the size of the batch and
                f is the hidden dimension.
            q0 (Array): reference hidden coordinates with shape
                (N-1,) where N is the basis size.

        Returns:
            list[Array]: [∂φ(wq + b) / ∂q]_{p=0}^{f}
                with shape (D, N)
                where D is the size of the batch
                and N is the basis size.

        """  # noqa: E501
        return [
            self.phis[i].partial(q[:, i], q0[:, i])
            for i in range(self.hidden_size)
        ]

    def __call__(self, q: Array, q0: Array) -> list[Array]:
        return self.forward(q, q0)

    def __getitem__(self, index: int) -> Phi:
        return self.phis[index]

    def __setitem__(self, index: int, phi: Phi):
        self.phis[index] = phi

    def __repr__(self):
        return (
            f"Basis(hidden_size={self.hidden_size}, "
            + f"basis_size={self.basis_size})"
        )

    @property
    def activations(self) -> tuple[Callable, ...]:
        """
        JAX cannot compile list[Callable], so use tuple[Callable] instead.
        """
        return tuple([phi.activation for phi in self.phis])

    def plot_data(self):
        fig = plt.figure(figsize=(5, 2 * self.hidden_size))
        for i_mode in range(self.hidden_size):
            w = getattr(getattr(self, f"phi{i_mode}"), f"w{i_mode}").data
            b = getattr(getattr(self, f"phi{i_mode}"), f"b{i_mode}").data
            ax = fig.add_subplot(self.hidden_size, 1, i_mode + 1)
            ax.bar(np.arange(len(w)), w, label=f"w{i_mode}")
            ax.bar(np.arange(len(b)), b, label=f"b{i_mode}", alpha=0.7)
            ax.legend()
        plt.show()

    def plot_basis(self, q: Array, q0: Array):
        """
        Monitor the distribution of the basis to
        avoid the saturation of the activation function.
        """
        fig = plt.figure(figsize=(10, 3 * self.hidden_size))
        basis = self.forward(q, q0)
        for i_mode in range(self.hidden_size):
            ax = fig.add_subplot(self.hidden_size, 1, i_mode + 1)
            ax.set_title(f"q{i_mode}")
            ax.hist(
                np.array(basis[i_mode]).flatten(),
                bins=100,
                label=f"q{[i_mode]}",
            )
            ax.legend()
        plt.tight_layout()
        plt.show()


class Phi(Layer):
    r"""Phi (1-Basis) layer class

    This class has two vector data $\boldsymbol{w}^{(p)}$ and
    $\boldsymbol{b}^{(p)}$ which are the weight and bias, respectively.

    The forward transformation is given by

    $$
       D @ \phi^{(p)}_{\rho_p}
       = \phi(w^{(p)}_{\rho_p} (D @ \boldsymbol{q}[p]) + b^{(p)}_{\rho_p})
    $$

    where $\phi$ is the activation function,
    $D$ is the size of the batch,
    $\boldsymbol{q}[p]$ is the hidden coordinates of the p-th mode,
    $\rho_p = 1, 2, \cdots, N$ is the index of the basis,
    $w^{(p)}_{\rho_p}$ is the weight, and
    $b^{(p)}_{\rho_p}$ is the bias.

    :::{ .callout-note }
       $\phi^{(p)}_{0}$ is fixed to 1.
    :::

    Args:
        basis_size (int): number of basis N
        activation (str): activation function
        key (Array, optional): random key. Defaults to None.
        w_dist (str): distribution of the weight.
            Available distributions are "uniform", "normal", "ones".
        w_scale (float): scale of the weight. Defaults to 1.0.
        b_dist (str): distribution of the bias.
        b_scale (float): scale of the bias. Defaults to 1.0.
            Available distributions are "uniform", "normal", "linspace".
        imode (int): index of the mode p

    """

    def __init__(
        self,
        basis_size: int,
        activation: str,
        key: Array | None = None,
        w_dist: str = "uniform",
        w_scale: float = 1.0,
        b_dist: str = "linspace",
        b_scale: float = 1.0,
        imode: int = 0,
    ):
        super().__init__()
        self.basis_size = basis_size
        self.activation: Callable[[Array], Array] = self._get_activation(
            activation
        )
        self.imode = imode
        if key is None:
            key = jax.random.PRNGKey(0)
        key_w, key_b = jax.random.split(key)
        if w_dist.lower() == "uniform":
            data_w = jax.random.uniform(
                key_w,
                (basis_size - 1,),
                dtype=pompon.DTYPE,
                minval=-w_scale / 2,
                maxval=w_scale / 2,
            )
        elif w_dist.lower() in ["normal", "gauss", "exp"]:
            data_w = (
                jax.random.normal(
                    key_w,
                    (basis_size - 1,),
                    dtype=pompon.DTYPE,
                )
                * w_scale
            )
        elif w_dist.lower() in ["ones", "one"]:
            data_w = jnp.ones((basis_size - 1,), dtype=pompon.DTYPE) * w_scale
        else:
            raise NotImplementedError(
                f"{w_dist=} is not yet implemented. Set normal, ones or uniform"
            )
        setattr(self, f"w{imode}", Parameter(data_w, f"w{imode}"))
        if b_dist.lower() == "uniform":
            data_b = jax.random.uniform(
                key_b,
                (basis_size - 1,),
                dtype=pompon.DTYPE,
                minval=-b_scale / 2,
                maxval=b_scale / 2,
            )
        elif b_dist.lower() == "normal":
            data_b = (
                jax.random.normal(
                    key_b,
                    (basis_size - 1,),
                    dtype=pompon.DTYPE,
                )
                * b_scale
            )
        elif b_dist.lower() == "linspace":
            data_b = jnp.linspace(-b_scale / 2, b_scale / 2, basis_size - 1)
            # shuffle data_b in the case of combination of activation functions
            data_b = jax.random.permutation(key_b, data_b)
        else:
            raise NotImplementedError(
                f"{b_dist=} is not yet implemented. "
                + "Set normal, uniform or linspace"
            )
        setattr(self, f"b{imode}", Parameter(data_b, f"b{imode}"))

    def forward(self, q: Array, q0: Array) -> Array:
        """Forward transformation

        Args:
            q (Array): hidden coordinates
                of the p-th mode with shape (D,)
                where D is the size of the batch.

        Returns:
            Array: basis with shape (D, N)
                where D is the size of the batch
                and N is the basis size.
        """
        w = getattr(self, f"w{self.imode}").data
        b = getattr(self, f"b{self.imode}").data
        return _forward_q2phi(q=q, q0=q0, activation=self.activation, w=w, b=b)

    def partial(self, q: Array, q0: Array) -> Array:
        """
        Partial derivative of the basis
        with respect to the q-th hidden coordinate.

        Args:
            q (Array): hidden coordinates with shape
                (D,) where D is the size of the batch.
            q0 (Array): hidden coordinates with shape
                (N-1,) where N is the basis size.

        Returns:
            Array: ∂φ(wq + b) / ∂q
                with shape (D, N)
                where D is the size of the batch
                and N is the basis size.

        """
        w = getattr(self, f"w{self.imode}").data
        b = getattr(self, f"b{self.imode}").data
        grad_q = _partial_q2phi(
            q=q, q0=q0, activation=self.activation, w=w, b=b
        )
        assert (
            grad_q[:, 0].all() == 0.0
        ), f"∂1/∂q must be zero: but {grad_q[:, 0]=}"
        return grad_q

    def _get_activation(self, activation: str) -> Callable[[Array], Array]:
        match activation.lower():
            case "tanh":
                return pompon.layers.activations.tanh
            case "exp" | "exponential":
                return pompon.layers.activations.exp
            case "gauss" | "gaussian":
                return pompon.layers.activations.gaussian
            case "polynomial":
                return partial(
                    pompon.layers.activations.polynomial, N=self.basis_size
                )
            case "erf":
                return pompon.layers.activations.erf
            case "moderate":
                return partial(pompon.layers.activations.moderate, ϵ=0.05)
            case "softplus" | "soft_plus":
                return pompon.layers.activations.softplus
            case "relu":
                return pompon.layers.activations.relu
            case "leakyrelu" | "leaky_relu":
                return partial(pompon.layers.activations.leakyrelu, alpha=0.01)
            case "swish" | "silu":
                return pompon.layers.activations.silu
            case "chebyshev":
                logger.warning(
                    "Chebyshev activation is experimental and "
                    + "must be used with all w=constant, b=0 and |wx|<=1"
                )
                return partial(
                    pompon.layers.activations.chebyshev_recursive,
                    N=self.basis_size,
                    k=1,
                )
            case "legendre":
                logger.warning(
                    "Legendre activation is experimental and "
                    + "must be used with all w=constant, b=0 and |wx|<=1"
                )
                return partial(
                    pompon.layers.activations.legendre_recursive,
                    N=self.basis_size,
                    k=1,
                )
            case (
                "moderate+silu"
                | "moderate+swish"
                | "silu+moderate"
                | "swish+moderate"
            ):
                moderate = partial(pompon.layers.activations.moderate, ϵ=0.05)
                silu = pompon.layers.activations.silu
                return partial(
                    pompon.layers.activations.combine,
                    funcs=(
                        moderate,
                        silu,
                    ),
                    split_indices=self._get_split_indices(2),
                )
            case _:
                raise NotImplementedError(f"{activation} is not implemented.")

    def __repr__(self):
        return (
            f"Phi(basis_size={self.basis_size}, "
            + f"activation={self.activation.__name__}, "
            + f"imode={self.imode})"
        )

    def _get_split_indices(self, n_funcs: int) -> tuple[int, ...]:
        return tuple(
            np.linspace(0, self.basis_size, n_funcs + 1, dtype=int)[1:-1]
        )

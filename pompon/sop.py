"""
Sum of products (SOP) model
"""

import logging
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import pompon
from pompon._jittables import (
    _flatten_basis2onebody,
    _forward_x2onebody2f,
    _mse_x2onebody2y,
    _mse_x2onebody2yf,
)
from pompon.layers.basis import Basis
from pompon.layers.coordinator import Coordinator
from pompon.layers.linear import Linear
from pompon.layers.parameters import Parameter
from pompon.model import Model

logger = logging.getLogger("pompon").getChild(__name__)


class SumOfProducts(Model, ABC):
    r"""
    Function given by sum of products

    .. math::
       f(x_1, x_2, \ldots, x_d) =
       \sum_{\rho} w_{\rho} \prod_{i=1}^{d} \phi_{\rho, i}(x_i)

    """

    x0: Parameter
    coordinator: Coordinator
    basis: Basis
    linear: Linear
    _required_args = [
        "input_size",
        "hidden_size",
        "basis_size",
        "output_size",
        "activation",
        "fix_bias",
    ]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        basis_size: int,
        output_size: int = 1,
        w_scale: float = 1.0,
        b_scale: float = 1.0,
        w_dist: str = "uniform",
        b_dist: str = "linspace",
        x0: Array | None = None,
        activation: str = "silu+moderate",
        key: Array | None = None,
        X_out: np.ndarray | None = None,
        fix_bias: bool = False,
    ):
        if key is None:
            _key = jax.random.PRNGKey(0)
        else:
            _key = key
        super().__init__(input_size, output_size, fix_bias)
        self.hidden_size = hidden_size
        self.basis_size = basis_size
        if self.output_size != 1:
            raise NotImplementedError
        self.coordinator = Coordinator(
            input_size=input_size,
            hidden_size=hidden_size,
            X_out=X_out,
        )
        if x0 is not None:
            assert x0.shape == (basis_size - 1, input_size)
            self.x0 = Parameter(jnp.asarray(x0, dtype=pompon.DTYPE), "x0")
        else:
            self.x0 = Parameter(
                jnp.zeros((basis_size - 1, input_size), dtype=pompon.DTYPE),
                "x0",
            )
        basis_key, _key = jax.random.split(_key)
        self.basis = Basis(
            hidden_size=hidden_size,
            basis_size=basis_size,
            activation=activation,
            key=basis_key,
            w_dist=w_dist,
            w_scale=w_scale,
            b_dist=b_dist,
            b_scale=b_scale,
        )
        self.activation = activation

    @abstractmethod
    def flatten_basis(self, basis: list[Array]) -> Array:
        raise NotImplementedError

    @abstractmethod
    def to_nnmpo(self) -> pompon.NNMPO:
        """
        Convert to NNMPO model
        """
        raise NotImplementedError


class OneBody(SumOfProducts):
    r"""
    Function given by sum of one-body functions

    $$
       f(q_1, q_2, \ldots, q_f) =
       \sum_{p=1}^{f} \sum_{\rho_p} W_{\rho_p}^{(p)}
       \phi_{\rho_p}(w_{\rho_p}^{(p)} q_p+b_{\rho_p}^{(p)})
    $$
    """

    _required_args = [
        "input_size",
        "hidden_size",
        "basis_size",
        "output_size",
        "activation",
        "fix_bias",
    ]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        basis_size: int,
        output_size: int = 1,
        w_scale: float = 1.0,
        b_scale: float = 1.0,
        w_dist: str = "uniform",
        b_dist: str = "linspace",
        x0: Array | None = None,
        activation: str = "moderate+silu",
        key: Array | None = None,
        X_out: np.ndarray | None = None,
        fix_bias: bool = False,
    ):
        if key is None:
            _key: Array = jax.random.PRNGKey(0)
        else:
            _key = key
        super_key, _key = jax.random.split(_key)
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            basis_size=basis_size,
            output_size=output_size,
            w_scale=w_scale,
            b_scale=b_scale,
            w_dist=w_dist,
            b_dist=b_dist,
            x0=x0,
            activation=activation,
            key=super_key,
            X_out=X_out,
            fix_bias=fix_bias,
        )
        # Duplicated constant basis 1 are excluded to avoid multicollinearity
        linear_key, _key = jax.random.split(_key)
        self.linear = Linear(
            in_dim=hidden_size * basis_size - hidden_size + 1,
            out_dim=output_size,
            key=linear_key,
        )

    def grad(
        self,
        x: Array,
        y: Array,
        *,
        f: Array | None = None,
        wf: float = 1.0,
        basis_grad: bool = True,
        coordinator_grad: bool = True,
        **kwargs,
    ) -> list[Parameter]:
        U = self.coordinator.U.data
        x0 = self.x0.data
        w = [
            getattr(self.basis.phis[i], f"w{i}").data
            for i in range(self.hidden_size)
        ]
        b = [
            getattr(self.basis.phis[i], f"b{i}").data
            for i in range(self.hidden_size)
        ]
        A = self.linear.A.data
        U_index, w_index, b_index, A_index = None, None, None, None
        argnums: tuple[int, ...]
        if f is not None:
            if basis_grad and coordinator_grad:
                if not self.fix_bias:
                    # argnums = (3, 5, 6, 7)
                    argnums = (4, 6, 7, 8)
                    U_index, w_index, b_index, A_index = 0, 1, 2, 3
                else:
                    # argnums = (3, 5, 7)
                    argnums = (4, 6, 8)
                    U_index, w_index, A_index = 0, 1, 2
            elif basis_grad:
                if not self.fix_bias:
                    # argnums = (5, 6, 7)
                    argnums = (6, 7, 8)
                    w_index, b_index, A_index = 0, 1, 2
                else:
                    # argnums = (5, 7)
                    argnums = (6, 8)
                    w_index, A_index = 0, 1
            elif coordinator_grad:
                # argnums = (3, 7)
                argnums = (4, 8)
                U_index, A_index = 0, 1
            else:
                # argnums = (7,)
                argnums = (8,)
                A_index = 0
            fun = partial(_mse_x2onebody2yf, wf=wf)
            grad = jax.grad(
                fun=fun,
                argnums=argnums,
            )(x, x0, y, f, U, self.basis.activations, w, b, A)
        else:
            if basis_grad and coordinator_grad:
                if not self.fix_bias:
                    # argnums = (2, 4, 5, 6)
                    argnums = (3, 5, 6, 7)
                    U_index, w_index, b_index, A_index = 0, 1, 2, 3
                else:
                    # argnums = (2, 4, 6)
                    argnums = (3, 5, 7)
                    U_index, w_index, A_index = 0, 1, 2
            elif basis_grad:
                if not self.fix_bias:
                    # argnums = (4, 5, 6)
                    argnums = (5, 6, 7)
                    w_index, b_index, A_index = 0, 1, 2
                else:
                    # argnums = (4, 6)
                    argnums = (5, 7)
                    w_index, A_index = 0, 1
            elif coordinator_grad:
                # argnums = (2, 6)
                argnums = (3, 7)
                U_index, A_index = 0, 1
            else:
                # argnums = (6,)
                argnums = (7,)
                A_index = 0
            grad = jax.grad(
                fun=_mse_x2onebody2y,
                argnums=argnums,
            )(x, x0, y, U, self.basis.activations, w, b, A)
        params_with_grad = []
        for param in self.params():
            if param.name == "U" and U_index is not None:
                param.grad = grad[U_index]
                params_with_grad.append(param)
            elif param.name[0] == "w" and w_index is not None:
                imode = int(param.name[1:])
                param.grad = grad[w_index][imode]
                params_with_grad.append(param)
            elif param.name[0] == "b" and b_index is not None:
                imode = int(param.name[1:])
                param.grad = grad[b_index][imode]
                params_with_grad.append(param)
            elif param.name == "A" and A_index is not None:
                param.grad = grad[A_index]
                params_with_grad.append(param)
            else:
                param.grad = None
        return params_with_grad

    def flatten_basis(self, basis: list[Array]) -> Array:
        return _flatten_basis2onebody(basis)

    def forward(self, x: Array) -> Array:
        q = self.coordinator.forward(x)
        q0 = self.coordinator.forward(self.x0.data)
        basis = self.basis.forward(q, q0)
        flattened_basis = _flatten_basis2onebody(basis)
        y = self.linear(flattened_basis)
        return y

    def force(self, x: Array) -> Array:
        U = self.coordinator.U.data
        x0 = self.x0.data
        w = [
            getattr(self.basis.phis[i], f"w{i}").data
            for i in range(self.hidden_size)
        ]
        b = [
            getattr(self.basis.phis[i], f"b{i}").data
            for i in range(self.hidden_size)
        ]
        A = self.linear.A.data
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
            jacobian = _forward_x2onebody2f(
                x, x0, U, self.basis.activations, w, b, A
            ).squeeze(0)
        else:
            jacobian = _forward_x2onebody2f(
                x, x0, U, self.basis.activations, w, b, A
            )
        return jacobian.squeeze(-2)

    def to_nnmpo(self) -> pompon.NNMPO:
        """
        Convert to NNMPO model
        """
        logger.info("OneBody function is converted to NNMPO")
        if self.output_size != 1:
            raise NotImplementedError

        nnmpo = pompon.NNMPO(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            basis_size=self.basis_size,
            bond_dim=2,
            output_size=self.output_size,
            activation=self.basis.activation,
            X_out=self.coordinator.X_out,
            fix_bias=self.fix_bias,
        )
        pytree_self: dict[str, Array] = {
            param.name: param.data.copy() for param in self.params()
        }
        for param in nnmpo.params():
            if param.name in pytree_self:
                param.data = pytree_self[param.name]
        split_index = np.cumsum([1] + [self.basis_size - 1] * self.hidden_size)
        logger.debug(f"{split_index=}")
        for i, core in enumerate(nnmpo.tt.cores):
            new_core = np.zeros_like(core.data, dtype=np.float64)
            if i == 0:
                new_core[0, 0, 0] = 1.0
                if self.hidden_size == 1:
                    new_core[0, :, 0] = pytree_self["A"][
                        : split_index[i + 1], 0
                    ]
                else:
                    new_core[0, :, 1] = pytree_self["A"][
                        : split_index[i + 1], 0
                    ]
            elif i == self.hidden_size - 1:
                new_core[0, 1:, 0] = pytree_self["A"][
                    split_index[i] : split_index[i + 1], 0
                ]
                new_core[1, 0, 0] = 1.0
            else:
                new_core[0, 1:, 1] = pytree_self["A"][
                    split_index[i] : split_index[i + 1], 0
                ]
                new_core[0, 0, 0] = 1.0
                new_core[1, 0, 1] = 1.0
            core.data = jnp.array(new_core, dtype=pompon.DTYPE)
        return nnmpo


class NearestNeighbor(SumOfProducts):
    r"""
    Function given by sum of nearest-neighbor functions

    $$
       f(x_1, x_2, \ldots, x_d) =
       \sum_{p=1}^{d-1} \sum_{\rho_p}
       w_{\rho_p} \phi_{\rho_p}(x_p)\phi_{\rho_p+1}(x_{p+1})
    $$

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        basis_size: int,
        output_size: int = 1,
        basis_twobody_size: int | None = None,
        w_scale: float = 1.0,
        b_scale: float = 0.0,
        w_dist: str = "uniform",
        b_dist: str = "uniform",
        activation: str = "tanh",
        key: Array | None = None,
        X_out: np.ndarray | None = None,
    ):
        if key is None:
            _key = jax.random.key(0)
        else:
            _key = key
        super_key, _key = jax.random.split(_key)
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            basis_size=basis_size,
            output_size=output_size,
            w_scale=w_scale,
            b_scale=b_scale,
            w_dist=w_dist,
            b_dist=b_dist,
            activation=activation,
            key=super_key,
            X_out=X_out,
        )
        if basis_twobody_size is None:
            # basis_twobody_size is a number of basis used in two-body functions
            basis_twobody_size = basis_size // 2
        assert basis_twobody_size <= basis_size
        self.basis_twobody_size = basis_twobody_size

        # Duplicated constant basis 1 are excluded to avoid multicollinearity
        linear_key, _key = jax.random.split(_key)
        self.linear = Linear(
            in_dim=hidden_size * basis_size - hidden_size + 1,
            out_dim=output_size,
            key=linear_key,
        )
        # Bond dimension increases linearly with `basis_twobody_size`
        raise NotImplementedError

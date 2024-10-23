from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

import pompon
from pompon import DTYPE
from pompon._jittables import (
    _forward_q2y,
    _forward_x2y,
    _grad_block_and_basis2y_onedot,
    _grad_block_and_basis2y_twodot,
    _mse_block_and_basis2y_onedot,
    _mse_block_and_basis2y_twodot,
    _mse_q2y,
    _mse_q2yf,
    _mse_x2y,
    _mse_x2yf,
    _total_loss_q2y,
    _total_loss_x2y,
)
from pompon.layers.basis import Basis
from pompon.layers.coordinator import Coordinator
from pompon.layers.layers import Layer
from pompon.layers.parameters import Parameter
from pompon.layers.tt import TensorTrain

logger = logging.getLogger("pompon").getChild(__name__)


class Model(Layer, ABC):
    """Abstract Model class"""

    coordinator: Coordinator
    basis: Basis
    x0: Parameter
    _required_args = ["input_size", "output_size", "fix_bias"]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        fix_bias: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fix_bias = fix_bias

    @property
    def q0(self):
        """Get initial hidden coordinates $q_0=x_0U$"""
        return self.coordinator.forward(self.x0.data)

    @staticmethod
    def _mse(pred, true) -> float:
        mse = pompon.losses.mse(true, pred)
        if jnp.isnan(mse):
            raise ValueError("Mean squared error is NaN")
        return float(mse)

    def mse(self, x: Array, y: Array) -> float:
        r"""Mean squared error

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.
            y (Array): output tensor with shape $(D,1)$

        Returns:
            float: mean squared error

        """
        y_pred = self.forward(x)
        return self._mse(y_pred, y)

    def mse_force(self, x: Array, f: Array) -> float:
        r"""Mean squared error with force

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.
            f (Array): force tensor with shape $(D,n)$
        """
        f_pred = self.force(x)
        return self._mse(f_pred, f)

    @abstractmethod
    def forward(self, x: Array) -> Array:
        r"""Forward propagation

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.

        Returns:
            Array: output tensor with shape $(D,1)$

        """
        raise NotImplementedError

    @abstractmethod
    def force(self, x: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def grad(
        self,
        x: Array,
        y: Array,
        *,
        f: Array | None = None,
        basis_grad: bool = True,
        coordinator_grad: bool = True,
        lambda1: float = 0.0,
        mu1: float = 1.0,
        mu2: float = 1.0,
        wf: float = 1.0,
    ) -> list[Parameter]:
        r"""Gradient of loss function

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.
            y (Array): output tensor with shape $(D,1)$
            f (Array, optional): force tensor with shape $(D,n)$
            basis_grad (bool, optional):  calculate $w,b$ grad
            coordinator_grad (bool, optional): calculate $U$ grad
            wf (float, optional): Weight $w_f$ of force term in loss function.

        Returns:
            list[Parameter]: list of parameters with gradients

        """
        raise NotImplementedError

    def basis_entropy(self, x: Array) -> float:
        q = self.coordinator.forward(x)
        q0 = self.coordinator.forward(self.x0.data)
        basis = self.basis.forward(q, q0)
        _l1_norms_phis = [pompon.losses.L1_norm(phi) for phi in basis]
        _l1_entropies_basis = [
            float(pompon.losses.L1_entropy(l1_norms))
            for l1_norms in _l1_norms_phis
        ]
        return sum(_l1_entropies_basis)

    def basis_L1_norm(self, x: Array) -> float:
        q = self.coordinator.forward(x)
        q0 = self.coordinator.forward(self.x0.data)
        basis = self.basis.forward(q, q0)
        _L1_norms_phis = [pompon.losses.L1_norm(phi) for phi in basis]
        _L1_norms_Phi = [
            float(jnp.sum(L1_norms)) for L1_norms in _L1_norms_phis
        ]
        return sum(_L1_norms_Phi)

    def plot_basis(self, x: Array):
        r"""
        Plot distribution of $\phi$

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.

        Examples:
           ```{python}
           import numpy as np
           import pompon
           x = np.random.rand(10, 3)
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5,
                                activation="gauss", b_scale=1.0, w_scale=1.0)
           model.plot_basis(x)
           ```
        """
        Q = self.coordinator.forward(x)
        Q0 = self.coordinator.forward(self.x0.data)
        self.basis.plot_basis(Q, Q0)

    def show_onebody(self):
        """
        Visualize one-dimensional cut.

        Examples:
           ```{python}
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           model.show_onebody()
           ```

        """
        d = self.input_size
        assert d > 1
        fig, axs = plt.subplots(1, d, figsize=(4 * d, 4))
        x = np.linspace(-3, 3, 100)

        for i in range(d):
            _x = np.zeros((100, d))
            _x[:, i] = x
            y = self.forward(jnp.array(_x, dtype=DTYPE))[:, 0]
            axs[i].set_title(f"{i}-site")
            axs[i].set_xlabel(f"$x_{i}$")
            axs[i].plot(x, y)
        plt.show()

    def export_h5(self, path: str) -> None:
        """
        Export the model to a HDF5 file

        Args:
           path (str): path to the HDF5 file

        Examples:
           ```python
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           model.export_h5("/path/to/model.h5")
           ```

        See Also:
           [import_h5()](#pompon.model.NNMPO.import_h5)

        """
        logger.info(f"Model is exported to {path}")
        pytree: dict[str, dict[str, Any]] = dict()
        pytree["params"] = dict()
        pytree["args"] = dict()
        for param in self.params():
            pytree["params"][param.name] = param.data
        for arg in self._required_args:
            pytree["args"][arg] = getattr(self, arg)

        def save_pytree_to_hdf5(pytree, f, group_name="/"):
            for key, value in pytree.items():
                if isinstance(value, dict):
                    save_pytree_to_hdf5(value, f, group_name + key + "/")
                else:
                    f.create_dataset(group_name + key, data=value)

        with h5py.File(path, "w") as f:
            save_pytree_to_hdf5(pytree, f)

    @classmethod
    def import_h5(cls, path: str) -> Model:
        """Import the model from a HDF5 file

        Args:
           path (str): path to the HDF5 file

        Returns:
           Model: model instance

        Examples:
           ```python
           import pompon
           model = pompon.NNMPO.import_h5("/path/to/model.h5")
           ```

        See Also:
           [export_h5()](#pompon.model.NNMPO.export_h5)

        """
        logger.info(f"Model is imported from {path}")

        def load_pytree_from_hdf5(f, group_name="/"):
            pytree = dict()
            for key, value in f[group_name].items():
                logger.debug(f"Loading {key}")
                if isinstance(value, h5py.Group):
                    pytree[key] = load_pytree_from_hdf5(
                        f, group_name + key + "/"
                    )
                elif isinstance(value[()], bytes):
                    pytree[key] = value[()].decode("utf-8")
                else:
                    pytree[key] = value[()]
            return pytree

        with h5py.File(path, "r") as f:
            pytree = load_pytree_from_hdf5(f)
        model = cls(**pytree["args"])
        for param in model.params():
            if param.name in pytree["params"]:
                param.data = jnp.asarray(
                    pytree["params"][param.name], dtype=DTYPE
                )
            else:
                logger.warning(
                    f"{param.name} is not found in the file. set default value."
                )
        return model


class NNMPO(Model):
    r"""Neural Network Matrix Product Operator

    ![](nnmpo.svg)

    $$
    \begin{align}
        &V_{\text{NN-MPO}}(\mathbf{x}) = \widetilde{V}_{\text{NN-MPO}}(\mathbf{q}) \notag \\
        &=
        \label{eq:nnmpo-full}
        \sum_{\substack{\rho_1,\rho_2,\cdots\rho_f\\
                \beta_1,\beta_2,\cdots\beta_{f-1}}}
        \phi_{\rho_1}(q_1) \cdots \phi_{\rho_f}(q_f)
        W\substack{\rho_1\\1\beta_1}W\substack{\rho_2\\\beta_1\beta_2}
        \cdots W\substack{\rho_f\\\beta_{f-1}1}.
    \end{align}
    $$

    where $\phi$ is an activation and
    $[q_1, \cdots, q_n] = [x_1, \cdots, x_n]U$ is a linear transformation.

    This class mainly consists of three layers:

    - [`Coordinator`](layers.coordinator.Coordinator.qmd)
    - [`Basis`](layers.basis.Basis.qmd)
    - [`TensorTrain`](layers.tt.TensorTrain.qmd)

    Args:
       input_size (int): Input size $n$
       hidden_size (int): Hidden size $f$
       basis_size (int): Number of basis $N$ per mode. ($\rho_i=1,2,\cdots,N$)
       bond_dim (int): Bond dimension $M$ ($\beta_i=1,2,\cdots,M$).
       output_size (int): Output size. Only `output_size=1` is supported so far.
       x0 (Array, optional): Reference point for the input coordinates.
            $q_0=x_0U$ and $w(q-q_0)+b$ will be argument of the basis function.
            If None, `x0` will be zeros.
       activation (str): activation function. See also [`activations`](layers.activations.qmd).
       w_scale (float): scaling factor of weights.
       w_dist (str): weight distribution. Available options is written in [`basis`](layers.basis.Basis.qmd).
       b_scale (float): scaling factor of biases.
       b_dist (str): bias distribution. Available options is written in [`basis`](layers.basis.Basis.qmd).
       random_tt (bool): if True, initialize tensor-train randomly.
       X_out (Array, optional): Project out vector from the hidden coordinates. See details on [`Coordinator`](layers.coordinator.Coordinator.qmd).
       fix_bias (bool): Whether or not fix $b$.

    Examples:
       ```{python}
       import numpy as np
       import pompon
       model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
       x = np.random.rand(10, 3)
       y = model.forward(x)
       y.shape
       ```

    """  # noqa: E501

    _required_args = [
        "input_size",
        "hidden_size",
        "basis_size",
        "output_size",
        "bond_dim",
        "activation",
        "fix_bias",
    ]

    def __init__(
        self,
        input_size: int,
        basis_size: int,
        *,
        hidden_size: int | None = None,
        bond_dim: int = 2,
        output_size: int = 1,
        w_scale: float = 1.0,
        b_scale: float = 0.0,
        w_dist: str = "uniform",
        b_dist: str = "linspace",
        x0: Array | None = None,
        activation: str = "silu+moderate",
        random_tt: bool = True,
        key: Array | None = None,
        X_out: np.ndarray | None = None,
        fix_bias: bool = False,
    ):
        if key is None:
            _key: Array = jax.random.PRNGKey(0)
        else:
            _key = key

        super().__init__(
            input_size=input_size, output_size=output_size, fix_bias=fix_bias
        )
        if hidden_size is None:
            hidden_size = input_size
        if hidden_size > input_size:
            logger.warning(
                f"{hidden_size=} > {input_size=} is not well tested especially for force calculation"  # noqa: E501
            )
        self.hidden_size: int = hidden_size
        self.basis_size = basis_size
        self.activation = activation
        self.fix_bias = fix_bias
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
        if self.output_size != 1:
            raise NotImplementedError
        self.limit_bond_dim = bond_dim
        if random_tt:
            tt_key, _key = jax.random.split(_key)
            self.tt = TensorTrain.set_random(
                shape=(basis_size,) * hidden_size,
                rank=bond_dim,
                key=tt_key,
            )
        else:
            self.tt = TensorTrain.set_ones(
                shape=(basis_size,) * hidden_size,
                rank=bond_dim,
            )
        self.tt.norm.data = jnp.array(1.0, dtype=DTYPE)

    @property
    def bond_dim(self) -> int:
        r"""
        Get maximum bond dimension $M_{\text{max}}$.

        Returns:
           int: maximum bond dimension

        Examples:
           ```{python}
           import numpy as np
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           model.bond_dim
           ```
        """
        if len(self.tt.ranks) == 0:
            return 1
        return max(self.tt.ranks)

    def _get_args(
        self,
    ):
        U = self.coordinator.U.data
        x0 = self.x0.data
        activations = self.basis.activations
        w = [
            getattr(self.basis.phis[i], f"w{i}").data
            for i in range(self.hidden_size)
        ]
        b = [
            getattr(self.basis.phis[i], f"b{i}").data
            for i in range(self.hidden_size)
        ]
        W = [getattr(self.tt, f"W{i}").data for i in range(self.hidden_size)]
        norm = self.tt.norm.data
        return U, x0, activations, w, b, W, norm

    def _check_valid_input(self, x: Array) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"ndim of input must be 2 but got {x.ndim} "
                + "when execution with scalar input, please use x[None, :]"
            )
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"input_size must be {self.input_size} but got {x.shape[1]}"
            )

    def forward(self, x: Array) -> Array:
        r"""
        Compute energy (forward propagation) $V_{\text{NN-MPO}}(\mathbf{x})$.

        Args:
            x (Array): input tensor with shape $(D,n)$
                       where $D$ is the batch size
                       and $n$ is the input size.

        Returns:
            Array: output tensor with shape $(D,1)$

        Examples:
            ```{python}
            import numpy as np
            import pompon
            model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
            x = np.random.rand(10, 3)
            y = model.forward(x)
            y.shape
            ```

        """
        x = jnp.asarray(x, dtype=DTYPE)
        self._check_valid_input(x)
        U, x0, activations, w, b, W, norm = self._get_args()
        return _forward_x2y(
            x=x, x0=x0, U=U, activations=activations, w=w, b=b, W=W, norm=norm
        )

    def force(self, x: Array) -> Array:
        r"""
        Compute force $-\nabla V_{\text{NN-MPO}}(\mathbf{x})$.

        Args:
           x (Array): input tensor with shape $(D,n)$

        Returns:
           force (Array): force tensor with shape $(D,n)$

        Examples:
           ```{python}
           import numpy as np
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           x = np.random.rand(10, 3)
           f = model.force(x)
           f.shape
           ```

        """
        if no_batch := (x.ndim == 1):
            x = x[jnp.newaxis, :]
        elif x.ndim != 2:
            raise ValueError("x must be 1D or 2D array")
        x = jnp.asarray(x, dtype=DTYPE)
        self._check_valid_input(x)
        U, x0, activations, w, b, W, norm = self._get_args()
        q = self.coordinator.forward(x)
        assert q.ndim == 2, f"{q.shape=}"
        q0 = self.coordinator.forward(x0)

        def func(q, q0, activations, w, b, W, norm):
            if q.ndim == 1:
                q = q[jnp.newaxis, :]
                return (
                    -1.0 * _forward_q2y(q, q0, activations, w, b, W, norm)
                ).squeeze(0)
            elif q.ndim == 2:
                return -1.0 * _forward_q2y(q, q0, activations, w, b, W, norm)
            else:
                raise ValueError(f"{q.ndim=}")

        func = partial(
            func, q0=q0, activations=activations, w=w, b=b, W=W, norm=norm
        )
        if no_batch:
            jacobian = (jax.jacrev(func)(q) @ U.T).squeeze(0)
        else:
            jacobian = jax.vmap(jax.jacrev(func))(q) @ U.T

        # jacobian has shape (batch_size, output_size, input_size)
        # while force has shape (batch_size, input_size)
        return jacobian.squeeze(-2)

    def grad(
        self,
        x: Array | None,
        y: Array,
        *,
        loss: str = "mse",
        twodot_grad: bool = False,
        onedot_grad: bool = False,
        basis_grad: bool = False,
        coordinator_grad: bool = False,
        q: Array | None = None,
        basis: list[Array] | None = None,
        use_auto_diff: bool = False,
        lambda1: float = 1.0e-04,
        mu1: float = 0.1,
        mu2: float = 1.0,
        f: Array | None = None,
        wf: float = 1.0,
        to_right: bool = True,
    ) -> list[Parameter]:
        r"""
        Gradient of loss function with respect to $W$, $w$, $b$ and $U$

        Args:
           x (Array): input tensor with shape $(D,n)$
                      where $D$ is the batch size and $n$ is the input size.
           y (Array): output tensor with shape $(D,1)$
           loss (str, optional): loss function.
           twodot_grad (bool, optional): if True, compute gradient with respect to $B$.
                                         Defaults to False.
           onedot_grad (bool, optional): if True, compute gradient with respect to $C$.
                                         Defaults to False.
           basis_grad (bool, optional): if True, compute gradient with respect to $w$ and $b$.
                                        Defaults to False.
           coordinator_grad (bool, optional): if True, compute gradient with respect to $U$.
                                              Defaults to False.
           q (Array, optional): hidden coordinates with shape $(D,f)$
                                where $f$ is the hidden dimension.
                                Defaults to None. If None, it is computed from $x$.
           basis (list[Array], optional): basis with shape $f\times(D,N)$
                                          where $N$ is the basis size.
                                          Defaults to None.
                                          If None, it is computed from $q$.
           use_auto_diff (bool, optional): if True, use auto differentiation.
                                           Otherwise, use analytical formula.
                                           Defaults to False.
           lambda1 (float, optional): EXPERIMENTAL FEATURE!  regularization parameter.
                if not 0, add L1 regularization + entropy penalty.
           mu1 (float, optional): EXPERIMENTAL FEATURE! L1 penalty parameter.
           mu2 (float, optional): EXPERIMENTAL FEATURE! entropy penalty parameter.
           f (Array, optional): force with shape $(D,n)$.
           wf (float, optional): Weight $w_f$ of force term in loss function.
           to_right (bool, optional): if True, twodot core index is ``(tt.center, tt.center+1)``
                otherwise ``(tt.center-1, tt.center)``.

        Returns:
            list[Parameter]: list of parameters with gradients

        Examples:
           ```{python}
           import numpy as np
           x = np.random.rand(10, 3)
           y = np.random.rand(10, 1)
           f = np.random.rand(10, 3)
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           params = model.grad(x, y, f=f,
                               basis_grad=True, coordinator_grad=True)
           for param in params:
              print(f"{param.name=}, {param.data.shape=}")
              param.data -= 0.01 * param.grad
           ```

        """  # noqa: E501
        if loss.lower() != "mse":
            raise NotImplementedError

        w = [
            getattr(self.basis.phis[i], f"w{i}").data
            for i in range(self.hidden_size)
        ]
        b = [
            getattr(self.basis.phis[i], f"b{i}").data
            for i in range(self.hidden_size)
        ]
        U = self.coordinator.U.data
        x0 = self.x0.data
        q0 = self.coordinator.forward(x0)
        if basis_grad or coordinator_grad:
            W = [
                getattr(self.tt, f"W{i}").data for i in range(self.hidden_size)
            ]
        else:
            W = None
        norm = self.tt.norm.data
        params_with_grad = []

        if twodot_grad & onedot_grad:
            raise ValueError(
                "twodot and onedot cannot be True at the same time"
            )

        match (twodot_grad ^ onedot_grad, basis_grad, coordinator_grad):
            case (False, False, False):
                raise ValueError(
                    "At least one of twodot, basis and coordinator must be True"
                )
            case (False, False, True):
                # jax.grad cannot support key=arg,
                # so we need to use positional argument
                if lambda1 > 0.0:
                    fun: Callable[..., Any] = partial(
                        _total_loss_x2y, lambda1=lambda1, mu1=mu1, mu2=mu2
                    )
                elif f is not None and wf > 0.0:
                    fun = partial(_mse_x2yf, f=f, wf=wf)
                else:
                    fun = _mse_x2y

                grad: list[Array | list[Array]] = jax.grad(
                    fun=fun, argnums=[3]
                )(x, x0, y, U, self.basis.activations, w, b, W, norm)
                for param in self.params():
                    if param.name == "U":
                        param.grad = grad[0]
                        params_with_grad.append(param)
                    else:
                        param.grad = None
            case (False, True, False):
                if q is None:
                    if x is None:
                        raise ValueError("x must be provided when q is None")
                    else:
                        q = self.coordinator.forward(x)
                q0 = self.coordinator.forward(self.x0.data)
                if lambda1 > 0.0:
                    fun = partial(
                        _total_loss_q2y,
                        q0=q0,
                        x0=x0,
                        lambda1=lambda1,
                        mu1=mu1,
                        mu2=mu2,
                    )
                elif f is not None and wf > 0.0:
                    if x is None:
                        x = q @ U.T
                    fun = partial(_mse_q2yf, x=x, x0=x0, f=f, wf=wf, U=U)
                else:
                    fun = partial(_mse_q2y, q0=q0)
                grad = jax.grad(fun=fun, argnums=[3, 4])(
                    q, y, self.basis.activations, w, b, W, norm
                )
                grad_dict: dict[str, Array | list[Array]] = {}
                for i in range(self.hidden_size):
                    grad_dict[f"w{i}"] = grad[0][i]
                    grad_dict[f"b{i}"] = grad[1][i]
                for param in self.params():
                    if param.name[0] == "w":
                        imode = int(param.name[1:])
                        param.grad = grad[0][imode]
                        params_with_grad.append(param)
                    elif param.name[0] == "b" and not self.fix_bias:
                        imode = int(param.name[1:])
                        param.grad = grad[1][imode]
                        params_with_grad.append(param)
                    else:
                        param.grad = None
            case (False, True, True):
                if lambda1 > 0.0:
                    fun = partial(
                        _total_loss_x2y, lambda1=lambda1, mu1=mu1, mu2=mu2
                    )
                elif f is not None and wf > 0.0:
                    fun = partial(_mse_x2yf, f=f, wf=wf)
                else:
                    fun = _mse_x2y
                grad = jax.grad(fun=fun, argnums=[3, 5, 6])(
                    x, x0, y, U, self.basis.activations, w, b, W, norm
                )
                grad_dict = {"U": grad[0]}
                for i in range(self.hidden_size):
                    grad_dict[f"w{i}"] = grad[1][i]
                    grad_dict[f"b{i}"] = grad[2][i]
                for param in self.params():
                    if param.name == "U":
                        param.grad = grad[0]
                        params_with_grad.append(param)
                    elif param.name[0] == "w":
                        imode = int(param.name[1:])
                        param.grad = grad[1][imode]
                        params_with_grad.append(param)
                    elif param.name[0] == "b" and not self.fix_bias:
                        imode = int(param.name[1:])
                        param.grad = grad[2][imode]
                        params_with_grad.append(param)
                    else:
                        param.grad = None
            case (True, False, False):
                if basis is None:
                    if q is None:
                        if x is None:
                            raise ValueError(
                                "x must be provided when q is None"
                            )
                        else:
                            q = self.coordinator.forward(x)
                    basis = self.basis.forward(q, q0)
                if twodot_grad:
                    if isinstance(self.tt.left_blocks_batch, list):
                        if to_right:
                            left_block_batch = self.tt.left_blocks_batch[
                                -1
                            ].data
                        else:
                            left_block_batch = self.tt.left_blocks_batch[
                                -2
                            ].data
                    else:
                        raise ValueError("left_block_batch is not yet computed")
                    if isinstance(self.tt.right_blocks_batch, list):
                        if to_right:
                            right_block_batch = self.tt.right_blocks_batch[
                                -2
                            ].data
                        else:
                            right_block_batch = self.tt.right_blocks_batch[
                                -1
                            ].data
                    else:
                        raise ValueError(
                            "right_block_batch is not yet computed"
                        )
                    B = self.tt.B
                    if B is None:
                        raise ValueError("B is not yet computed")
                    center_twodot = B.data
                    if self.tt.center is None:
                        raise ValueError("center is not yet computed")
                    else:
                        if to_right:
                            left_phi_batch: Array = basis[self.tt.center]
                            right_phi_batch: Array = basis[self.tt.center + 1]
                        else:
                            left_phi_batch = basis[self.tt.center - 1]
                            right_phi_batch = basis[self.tt.center]
                    if use_auto_diff:
                        grad = jax.grad(
                            fun=_mse_block_and_basis2y_twodot, argnums=[0]
                        )(
                            center_twodot,
                            y,
                            left_phi_batch,
                            right_phi_batch,
                            left_block_batch,
                            right_block_batch,
                            norm,
                        )
                    else:
                        grad = [
                            _grad_block_and_basis2y_twodot(
                                center_twodot=center_twodot,
                                y=y,
                                left_phi_batch=left_phi_batch,
                                right_phi_batch=right_phi_batch,
                                left_block_batch=left_block_batch,
                                right_block_batch=right_block_batch,
                                norm=norm,
                            )
                        ]
                    B.grad = grad[0]
                    params_with_grad.append(B)
                elif onedot_grad:
                    if isinstance(self.tt.left_blocks_batch, list):
                        left_block_batch = self.tt.left_blocks_batch[-1].data
                    else:
                        raise ValueError("left_block_batch is not yet computed")
                    if isinstance(self.tt.right_blocks_batch, list):
                        right_block_batch = self.tt.right_blocks_batch[-1].data
                    else:
                        raise ValueError(
                            "right_block_batch is not yet computed"
                        )
                    C = self.tt.C
                    if C is None:
                        raise ValueError("C is not yet computed")
                    center_onedot = C.data
                    center_phi_batch = basis[self.tt.center]
                    if use_auto_diff:
                        grad = jax.grad(
                            fun=_mse_block_and_basis2y_onedot, argnums=[0]
                        )(
                            center_onedot,
                            y,
                            center_phi_batch,
                            left_block_batch,
                            right_block_batch,
                            norm,
                        )
                    else:
                        grad = [
                            _grad_block_and_basis2y_onedot(
                                center_onedot=center_onedot,
                                y=y,
                                center_phi_batch=center_phi_batch,
                                left_block_batch=left_block_batch,
                                right_block_batch=right_block_batch,
                                norm=norm,
                            )
                        ]
                    C.grad = grad[0]
                    params_with_grad.append(C)
                else:
                    raise ValueError("Invalid combination of twodot and onedot")

            case (True, False, True):
                raise ValueError(
                    "twodot and coordinator cannot be True at the same time"
                )
            case (True, True, False):
                raise ValueError(
                    "twodot and basis cannot be True at the same time"
                )
            case (True, True, True):
                raise ValueError(
                    "twodot, basis and coordinator "
                    + "cannot be True at the same time"
                )
            case _:
                raise ValueError(
                    "Invalid combination of twodot, basis and coordinator"
                )

        return params_with_grad

    def update_blocks_batch(
        self,
        x: Array | None,
        q: Array | None = None,
        basis: list[Array] | None = None,
        is_onedot_center: bool = False,
    ) -> None:
        r"""
        Update Left and Right blocks batch of $W$ (tensor-train) with $\phi$.

        Args:
           x (Array): input tensor with shape $(D,n)$
                      where $D$ is the batch size
           q (Array, optional): hidden coordinates with shape
                                $(D,f)$
                                where $f$ is the hidden dimension.
                                If already computed, set this argument to
                                avoid redundant computation.
           basis (list[Array], optional): $\phi_{\rho_i}(q_i)$ with shape $(D,N)$.
                                          If already computed, set this argument
                                          to avoid redundant computation.
           is_onedot_center (bool, optional): if True,
                update $L[1],...,L[p-1],R[p+1],...R[f]$
                with the new basis. Otherwise,
                update $L[1],...,L[p-1],R[p+2],...R[f]`$.

        Examples:
           ```{python}
           import numpy as np
           x = np.random.rand(10, 3)
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           model.update_blocks_batch(x, is_onedot_center=True)
           print(f"{model.tt.center=}")
           print(f"{model.tt.left_blocks_batch=}")
           print(f"{model.tt.right_blocks_batch=}")
           ```

        """  # noqa: E501
        if basis is None:
            if q is None:
                if x is None:
                    raise ValueError("x must be provided when q is None")
                else:
                    q = self.coordinator.forward(x)
            q0 = self.coordinator.forward(self.x0.data)
            basis = self.basis.forward(q, q0)
        self.tt.set_blocks_batch(basis)

    def rescale(self, input_scale: float, output_scale: float) -> None:
        r"""Rescale the model

        Learning should be done with the normalized input and output.
        But, when the model is used for prediction, it is better to
        rescale the input and output to the original scale.

        Input scale and output scale are attributed to the
        ``basis.phi.w.data`` and ``tt.norm.data``, respectively.

        Args:
            input_scale (float): scaling factor of input
            output_scale (float): scaling factor of output

        Examples:
           ```{python}
           import numpy as np
           import pompon
           x = np.random.rand(10, 3)
           y = np.random.rand(10, 1)
           x_scale = x.std()
           y_scale = y.std()
           x /= x_scale
           y /= y_scale
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           # Some learning process with normalized input and output
           model.rescale(input_scale=x_scale, output_scale=y_scale)
           ```
        """
        for i in range(self.hidden_size):
            getattr(getattr(self.basis, f"phi{i}"), f"w{i}").data /= input_scale
        self.x0.data *= input_scale
        self.tt.norm.data *= output_scale

    def convert_to_mpo(self, basis_ints: list[np.ndarray]) -> list[np.ndarray]:
        r"""Convert to Matrix Product Operator (MPO)

        $$
           \mathcal{W}\substack{\sigma_i^\prime\\\beta_{i-1}\beta_i \\ \sigma_{i}}
           = \sum_{\rho_i=1}^{N_i}
           W\substack{\rho_i\\\beta_{i-1}\beta_i}
           \langle\sigma_i^\prime|\phi_{\rho_i}^{(i)}|\sigma_i\rangle
        $$
        $$
           \hat{V}_{\mathrm{NNMPO}}\left(\pmb{Q}\right)
           = \sum_{\{\pmb{\beta}\},\{\pmb{\sigma}\},\{\pmb{\sigma}^\prime\}}
           \mathcal{W}\substack{\sigma_1^\prime\\1\beta_1\\\sigma_1}
           \mathcal{W}\substack{\sigma_2^\prime\\\beta_1\beta_2\\\sigma_2}
           \cdots
           \mathcal{W}\substack{\sigma_f^\prime\\\beta_{f-1}1\\\sigma_f}
           |\sigma_1^\prime\sigma_2^\prime\cdots\sigma_f^\prime\rangle
           \langle\sigma_1\sigma_2\cdots\sigma_f|
        $$


        Args:
            basis_ints (list[np.ndarray], optional): List of the integrals between
                potential basis function and wave function basis function
                $\langle\sigma_i|\phi_{\rho_i}^{(i)}|\sigma_i\rangle$.
                The length of the list must be equal to the hidden size $f$.
                The list element is an array with shape $(d_i, N_i, d_i)$
                where $d_i$ is the number of basis functions of the wave function
                and $N_i$ is the number of basis functions of the potential.
                If you want raw tensor-train data, you can address by ``nnmpo.tt.W.data``.

        Returns:
            list[np.ndarray]: MPO. The length of the list is equal to the hidden size.
                The $i$-th element is an array with shape $(M_i, d_i, d_i, M_{i+1})$
                where $M_i$ is the bond dimension.

        Examples:
           ```{python}
           import numpy as np
           import pompon
           model = pompon.NNMPO(input_size=3, hidden_size=3, basis_size=5)
           # Basis functions can be evaluated ``by model.basis.phis.forward(q, model.q0)``
           # This is just an dummy example.
           basis_ints = [np.random.rand(4, 5, 4) for _ in range(3)]
           mpo = model.convert_to_mpo(basis_ints)
           for i in range(3):
               print(f"{mpo[i].shape=}")
           ```

        """  # noqa: E501
        mpo = []
        for i in range(self.hidden_size):
            W = np.array(getattr(self.tt, f"W{i}").data)
            mpo.append(np.einsum("abc,dbe->adec", W, basis_ints[i]))
        mpo[0] *= float(self.tt.norm.data)

        return mpo

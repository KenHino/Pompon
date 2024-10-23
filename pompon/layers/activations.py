r"""
Activation functions for neural networks.

## Supported Activations

| Argument name   | Implementation (Docstring) |
| :--:            | :--:                       |
| `silu+moderate` | [`combine`](`pompon.layers.activations.combine`) [`silu`](#pompon.layers.activations.silu) and [`moderate`](#pompon.layers.activations.moderate)|
| `tanh`          | [`pompon.layers.activations.tanh`](#pompon.layers.activations.tanh) |
| `exp`           | [`pompon.layers.activations.exp`](#pompon.layers.activations.exp) |
| `gauss`         | [`pompon.layers.activations.gaussian`](#pompon.layers.activations.gaussian) |
| `erf`           | [`pompon.layers.activations.erf`](#pompon.layers.activations.erf) |
| `moderate`      | [`pompon.layers.activations.moderate`](#pompon.layers.activations.moderate) |
| `silu`          | [`pompon.layers.activations.silu`](#pompon.layers.activations.silu) |


## How to add custom activation function?

- Modify pompon (**recommended**)
    1. Implement JAX function in `pompon.layers.activations`.
    2. Add custom name in `pompon.layers.basis.Phi._get_activation`.
    3. Specify the name as NNMPO argument.
    4. Give us your pull requests! (Optional)
- Override `activation` attribute
    1. Define `func: Callable[[jax.Array], jax.Array]` object by JAX.
    2. Set attribute `NNMPO.basis.phi{i}.activation = func` for i=0,1,...,f-1.

::: { .callout-warning }
The 0-th basis is always 1
because of the implementation in
`pompon._jittables._forward_q2phi`
:::

"""  # noqa: E501

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from pompon import DTYPE


@jax.jit
def tanh(x: Array) -> Array:
    r"""Hyperbolic tangent activation function

    $$
       \phi(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$
    """
    return jnp.tanh(x)


@jax.jit
def exp(x: Array) -> Array:
    r"""Exponential activation function

    $$
       \phi(x) = e^{|-x|}
    $$

    """
    return jnp.exp(-jnp.abs(x))


@jax.jit
def relu(x: Array) -> Array:
    r"""Rectified linear unit activation function

    $$
       \phi(x) = \max(0, x)
    $$

    ::: { .callout-note }
    This function is not suitable for force field regression.
    :::

    """
    return jnp.maximum(0, x)


@jax.jit
def silu(x: Array) -> Array:
    r"""Sigmoid linear unit activation function

    $$
       \phi(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
    $$

    """
    return x * jax.scipy.special.expit(x)


@jax.jit
def leakyrelu(x: Array, alpha: float = 0.01) -> Array:
    r"""Leaky rectified linear unit activation function

    $$
       \phi(x) = \max(\alpha x, x)
    $$

    """
    return jnp.maximum(alpha * x, x)


@jax.jit
def softplus(x: Array) -> Array:
    r"""Softplus activation function

    $$
       \phi(x) = \log(1 + e^x)
    $$

    """
    return jnp.log(1 + jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


@jax.jit
def gaussian(x: Array) -> Array:
    r"""Gaussian activation function

    $$
       \phi(x) = -e^{-x^2}
    $$
    """
    return -jnp.exp(-(x**2))


@partial(jax.jit, static_argnums=(1,))
def polynomial(x: Array, N: int) -> Array:
    r"""Polynomial basis function

    :::{ .callout-caution }
    - This activation is experimental and may not be stable.
    - One should fix `w` and `b` to 1.0 and 0.0, respectively.
    :::

    $$
       \phi_n(x) = x^n \quad (n=1,2,\cdots,N-1)
    $$

    - When $N$ is too large, this function is numerically unstable.
    - When $N=1$, it is equivalent to linear activation function
    - By using this function, the model can be regressed to a polynomial function.
    - This function should be used with \
      `functools.partial <https://docs.python.org/3/library/functools.html#functools.partial>`_ as follows:

    ```python
    func = functools.partial(polynomial, N=3)
    ```


    """  # noqa: E501
    retval_list = []
    for i in range(1, N):
        retval_list.append(x[..., i] ** i)
    retval = jnp.stack(jnp.array(retval_list, dtype=DTYPE), axis=-1)
    assert (
        retval.shape == x.shape
    ), f"retval.shape = {retval.shape}, x.shape = {x.shape}"
    return retval


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def polynomial_recursive(x: Array, N: int, k: int = 1) -> Array:
    r"""Calculate polynomial basis recursively

    :::{ .callout-caution }
    - This activation is experimental and may not be stable.
    - One should fix `w` and `b` to 1.0 and 0.0, respectively.
    :::

    $$
        \phi_n(x) = x^n = x^{n-1} \cdot x
    $$

    Args:
        x (Array): input with shape (D, f)
            where D is the number of data points.
        N (int): maximum degree of polynomial basis
        k (int): current degree of polynomial basis

    Returns:
        Array: ϕ = output with shape (D, f)

    ϕ = D @ [x^1, x^2, ..., x^N]
    """
    if k == N - 1:
        return x
    else:
        # x[:, k] = x[:, k-1] * x[:, 0]
        x = x.at[:, k].set(x[:, k - 1] * x[:, 0])
        return polynomial_recursive(x, N, k + 1)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def chebyshev_recursive(x: Array, N: int, k=1) -> Array:
    r"""Chebyshev polynomial basis function

    :::{ .callout-caution }
    - This activation is experimental and may not be stable.
    - One should fix `w` and `b` to 1.0 and 0.0, respectively.
    - The input `x` must be in [-1, 1].
    :::

    $$
       \phi_n(x) = T_n(x) \quad (n=1,2,\cdots,N-1)
    $$

    - By using this function, the model can be regressed to a Chebyshev polynomial function.
    - This function should be used with \
      `functools.partial <https://docs.python.org/3/library/functools.html#functools.partial>`_ as follows:
    """  # noqa: E501
    # assert jnp.abs(x).max() <= 1.0, "x must be in [-1, 1]"
    if x.ndim == 1:
        x = x[jnp.newaxis, :]
    if k == N - 1:
        return x  # .squeeze()
    elif k == 1:
        x = x.at[:, k].set(2 * x[:, 0])
    else:
        x = x.at[:, k].set(2 * x[:, 0] * x[:, k - 1] - x[:, k - 2])
    return chebyshev_recursive(x, N, k + 1)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def legendre_recursive(x: Array, N: int, k=1) -> Array:
    r"""Legendre polynomial basis function

    :::{ .callout-caution }
    - This activation is experimental and may not be stable.
    - One should fix `w` and `b` to 1.0 and 0.0, respectively.
    - The input `x` must be in [-1, 1].
    :::

    $$
       \phi_n(x) = P_n(x) \quad (n=1,2,\cdots,N-1)
    $$

    - By using this function, the model can be regressed to a Legendre polynomial function.
    - This function should be used with \
      `functools.partial <https://docs.python.org/3/library/functools.html#functools.partial>`_ as follows:
    """  # noqa: E501
    # assert jnp.abs(x).max() <= 1.0, "x must be in [-1, 1]"
    if x.ndim == 1:
        x = x[jnp.newaxis, :]
    if k == N - 1:
        return x  # .squeeze()
    elif k == 1:
        x = x.at[:, k].set((2 * k + 1) / (k + 1) * x[:, 0] - k / (k + 1))
    else:
        # (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        x = x.at[:, k].set(
            (2 * k + 1) / (k + 1) * x[:, 0] * x[:, k - 1]
            - k / (k + 1) * x[:, k - 2]
        )
    return legendre_recursive(x, N, k + 1)


@partial(jax.jit, static_argnums=(2,))
def Bspline(x: Array, grid: Array, k: int = 0):
    r"""B-spline basis function

    :::{ .callout-caution }
    - This activation is experimental and may not be stable.
    - One should fix `w` and `b` to 1.0 and 0.0, respectively.
    - The input `x` must be in [-1, 1].
    :::

    $$
       \phi_n(x) = B_{n,k}(x)
    $$

    Args:
        x (Array): input with shape (D, f)
            where D is the number of data points.
        grid (Array): grid points with shape (f, N)
            where N is the number of grid points.
        k (int): order of B-spline basis function

    """  # noqa: E501

    assert (
        x.shape[1] == grid.shape[0]
    ), f"x.shape = {x.shape}, grid.shape = {grid.shape}"
    x = x[:, :, jnp.newaxis]  # (D, f) -> (D, f, 1)
    # x = jnp.expand_dims(x, axis=2)
    grid = grid[jnp.newaxis, :, :]  # (f, N) -> (1, f, N)
    # grid = jnp.expand_dims(grid, axis=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = Bspline(x[:, :, 0], grid[0], k - 1)
        value = (x - grid[:, :, : -(k + 1)]) / (
            grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
        ) * B_km1[:, :, :-1] + (grid[:, :, k + 1 :] - x) / (
            grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]
        ) * B_km1[:, :, 1:]
    value = jnp.nan_to_num(value)
    return value


@jax.jit
def erf(x: Array) -> Array:
    r"""Error function activation function

    $$
       \phi(x) = \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
    $$

    - [W. Koch et al., J. Chem. Phys. 141(2), 021101 (2014)](https://doi.org/10.1063/1.4887508)
      adopted this function as multiplicative artificial neural networks.
    - Can be analytically integrated with Gaussian basis
    - Need large number of basis functions (empirically)
    - Almost the same as sigmoid

    """
    return jax.scipy.special.erf(x)


@jax.jit
def moderate(x: Array, ϵ: float = 0.05) -> Array:
    r"""Moderate activation function

    $$
       \phi(x) = 1 - e^{-x^2} + \epsilon x^2
    $$

    - [W. Koch et al. J. Chem. Phys. 151, 064121 (2019)](https://doi.org/10.1063/1.5113579)
      adopted this function as multiplicative neural network potentials.
    - Moderate increase outside the region
      spanned by the ab initio sample points

    """
    return 1 - jnp.exp(-(x**2)) + ϵ * (x**2)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def combine(
    x: Array,
    funcs: tuple[Callable[[Array], Array], ...],
    split_indices: tuple[int, ...],
) -> Array:
    r"""Combine activation functions

    Args:
        x (Array): input with shape (D, f)
            where D is the number of data points.
        funcs (tuple): list of activation functions

    Returns:
        Array: output with shape (D, f)

    """
    n_funcs = len(funcs)
    x_split = jnp.split(x, split_indices, axis=-1)
    retval_list = []
    for i in range(n_funcs):
        retval_list.append(funcs[i](x_split[i]))
    retval = jnp.concatenate(retval_list, axis=-1)
    assert (
        retval.shape == x.shape
    ), f"retval.shape = {retval.shape}, x.shape = {x.shape}"
    return retval


def extend_grid(grid, k_extend=0):
    r"""Extend grid points for B-spline basis function

    Args:
        grid (Array): grid points with shape (f, N)
            where N is the number of grid points.
        k_extend (int): order of B-spline basis function

    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for _ in range(k_extend):
        grid = jnp.concatenate([grid[:, [0]] - h, grid], axis=1)
        grid = jnp.concatenate([grid, grid[:, [-1]] + h], axis=1)
    return grid

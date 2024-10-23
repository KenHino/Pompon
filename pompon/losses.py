import logging

import jax
import jax.numpy as jnp
from jax import Array

logger = logging.getLogger(__name__)


def mse(y_train: Array, y_pred: Array) -> Array:
    r"""
    Mean squared error

    $$
        \mathrm{MSE} = \frac{1}{D} \sum_{i=1}^D (y_i - \hat{y}_i)^2
    $$

    Args:
        y_train (Array): target value with shape (D, 1)
        y_pred (Array): prediction with shape (D, 1)

    Returns:
        Array: mean squared error with shape (1,)

    """
    return (
        jnp.sum((y_train.flatten() - y_pred.flatten()) ** 2) / y_train.shape[0]
    )


@jax.jit
def _deviation_for_mse(y_train: Array, y_pred: Array) -> Array:
    r"""
    You may want to use this function when you want to compute the deviation for MSE analytically.

    Args:
        y_train (Array): target value with shape (D, 1)
        y_pred (Array): prediction with shape (D, 1)

    Returns:
        Array: deviation for MSE with shape (D, 1) $\frac{\partial}{\partial \hat{y}} \mathrm{MSE}$

    """  # noqa: E501
    mean_deviation = y_pred.flatten() - y_train.flatten()
    mean_deviation *= 2.0 / y_train.size
    return mean_deviation


def rmse(y_train: Array, y_pred: Array) -> Array:
    r"""
    Root mean squared error

    $$
        \mathrm{RMSE} = \sqrt{\frac{1}{D} \sum_{i=1}^D (y_i - \hat{y}_i)^2}
    $$

    Args:
        y_train (Array): target value with shape (D, 1)
        y_pred (Array): prediction with shape (D, 1)

    Returns:
        Array: root mean squared error with shape (1,)

    """
    return jnp.sqrt(mse(y_train, y_pred))


def mae(y_train: Array, y_pred: Array) -> Array:
    r"""
    Mean absolute error

    $$
        \mathrm{MAE} = \frac{1}{D} \sum_{i=1}^D |y_i - \hat{y}_i|
    $$

    Args:
        y_train (Array): target value with shape (D, 1)
        y_pred (Array): prediction with shape (D, 1)

    Returns:
        Array: mean absolute error with shape (1,)

    """
    return (
        jnp.sum(jnp.abs(y_train.flatten() - y_pred.flatten()))
        / y_train.shape[0]
    )


def L1_norm(phi: Array) -> Array:
    r"""
    L1 norm of the basis $\phi$.

    Args:
        phi (Array): basis with shape ($D$, $N$)

    Returns:
        Array: L1 norm of the basis with shape ($N$,)

    """
    return jnp.mean(jnp.abs(phi), axis=0)


def L1_entropy(L1_norm: Array) -> Array:
    r"""
    Entropy of the basis $\phi$.

    $$
        - \sum_{i=1}^{N} \phi_i \log \left( \phi_i \right)
    $$

    Args:
        L1_norm (Array): L1 norm of the basis with shape ($N$,)

    Returns:
        Array: entropy of the basis with shape ($N$,)

    """
    p = L1_norm / jnp.sum(L1_norm)
    return jnp.sum(-p * jnp.log(p))


@jax.jit
def total_loss(
    y_train: Array,
    y_pred: Array,
    basis: list[Array],
    lambda1: float = 1.0e-02,
    mu1: float = 1.0,
    mu2: float = 1.0,
) -> Array:
    r"""
    Total loss function.

    $$
        \mathrm{total\_loss} = \mathrm{MSE}
        + \lambda (\mu_1 \mathrm{L1\_entropy} + \mu_2 \mathrm{L1\_entropy})
    $$

    Args:
        y_train (Array): target value with shape (D, 1)
        y_pred (Array): prediction with shape (D, 1)
        basis (list[Array]): basis with shape (D, N)

    Returns:
        Array: total loss with shape (1,)

    """
    _L1_norms_phis = [L1_norm(phi) for phi in basis]
    _L1_norms_Phi = [jnp.sum(L1_norms) for L1_norms in _L1_norms_phis]
    _L1_entropies_basis = [L1_entropy(L1_norms) for L1_norms in _L1_norms_phis]
    return mse(y_train, y_pred) + lambda1 * (
        mu1 * sum(_L1_norms_Phi) - mu2 * sum(_L1_entropies_basis)
    )

import jax
import jax.numpy as jnp
from jax import Array

from pompon.layers.parameters import Parameter

from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, alpha: float = 0.9):
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def update_one(self, param: Parameter) -> None:
        if not hasattr(param, "momentum"):
            param.momentum = jnp.zeros_like(param.data)
        elif param.momentum is None or param.v.shape != param.data.shape:
            param.momentum = jnp.zeros_like(param.data)

        param.data, momentum = _update_one(
            param.data, param.grad, self.lr, self.alpha, param.momentum
        )
        # In the case of Riemannian optimization,
        # the momentum should be transported to the new point
        # therefore param.momentum should be set after new param.data is set
        param.momentum = momentum


@jax.jit
def _update_one(
    data: Array,
    grad: Array,
    lr: float,
    alpha: float,
    momentum: Array,
) -> tuple[Array, Array]:
    momentum = alpha * momentum - lr * grad
    return data + momentum, momentum

import jax
from jax import Array

from pompon.layers.parameters import Parameter

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Steepest Gradient Descent (SGD) optimizer class
    'S' also stands for Stochastic
    """

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        param.data = _update_one(param.data, param.grad, self.lr)


@jax.jit
def _update_one(
    data: Array,
    grad: Array,
    lr: float,
) -> Array:
    return data - lr * grad

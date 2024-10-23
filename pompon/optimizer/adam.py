import jax
import jax.numpy as jnp
import polars as pl
from jax import Array

from pompon.layers.parameters import Parameter

from .optimizer import Optimizer


class Adam(Optimizer):
    r"""
    Adam optimizer class

    See also [Optax documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam)

    $$
       \begin{align*}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
       \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
       \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
       \Delta \theta_t &= -\frac{\eta}{\sqrt{\hat{v}_t + \bar{\epsilon}} + \epsilon} \hat{m}_t \\
       \theta_{t+1} &= \theta_t + \Delta \theta_t
       \end{align*}
    $$
    """  # noqa

    def __init__(
        self,
        lr: float = 1.0e-02,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1.0e-08,
        eps_root: float = 0.0,
    ):
        super().__init__()
        self.lr = lr
        self.t = 0
        self.b1 = b1
        assert 0 < self.b1 < 1
        self.b2 = b2
        assert 0 < self.b2 < 1
        self.eps = eps
        self.eps_root = eps_root

    def update(self, params: list[Parameter]) -> None:
        self.t += 1
        super().update(params)

    def update_one(self, param: Parameter) -> None:
        if not hasattr(param, "v") or self.t == 1:
            param.v = jnp.zeros_like(param.data)
        elif param.v is None or param.v.shape != param.data.shape:
            param.v = jnp.zeros_like(param.data)

        if not hasattr(param, "m") or self.t == 1:
            param.m = jnp.zeros_like(param.data)
        elif param.m is None or param.m.shape != param.data.shape:
            param.m = jnp.zeros_like(param.data)

        param.data, param.v, m = _update_one(
            data=param.data,
            grad=param.grad,
            lr=self.lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            eps_root=self.eps_root,
            v=param.v,
            m=param.m,
        )
        # In the case of Riemannian optimization,
        # the vector m should be transported to the new point
        # therefore param.m should be set after new param.data is set
        param.m = m

    def optimize(
        self,
        **kwargs,
    ) -> pl.DataFrame:
        self.t = 0
        return super().optimize(**kwargs)


@jax.jit
def _update_one(
    data: Array,
    grad: Array,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    v: Array,
    m: Array,
) -> tuple[Array, Array, Array]:
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * jnp.square(grad)
    m_hat = m / (1 - b1)
    v_hat = v / (1 - b2)
    data -= lr * m_hat / (jnp.sqrt(v_hat + eps_root) + eps)
    return data, v, m

"""Linear layers"""

from __future__ import annotations

from logging import getLogger

import jax
from jax import Array

from pompon import DTYPE
from pompon._jittables import _forward_linear
from pompon.layers.layers import Layer
from pompon.layers.parameters import Parameter

logger = getLogger("pompon").getChild(__name__)


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, key: Array):
        super().__init__()
        data = jax.random.normal(key, (in_dim, out_dim), dtype=DTYPE)
        self.A = Parameter(data, "A")  # y = xA

    def __call__(self, vector: Array) -> Array:
        return self.forward(vector)

    def forward(self, vector: Array) -> Array:
        r"""

        y = Ax
        where A is the weight matrix x is basis

        """  # noqa: E501
        return _forward_linear(self.A.data, vector)

from __future__ import annotations

from jax import Array


class Parameter:
    """
    Parameter abstract object

    Args:
       data (Array): Data array
       name (str): Name of parameter

    """

    def __init__(self, data: Array, name: str):
        self.data = data
        self.name = name
        self.grad: list[Array] | Array | None = None
        self.v: Array  # For Adam optimizer g^2
        self.m: Array  # For Adam optimizer g
        self.momentum: Array  # For Momentum optimizer

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Parameter({self.name}, shape={self.shape})"

    def clear_grad(self) -> None:
        self.grad = None

    def __mul__(self, other: Parameter | Array) -> Array:
        if isinstance(other, Parameter):
            return self.data * other.data
        elif isinstance(other, Array):
            return self.data * other
        else:
            raise ValueError(f"Invalid type: {type(other)}")

    def __rmul__(self, other: Parameter | Array) -> Array:
        if isinstance(other, Parameter):
            return other.data * self.data
        elif isinstance(other, Array):
            return other * self.data
        else:
            raise ValueError(f"Invalid type: {type(other)}")

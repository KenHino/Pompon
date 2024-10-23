"""
Optimizer module
================

## For tensor-train optimization.
- [`Sweeper`](optimizer.sweeper.Sweeper.qmd): Sweeper for tensor-train optimization.

## For basis & coordinator optimization.
- [`Adam`](optimizer.adam.Adam.qmd): Adam optimizer.
- [`Momentum`](optimizer.momentum.Momentum.qmd): Momentum optimizer.
- [`SGD`](optimizer.sgd.SGD.qmd): Stochastic gradient descent optimizer.

"""  # noqa: E501

from .adam import Adam
from .lin_reg import LinearRegression
from .momentum import Momentum
from .optimizer import Optimizer
from .sgd import SGD
from .sweeper import Sweeper

__all__ = [
    "Adam",
    "Momentum",
    "Optimizer",
    "SGD",
    "Sweeper",
    "LinearRegression",
]

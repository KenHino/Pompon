import logging
import math

import jax.numpy as jnp
import numpy as np
from jax import Array

from pompon import DTYPE

logger = logging.getLogger("pompon").getChild("dataloader")


class DataLoader:
    """
    DataLoader class for mini-batch training.

    Examples:
        >>> x = np.random.rand(100, 2)
        >>> y = np.random.rand(100, 1)
        >>> loader = DataLoader(x, y, batch_size=10)
        >>> for x_batch, y_batch in loader:
        ...     print(x_batch.shape, y_batch.shape)


    """

    def __init__(
        self,
        arrays: tuple[Array | None, ...],
        batch_size: int = 10000,
        shuffle: bool = False,
    ) -> None:
        self.arrays = []
        for array in arrays:
            if array is None:
                continue
            if not hasattr(self, "data_size"):
                self.data_size = array.shape[0]
            assert self.data_size == array.shape[0]
            self.arrays.append(jnp.asarray(array, dtype=DTYPE))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = math.ceil(self.data_size / self.batch_size)
        self.i_batch = 0
        self.indices = np.arange(self.data_size)
        self.reset()

    def reset(self) -> None:
        self.i_batch = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.data_size)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Array, ...]:
        if self.i_batch >= self.batches:
            self.reset()
            raise StopIteration
        batch_indices = self.indices[
            self.i_batch * self.batch_size : (self.i_batch + 1)
            * self.batch_size
        ]
        self.i_batch += 1
        return tuple(array[batch_indices] for array in self.arrays)

    def __len__(self) -> int:
        return self.batches

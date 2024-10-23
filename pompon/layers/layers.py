"""
Base Layer class
"""

from __future__ import annotations

import logging
from typing import Generator

from pompon.layers.parameters import Parameter

logger = logging.getLogger("pompon").getChild("layers")


class Layer:
    def __init__(self) -> None:
        self._parameters: set[str] = set()

    def __setattr__(self, name: str, value: Parameter | Layer) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._parameters.add(name)
            if isinstance(value, Layer):
                logger.debug(
                    f"Setting {name} as a sublayer of {self.__class__.__name__}"
                )
            else:
                logger.debug(
                    f"Added {name} as a parameter to {self.__class__.__name__} "
                    + f"with shape {value.data.shape} "
                    + f"on device {value.data.devices()}"
                )
        super().__setattr__(name, value)

    def params(self) -> Generator[Parameter, None, None]:
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            elif isinstance(obj, Parameter):
                logger.debug(f"Yielding {obj}")
                yield obj
            else:
                raise ValueError(
                    f"Invalid object found: {obj} in {self.__class__.__name__}"
                )

    def clear_grads(self) -> None:
        for param in self.params():
            param.clear_grad()

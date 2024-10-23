import logging

from jax import config

# from . import losses, utils
from .__dtype__ import DTYPE
from .__index__ import BASIS_INDEX, BATCH_INDEX, BOND_INDEX, SUB_BOND_INDEX
from .__version__ import __version__
from .dataloader import DataLoader
from .layers import activations, basis, coordinator, tensor, tt
from .layers.tt import TensorTrain
from .model import NNMPO
from .sop import NearestNeighbor, OneBody

__all__ = [
    "__version__",
    "BOND_INDEX",
    "SUB_BOND_INDEX",
    "BASIS_INDEX",
    "BATCH_INDEX",
    "DTYPE",
    "activations",
    "basis",
    "coordinator",
    "tensor",
    "tt",
    "TensorTrain",
    "NNMPO",
    "DataLoader",
    "OneBody",
    "NearestNeighbor",
    "losses",
    "utils",
]


# See also JAX's GitHub https://github.com/google/jax#current-gotchas
config.update("jax_enable_x64", True)

# Set the logging level of the pompon to INFO
logging.getLogger("pompon").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s:%(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
stream_handler.setFormatter(formatter)
logging.getLogger("pompon").addHandler(stream_handler)

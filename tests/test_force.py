import logging

import jax
import pytest

from pompon import NNMPO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "d, f, N, D, M, activation",
    [
        (5, 4, 6, 1, 10, "tanh"),
        (4, 4, 6, 2, 10, "silu"),
        (4, 4, 6, 3, 5, "polynomial"),
        (4, 4, 6, 1, 5, "chebyshev"),
        (4, 4, 6, 3, 5, "legendre"),
        (4, 4, 6, 3, 5, "moderate+silu"),
        (
            20,
            20,
            10,
            3,
            5,
            "tanh",
        ),  # naive algorithm suffers from the curse of dimensionality
    ],
)
def test_force(d, f, N, D, M, activation):
    nnmpo = NNMPO(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (D, d))
    f = nnmpo.force(x)
    logger.debug(f)
    assert f.shape == (D, d)


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.layers").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.coordinator").setLevel(logging.DEBUG)
    pytest.main([__file__])

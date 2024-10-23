import logging

import jax
import pytest

from pompon import NNMPO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "d, f, N, D, M, activation",
    [
        (5, 4, 6, 1000, 10, "tanh"),
        (4, 4, 6, 1000, 10, "silu"),
        (4, 4, 6, 1000, 5, "polynomial"),
        (4, 4, 6, 1000, 5, "chebyshev"),
        (4, 4, 6, 1000, 5, "legendre"),
        (4, 4, 6, 1000, 5, "moderate+silu"),
    ],
)
def test_nnmpo_forward(d, f, N, D, M, activation):
    nnmpo = NNMPO(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (D, d))
    y = nnmpo.forward(x)
    assert y.shape == (D, 1)


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.layers").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.coordinator").setLevel(logging.DEBUG)
    pytest.main([__file__])

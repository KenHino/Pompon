import logging

import numpy as np
import pytest

from pompon import NNMPO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "d, f, N, n, M, activation",
    [
        (5, 4, 6, 100, 10, "relu"),
        (4, 4, 6, 100, 10, "tanh"),
        (4, 4, 6, 100, 5, "moderate"),
    ],
)
def test_convert_mpo(d, f, N, n, M, activation):
    nnmpo = NNMPO(
        input_size=d,
        hidden_size=f,
        basis_size=N,
        bond_dim=M,
        activation=activation,
    )
    # Debug by integral of grid basis
    grid = np.linspace(-1, 1, n)
    q0 = nnmpo.q0
    basis_ints = []
    for i in range(f):
        grid_value = nnmpo.basis[i].forward(grid, q0[:, i])
        assert grid_value.shape == (n, N)
        kronecker = np.eye(n)
        basis_int = np.einsum("ij,jk->ikj", kronecker, grid_value)
        basis_ints.append(basis_int)
    mpo = nnmpo.convert_to_mpo(basis_ints)
    for core_mpo, core_nnmpo in zip(mpo, nnmpo.tt, strict=True):
        M_left, N, M_right = core_nnmpo.shape
        assert core_mpo.shape == (M_left, n, n, M_right)


if __name__ == "__main__":
    # logging DEBUG in pompon/model.py
    logging.getLogger("pompon.model").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.basis").setLevel(logging.DEBUG)
    logging.getLogger("pompon.layers.tt").setLevel(logging.DEBUG)
    pytest.main([__file__])

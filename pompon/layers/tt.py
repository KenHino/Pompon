"""Tensor Train (TT) layer"""

from __future__ import annotations

from logging import getLogger
from typing import Generator

import jax
import jax.numpy as jnp
from jax import Array

from pompon import BASIS_INDEX, BATCH_INDEX, BOND_INDEX, DTYPE
from pompon._jittables import _forward_basis2y
from pompon.layers.layers import Layer
from pompon.layers.parameters import Parameter
from pompon.layers.tensor import (
    BasisBatch,
    Core,
    LeftBlockBatch,
    RightBlockBatch,
    Tensor,
    TwodotCore,
    dot,
)

logger = getLogger("pompon").getChild(__name__)


class TensorTrain(Layer):
    r"""Tensor Train (TT) class

    $$
       A(i_1, i_2, \cdots, i_f) = \sum_{\beta_1,\beta_2,\cdots,\beta_{f-1}} \
       W^{[1]}_{i_1\beta_1}
       W^{[2]}_{\beta_1 i_2 \beta_2} \cdots
       W^{[f]}_{\beta_{f-1}i_f}
    $$

    This class object is initialized by either following methods:

    1. [`TensorTrain.decompose(tensor)`](#pompon.layers.tt.TensorTrain.decompose): exact tensor train decomposition

        ```{.python}
        import jax
        from pompon import TensorTrain
        tensor = jax.random.normal(jax.random.PRNGKey(0), (3, 3, 3))
        tt = TensorTrain.decompose(tensor)
        ```

    2. [`TensorTrain.set_custom(cores)`](#pompon.layers.tt.TensorTrain.set_custom): set custom cores

        ```{.python}
        import jax
        from pompon import TensorTrain
        cores = [jax.random.normal(jax.random.PRNGKey(0), (1, 3, 2)),
                 jax.random.normal(jax.random.PRNGKey(1), (2, 3, 2)),
                 jax.random.normal(jax.random.PRNGKey(2), (2, 3, 1))]
        tt = TensorTrain.set_custom(cores)
        ```

    3. [`TensorTrain.set_random(shape, rank)`](#pompon.layers.tt.TensorTrain.set_random): set random tensor train

        ```{.python}
        from pompon import TensorTrain
        tt = TensorTrain.set_random(shape=(3, 3, 3), rank=2)
        ```

    """  # noqa: E501

    shape: tuple[int, ...]
    ndim: int
    valid_ranks: list[int]

    def __init__(self):
        super().__init__()
        self.cores: list[Core] = []
        self.center: int = 0
        self.left_blocks_batch: list[LeftBlockBatch] = []
        self.right_blocks_batch: list[RightBlockBatch] = []
        self.B: TwodotCore | None = None
        self.C: Core | None = None
        data = jnp.array(1.0, dtype=DTYPE)
        self.norm = Parameter(data, name="norm")

    def __call__(self, basis: list[Array] | list[Tensor]) -> Array:
        return self.forward(basis)

    def forward(self, basis: list[Array] | list[Tensor]) -> Array:
        r"""
        Evaluate the contraction of the tensor train $A(i_1, i_2, \cdots, i_f)$
        with the input tensor $\Phi(i_1, i_2, \cdots, i_f)$

        Args:
            basis (list[Array] | list[Tensor]): Input tensor
                $D$ @ $\phi^{[p]}_{i_p}$
                with shape $f\times(D, N)$ where $D$ is the batch size.

        Returns:
            Array: Output tensor $D$ @
                $\sum_{i_1,\cdots,i_f} A(i_1,\cdots,i_f) \phi^{[1]}_{i_1} \cdots \phi^{[f]}_{i_f}$
                with shape $(D,1)$

        """  # noqa: E501
        _basis = []
        for i_basis in basis:
            if isinstance(i_basis, Array):
                _basis.append(i_basis)
            elif isinstance(i_basis, Tensor):
                _basis.append(i_basis.data)
            else:
                raise ValueError("basis must be a list of Tensor or Array")
        W = [getattr(self, f"W{i}").data for i in range(self.ndim)]
        return _forward_basis2y(basis=_basis, W=W, norm=self.norm.data)

    def _initialize(self):
        self.center = 0
        self.to_canonical(gauge="CR")
        for i, core in enumerate(self.cores):
            setattr(self, f"W{i}", core)
        if self.ndim > 1:
            self.set_center_twodot()
        self.set_center_onedot()
        self.left_blocks_batch = []
        self.right_blocks_batch = []

    @property
    def ranks(self) -> list[int]:
        r"""
        List of ranks [$M_1, M_2, \cdots, M_{f-1}$]
        """
        return [core.shape[0] for core in self.cores][1:]

    @classmethod
    def decompose(cls, tensor: Array) -> TensorTrain:
        """
        Initialize with a given tensor by exact tensor train decomposition

        Args:
           tensor (Array): tensor with shape (N, N, ..., N)

        Returns:
           TensorTrain: TensorTrain object

        """
        tensor_train = cls()
        tensor_train.cores = exact_tensor_train_decomposition(tensor)
        tensor_train.shape = tuple(core.shape[1] for core in tensor_train.cores)
        tensor_train.ndim = len(tensor_train.shape)
        tensor_train.valid_ranks = [
            core.shape[0] for core in tensor_train.cores
        ] + [1]
        tensor_train._initialize()
        return tensor_train

    @classmethod
    def set_ones(
        cls, shape: tuple[int, ...], rank: int | None = None
    ) -> TensorTrain:
        r"""
        Initialize with all ones tensor train
        """
        tensor_train = cls()
        tensor_train.shape = shape
        tensor_train.ndim = len(shape)
        tensor_train.valid_ranks = tensor_train._get_valid_ranks(rank)
        tensor_train.cores = tensor_train._init_ones_cores()
        tensor_train._initialize()
        return tensor_train

    @classmethod
    def set_custom(cls, cores: list[Core | Array]) -> TensorTrain:
        r"""
        Initialize with a given list of cores

        Args:
           cores (list[Core | Array]):
                list of cores with shape (M, N, M) like
                [$W^{[1]}, W^{[2]}, \cdots, W^{[f]}$]

        Returns:
           TensorTrain: TensorTrain object

        """
        tensor_train = cls()
        tensor_train.shape = tuple(core.shape[1] for core in cores)
        tensor_train.ndim = len(tensor_train.shape)
        tensor_train.cores = []
        for i, core in enumerate(cores):
            if isinstance(core, Core):
                core.name = f"W{i}"
                tensor_train.cores.append(core)
            else:
                tensor_train.cores.append(
                    Core(
                        data=core,
                        leg_names=(
                            f"{BOND_INDEX}{i}",
                            f"{BASIS_INDEX}{i + 1}",
                            f"{BOND_INDEX}{i + 1}",
                        ),
                        name=f"W{i}",
                    )
                )
        tensor_train.valid_ranks = [
            core.shape[0] for core in tensor_train.cores
        ] + [1]
        tensor_train._initialize()
        return tensor_train

    @classmethod
    def set_random(
        cls,
        shape: tuple[int, ...],
        rank: int | None = None,
        key: Array | None = None,
    ) -> TensorTrain:
        """
        Initialize with a random tensor train

        Args:
           shape (tuple[int, ...]): shape of the tensor like $(N, N, ..., N)$
           rank (int, optional): maximum tt-rank of the tensor train. Defaults to None.
           key (Array, optional): random key. Defaults to None.

        Returns:
           TensorTrain: TensorTrain object

        """  # noqa: E501
        tensor_train = cls()
        tensor_train.shape = shape
        tensor_train.ndim = len(shape)
        tensor_train.valid_ranks = tensor_train._get_valid_ranks(rank)
        if key is None:
            key = jax.random.PRNGKey(0)
        tensor_train.cores = tensor_train._init_random_cores(key)
        tensor_train._initialize()
        return tensor_train

    def __iter__(self) -> Generator:
        yield from self.cores

    def _get_valid_ranks(self, max_rank) -> list[int]:
        ranks = [1]
        for i in range(self.ndim - 1):
            ranks.append(min(max_rank, self.shape[i] * ranks[-1]))
        ranks.append(1)
        for i in range(self.ndim - 1):
            ranks[-i - 2] = min(
                ranks[-i - 2], self.shape[-i - 1] * ranks[-i - 1]
            )
        logger.debug(f"Valid ranks: {ranks}")
        return ranks

    def _init_random_cores(self, key: Array) -> list[Core]:
        cores = []
        for i in range(self.ndim):
            new_key, key = jax.random.split(key)
            core_shape = (
                self.valid_ranks[i],
                self.shape[i],
                self.valid_ranks[i + 1],
            )
            core_data = jax.random.normal(new_key, core_shape, dtype=DTYPE)
            core = Core(
                data=core_data,
                leg_names=(
                    f"{BOND_INDEX}{i}",
                    f"{BASIS_INDEX}{i + 1}",
                    f"{BOND_INDEX}{i + 1}",
                ),
                name=f"W{i}",
            )
            cores.append(core)
        return cores

    def _init_ones_cores(self) -> list[Core]:
        cores = []
        for i in range(self.ndim):
            core_shape = (
                self.valid_ranks[i],
                self.shape[i],
                self.valid_ranks[i + 1],
            )
            core_data = jnp.ones(core_shape, dtype=DTYPE)
            core = Core(
                data=core_data,
                leg_names=(
                    f"{BOND_INDEX}{i}",
                    f"{BASIS_INDEX}{i + 1}",
                    f"{BOND_INDEX}{i + 1}",
                ),
                name=f"W{i}",
            )
            cores.append(core)
        return cores

    def __str__(self) -> str:
        output = "["
        for core in self.cores:
            output += f"{core.__str__()}, "
        output += "]"
        return output

    def __repr__(self) -> str:
        return f"TensorTrain(shape={self.shape}, ranks={self.ranks})"

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, key):
        return self.cores.__getitem__(key)

    def to_canonical(self, gauge: str = "CR", ord: str = "fro"):
        r"""Convert tensor-train into canonical form

        Args:
            gauge (str, optional): gauge.
                 "LC" for left-canonical form, "CR" for right-canonical form.
            ord (str, optional): order of the norm.
                Defaults to "fro" which is Frobenius norm.

        """
        match gauge:
            case "CR":
                for i in range(self.ndim - 1, 1, -1):
                    left_core, right_core = self.cores[i - 1 : i + 1]
                    twodot_core = left_core @ right_core
                    if not isinstance(twodot_core, TwodotCore):
                        raise ValueError("The core is not TwodotCore")
                    Wl, Wr = twodot_core.svd(
                        gauge=gauge, rank=self.valid_ranks[i]
                    )
                    Wl.name = f"W{i - 1}"
                    Wr.name = f"W{i}"
                    self.cores[i - 1] = Wl
                    self.cores[i] = Wr
                self.norm.data *= self.cores[0].scale_to(1.0, ord=ord)
            case "LC":
                for i in range(self.ndim - 1):
                    left_core, right_core = self.cores[i : i + 2]
                    twodot_core = left_core @ right_core
                    if not isinstance(twodot_core, TwodotCore):
                        raise ValueError("The core is not TwodotCore")
                    Wl, Wr = twodot_core.svd(
                        gauge=gauge, rank=self.valid_ranks[i + 1]
                    )
                    Wl.name = f"W{i}"
                    Wr.name = f"W{i + 1}"
                    self.cores[i] = Wl
                    self.cores[i + 1] = Wr
                self.norm.data *= self.cores[-1].scale_to(1.0, ord=ord)
            case _:
                raise NotImplementedError(f"{gauge=} is not implemented")

    def set_blocks_batch(
        self,
        basis: list[Array],
    ):
        r"""Set left and right blocks for batch

        Args:
            basis (list[Array]): List of Input tensor
                $D$ @ $\phi^{[p]}_{i_p}$
                with shape (D, N) where D is the batch size

        """
        self._set_left_blocks_batch(basis, self.center - 1)
        self._set_right_blocks_batch(basis, self.center + 1)
        assert (
            len(self.left_blocks_batch) + len(self.right_blocks_batch) + 1
            == self.ndim + 2
        )

    def _set_right_blocks_batch(self, basis: list[Array], terminal_site: int):
        r"""

        Right blocks for batch are calculated recursively as follows:

        $$
           \mathcal{R}^{[f]}_{\beta_{f-1}} =
           \sum_{i_f} W^{[f]}_{\beta_{f-1}i_f} \phi_{i_f}^{[f]}
        $$

        $$
           \mathcal{R}^{[p]}_{\beta_{p-1}} =
           \sum_{\beta_p}
           \sum_{i_{p}} W^{[p]}_{\beta_{p-1} i_{p} \beta_{p}}
           \phi_{i_{p}}^{[p]} \mathcal{R}^{[p+1]}_{\beta_{p}}
        $$

        The result is stored in ``self.right_blocks_batch``
        with length ``self.ndim - self.center``.

        Args:
            basis (list[Array]): List of Input tensor
                $D$ @ $\phi^{[p]}_{i_p}$
                with shape (D,N) where D is the batch size
            terminal_site (int): terminal site of the right blocks.

        """  # noqa: E501
        p_site = terminal_site
        self.right_blocks_batch = [
            RightBlockBatch(
                data=jnp.ones((basis[-1].shape[0], 1), dtype=DTYPE),
                leg_names=(f"{BATCH_INDEX}", f"{BOND_INDEX}{self.ndim}"),
                name=f"R{self.ndim}",
            )
        ]
        for i_site in range(self.ndim - 1, p_site - 1, -1):
            logger.debug(f"i_site: {i_site}")
            phi = BasisBatch(
                data=basis[i_site],
                leg_names=(f"{BATCH_INDEX}", f"{BASIS_INDEX}{i_site + 1}"),
                name=f"phi{i_site}",
            )
            core = self.cores[i_site]

            self.right_blocks_batch.append(
                dot(
                    phi, core, self.right_blocks_batch[-1]
                ).as_right_block_batch(name=f"R{i_site}")
            )

    def _set_left_blocks_batch(self, basis: list[Array], terminal_site: int):
        r"""

        Left blocks for batch are calculated recursively as follows:

        $$
           \mathcal{L}^{[1]}_{\beta_{1}} =
           \sum_{i_1} W^{[1]}_{i_1\beta_{1}} \phi_{i_1}^{[1]}
        $$

        $$
           \mathcal{L}^{[p]}_{\beta_{p}} =
           \sum_{\beta_{p-1}} \sum_{i_{p}} W^{[p]}_{\beta_{p-1} i_{p} \beta_{p}}
           \phi_{i_{p}}^{[p]} \mathcal{L}^{[p-1]}_{\beta_{p-1}}
        $$

        The result is stored in ``self.left_blocks_batch``
        with length ``self.center + 1``.

        Args:
           basis (list[Array]): List of tensor
                :math:`D` @ :math:`\phi^{[p]}_{i_p}`
                with shape :math:`(D, N)` where :math:`D` is the batch size
           terminal_site (int): terminal site of the left blocks.

        """
        p_site = terminal_site
        self.left_blocks_batch = [
            LeftBlockBatch(
                data=jnp.ones((basis[0].shape[0], 1), dtype=DTYPE),
                leg_names=(f"{BATCH_INDEX}", f"{BOND_INDEX}{0}"),
                name=f"L{0}",
            )
        ]
        for i_site in range(p_site + 1):
            logger.debug(f"i_site: {i_site}")
            phi = BasisBatch(
                data=basis[i_site],
                leg_names=(f"{BATCH_INDEX}", f"{BASIS_INDEX}{i_site + 1}"),
            )
            core = self.cores[i_site]

            self.left_blocks_batch.append(
                dot(phi, core, self.left_blocks_batch[-1]).as_left_block_batch(
                    name=f"L{i_site + 1}"
                )
            )

    def shift_center(
        self,
        to_right: bool,
        basis: list[Array],
        is_onedot_center: bool = False,
    ) -> int:
        r"""Shift the center site to the left or right.

        When ``to_right`` is ``True``, the ``self.center`` is shifted to ``self.center + 1``,
        left blocks are updated as follows:

        $$
           \mathcal{L}^{[p]}_{\beta_{p}} =
           \sum_{\beta_{p-1}} \sum_{i_{p}} W^{[p]}_{\beta_{p-1} i_{p} \beta_{p}}
           \phi_{i_{p}}^{[p]} \mathcal{L}^{[p-1]}_{\beta_{p-1}}
        $$

        the last term of the right blocks is popped.

        Args:
            to_right (bool): If ``True``, the center site is shifted to the right.
                Otherwise, the center site is shifted to the left.
            basis (list[Array]): f-length list of tensor $D$ @ $\phi^{[p]}_{i_p}$
                with shape (D, N) where D is the batch size
            is_onedot_center (bool): If ``True``, the center site is the one-dot tensor.

        """  # noqa: E501

        if (not self.right_blocks_batch) or (not self.left_blocks_batch):
            self.set_blocks_batch(basis)

        if to_right:
            p = self.center
            assert p < self.ndim - 1
            basis_batch = BasisBatch(
                data=basis[p],
                leg_names=(
                    f"{BATCH_INDEX}",
                    f"{BASIS_INDEX}{p + 1}",  # Index starts from 1
                ),
            )
            left_block_batch = self.left_blocks_batch[-1]
            self.right_blocks_batch.pop()
            core_left = self.cores[p]
            left_block_batch = dot(
                left_block_batch, basis_batch, core_left
            ).as_left_block_batch(name=f"L{p + 1}")
            self.left_blocks_batch.append(left_block_batch)
            self.center = p + 1
            self.set_center_onedot()
            if self.center < self.ndim - 1:
                self.set_center_twodot(to_right=True)
            else:
                self.set_center_twodot(to_right=False)
        else:
            p = self.center
            assert p > 0
            self.left_blocks_batch.pop()
            right_block_batch = self.right_blocks_batch[-1]
            basis_batch = BasisBatch(
                data=basis[p],
                leg_names=(f"{BATCH_INDEX}", f"{BASIS_INDEX}{p + 1}"),
            )
            core_right = self.cores[p]
            new_name = f"R{p + 1}"
            right_block_batch = dot(
                right_block_batch, basis_batch, core_right
            ).as_right_block_batch(name=new_name)
            self.right_blocks_batch.append(right_block_batch)
            self.center = p - 1
            self.set_center_onedot()
            if self.center > 0:
                self.set_center_twodot(to_right=False)
            else:
                self.set_center_twodot(to_right=True)
        return self.center

    def switch_dot(self, to_onedot: bool, to_right: bool, basis: list[Array]):
        r"""
        When bond-dimension reaches the maximum, center cites should be switched to one-dot tensor.

        Args:
            to_onedot (bool, optional): If ``True``, the center site is switched to the one-dot tensor.
                Otherwise, the center site is switched to the two-dot tensor.
            basis (list[Array]): f-length list of tensor $D$ @ $\phi^{[p]}_{i_p}$
                with shape (D, N) where D is the batch size
        """  # noqa: E501
        raise NotImplementedError
        p = self.center
        match (to_onedot, to_right):
            case (True, True):
                # Wp, Wp+1 <- Bp,p+1
                # Cp <- Wp
                # Lp <- Lp
                self.set_center_onedot()
            case (True, False):
                # Wp, Wp+1 <- Bp,p+1
                # Cp+1 <- Wp+1
                # Lp <- Wp Φp Lp-1
                # Rp+1 <- pop(Rp+1)
                self.set_center_onedot()
            case (False, True):
                # Wp <- Cp
                # Bp,p+1 <- Wp Wp+1
                # Lp <- Lp
                # Rp+2 <- pop(Rp+1)
                if p == self.ndim - 1:
                    raise ValueError("The center site is the last site")
                self.set_center_twodot(to_right=True)
                self.right_blocks_batch.pop()
            case (False, False):
                # Wp <- Cp
                # Bp,p+1 <- Wp Wp+1
                # Lp-1 <- pop(Lp)
                # Rp <- Rp
                if p == 0:
                    raise ValueError("The center site is the first site")
                self.set_center_twodot(to_right=False)
                self.left_blocks_batch.pop()

    def set_center_twodot(self, to_right: bool = True):
        r"""Set the center two-dot tensor"""
        assert 0 <= self.center < self.ndim
        if to_right:
            assert self.center + 1 < self.ndim
            center_twodot = (
                self.cores[self.center] @ self.cores[self.center + 1]
            )
        else:
            assert self.center - 1 >= 0
            center_twodot = (
                self.cores[self.center - 1] @ self.cores[self.center]
            )
        center_twodot.name = "B"
        center_twodot.grad = None
        setattr(self, center_twodot.name, center_twodot)

    def set_center_onedot(self):
        r"""Set the center one-dot tensor"""
        assert 0 <= self.center < self.ndim
        center_onedot_W = self.cores[self.center]
        center_onedot_W.grad = None
        if hasattr(self, "C"):
            self.C = center_onedot_W
        else:
            self.C = center_onedot_W
        assert id(self.C) == id(getattr(self, f"W{self.center}"))

    def decompose_and_assign_center_twodot(
        self,
        *,
        to_right: bool = True,
        truncation: float = 1.0,
        rank: int | None = None,
        gauge: str = "CR",
        ord: str = "fro",
    ):
        B = self.B
        if B is None:
            raise AttributeError("The center site is not the two-dot tensor")
        self.norm.data *= B.scale_to(1.0, ord=ord)
        W_left, W_right = B.svd(truncation=truncation, rank=rank, gauge=gauge)
        p = self.center
        assert 0 <= p < self.ndim
        if to_right:
            assert p + 1 < self.ndim
            getattr(self, f"W{p}").data = W_left.data
            getattr(self, f"W{p + 1}").data = W_right.data
        else:
            assert p - 1 >= 0
            getattr(self, f"W{p - 1}").data = W_left.data
            getattr(self, f"W{p}").data = W_right.data

    def decompose_and_assign_center_onedot(
        self, to_right: bool, ord: str = "fro"
    ):
        C = self.C
        if C is None:
            raise AttributeError("The center site is not the one-dot tensor")
        self.norm.data *= C.scale_to(1.0, ord=ord)
        p = self.center
        assert 0 <= p < self.ndim
        if to_right:
            W_left, R = C.qr()
            W_left.leg_names = (
                f"{BOND_INDEX}{p}",
                f"{BASIS_INDEX}{p + 1}",
                f"{BOND_INDEX}{p + 1}",
            )
            getattr(self, f"W{p}").data = W_left.data
            if p == self.ndim - 1:
                assert R.data.size == 1, f"{self.ndim=}, {R.data=}"
                self.norm.data *= R.data.reshape(1)[0]
            else:
                W_right = getattr(self, f"W{p + 1}")
                W_right.data = (R @ W_right).data
        else:
            L, W_right = C.lq()
            W_right.leg_names = (
                f"{BOND_INDEX}{p}",
                f"{BASIS_INDEX}{p + 1}",
                f"{BOND_INDEX}{p + 1}",
            )
            getattr(self, f"W{p}").data = W_right.data
            if p == 0:
                assert L.data.size == 1, f"{self.ndim=}, {L.data=}"
                self.norm.data *= L.data.reshape(1)[0]
            else:
                W_left = getattr(self, f"W{p - 1}")
                W_left.data = (W_left @ L).data

    @property
    def center_twodot_indices(self) -> tuple[int, int]:
        if self.center == self.ndim - 1:
            return (self.center - 1, self.center)
        else:
            return (self.center, self.center + 1)


def exact_tensor_train_decomposition(tensor: Array) -> list[Core]:
    """
    Tensor train decomposition

    Args:
        tensor (Array): tensor with shape (N, N, ..., N)

    Returns:
        list[Core]: list of tensors with shape (M, N, M)

    """
    logger.debug("exact_tensor_train_decomposition")
    tensor = jnp.array(tensor, dtype=DTYPE)
    tensor_train = []
    n_mode = int(tensor.ndim)
    original_shape = tensor.shape
    left_bond_dim = 1
    for i_mode in range(n_mode - 1):
        logger.debug(f"i_mode = {i_mode}")
        Q, R = jnp.linalg.qr(
            tensor.reshape(left_bond_dim * original_shape[i_mode], -1)
        )
        logger.debug(f"tensor.shape = {tensor.shape}")
        logger.debug(f"Q.shape = {Q.shape}, R.shape = {R.shape}")
        core_data = Q.reshape(left_bond_dim, original_shape[i_mode], -1)
        core = Core(
            data=core_data,
            leg_names=(
                f"{BOND_INDEX}{i_mode}",
                f"{BASIS_INDEX}{i_mode + 1}",
                f"{BOND_INDEX}{i_mode + 1}",
            ),
            name=f"W{i_mode}",
        )
        tensor_train.append(core)
        left_bond_dim = tensor_train[-1].shape[2]
        tensor = R
    core_data = tensor.reshape(left_bond_dim, original_shape[-1], 1)
    core = Core(
        data=core_data,
        leg_names=(
            f"{BOND_INDEX}{n_mode - 1}",
            f"{BASIS_INDEX}{n_mode}",
            f"{BOND_INDEX}{n_mode}",
        ),
        name=f"W{n_mode - 1}",
    )
    tensor_train.append(core)
    return tensor_train

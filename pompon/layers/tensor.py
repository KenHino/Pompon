from __future__ import annotations

from logging import getLogger

import jax.numpy as jnp
from jax import Array

from pompon import BASIS_INDEX, BATCH_INDEX, BOND_INDEX, DTYPE, SUB_BOND_INDEX
from pompon.layers.parameters import Parameter

logger = getLogger("pompon").getChild(__name__)


class Tensor(Parameter):
    """Tensor class support "leg_names" for tensor network

    Examples:
       >>> import jax.numpy as jnp
       >>> from pompon.layers.core import Tensor
       >>> tensor_abc = Tensor(data=jnp.ones((2, 3, 4)), leg_names=("a", "b", "c"))
       >>> tensor_cde = Tensor(data=jnp.ones((4, 5, 6)), leg_names=("c", "d", "e"))
       >>> tensor_abde = tensor_abc @ tensor_cde  # contraction of "c"
       >>> print(tensor_abde)
       Tensor(shape=(2, 3, 5, 6), leg_names=('a', 'b', 'd', 'e'))
       >>> Δt = 0.01
       >>> print(tensor_abde * Δt)  # multiplication by a scalar
       Tensor(shape=(2, 3, 5, 6), leg_names=('a', 'b', 'd', 'e'))
       >>> tensor_abde -= tensor_abde * Δt  # subtraction
       >>> print(tensor_abde)
       Tensor(shape=(2, 3, 5, 6), leg_names=('a', 'b', 'd', 'e'))
       >>> tensor_Dab = Tensor(data=jnp.ones((100, 2, 3)), leg_names=("D", "a", "b")) # "D" means batch dimension
       >>> tensor_Dbc = Tensor(data=jnp.ones((100, 3, 4)), leg_names=("D", "b", "c"))
       >>> tensor_Dac = tensor_Dab @ tensor_Dbc
       >>> print(tensor_Dac)  # The batch dimension "D" is kept.
       Tensor(shape=(100, 2, 4), leg_names=('D', 'a', 'c'))

    """  # noqa: E501

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "T"
    ):
        super().__init__(data, name)
        self.leg_names = leg_names
        if len(self.leg_names) != self.data.ndim:
            raise ValueError(
                "The number of leg names must be the same as "
                + "the number of dimensions."
                + f" Got leg_names = {(self.leg_names)} leg names "
                + f"and {self.data.ndim} dimensions data"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Note that JAX ndarray is immutable.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Tensor(shape={self.shape}, "
            + f"leg_names={self.leg_names}, "
            + f"dtype={self.dtype})"
        )

    def __mul__(self, other: float | Array | Parameter) -> Tensor:  # type: ignore[override]
        if isinstance(other, float):
            ret_class = type(self)
            return ret_class(data=self.data * other, leg_names=self.leg_names)
        elif isinstance(other, Array):
            ret_class = type(self)
            return ret_class(data=self.data * other, leg_names=self.leg_names)
        elif isinstance(other, Parameter):
            ret_class = type(self)
            return ret_class(
                data=self.data * other.data, leg_names=self.leg_names
            )
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")

    def __rmul__(self, other: float) -> Tensor:  # type: ignore[override]
        return self.__mul__(other)

    def __matmul__(self, other: Tensor) -> Tensor:
        if isinstance(other, Tensor):
            tensor = dot(self, other)
            if isinstance(self, Core) and isinstance(other, Core):
                return TwodotCore(data=tensor.data, leg_names=tensor.leg_names)
            elif (
                isinstance(self, CoreBasisBatch)
                and isinstance(other, RightBlockBatch)
            ) or (
                isinstance(self, RightBlockBatch)
                and isinstance(other, CoreBasisBatch)
            ):
                return RightBlockBatch(
                    data=tensor.data, leg_names=tensor.leg_names
                )
            elif (
                isinstance(self, LeftBlockBatch)
                and isinstance(other, CoreBasisBatch)
            ) or (
                isinstance(self, CoreBasisBatch)
                and isinstance(other, LeftBlockBatch)
            ):
                return LeftBlockBatch(
                    data=tensor.data, leg_names=tensor.leg_names
                )
            elif (isinstance(self, Core) and isinstance(other, BasisBatch)) or (
                isinstance(self, BasisBatch) and isinstance(other, Core)
            ):
                return CoreBasisBatch(
                    data=tensor.data, leg_names=tensor.leg_names
                )
            else:
                return tensor
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")

    def __add__(self, other: Tensor | Array) -> Tensor:
        ret_class = type(self)
        if isinstance(other, Tensor):
            if set(self.leg_names) != set(other.leg_names):
                raise ValueError(
                    f"Cannot add {self.leg_names} and {other.leg_names}"
                )
            # swap axes of other to match leg_names order of self
            swap_axes = tuple(
                [other.leg_names.index(leg_name) for leg_name in self.leg_names]
            )
            return ret_class(
                data=self.data + jnp.transpose(other.data, swap_axes),
                leg_names=self.leg_names,
            )
        elif isinstance(other, Array):
            return ret_class(data=self.data + other, leg_names=self.leg_names)

    def __radd__(self, other: Tensor | Array) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | Array) -> Tensor:
        return self.__add__(other.__mul__(-1.0))

    def __rsub__(self, other: Tensor | Array) -> Tensor:
        return self.__sub__(other).__mul__(-1.0)

    def __array__(self) -> Array:
        return self.data

    def normalize(self) -> Array:
        """Normalize tensor

        Tensor is normalized and return the norm of the tensor.

        Returns:
            Array: norm of the tensor before normalization

        """
        norm = jnp.linalg.norm(self.data)
        self.data /= norm
        return norm

    def scale_to(
        self, scale: float | Array | None = None, ord: str = "fro"
    ) -> Array:
        """Scale maximum abs element of the tensor to the given scale

        Args:
            scale (float | Array): scale factor. Defaults to jnp.array(1.0).
            ord (str, optional): norm type to scale either "fro" or "max".
                Defaults to "fro" (Frobenius norm).
                "fro" : Frobenius norm
                "max" : maximum absolute value

        Returns:
            Array: multiplication factor to scale the tensor
        """
        if scale is None:
            scale = jnp.array(1.0, dtype=DTYPE)
        elif isinstance(scale, float):
            scale = jnp.array(scale, dtype=DTYPE)
        elif isinstance(scale, Array):
            assert scale.ndim == 0

        match ord.lower():
            case "fro":
                _norm = jnp.linalg.norm(self.data)
            case "max":
                _norm = jnp.max(jnp.abs(self.data))
            case _:
                raise ValueError(f"Invalid norm type: {ord}")
        norm = _norm / scale
        self.data /= norm
        assert (
            norm.ndim == 0
        ), f"{self.data=}, {_norm=}, {scale=}, {norm.ndim=}, {norm=}"
        return norm

    def as_core(self, name="W") -> Core:
        """Convert to Core

        Returns:
           Core: Core tensor

        """
        return Core(data=self.data, leg_names=self.leg_names, name=name)

    def as_twodot_core(self, name: str = "B") -> TwodotCore:
        """Convert to TwodotCore

        Returns:
           TwodotCore: TwodotCore tensor

        """
        return TwodotCore(data=self.data, leg_names=self.leg_names, name=name)

    def as_left_block_batch(self, name="L") -> LeftBlockBatch:
        """
        Convert to LeftBlockBatch

        Returns:
           LeftBlockBatch: LeftBlockBatch tensor
        """
        return LeftBlockBatch(
            data=self.data, leg_names=self.leg_names, name=name
        )

    def as_right_block_batch(self, name: str = "R") -> RightBlockBatch:
        """
        Convert to RightBlockBatch

        Returns:
           RightBlockBatch: RightBlockBatch tensor

        """
        return RightBlockBatch(
            data=self.data, leg_names=self.leg_names, name=name
        )

    def as_basis_batch(self, name="Phi") -> BasisBatch:
        """
        Convert to BasisBatch

        Returns:
           BasisBatch: BasisBatch tensor

        """
        return BasisBatch(data=self.data, leg_names=self.leg_names, name=name)

    def as_core_basis_batch(self, name="WPhi") -> CoreBasisBatch:
        """
        Convert to CoreBasisBatch

        Returns:
           CoreBasisBatch: CoreBasisBatch tensor

        """
        return CoreBasisBatch(
            data=self.data, leg_names=self.leg_names, name=name
        )

    def as_tensor(self, name="T") -> Tensor:
        """
        Convert to Tensor

        Returns:
           Tensor: Tensor tensor

        """
        return Tensor(data=self.data, leg_names=self.leg_names, name=name)

    def as_ndarray(self) -> Array:
        """
        Convert to jax.Array (Array)

        Returns:
           Array: Array tensor

        """
        return self.data


class Core(Tensor):
    r"""
    TT-Core tensor

    $$
       W^{[p]}_{\beta_{p-1} i_p \beta_p}
    $$

    Examples:
       >>> import jax.numpy as jnp
       >>> from pompon.layers.tt import TensorTrain
       >>> tt = TensorTrain.decompose(original_tensor=jnp.ones(4, 4, 4, 4))
       >>> tt[0]
       Core(shape=(1, 4, 4), leg_names=('β0', 'i0', 'β1'))
       >>> tt[1]
       Core(shape=(4, 4, 16), leg_names=('β1', 'i1', 'β2'))
       >>> print(B := tt[0] @ tt[1])
       TwodotCore(shape=(1, 4, 4, 16), leg_names=('β0', 'i0', 'i1', 'β2'))
       >>> print(B.svd(rank=2))
       (Core(shape=(1, 4, 2), leg_names=('β0', 'i0', 'β1')), Core(shape=(2, 4, 16), leg_names=('β1', 'i1', 'β2')))

    """  # noqa: E501

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "W"
    ):
        super().__init__(data, leg_names, name)
        assert self.ndim == 3
        assert self.leg_names[0].startswith(BOND_INDEX) or self.leg_names[
            0
        ].startswith(SUB_BOND_INDEX)
        assert self.leg_names[1].startswith(BASIS_INDEX)
        assert self.leg_names[2].startswith(BOND_INDEX) or self.leg_names[
            2
        ].startswith(SUB_BOND_INDEX)

    def __str__(self) -> str:
        return (
            f"Core(shape={self.shape}, "
            + f"leg_names={self.leg_names}, "
            + f"dtype={self.dtype})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def qr(self) -> tuple[Core, Tensor]:
        r"""QR decomposition

        Returns:
            Tuple[Core, Tensor]: left core and right core

        Examples:
            >>> import jax.numpy as jnp
            >>> from pompon.layers.tt import TensorTrain
            >>> tt = TensorTrain.decompose(original_tensor=jnp.ones((4, 4, 4, 4)))
            >>> W = tt[0]
            >>> print(W)
            Core(shape=(1, 4, 4), leg_names=('β0', 'i1', 'β1'))
            >>> print(W.qr())
            (Core(shape=(1, 4, 4), leg_names=('β0', 'i1', 'γ1')),
             Tensor(shape=(4, 4), leg_names=('γ1', 'β1')))

        """  # noqa: E501
        Ml, N, Mr = self.shape
        data = self.data.reshape(Ml * N, Mr)
        q, r = jnp.linalg.qr(data)
        left_core_data = q.reshape(Ml, N, Mr)
        right_tensor_data = r
        new_leg_name = f"γ{self.leg_names[2][1:]}"
        left_core = Core(
            data=left_core_data,
            leg_names=(self.leg_names[0], self.leg_names[1], new_leg_name),
        )
        right_tensor = Tensor(
            data=right_tensor_data, leg_names=(new_leg_name, self.leg_names[2])
        )
        return (left_core, right_tensor)

    def lq(self) -> tuple[Tensor, Core]:
        r"""LQ decomposition


        A.T = qr(A.T) = QR

        A = (QR).T = R.T Q.T =: L Q'


        Returns:
            Tuple[Tensor, Core]: left core and right core

        Examples:
            >>> import jax.numpy as jnp
            >>> from pompon.layers.tt import TensorTrain
            >>> tt = TensorTrain.decompose(original_tensor=jnp.ones((4, 4, 4, 4)))
            >>> W = tt[1]
            >>> print(W)
            Core(shape=(4, 4, 16), leg_names=('β1', 'i2', 'β2'))
            >>> print(W.rq())
            (Tensor(shape=(4, 4), leg_names=('β1', 'γ1')),
             Core(shape=(4, 4, 16), leg_names=('γ1', 'i2', 'β2')))

        """  # noqa: E501
        Ml, N, Mr = self.shape
        # swap axes (β0, i0, β1) -> (β1, i0, β0)
        data = jnp.transpose(self.data, (2, 1, 0)).reshape(Mr * N, Ml)
        q, r = jnp.linalg.qr(data)
        left_tensor_data = jnp.transpose(r)
        right_core_data = jnp.transpose(q.reshape(Mr, N, Ml), (2, 1, 0))
        new_leg_name = f"γ{self.leg_names[0][1:]}"
        left_tensor = Tensor(
            data=left_tensor_data, leg_names=(self.leg_names[0], new_leg_name)
        )
        right_core = Core(
            data=right_core_data,
            leg_names=(new_leg_name, self.leg_names[1], self.leg_names[2]),
        )

        return (left_tensor, right_core)


class TwodotCore(Tensor):
    r"""Two-dot tensor

    $$
       B\substack{i_pi_{p+1} \\ \beta_{p-1} \beta_{p+1}}
       = \sum_{\beta_p=1}^{M}
       W^{[p]}_{\beta_{p-1} i_p \beta_p}
       W^{[p+1]}_{\beta_p i_{p+1} \beta_{p+1}}
    $$

    """

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "B"
    ):
        super().__init__(data, leg_names, name)
        if self.ndim != 4:
            raise ValueError(
                "TwodotCore must be 4-dimensional tensor."
                + f" Got {self.ndim} dimensions "
                + f"and leg_names = {self.leg_names}"
            )
        assert self.leg_names[0].startswith(BOND_INDEX) or self.leg_names[
            0
        ].startswith(SUB_BOND_INDEX)
        assert self.leg_names[1].startswith(BASIS_INDEX)
        assert self.leg_names[2].startswith(BASIS_INDEX)
        assert self.leg_names[3].startswith(BOND_INDEX) or self.leg_names[
            3
        ].startswith(SUB_BOND_INDEX)

    def __str__(self):
        return f"TwodotCore(shape={self.shape}, leg_names={self.leg_names})"

    def svd(
        self,
        rank: int | None = None,
        new_leg_name: str | None = None,
        truncation: float = 1.0,
        gauge: str = "CR",
    ) -> tuple[Core, Core]:
        r"""Singular value decomposition between (0,1) and (2,3) legs

        Args:
            rank (int, optional): bond dimension (rank). Defaults to None.
            new_leg_name (str, optional): new leg name. Defaults to None.
            truncation (float, optional): singular value truncation. Defaults to 1.0.
            gauge (str, optional): gauge. Defaults to "CR".

        Returns:
            Tuple[Core, Core]: left core and right core

        Examples:
            >>> import jax.numpy as jnp
            >>> from pompon.tt import TensorTrain
            >>> tt = TensorTrain.decompose(original_tensor=jnp.ones((4, 4, 4, 4)))
            >>> B = tt[0] @ tt[1]
            >>> print(B)
            TwodotCore(shape=(1, 4, 4, 16), leg_names=('β0', 'i0', 'i1', 'β2'))
            >>> print(B.svd(rank=2))
            (Core(shape=(1, 4, 2), leg_names=('β0', 'i0', 'β1')), Core(shape=(2, 4, 16), leg_names=('β1', 'i1', 'β2')))

        """  # noqa: E501
        Ml, Nl, Nr, Mr = self.shape
        data = self.data.reshape(Ml * Nl, Nr * Mr)
        norm = jnp.linalg.norm(data)
        data /= norm
        u, s, v = jnp.linalg.svd(data, full_matrices=False)
        if rank is None:
            rank = 100_000_000  # Dummy large number
        rank = min(rank, self.truncate_rank(s, truncation))
        s = s[:rank]
        u = u[:, :rank]
        v = v[:rank, :]
        if gauge == "CR":
            left_matrix = u @ jnp.diag(s) * norm
            right_matrix = v
        elif gauge == "LC":
            left_matrix = u
            right_matrix = jnp.diag(s) @ v * norm
        else:
            sqrt_s = jnp.diag(jnp.sqrt(s) * norm)
            left_matrix = u @ sqrt_s
            right_matrix = sqrt_s @ v

        left_core_data = left_matrix.reshape(Ml, Nl, rank)
        right_core_data = right_matrix.reshape(rank, Nr, Mr)
        if new_leg_name is None:
            new_leg_name = f"{BOND_INDEX}{self.leg_names[1][1:]}"
        left_core = Core(
            data=left_core_data,
            leg_names=(self.leg_names[0], self.leg_names[1], new_leg_name),
            name="W",
        )
        right_core = Core(
            data=right_core_data,
            leg_names=(new_leg_name, self.leg_names[2], self.leg_names[3]),
            name="W",
        )
        return (left_core, right_core)

    @staticmethod
    def truncate_rank(s: Array, truncation: float) -> int:
        r"""Get new bond dimension

        Args:
            s (Array): singular values in descending order
                       with shape (M,)
            truncation (float): singular value truncation

        Returns:
            int: new bond dimension (rank)
        """
        if truncation == 1.0:
            return len(s)
        else:
            singular_cumsum = jnp.cumsum(s)
            singular_sum = singular_cumsum[-1]
            new_bond_dim = (
                int(
                    jnp.searchsorted(
                        singular_cumsum > singular_sum * truncation,
                        True,
                        side="left",
                    )
                )
                + 1
            )
            return min(new_bond_dim, len(s))


class LeftBlockBatch(Tensor):
    r"""

    Left blocks for batch are calculated
    recursively as follows:

    $$
       \mathcal{L}^{[1]}_{\beta_{1}} =
       \sum_{i_1} W^{[1]}_{i_1\beta_{1}} \phi_{i_1}^{[1]}
    $$

    $$
       \mathcal{L}^{[p]}_{\beta_{p}} =
       \sum_{\beta_{p-1}} \sum_{i_{p}} W^{[p]}_{\beta_{p-1} i_{p} \beta_{p}}
       \phi_{i_{p}}^{[p]} \mathcal{L}^{[p-1]}_{\beta_{p-1}}
    $$

    ::: {.callout-note}
       The batch dimension ``"D"`` must be the first index.
    :::

    """

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "L"
    ):
        super().__init__(data, leg_names, name)
        assert self.ndim == 2
        # "D" means batch dimension
        assert self.leg_names[0] == BATCH_INDEX
        assert self.leg_names[1].startswith(BOND_INDEX)

    def __str__(self):
        return (
            f"LeftBlockBatch(shape={self.shape}, "
            + f"leg_names={self.leg_names}, dtype={self.dtype})"
        )


class RightBlockBatch(Tensor):
    r"""

    Right blocks for batch are calculated
    recursively as follows:

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

    ::: {.callout-note}
       The batch dimension ``"D"`` must be the first index.
    :::

    """

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "R"
    ):
        super().__init__(data, leg_names, name)
        assert self.ndim == 2
        # "D" means batch dimension
        assert self.leg_names[0] == BATCH_INDEX
        assert self.leg_names[1].startswith(BOND_INDEX) or self.leg_names[
            1
        ].startswith(SUB_BOND_INDEX)

    def __str__(self):
        return (
            f"RightBlockBatch(shape={self.shape}, "
            + f"leg_names={self.leg_names}, dtype={self.dtype})"
        )


class BasisBatch(Tensor):
    r"""
    Basis batch $D$ @ $\phi^{[p]}_{i_p}$.

    ::: {.callout-note}
       The batch dimension ``"D"`` must be the first index.
    :::

    """

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "Phi"
    ):
        super().__init__(data, leg_names, name)
        assert self.ndim == 2
        # "D" means batch dimension
        assert self.leg_names[0] == BATCH_INDEX
        assert self.leg_names[1].startswith(BASIS_INDEX)

    def __str__(self):
        return (
            f"BasisBatch(shape={self.shape}, "
            + f"leg_names={self.leg_names}, dtype={self.dtype})"
        )


class CoreBasisBatch(Tensor):
    r"""

    Core basis batch $D$ @ $\sum_{i_p}W^{[p]}_{\beta_{p-1} i_p \beta_p} \phi^{[p]}_{i_p}$.

    ::: {.callout-note}
       The batch dimension ``"D"`` must be the first index.
    :::

    """  # noqa: E501

    def __init__(
        self, data: Array, leg_names: tuple[str, ...], name: str = "WPhi"
    ):
        super().__init__(data, leg_names, name)
        assert self.ndim == 3
        # "D" means batch dimension
        assert self.leg_names[0] == BATCH_INDEX
        assert self.leg_names[1].startswith(BOND_INDEX) or self.leg_names[
            1
        ].startswith(SUB_BOND_INDEX)
        assert self.leg_names[2].startswith(BOND_INDEX) or self.leg_names[
            2
        ].startswith(SUB_BOND_INDEX)

    def __str__(self):
        return (
            f"CoreBasisBatch(shape={self.shape}, "
            + f"leg_names={self.leg_names}, dtype={self.dtype})"
        )


def get_einsum_subscripts_and_new_legs(
    *tensors: Tensor,
) -> tuple[str, tuple[str, ...]]:
    r"""Get einsum subscripts from tensors

    Args:
        tensors (Tensor): list of tensors

    Returns:
        Tuple[str, Tuple[str, ...]]: einsum subscripts and new leg names

    ::: {.callout-note}
       The batch dimension ``"D"`` must be the first index and ``"D"`` is kept even if it is duplicated.
    :::

    """  # noqa: E501
    # assign new leg names from 'a', 'b', 'c', ...
    leg_names_orig2abc = dict()
    concat_leg_names_abc = []
    concat_leg_names_orig = []
    ord_abc = 97  # ord('a')
    for i, leg_name_orig in enumerate(tensors[0].leg_names):
        if leg_name_orig == BATCH_INDEX:
            assert i == 0, (
                f"{BATCH_INDEX} index, "
                + "which is batch dimension, must be the first index."
            )
            leg_name_abc = BATCH_INDEX
        else:
            leg_name_abc = chr(ord_abc)
            ord_abc += 1
        leg_names_orig2abc[leg_name_orig] = leg_name_abc
        concat_leg_names_abc.append(leg_name_abc)
        concat_leg_names_orig.append(leg_name_orig)
    subscripts = "".join(concat_leg_names_abc)
    subscripts += ","
    for j, tensor in enumerate(tensors[1:]):
        for i, leg_name_orig in enumerate(tensor.leg_names):
            if leg_name_orig == BATCH_INDEX:
                assert i == 0, (
                    f"{BATCH_INDEX} index, "
                    "which is batch dimension, must be the first index."
                )
                leg_name_abc = BATCH_INDEX
                if concat_leg_names_abc[0] != BATCH_INDEX:
                    assert concat_leg_names_orig[0] != BATCH_INDEX
                    concat_leg_names_abc = [BATCH_INDEX] + concat_leg_names_abc
                    concat_leg_names_orig = [
                        BATCH_INDEX
                    ] + concat_leg_names_orig
                else:
                    assert concat_leg_names_orig[0] == BATCH_INDEX
            elif leg_name_orig in leg_names_orig2abc:
                leg_name_abc = leg_names_orig2abc[leg_name_orig]
                # remove leg_name_orig from concat_leg_names_orig
                if leg_name_orig in concat_leg_names_orig:
                    concat_leg_names_orig.remove(leg_name_orig)
                # remove leg_name_abc from concat_leg_names_abc
                if leg_name_abc in concat_leg_names_abc:
                    concat_leg_names_abc.remove(leg_name_abc)
            else:
                leg_name_abc = chr(ord_abc)
                leg_names_orig2abc[leg_name_orig] = leg_name_abc
                ord_abc += 1
                concat_leg_names_abc.append(leg_name_abc)
                concat_leg_names_orig.append(leg_name_orig)
            subscripts += leg_name_abc
        if j != len(tensors) - 2:
            subscripts += ","
    subscripts += "->"
    subscripts += "".join(concat_leg_names_abc)
    concat_leg_names_orig_tuple = tuple(concat_leg_names_orig)
    assert len(concat_leg_names_orig_tuple) == len(concat_leg_names_abc)
    return (subscripts, concat_leg_names_orig_tuple)


def dot(*tensors: Tensor) -> Tensor:
    """Dot product of tensors

    Args:
        tensors (Tensor): list of tensors

    Returns:
        Tensor: result tensor

    Examples:
        >>> import jax.numpy as jnp
        >>> from pompon.layers.core import Tensor, dot
        >>> tensor_abc = Tensor(data=jnp.ones((2, 3, 4)), leg_names=("a", "b", "c"))
        >>> tensor_bcd = Tensor(data=jnp.ones((3, 4, 5)), leg_names=("b", "c", "d"))
        >>> tensor_de = Tensor(data=jnp.ones((5, 6)), leg_names=("d", "e"))
        >>> tensor_ae = dot(tensor_abc, tensor_bcd, tensor_de)
        >>> print(tensor_ae)
        Tensor(shape=(2, 6), leg_names=('a', 'e'))

    """  # noqa: E501

    subscripts, new_legs = get_einsum_subscripts_and_new_legs(*tensors)
    return Tensor(
        data=jnp.einsum(subscripts, *[tensor.data for tensor in tensors]),
        leg_names=new_legs,
    )

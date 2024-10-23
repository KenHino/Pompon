import jax
import jax.numpy as jnp
import pytest

from pompon.layers.tensor import Tensor, dot


@pytest.mark.parametrize(
    "multi_shape, multi_leg_names, expected_leg_names, expected_subscripts",
    [
        (((2, 3), (3, 4)), (("a", "b"), ("b", "c")), ("a", "c"), "ab,bc->ac"),
        (
            ((2, 3, 4), (4, 5, 6)),
            (("β0", "i1", "β1"), ("β1", "i2", "β2")),
            ("β0", "i1", "i2", "β2"),
            "abc,cde->abde",
        ),
        (
            ((2, 3), (3, 4), (4, 5)),
            (("a", "b"), ("b", "c"), ("c", "d")),
            ("a", "d"),
            "ab,bc,cd->ad",
        ),
        (
            ((10, 3), (10, 3, 4), (4, 5)),
            (("D", "a"), ("D", "a", "b"), ("b", "c")),
            ("D", "c"),
            "Da,Dab,bc->Dc",
        ),
    ],
)
def test_tensor_core(
    multi_shape, multi_leg_names, expected_leg_names, expected_subscripts
):
    multi_tensor = []
    multi_data = []
    for shape, leg_names in zip(multi_shape, multi_leg_names, strict=False):
        multi_data.append(
            data := jax.random.normal(jax.random.PRNGKey(0), shape)
        )
        multi_tensor.append(Tensor(data=data, leg_names=leg_names))

    contracted_tensor = dot(*multi_tensor)
    assert contracted_tensor.leg_names == expected_leg_names
    assert jnp.allclose(
        contracted_tensor.data, jnp.einsum(expected_subscripts, *multi_data)
    )

    if (
        len(multi_tensor) == 2
        and multi_tensor[0].ndim == 3
        and multi_tensor[1].ndim == 3
        and len(set(multi_tensor[0].leg_names) & set(multi_tensor[1].leg_names))
        == 1
        and multi_tensor[0].leg_names[-1] == multi_tensor[1].leg_names[0]
    ):
        core1 = multi_tensor[0].as_core()
        core2 = multi_tensor[1].as_core()
        core12 = core1 @ core2
        assert core12.leg_names == expected_leg_names
        core1, core2 = core12.svd()
        print(core1)
        assert core1.leg_names[-1] == core2.leg_names[0]
        zero = core1 @ core2 - core12
        assert jnp.allclose(zero.data, jnp.zeros_like(zero.data), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])

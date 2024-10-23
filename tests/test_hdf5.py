import jax
import jax.numpy as jnp

import pompon


def test_hdf5():
    nnmpo = pompon.NNMPO(
        input_size=4,
        hidden_size=3,
        basis_size=4,
        bond_dim=2,
        activation="tanh",
    )
    nnmpo.export_h5("test.h5")
    param_dict = {}
    for param in nnmpo.params():
        param_dict[param.name] = param.data

    nnmpo2 = pompon.NNMPO.import_h5("test.h5")
    for param in nnmpo2.params():
        assert jnp.allclose(param.data, param_dict[param.name])
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    y = nnmpo.forward(x)
    y2 = nnmpo2.forward(x)
    assert jnp.allclose(y, y2)

    onebody = pompon.OneBody(
        input_size=4,
        hidden_size=3,
        basis_size=4,
        activation="tanh",
    )
    onebody.export_h5("test.h5")
    param_dict = {}
    for param in onebody.params():
        param_dict[param.name] = param.data

    onebody2 = pompon.OneBody.import_h5("test.h5")
    for param in onebody2.params():
        assert jnp.allclose(param.data, param_dict[param.name])
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    y = onebody.forward(x)
    y2 = onebody2.forward(x)
    assert jnp.allclose(y, y2)

    nnmpo = onebody.to_nnmpo()
    nnmpo.export_h5("test.h5")
    param_dict = {}
    for param in nnmpo.params():
        param_dict[param.name] = param.data
    nnmpo2 = pompon.NNMPO.import_h5("test.h5")
    for param in nnmpo2.params():
        assert jnp.allclose(param.data, param_dict[param.name])
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
    y = nnmpo.forward(x)
    y2 = nnmpo2.forward(x)
    assert jnp.allclose(y, y2)

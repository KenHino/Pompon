import platform
import sys

import pompon

print(sys.version)
print(f"pompon version = {pompon.__version__}")
print(platform.platform())
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm.auto import tqdm

from pompon.optimizer import Adam, Sweeper

x_train = np.load("data/x_train.npy")
x_valid = np.load("data/x_validation.npy")
x_test = np.load("data/x_test.npy")

y_train = np.load("data/y_train.npy")
y_valid = np.load("data/y_validation.npy")
y_test = np.load("data/y_test.npy")

f_train = np.load("data/f_train.npy")
f_valid = np.load("data/f_validation.npy")
f_test = np.load("data/f_test.npy")

y_shift = -3112.1604302407663  # eV (eq. position)
y_min = -3113.4044750979383  # eV (mean)


x_scale = x_train.std()
y_scale = y_train.std()

x_train /= x_scale
x_valid /= x_scale
x_test /= x_scale
y_train /= y_scale
y_valid /= y_scale
y_test /= y_scale
f_train /= y_scale / x_scale
f_valid /= y_scale / x_scale
f_test /= y_scale / x_scale


def plot_trace(trace):
    return None
    plt.plot(
        trace["epoch"], np.sqrt(trace["mse_train"]) * y_scale, label="train"
    )
    plt.plot(trace["epoch"], np.sqrt(trace["mse_test"]) * y_scale, label="test")
    plt.xticks(np.arange(0, trace["epoch"][-1], 4000), rotation=45)
    plt.xlabel("epochs")
    plt.ylabel("RMSE Energy [eV]")
    plt.yscale("log")
    plt.legend()
    plt.show()


restart = False

if not restart:
    n = 6
    f = 6
    N = 21
    rng = np.random.default_rng(0)
    indices = rng.choice(len(x_train), N - 1)
    print(indices)
    x0_array = x_train[indices]

    nnmpo = pompon.NNMPO(
        input_size=n,
        hidden_size=f,
        basis_size=N,
        bond_dim=2,
        activation="moderate+silu",
        b_scale=0.0,
        w_scale=1.0,
        b_dist="uniform",
        w_dist="uniform",
        x0=x0_array,
        key=jax.random.key(3),
        fix_bias=False,
        random_tt=True,
    )

    optimizer = Adam(lr=1.0e-03).setup(
        model=nnmpo,  # <- changed!!
        x_train=x_train,
        y_train=y_train,
        f_train=f_train,
        batch_size=125,
        x_test=x_valid,
        y_test=y_valid,
        outdir="data",
    )

    sweeper = Sweeper(optimizer)
    for _ in tqdm(range(100)):
        lr = 1.0e-03 if _ < 50 else 1.0e-04
        optimizer.lr = lr
        trace = optimizer.optimize(epochs=10, epoch_per_log=10)
        solver = optax.adam(lr)
        sweeper.sweep(
            nsweeps=1,
            opt_tol=1.0e-10,
            opt_maxiter=50,
            opt_batchsize=625,
            optax_solver=solver,
            onedot=True,
        )

    cutoffs = np.logspace(np.log10(1.0e-02), np.log10(1.0e-03), 15)
    maxdim = [8] * 5 + [16] * 5 + [24] * 5
    lr = [1.0e-04] * 5 + [1.0e-04] * 5 + [1.0e-04] * 5

    for i in tqdm(range(15)):
        optimizer.lr = lr[i]
        solver = optax.adam(lr[i])
        sweeper.sweep(
            nsweeps=10,
            opt_tol=1.0e-10,
            opt_maxiter=200,
            opt_batchsize=625,
            cutoff=cutoffs[i],
            optax_solver=solver,
            auto_onedot=True,
            maxdim=14,
        )
        trace = optimizer.optimize(epochs=500, epoch_per_log=100)
        plot_trace(trace)
        nnmpo.export_h5(
            f'data/nnmpo_step_{i}_rmse_{np.sqrt(trace["mse_test"][-1]) * y_scale:.3e}.h5'
        )
    solver = optax.adam(1.0e-05)
    optimizer.lr = 1.0e-05
    for i in tqdm(range(15, 30)):
        sweeper.sweep(
            nsweeps=10,
            opt_tol=1.0e-12,
            opt_maxiter=500,
            opt_batchsize=625,
            optax_solver=solver,
            # use_CG=True,
            onedot=True,
        )
        trace = optimizer.optimize(epochs=500, epoch_per_log=100)
        plot_trace(trace)
        nnmpo.export_h5(
            f'data/nnmpo_step_{i}_rmse_{np.sqrt(trace["mse_test"][-1]) * y_scale:.3e}.h5'
        )

    sweeper.sweep(
        nsweeps=500,
        opt_tol=1.0e-13,
        opt_maxiter=100000,
        opt_batchsize=625,
        use_CG=True,
        onedot=True,
    )
    optimizer.lr = 0.0
    trace = optimizer.optimize(epochs=200, epoch_per_log=100)
else:
    # Change name as needed !!
    nnmpo = pompon.NNMPO.import_h5("data/nnmpo_step_209_rmse_1.893e-03.h5")
    optimizer = Adam(lr=1.0e-03).setup(
        model=nnmpo,
        x_train=x_train,
        y_train=y_train,
        f_train=f_train,
        batch_size=125,
        x_test=x_test,
        y_test=y_test,
        outdir="data",
    )
    sweeper = Sweeper(optimizer)
    solver = optax.adam(1.0e-07)
    optimizer.lr = 1.0e-12
    for i in range(210, 240):
        sweeper.sweep(
            nsweeps=10,
            opt_tol=1.0e-10,
            opt_maxiter=500,
            opt_batchsize=625,
            optax_solver=solver,
            onedot=True,
        )
        trace = optimizer.optimize(epochs=500, epoch_per_log=100)
        plot_trace(trace)
        nnmpo.export_h5(
            f'data/nnmpo_step_{i}_rmse_{np.sqrt(trace["mse_test"][-1]) * y_scale:.3e}.h5'
        )

rmse = float(np.sqrt(nnmpo.mse(x_test, y_test)) * y_scale)
nnmpo.rescale(x_scale, y_scale)
nnmpo.export_h5(f"data/nnmpo_final_rmse_{rmse:.3e}.h5")


# Plot as needed
nnmpo = pompon.NNMPO.import_h5(
    f'data/nnmpo_final_rmse_{np.sqrt(trace["mse_test"][-1]) * y_scale:.3e}.h5'
)

y_pred = nnmpo.forward(x_test * x_scale)

fig, ax = plt.subplots()

ax.scatter((y_test + y_shift) * 1000, (y_pred + y_shift) * 1000)
plt.xlabel("Test [meV]")
plt.ylabel("Prediction [meV]")
ax.set_xscale("log")
ax.set_yscale("log")
plt.title("Energy accuracy")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(dpi=500)
ax.scatter(
    (y_test + y_shift) * 8.065544e3, (y_pred + y_shift) * 8.065544e3, s=1
)
plt.xlabel("Test [cm-1]")
plt.ylabel("Prediction [cm-1]")
plt.xlim(0, 20000)
plt.ylim(0, 20000)
plt.title("Energy accuracy")
plt.tight_layout()
plt.show()

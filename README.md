![Static Badge](https://img.shields.io/badge/Version-v0.0.9-brightgreen)
[![unittest-uv](https://github.com/KenHino/Pompon/actions/workflows/unittest-uv.yml/badge.svg)](https://github.com/KenHino/Pompon/actions/workflows/unittest-uv.yml)
# Pompon

![](docs/notebook/assets/pompon-logo.svg)

Welcome to Pompon (Potential Optimizer in Matrix Product Operator Networks)!


Documentation is [here](https://kenhino.github.io/Pompon/)!

## What is Pompon?
Pompon is a Python library for optimizing matrix product operator (MPO) to randomly sampled points on a potential energy surface (PES).
MPO is suitable for high-dimensional integral and facilitates first-principles calculation in both time-dependent and time-independent manners.

## Installation

- The easiest way to install `pompon` is to use `pip`.
    Prepare Python 3.10 or later and execute;
    ```bash
    $ python -m venv pompon-env
    $ source pompon-env/bin/activate
    $ pip install git+https://github.com/KenHino/Pompon
    ```

- We recommend install `pompon` from source using [`uv`](https://docs.astral.sh/uv/)

    ```bash
    $ git clone https://github.com/KenHino/Pompon.git
    $ cd Pompon
    $ uv version
    uv 0.4.18 (7b55e9790 2024-10-01)
    $ uv sync --all-extras
    ```
    will install all dependencies including development tools.
    If you need only the runtime dependencies, you can use `uv sync --no-dev`.

    Then, you can execute `pompon` by
    ```bash
    $ uv run python xxx.py
    ```
    or
    ```bash
    $ souce .venv/bin/activate
    $ python
    >>> import pompon
    ```

    For jupyter notebook tutorials, you can use
    ```bash
    $ uv run jupyter lab
    ```

### For GPU users

`Pompon` works both on CPU and GPU.
If you treat large-scale batch or model, we recommend using GPU.
See also [JAX's GPU support](https://jax.readthedocs.io/en/latest/installation.html).

1. Make sure the latest NVIDIA driver is installed.

    ```bash
    $ /usr/local/cuda/bin/nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Wed_Apr_17_19:19:55_PDT_2024
    Cuda compilation tools, release 12.5, V12.5.40
    Build cuda_12.5.r12.5/compiler.34177558_0
    ```

2. Install GPU-supported JAX in your virtual envirionment.

    ```bash
    $ uv pip install -U "jax[cuda12]"
    $ uv run python -c "import jax; print(jax.default_backend())"
    'gpu'
    ```

### Testing

```bash
$ cd tests/build
$ uv run pytest ..
```


### For developers

You should install pre-commit hooks including ruff formatting and linting, mypy type checking, pytest testing, and so on.
```bash
$ uv run pre-commit install
$ git add .
$ uv run pre-commit
```
Before push, you must fix problems!!

Please feel free to give us feedback or pull requests.

### Results of manuscript

- Reference geometry information
    - [docs/notebook/bagel_h2co_dft.s0.harmonic.json](docs/notebook/data/bagel_h2co_dft.s0.harmonic.json)

- Training data
    - [docs/notebook/x_train.npy](docs/notebook/x_train.npy)
    - [docs/notebook/y_train.npy](docs/notebook/y_train.npy)
    - [docs/notebook/f_train.npy](docs/notebook/f_train.npy)
    - [docs/notebook/x_validation.npy](docs/notebook/x_validation.npy)
    - [docs/notebook/y_validation.npy](docs/notebook/y_validation.npy)
    - [docs/notebook/f_validation.npy](docs/notebook/f_validation.npy)
    - [docs/notebook/x_test.npy](docs/notebook/x_test.npy)
    - [docs/notebook/y_test.npy](docs/notebook/y_test.npy)
    - [docs/notebook/f_test.npy](docs/notebook/f_test.npy)

- Optimization
    - [docs/notebook/_h2co_opt.py](docs/notebook/h2co_opt.py)

- MPO encoding
    - [docs/notebook/create-random-mpo.ipynb](docs/notebook/create-random-mpo.ipynb)
    - [docs/notebook/nnmpo_to_itensor_mpo.ipynb](docs/notebook/nnmpo_to_itensor_mpo.ipynb)

- DMRG calculation
    - [docs/notebook/itensor_vDMRG.ipynb](docs/notebook/itensor_vDMRG.ipynb)
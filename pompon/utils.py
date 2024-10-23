import os
import shutil
from itertools import chain

import h5py
import numpy as np
from jax import Array


def train_test_split(
    *arrays: Array | np.ndarray,
    test_size: float | None = None,
    train_size: float | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
) -> list[np.ndarray]:
    """Split the dataset into training and test sets

    Almost same interface as sklearn.model_selection.train_test_split

    Args:
        *arrays (np.array, Array): arrays to be split
        test_size (float, optional): the proportion of the dataset
                                     to include in the test split.
                                     Defaults to None.
        train_size (float, optional): the proportion of the dataset
                                      to include in the train split.
                                      Defaults to None.
        random_state (int, optional): random seed. Defaults to None.
        shuffle (bool, optional): whether to shuffle the data before splitting.
                                  Defaults to True.

    Returns:
        list[np.ndarray]: x_train, x_test, y_train, y_test, ...

    Examples:
        >>> x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(
        ...     x, y, f, test_size=0.2
        ... )

    """  # noqa: E501
    if test_size is None and train_size is None:
        raise ValueError("test_size or train_size should be specified")
    if test_size is not None and train_size is not None:
        raise ValueError(
            "test_size and train_size cannot be specified at the same time"
        )
    if test_size is not None:
        if not 0 < test_size < 1:
            raise ValueError("test_size should be in the range (0, 1)")
        train_size = 1 - test_size
    if train_size is not None:
        if not 0 < train_size < 1:
            raise ValueError("train_size should be in the range (0, 1)")
        test_size = 1 - train_size
    assert isinstance(train_size, float)
    assert isinstance(test_size, float)
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    n_train = int(n_samples * train_size)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    # return_list = []
    # for array in arrays:
    #    return_list += np.array(array)[train_indices]
    #    return_list += np.array(array)[test_indices]
    # return return_list
    return list(
        chain(
            *zip(
                (np.array(array)[train_indices] for array in arrays),
                (np.array(array)[test_indices] for array in arrays),
                strict=False,
            )
        )
    )


def export_mpo_to_itensor(
    mpo: list[np.ndarray | Array], path: str, name: str
) -> str:
    """Export MPO to ITensor format

    Args:
        mpo (list[np.array | Array]): MPO
        path (str): path to the output file. For example, "/path/to/mpo.h5"
        name (str): name of the MPO. For example, "V"

    Returns:
        str: path to the filled mpo file (e.g. "/path/to/mpo_filled.h5")

    Examples:
        See also [`docs/notebook/nnmpo_to_itensor_mpo.ipynb`](../notebook/nnmpo_to_itensor_mpo.ipynb)
        and [ITensors.jl](https://itensor.github.io/ITensors.jl/stable/examples/MPSandMPO.html#Write-and-Read-an-MPS-or-MPO-to-Disk-with-HDF5)

    """  # noqa: E501
    # copy fpath to avoid modifying the original path
    if not path.endswith(".h5"):
        path += ".h5"
    new_path = path.replace(".h5", "_filled.h5")
    if os.path.exists(path):
        shutil.copy(path, new_path)
    f = h5py.File(new_path, "r+")

    N = f[name]["length"][()]  # number of sites
    assert N == len(mpo)
    M = []
    for core in mpo:
        assert core.ndim == 4
        M.append(core.shape[0])
    M.append(mpo[-1].shape[-1])

    for i in range(1, N + 1):
        site = f[name][f"MPO[{i}]"]
        n_index = site["inds"]["length"][()]
        data = mpo[i - 1]
        orig2tmp = [0, 1, 2, 3]
        for j in range(n_index):
            index = site["inds"][f"index_{j+1}"]
            tags = index["tags"]["tags"][()]
            changed_index = orig2tmp.index(j)
            if tags == f"Link,l={i}".encode():
                index["dim"][()] = M[i]
                data = np.swapaxes(data, orig2tmp[3], j)
                orig2tmp[changed_index], orig2tmp[3] = (
                    orig2tmp[3],
                    orig2tmp[changed_index],
                )
            elif tags == f"Boson,Site,n={i}".encode():
                assert (
                    index["dim"][()]
                    == mpo[i - 1].shape[1]
                    == mpo[i - 1].shape[2]
                )
                if index["plev"][()] == 1:
                    data = np.swapaxes(data, orig2tmp[1], j)
                    orig2tmp[changed_index], orig2tmp[1] = (
                        orig2tmp[1],
                        orig2tmp[changed_index],
                    )
                else:
                    data = np.swapaxes(data, orig2tmp[2], j)
                    orig2tmp[changed_index], orig2tmp[2] = (
                        orig2tmp[2],
                        orig2tmp[changed_index],
                    )
            elif tags == f"Link,l={i-1}".encode():
                index["dim"][()] = M[i - 1]
                data = np.swapaxes(data, orig2tmp[0], j)
                orig2tmp[changed_index], orig2tmp[0] = (
                    orig2tmp[0],
                    orig2tmp[changed_index],
                )
            else:
                raise ValueError(f"Unknown tag: {tags}")
        data_reshaped = data.reshape(-1, order="f")
        if "data" not in site["storage"]:
            raise ValueError(
                "No data in storage. use randomMPO[sites] instead of MPO[sites]"
            )
        del site["storage"]["data"]
        site["storage"].create_dataset("data", data=data_reshaped, chunks=True)
    f.close()
    return new_path

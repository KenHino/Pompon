import itertools

import h5py
import numpy as np

from pompon.utils import export_mpo_to_itensor


def test_export_itensor():
    mpo = []
    M = [1, 2, 3, 3, 2, 1]
    for i in range(5):
        core = np.zeros((M[i - 1], 6, 6, M[i]))
        for left, right in itertools.product(range(M[i - 1]), range(M[i])):
            core[left, :, :, right] = (left + 1) * 10 + (right + 1)
        mpo.append(core)
    # get path of this file
    path = __file__
    print(path)
    # mpo-empty.h5 is in the same directory as this file
    path = path[: path.rfind("/")] + "/mpo-empty.h5"
    print(path)
    new_path = export_mpo_to_itensor(mpo, path, "V")
    print(new_path)
    # Show the exported_file
    f = h5py.File(new_path, "r")
    for key, value in f["V"].items():
        # check if value is a group
        if isinstance(value, h5py.Group):
            print(key)
            for key_child, value_child in value.items():
                if isinstance(value_child, h5py.Group):
                    print(key_child)
                    for key_grandchild, value_grandchild in value_child.items():
                        if not isinstance(value_grandchild, h5py.Group):
                            print(key_grandchild, value_grandchild[()])
                        else:
                            print(key_grandchild)
                else:
                    print(key_child, value_child[()])
        else:
            print(key, value[()])


if __name__ == "__main__":
    test_export_itensor()

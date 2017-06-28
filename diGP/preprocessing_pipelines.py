# -*- coding: utf-8 -*-


import numpy as np
from dipy.core.gradients import gradient_table
from diGP.preprocessing import (readSPARC, readHCP, replaceNegativeData,
                                averageb0Volumes, normalize_data,
                                createBrainMaskFromb0Data)


def preprocess_SPARC(directory):
    print("Loading SPARC data located at {}".format(directory))
    gtab, data, voxelSize = readSPARC(directory)

    print("Checking for negative data.")
    data = replaceNegativeData(data, gtab)

    print("Extracting b0 image.")
    b0 = averageb0Volumes(data, gtab)

    print("Creating mask.")
    mask = np.ones_like(b0)

    print("Normalizing data.")
    data = normalize_data(data, b0, mask)

    return gtab, data, voxelSize


def preprocess_HCP(directory):
    print("Loading HCP data located at {}".format(directory))
    gtab, data, voxelSize = readHCP(directory)

    print("Checking for negative data.")
    data = replaceNegativeData(data, gtab)

    print("Extracting b0 image.")
    b0 = averageb0Volumes(data, gtab)

    print("Creating mask.")
    mask = createBrainMaskFromb0Data(b0)

    print("Normalizing data.")
    data = normalize_data(data, b0, mask)

    return gtab, data, voxelSize


def get_SPARC_train_and_test(train_dir, test_dir, q_test_path):

    gtab, data, voxelSize = preprocess_SPARC(train_dir)
    data = data[:, :, 0, :]  # Remove singleton dimension

    _, data_test, _ = preprocess_SPARC(test_dir)
    data_test = data_test[:, :, 0, 1:]  # Remove b0 and singleton dimension
    # Transpose to get same shape as training data
    data_test = data_test.transpose((1, 0, 2))

    # Would have been simple to use the gradient table from the gold standard
    # if it hadn't contained b0. Nevertheless, this is more fair since it is
    # the same information as given to the contestants in SPARC
    q_test = load_q_test(q_test_path)
    gtab_test = get_gtab_test(gtab, q_test)

    gtab_dict = {'train': gtab, 'test': gtab_test}
    data_dict = {'train': data, 'test': data_test}
    return gtab_dict, data_dict, voxelSize


def load_q_test(path):
    with open(path, 'r') as f:
        out = [line.split(' ') for line in f.readlines()]
    return np.array(out, dtype=float)


def get_gtab_test(gtab, q_test):
    small_delta = gtab.small_delta
    big_delta = gtab.big_delta
    tau = (big_delta - small_delta/3) * 1e-3
    q_magnitude = np.sqrt(np.sum(q_test ** 2, 1))
    bvals = (2*np.pi*q_magnitude) ** 2 * tau
    bvecs = q_test / q_magnitude[:, None]
    return gradient_table(bvals=bvals, bvecs=bvecs,
                          small_delta=small_delta, big_delta=big_delta)

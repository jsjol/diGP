# -*- coding: utf-8 -*-


import numpy as np
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

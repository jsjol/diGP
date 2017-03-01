# -*- coding: utf-8 -*-


import numpy as np
from diGP.averageb0Volumes import averageb0Volumes


def replaceNegativeData(data, gtab):
    """ Replace negative data points with mean if b = 0 and zero otherwise.

    Parameters:
    -------------
    data : ndarray
        4D diffusion data to process.
    gtab : Gradient table
        Gradient table that specifies the b-values used.

    Returns:
    --------
    processedData : ndarray
        Data of the same size as input data but with negative values replaced.
    """

    data[data < 0] = 0
    b0 = averageb0Volumes(data, gtab)
    zeroIdx = np.nonzero(data == 0)
    for b0Idx in np.nditer(np.nonzero(gtab.b0s_mask)):
        match = (zeroIdx[-1] == b0Idx)
        x = zeroIdx[0][match]
        y = zeroIdx[1][match]
        z = zeroIdx[2][match]
        data[x, y, z, b0Idx] = b0[x, y, z]

    return data

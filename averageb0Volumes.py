# -*- coding: utf-8 -*-


import numpy as np


def averageb0Volumes(data, gtab):
    maskedData = data[:, :, :, gtab.b0s_mask]
    averagedMaskedData = np.mean(maskedData, axis=3)
    return np.squeeze(averagedMaskedData)

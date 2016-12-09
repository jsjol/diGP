# -*- coding: utf-8 -*-


import numpy as np
from dipy.segment.mask import median_otsu


def createBrainMaskFromb0Data(b0Data):
    medianRadius = 4
    numIterations = 4
    # Call median_otsu and discard first return value
    mask = median_otsu(b0Data, medianRadius, numIterations)[1]
    mask = np.logical_and(mask == 1, b0Data > 0)
    return mask

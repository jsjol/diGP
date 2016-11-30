# -*- coding: utf-8 -*-


from dipy.segment.mask import median_otsu


def createBrainMaskFromb0Data(b0Data):
    medianRadius = 4
    numIterations = 4
    # Call median_otsu and discard first return value
    mask = median_otsu(b0Data, medianRadius, numIterations)[1]
    return mask

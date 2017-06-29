# -*- coding: utf-8 -*-

import numpy as np


def get_SPARC_metrics(gtab, target, pred, verbose=False):
    # 3500 instead of 3000 just to avoid round-off problems
    bvals_in_range = (gtab.bvals <= 3500)

    lowIdx = np.nonzero(bvals_in_range)
    highIdx = np.nonzero(np.invert(bvals_in_range))

    NMSE_low = NMSE(target[:, :, lowIdx], pred[:, :, lowIdx])
    NMSE_high = NMSE(target[:, :, highIdx], pred[:, :, highIdx])
    NMSE_all = NMSE(target, pred)

    if verbose:
        print("NMSE low: {}\nNMSE high: {}\nNMSE all: {}"
              .format(NMSE_low, NMSE_high, NMSE_all))

    return NMSE_low, NMSE_high, NMSE_all


def NMSE(target, pred):
    # (target - pred)/target = 1 - pred/target
    ratio = pred / target
    return MSE(1., ratio)


def MSE(target, pred):
    return np.mean((target - pred) ** 2)

# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats
from dipy.denoise.noise_estimate import piesno


class DataHandler:

    def __init__(self, gtab, data, voxelSize=(1, 1, 1), spatialIdx=None,
                 box_cox_lambda=None, qMagnitudeTransform=lambda x: x):
        self.gtab = gtab
        self.originalShape = data.shape
        if spatialIdx is None:
            self.spatialIdx = None
            self.data = data.reshape(-1, 1)
        else:
            self.spatialIdx = spatialIdx
            self.data = data[self.spatialIdx[0],
                             self.spatialIdx[1],
                             self.spatialIdx[2],
                             :]
        self.voxelSize = voxelSize
        self.box_cox_lambda = box_cox_lambda
        self.qMagnitudeTransform = qMagnitudeTransform
        self._X_coordinates = None
        self.X_q = self.getqFeatures()
        self._X = None

    @property
    def X(self):
        if self._X is None:
            X = get_separate_coordinate_arrays(self.originalShape[:-1],
                                               self.voxelSize)
            X = [a[:, None] for a in X]
            X.append(self.X_q)
            self._X = X

        return self._X

    @property
    def y(self):
        if self.box_cox_lambda is None:
            return self.data
        else:
            out = scipy.stats.boxcox(self.data.flatten(),
                                     lmbda=self.box_cox_lambda)
            return out.reshape(self.data.shape)

    @property
    def X_coordinates(self):
        if self._X_coordinates is None:
            self._X_coordinates = self.getCoordinates()

        return self._X_coordinates

    def optimize_box_cox_lambda(self):
        _, lmbda = scipy.stats.boxcox(self.data.flatten(), lmbda=None)
        self.box_cox_lambda = lmbda

    def getqFeatures(self):
        qvecs = self.gtab.bvecs
        qMagnitudesFeature = self.qMagnitudeTransform(self.gtab.qvals)
        return np.column_stack((qMagnitudesFeature, qvecs))

    def getCoordinates(self):
        coordinates_cube = generateCoordinates(self.originalShape[:-1],
                                               self.voxelSize)
        if self.spatialIdx is None:
            return coordinates_cube.reshape(-1, 3)
        else:
            linearIdx = np.ravel_multi_index(self.spatialIdx,
                                             self.originalShape[:-1])
            return coordinates_cube[linearIdx, :]


def inverseBoxCox(data, lmbda):
    if lmbda == 0:
        out = np.exp(data)
    else:
        out = (lmbda*data + 1) ** (1/lmbda)
    return out


def estimateBoxCoxLambdaFromBackground(data):
    idx = getBackgroundIdxUsingPIESNO(data)
    x = data[idx[0], idx[1], idx[2], :].flatten()

    # Box-Cox requires positive values
    x = x[x > 0]

    _, lmbda = scipy.stats.boxcox(x)
    return lmbda


def getBackgroundIdxUsingPIESNO(data):
    _, mask = piesno(data, N=1, return_mask=True)
    return np.nonzero(mask)


def get_separate_coordinate_arrays(voxelsInEachDim, voxelSize):
    return [voxelSize[i]*np.arange(voxelsInEachDim[i])
            for i in range(len(voxelsInEachDim))]


def generateCoordinates(voxelsInEachDim, voxelSize=None, image_origin=None):
    """ Generate coordinates for a grid.

    Parameters:
    ----------
    voxelsInEachDim : array-like
        The shape of an ND grid.

    voxelSize : array-like
        The voxel sizes in mm

    image_origin : array-like
        The coordinates to start counting from

    Returns:
    --------
    coordinates : ndarray
        Array of coordinates in ND, with dimensions prod(voxelsInEachDim) x ND.

    """
    if voxelSize is None:
        voxelSize = np.ones_like(voxelsInEachDim)

    separateCoordinateArrays = get_separate_coordinate_arrays(voxelsInEachDim,
                                                              voxelSize)
    meshedCoordinates = np.meshgrid(*separateCoordinateArrays, indexing='ij')

    coordinates = np.column_stack(list(map(np.ravel, meshedCoordinates)))

    if image_origin is not None:
        coordinates += image_origin

    return coordinates


def combineCoordinatesAndqVecs(coordinates, qVecs):
    """ Computes every combination of coordinates and qVecs.

    Parameters
    ----------
    coordinates : ndarray
        n x D array of coordinates
    qVecs : ndarray
        m x 3 array of q-vectors

    Returns
    -------
    out : ndarray
        nm x (D+3) array containing every combination of coordinates and qVecs

    """
    return np.column_stack((np.repeat(coordinates, qVecs.shape[0], axis=0),
                            np.tile(qVecs, (coordinates.shape[0], 1))))


def log_q_squared(q, c=1.):
    return np.log(c**2 + q**2)

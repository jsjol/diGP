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
            self.spatialIdx = np.nonzero(np.ones(data.shape[:-1]))
        else:
            self.spatialIdx = spatialIdx
        self.data = data[self.spatialIdx[0],
                         self.spatialIdx[1],
                         self.spatialIdx[2],
                         :]
        self.voxelSize = voxelSize
        self.box_cox_lambda = box_cox_lambda
        self.qMagnitudeTransform = qMagnitudeTransform
        self.X_coordinates = self.getCoordinates()
        self.X_q = self.getqFeatures()

    @property
    def y(self):
        out = scipy.stats.boxcox(self.data.flatten(),
                                 lmbda=self.box_cox_lambda)
        return out.reshape(self.data.shape)

    def getqFeatures(self):
        qvecs = self.gtab.bvecs
        qMagnitudesFeature = self.qMagnitudeTransform(self.gtab.qvals)
        return np.column_stack((qMagnitudesFeature, qvecs))

    def getCoordinates(self):
        coordinates_cube = generateCoordinates(self.originalShape[:-1],
                                               self.voxelSize)
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


def generateCoordinates(voxelsInEachDim, voxelSize=np.array([1, 1, 1])):
    """ Generate coordinates for a cube.

    Parameters:
    ----------
    voxelsInEachDim : array-like
        The dimensions, (nx, ny, nz), of a 3D grid.

    voxelSize : array-like
        The voxel sizes in mm

    Returns:
    --------
    coordinates : ndarray
        Array of coordinates in 3D, flattened to shape (nx*ny*nz) x 3.

    """
    separateCoordinateArrays = [voxelSize[i]*np.arange(voxelsInEachDim[i])
                                for i in range(3)]
    xMesh, yMesh, zMesh = np.meshgrid(*separateCoordinateArrays, indexing='ij')
    coordinates = np.column_stack((xMesh.flatten(),
                                   yMesh.flatten(),
                                   zMesh.flatten()))
    return coordinates


def combineCoordinatesAndqVecs(coordinates, qVecs):
    """ Computes every combination of coordinates and qVecs.

    Parameters
    ----------
    coordinates : ndarray
        n x 3 array of coordinates
    qVecs : ndarray
        m x 3 array of q-vectors

    Returns
    -------
    out : ndarray
        nm x 6 array containing every combination of coordinates and qVecs

    """
    return np.column_stack((np.repeat(coordinates, qVecs.shape[0], axis=0),
                            np.tile(qVecs, (coordinates.shape[0], 1))))


def log_q_squared(q, c=1.):
    return np.log(c**2 + q**2)

# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats


class DataHandler:

    def __init__(self, gtab, data=None, spatial_shape=None, voxelSize=None,
                 image_origin=None, spatialIdx=None, box_cox_lambda=None,
                 qMagnitudeTransform=lambda x: x):
        self.gtab = gtab

        if data is not None:
            self.originalShape = data.shape
        elif spatial_shape is not None:
            self.originalShape = (*spatial_shape, len(gtab.bvals))
        else:
            raise ValueError("To instantiate a DataHandler, either data or \
                             spatial shape needs to be provided.")

        self.spatialIdx = spatialIdx
        self.data = data
        self.voxelSize = voxelSize
        self.image_origin = image_origin
        self.box_cox_lambda = box_cox_lambda
        self.qMagnitudeTransform = qMagnitudeTransform
        self._X_coordinates = None
        self.X_q = self.getqFeatures()
        self._X = None

    @property
    def X(self):
        if self._X is None:
            X = get_separate_coordinate_arrays(self.originalShape[:-1],
                                               self.voxelSize,
                                               self.image_origin)
            X = [a[:, None] for a in X]
            X.append(self.X_q)
            self._X = X

        return self._X

    @property
    def y(self):
        if self.data is None:
            raise ValueError("DataHandler has no associated data.")

        if self.spatialIdx is None:
            out = self.data
        else:
            out = self.data[self.spatialIdx[0],
                            self.spatialIdx[1],
                            self.spatialIdx[2],
                            :]
            out_shape = out.shape

        if self.box_cox_lambda is not None:
            out = scipy.stats.boxcox(out.flatten(),
                                     lmbda=self.box_cox_lambda)

        if self.spatialIdx is None:
            return out.reshape(-1, 1)
        else:
            return out.reshape(out_shape)

    @property
    def X_coordinates(self):
        if self._X_coordinates is None:
            self._X_coordinates = self.getCoordinates()

        return self._X_coordinates

    def untransform(self, transformed):
        if self.box_cox_lambda is None:
            out = transformed
        else:
            out = inverseBoxCox(transformed, self.box_cox_lambda)

        return out.reshape(self.originalShape)

    def getqFeatures(self):
        qvecs = self.gtab.bvecs
        qMagnitudesFeature = self.qMagnitudeTransform(self.gtab.qvals)
        return np.column_stack((qMagnitudesFeature, qvecs))

    def getCoordinates(self):
        coordinates_cube = generateCoordinates(self.originalShape[:-1],
                                               self.voxelSize,
                                               self.image_origin)
        if self.spatialIdx is None:
            return coordinates_cube
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


def get_separate_coordinate_arrays(voxelsInEachDim, voxelSize, image_origin):
    if voxelSize is None:
        voxelSize = np.ones_like(voxelsInEachDim)

    if image_origin is None:
        image_origin = np.zeros_like(voxelsInEachDim)

    return [image_origin[i] + voxelSize[i]*np.arange(voxelsInEachDim[i])
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
    separateCoordinateArrays = get_separate_coordinate_arrays(voxelsInEachDim,
                                                              voxelSize,
                                                              image_origin)
    meshedCoordinates = np.meshgrid(*separateCoordinateArrays, indexing='ij')

    coordinates = np.column_stack(list(map(np.ravel, meshedCoordinates)))

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

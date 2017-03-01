# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats


class DataHandler:

    def __init__(self, gtab, data, voxelSize=(1, 1, 1),
                 box_cox_lambda=None, qMagnitudeTransform=lambda x: x):
        self.gtab = gtab
        self.data = data
        self.voxelSize = voxelSize
        self.box_cox_lambda = box_cox_lambda
        self.qMagnitudeTransform = qMagnitudeTransform
        self.X_coordinates = self.getCoordinates()
        self.X_q = self.getqFeatures()

    @property
    def box_cox_transformed_data(self):
        y = scipy.stats.boxcox(self.data.flatten(),
                               lmbda=self.box_cox_lambda)
        return y.reshape(self.data.shape)

# TODO: function that outputs Y-matrix with rows corresponding to X_coordinates
#       and columns to X_q. Important that ordering matches between X and Y.

# TODO: Necessary to have ind2Sub-like coordinate retrieval?

    def getqFeatures(self):
        qvecs = self.gtab.bvecs
        qMagnitudesFeature = self.qMagnitudeTransform(self.gtab.qvals)
        return np.column_stack((qMagnitudesFeature, qvecs))

    def getCoordinates(self):
        return generateCoordinates(self.data.shape[:-1], self.voxelSize)

# TODO: inverse Box-Cox transformation

# TODO: estimation of Box-Cox lambda from PIESNO background

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


def log_q_squared(q, c=1.):
    return np.log(c**2 + q**2)

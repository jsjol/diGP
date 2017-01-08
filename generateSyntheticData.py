# -*- coding: utf-8 -*-


import numpy as np
from dipy.sims.voxel import multi_tensor


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


def generatebVecs(numbVecs):
    """Generate random b-vectors from a sphere

    Parameters:
    ----------
    numbVecs : array-like
        Number of b-vectors to generate

    Returns:
    --------
    bVecs : ndarray
        numbVecs x 3 array of b-vectors randomly sampled
        from the sphere.
    """
    bVecs = _samplesOnTheSphere(numbVecs)
    return bVecs


def generatebValsAndbVecs(uniquebVals, numbVals):
    """Generate synthetic bval- and bvec-arrays in a format suitable for
    gradient table creation.

    Parameters:
    ----------
    uniquebVals : array-like
        An array of unique b-values (defining the radii of b-value shells)
    numbVals : array-like
        The number of samples to generate for each b-value

    Returns:
    --------
    bVals : array
        Array of shape (N,) with each unique b-value repeated as many times as
        specified by numbVals
    bVecs : ndarray
        (N, 3) array of unit vectors

    """
    bVals = [np.ones(n) for n in numbVals] * uniquebVals
    bVals = np.concatenate(np.asarray(bVals))
    totalNumber = np.sum(numbVals)
    bVecs = generatebVecs(totalNumber)
    return bVals, bVecs


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


def generateSyntheticInputs(voxelsInEachDim, gtab):
    """Generate inputs (i.e. features) for the machine learning algorithm.

    Parameters:
    -----------
    voxelsInEachDim : array-like
        The dimensions, (nx, ny, nz), of a 3D grid.

    gtab : GradientTable
        Specification of the diffusion MRI measurement.

    Returns:
    --------
    inputs : ndarray
        (N, 7) = (nx*ny*nz*nq, 3 + 1 + 3) array of inputs.
        Each row corresponds to a single diffusion MRI measurement in
        a single voxel. It is formatted as [coordinates, qMagnitudes, bvecs].
    """
    coordinates = generateCoordinates(voxelsInEachDim)
    qMagnitudes = gtab.qvals
    bvecs = gtab.bvecs
    qVecs = np.column_stack((qMagnitudes[:, np.newaxis], bvecs))
    return combineCoordinatesAndqVecs(coordinates, qVecs)


def generateSyntheticOutputsFromMultiTensorModel(voxelsInEachDim,
                                                 gtab, eigenvalues, **kwargs):
    """Generate a signal simulated from a multi-tensor model for each voxel.

    Parameters:
    -----------
    voxelsInEachDim : array-like
        The dimensions, (nx, ny, nz), of a 3D grid.

    gtab : GradientTable
        Specification of the diffusion MRI measurement.

    eigenvalues : (K, 3) array
        Each tensor's eigenvalues in each row

    Returns:
    --------
    output : (nx*ny*nz*nq,) array
        Simulated outputs.
    """
    N = np.prod(voxelsInEachDim)
    numberOfMeasurements = len(gtab.bvals)
    output = np.zeros((N, numberOfMeasurements))
    for i in range(N):
        output[i, :] = multi_tensor(gtab, eigenvalues, **kwargs)[0]
    return output.flatten(order='C')


def _samplesOnTheSphere(n):
    """Generate random samples from the 3D unit sphere.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    out : ndarray
        n x 3 array where each row corresponds to a point on the unit sphere.
    """
    randomVectors = np.random.randn(n, 3)
    norms = np.linalg.norm(randomVectors, axis=1)
    return randomVectors / norms[:, np.newaxis]

# -*- coding: utf-8 -*-


import numpy as np
from dipy.sims.voxel import multi_tensor


def generateCoordinates(voxelsInEachDim, voxelSize=np.array([1, 1, 1])):
    """ Generate coordinates for a cube.

    Parameters:
    ----------
    voxelsInEachDim : array-like
        The dimensions, (nx, ny, nz), of a 3D grid.

    Returns:
    --------
    coordinates : ndarray
        Array of coordinates in 3D, flattened to shape (nx*ny*nz) x 3.

    """
    separateCoordinateArrays = [np.arange(n) for n in voxelsInEachDim]
    coordinates = _cartesian(separateCoordinateArrays *
                             voxelSize[:, np.newaxis])
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


def _cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    New BSD License

    Copyright (c) 2007–2016 The scikit-learn developers.
    All rights reserved.


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

      a. Redistributions of source code must retain the above copyright notice,
         this list of conditions and the following disclaimer.
      b. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
      c. Neither the name of the Scikit-learn Developers  nor the names of
         its contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.


    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out

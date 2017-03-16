# -*- coding: utf-8 -*-


import numpy as np
import unittest
import numpy.testing as npt
from dipy.core.gradients import gradient_table
from diGP.dataManipulations import (DataHandler,
                                    generateCoordinates,
                                    inverseBoxCox)


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        bvals = np.array([0., 1000., 3000.])
        bvecs = np.array([[1., 0., 0.], [0., 1., 0], [0., 0., 1.]])
        big_delta = 1
        small_delta = 0
        # Expect all voxels except the one at (x, y, z) = (1, 1, 0)
        # to be background
        data = 0.05*np.ones((2, 2, 1, 3))
        data[1, 1, 0, 0] += 1
        data[1, 1, 0, 1] += 0.6
        data[1, 1, 0, 2] += 0.3

        self.gtab = gradient_table(bvals,
                                   bvecs,
                                   big_delta=big_delta,
                                   small_delta=small_delta)
        self.data = data

    def test_defaultInit(self):
        handler = DataHandler(self.gtab, self.data)
        dummy = 47
        assert(handler.gtab == self.gtab)
        npt.assert_allclose(handler.data, np.reshape(self.data, (4, 3)))
        npt.assert_allclose(handler.originalShape, self.data.shape)
        npt.assert_almost_equal(handler.qMagnitudeTransform(dummy), dummy)
        assert(handler.box_cox_lambda is None)

    def test_spatialIdx(self):
        mask = np.zeros(self.data.shape[:-1])
        mask[0, 1, 0] = 1.
        mask[1, 0, 0] = 1.
        idx = np.nonzero(mask)
        handler = DataHandler(self.gtab, self.data, spatialIdx=idx)
        npt.assert_allclose(handler.spatialIdx, idx)
        npt.assert_allclose(handler.data.shape, (2, 3))

    def test_box_cox(self):
        lmbda = 2
        handler = DataHandler(self.gtab, self.data, box_cox_lambda=lmbda)
        expected = np.reshape((self.data**lmbda - 1)/lmbda, (4, 3))
        npt.assert_allclose(handler.y, expected)

    def test_inverse_box_cox(self):
        lmbda = 0
        handler = DataHandler(self.gtab, self.data, box_cox_lambda=lmbda)
        dataAfterInverse = inverseBoxCox(handler.y, lmbda)
        npt.assert_allclose(dataAfterInverse, handler.data)

        lmbda = 2
        handler = DataHandler(self.gtab, self.data, box_cox_lambda=lmbda)
        dataAfterInverse = inverseBoxCox(handler.y, lmbda)
        npt.assert_allclose(dataAfterInverse, handler.data)

    def test_qFeatures(self):
        handler = DataHandler(self.gtab, self.data)
        expected = np.column_stack((self.gtab.qvals, self.gtab.bvecs))
        npt.assert_allclose(handler.X_q, expected)

    def test_X_coordinates_ordering_matches_y(self):
        data = np.ones((1, 2, 2, 3))
        data[0, 0, 0, :] = np.array([0, 0, 0])
        data[0, 0, 1, :] = np.array([0, 0, 1])
        data[0, 1, 0, :] = np.array([0, 1, 0])
        data[0, 1, 1, :] = np.array([0, 1, 1])
        expected = np.reshape(data, (4, 3))

        handler = DataHandler(self.gtab, data)
        npt.assert_allclose(handler.X_coordinates, expected)


class TestCoordinates(unittest.TestCase):

    def test_sameOrdering(self):
        (nx, ny, nz) = (1, 2, 3)
        syntheticCoordinates = generateCoordinates((nx, ny, nz))

        # Expect coordinates in C-order
        expectedCoordinates = np.array([[0, 0, 0],
                                        [0, 0, 1],
                                        [0, 0, 2],
                                        [0, 1, 0],
                                        [0, 1, 1],
                                        [0, 1, 2]])

        npt.assert_array_equal(syntheticCoordinates, expectedCoordinates)

        voxelSize = np.array([2, 3, 4])
        coordinatesWhenVoxelSizeDiffers = generateCoordinates(
            (nx, ny, nz), voxelSize=voxelSize)
        expectedCoordinatesWhenVoxelSizeDiffers = np.array([[0, 0, 0],
                                                            [0, 0, 4],
                                                            [0, 0, 8],
                                                            [0, 3, 0],
                                                            [0, 3, 4],
                                                            [0, 3, 8]])
        npt.assert_array_equal(coordinatesWhenVoxelSizeDiffers,
                               expectedCoordinatesWhenVoxelSizeDiffers)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

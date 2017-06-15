# -*- coding: utf-8 -*-


import numpy as np
import unittest
import unittest.mock as mock
import numpy.testing as npt
from dipy.core.gradients import gradient_table
from diGP.dataManipulations import (DataHandler,
                                    generateCoordinates,
                                    combineCoordinatesAndqVecs,
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

        mask = np.zeros(self.data.shape[:-1])
        mask[0, 1, 0] = 1.
        mask[1, 0, 0] = 1.
        self.maskIdx = np.nonzero(mask)

    def test_defaultInit(self):
        handler = DataHandler(self.gtab, self.data)
        dummy = 47
        assert(handler.gtab == self.gtab)
        npt.assert_allclose(handler.data, self.data.reshape(-1, 1))
        npt.assert_allclose(handler.originalShape, self.data.shape)
        npt.assert_almost_equal(handler.qMagnitudeTransform(dummy), dummy)
        assert(handler.box_cox_lambda is None)

    def test_spatialIdx(self):
        handler = DataHandler(self.gtab, self.data, spatialIdx=self.maskIdx)
        npt.assert_allclose(handler.spatialIdx, self.maskIdx)
        npt.assert_allclose(handler.data.shape, (2, 3))

    def test_X(self):
        handler = DataHandler(self.gtab, self.data, voxelSize=(3, 2, 1))
        x = np.array([[0], [3]])
        y = np.array([[0], [2]])
        z = np.array([[0]])
        expected = [x, y, z, handler.X_q]

        npt.assert_array_almost_equal(handler.X[0], expected[0])
        npt.assert_array_almost_equal(handler.X[1], expected[1])
        npt.assert_array_almost_equal(handler.X[2], expected[2])
        npt.assert_array_almost_equal(handler.X[3], expected[3])

    def test_y(self):
        handler = DataHandler(self.gtab, self.data)
        npt.assert_array_equal(handler.y, self.data.reshape(-1, 1))

    def test_box_cox(self):
        lmbda = 2
        handler = DataHandler(self.gtab, self.data, box_cox_lambda=lmbda)
        expected = ((self.data**lmbda - 1)/lmbda).reshape(-1, 1)
        npt.assert_allclose(handler.y, expected)

        handler = DataHandler(self.gtab, self.data,
                              spatialIdx=self.maskIdx, box_cox_lambda=lmbda)

        expected = (self.data[self.maskIdx[0], self.maskIdx[1],
                              self.maskIdx[2], :]**lmbda - 1)/lmbda
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

    @mock.patch('scipy.stats.boxcox')
    def test_optimize_box_cox_lambda(self, mock_boxcox):
        lmbda = 3
        mock_boxcox.return_value = [np.array([]), lmbda]

        handler = DataHandler(self.gtab, self.data)
        handler.optimize_box_cox_lambda()

        npt.assert_array_equal(np.array(mock_boxcox.call_args[0]).flatten(),
                               self.data.flatten())
        npt.assert_(mock_boxcox.call_args[1]['lmbda'] is None)
        npt.assert_equal(handler.box_cox_lambda, lmbda)

    def test_qFeatures(self):
        handler = DataHandler(self.gtab, self.data)
        expected = np.column_stack((self.gtab.qvals, self.gtab.bvecs))
        npt.assert_allclose(handler.X_q, expected)

        def f(q):
            return q ** 2
        handler = DataHandler(self.gtab, self.data, qMagnitudeTransform=f)
        expected = np.column_stack((self.gtab.qvals ** 2, self.gtab.bvecs))
        npt.assert_allclose(handler.X_q, expected)

    def test_X_coordinates_ordering_matches_y(self):
        expected = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [1, 0, 0],
                             [1, 1, 0]])

        handler = DataHandler(self.gtab, self.data)
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

    def test_combinationOfCoordinatesAndqVecs(self):
        coordinates = np.array([[2, 0, 0],
                                [0, 3, 0],
                                [0, 0, 4]])
        qMagnitude = 10
        qVecs = np.array([[qMagnitude, 1, 0, 0],
                          [qMagnitude, 0, 0, 1]])
        expectedCombination = np.array([[2, 0, 0, qMagnitude, 1, 0, 0],
                                        [2, 0, 0, qMagnitude, 0, 0, 1],
                                        [0, 3, 0, qMagnitude, 1, 0, 0],
                                        [0, 3, 0, qMagnitude, 0, 0, 1],
                                        [0, 0, 4, qMagnitude, 1, 0, 0],
                                        [0, 0, 4, qMagnitude, 0, 0, 1]])
        combination = combineCoordinatesAndqVecs(coordinates, qVecs)
        npt.assert_array_equal(combination, expectedCombination)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-


import numpy as np
import numpy.testing as npt
import unittest
from dipy.core.gradients import gradient_table
from diGP.preprocessing import replaceNegativeData


class TestReadHCP(unittest.TestCase):

    def test_replaceNegativeData(self):
        bvals = np.array([0., 0., 0., 1000.])
        bvecs = np.array([[1., 0., 0.],
                          [0., 1., 0],
                          [0., 0., 1.],
                          [0., 0., 1.]])
        data = np.ones((2, 2, 3, 4))
        data[:, :, :, 1] = -2*np.ones((2, 2, 3))
        data[:, :, :, 3] = -3*np.ones((2, 2, 3))

        gtab = gradient_table(bvals, bvecs)
        processedData = replaceNegativeData(data, gtab)

        expectedProcessedData = np.ones((2, 2, 3, 4))
        expectedProcessedData[:, :, :, 1] = 2/3*np.ones((2, 2, 3))
        expectedProcessedData[:, :, :, 3] = np.zeros((2, 2, 3))

        npt.assert_allclose(processedData, expectedProcessedData)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

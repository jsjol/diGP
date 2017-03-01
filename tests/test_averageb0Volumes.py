# -*- coding: utf-8 -*-


import unittest
import numpy as np
import numpy.testing as npt
from dipy.core.gradients import gradient_table
from diGP.averageb0Volumes import averageb0Volumes


class TestAverageb0Volumes(unittest.TestCase):

    def test_averageb0Volumes(self):
        mockbvals = np.array([0., 0., 1000.])
        mockbvecs = np.array([[1., 0., 0.], [0., 1., 0], [0., 0., 1.]])
        mockData = np.ones((4, 5, 6, 3))
        mockData[:, :, :, 1] = 3*np.ones((4, 5, 6))

        gtab = gradient_table(mockbvals, mockbvecs)
        averagedVolume = averageb0Volumes(mockData, gtab)
        npt.assert_array_almost_equal(averagedVolume, 2*np.ones((4, 5, 6)))


def main():
    unittest.main()

if __name__ == '__main__':
    main()

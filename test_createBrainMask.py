# -*- coding: utf-8 -*-


import unittest
from os.path import join, expanduser, isdir
import numpy as np
from dipy.data.fetcher import fetch_scil_b0, read_siemens_scil_b0
from createBrainMask import createBrainMaskFromb0Data


class TestCreateBrainMask(unittest.TestCase):

    def test_createBrainMaskFromb0Data(self):
        homeDirectory = expanduser('~')
        dataDirectory = join(homeDirectory, '.dipy',
                             'datasets_multi-site_all_companies')

        if not isdir(dataDirectory):
            fetch_scil_b0()

        b0Img = read_siemens_scil_b0()
        b0Data = np.squeeze(b0Img.get_data())

        brainMask = createBrainMaskFromb0Data(b0Data)
        self.assertTrue(np.array_equal(brainMask.shape, b0Data.shape))
        # Want to test that brainMask is boolean, but couldn't find a built-in
        # function
        dataIsTrueOrFalse = np.logical_or(brainMask == 0, brainMask == 1)
        self.assertTrue(np.array_equal(dataIsTrueOrFalse,
                                       np.ones(brainMask.shape)))
        voxelSizes = b0Img.header.get_zooms()[:3]
        voxelVolumeInCubicCentimeters = 1e-3*np.prod(voxelSizes)
        brainVolume = voxelVolumeInCubicCentimeters*np.sum(brainMask)
        self.assertTrue(brainVolume > 500 and brainVolume < 1500)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

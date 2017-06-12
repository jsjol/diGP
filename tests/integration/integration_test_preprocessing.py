# -*- coding: utf-8 -*-


import unittest
from os.path import expanduser, join, isdir
import numpy as np
from dipy.data.fetcher import fetch_scil_b0, read_siemens_scil_b0
from diGP.preprocessing import createBrainMaskFromb0Data


class IntegrationTestPreprocessingdMRI(unittest.TestCase):
    dataDirectory = None

    def setUp(self):
        homeDirectory = expanduser('~')
        self.dataDirectory = join(homeDirectory, '.dipy',
                                  'datasets_multi-site_all_companies')

        if not isdir(self.dataDirectory):
            fetch_scil_b0()

    def tearDown(self):
        pass

    def test_brainMaskVolume(self):
        b0Img = read_siemens_scil_b0()
        b0Data = np.squeeze(b0Img.get_data())
        voxelSizes = b0Img.header.get_zooms()[:3]
        voxelVolumeInCubicCentimeters = 1e-3*np.prod(voxelSizes)

        brainMask = createBrainMaskFromb0Data(b0Data)
        brainVolume = voxelVolumeInCubicCentimeters*np.sum(brainMask)
        self.assertTrue(brainVolume > 500 and brainVolume < 1500)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

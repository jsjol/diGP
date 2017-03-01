# -*- coding: utf-8 -*-


import unittest
from os import remove
from os.path import join, expanduser
import numpy as np
import nibabel as nib
from diGP.createBrainMask import createBrainMaskFromb0Data


class TestCreateBrainMask(unittest.TestCase):
    dataDirectory = None
    b0Data = np.array([0])
    affine = np.eye(4)

    def setUp(self):
        homeDirectory = expanduser('~')
        self.dataDirectory = join(homeDirectory, '.dipy')
        self.b0Data = np.reshape(np.arange(3375.0), (15, 15, 15))
        self.affine = np.array([[1., 0., 0., 1.],
                                [0., 2., 0., 2.],
                                [0., 0., 3., 3.],
                                [0., 0., 0., 1.]])

    def tearDown(self):
        try:
            remove(join(self.dataDirectory, 'brainMask.nii.gz'))
        except OSError:
            pass

    def test_createBrainMaskFromb0Data(self):
        brainMask = createBrainMaskFromb0Data(self.b0Data)
        self.assertTrue(np.array_equal(brainMask.shape, self.b0Data.shape))
        # Want to test that brainMask is boolean, but couldn't find a built-in
        # function
        dataIsTrueOrFalse = np.logical_or(brainMask == 0, brainMask == 1)
        self.assertTrue(np.array_equal(dataIsTrueOrFalse,
                                       np.ones(brainMask.shape)))
        self.assertTrue(np.all(self.b0Data[brainMask == 1] > 0))

    def test_saveAndLoadBrainMask(self):
        brainMask = createBrainMaskFromb0Data(self.b0Data,
                                              affineMatrix=self.affine,
                                              saveDir=self.dataDirectory)
        loadedBrainMaskNifti = nib.load(join(self.dataDirectory,
                                             'brainMask.nii.gz'))
        self.assertTrue(np.allclose(brainMask,
                                    loadedBrainMaskNifti.get_data()))
        self.assertTrue(np.allclose(loadedBrainMaskNifti.affine, self.affine))


def main():
    unittest.main()

if __name__ == '__main__':
    main()

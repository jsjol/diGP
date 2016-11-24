# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:12:09 2016

@author: Jens Sjölund
"""

import unittest
from os import makedirs, rmdir, rename
from os.path import dirname, abspath, join, expanduser, exists
import numpy as np
from dipy.data import fetch_isbi2013_2shell
from dipy.core.gradients import GradientTable
from readHCP import readHCP


class TestReadHCP(unittest.TestCase):

    def test_fileNotFoundRaisesFileNotFoundException(self):
        pathToThisFile = dirname(abspath(__file__))
        pathToIntentionallyEmptyDirectory = join(pathToThisFile,
                                                 'intentionallyEmptyDirectory')
        makedirs(pathToIntentionallyEmptyDirectory)
        self.assertRaises(FileNotFoundError,
                          readHCP, pathToIntentionallyEmptyDirectory)
        rmdir(pathToIntentionallyEmptyDirectory)

    def test_returnsCorrect(self):
        homeDirectory = expanduser('~')
        dataDirectory = join(homeDirectory, '.dipy', 'isbi2013')
        
        if not exists(dataDirectory):
            fetch_isbi2013_2shell()
            
            oldFileNamebval = join(dataDirectory, 'phantom64.bval')
            newFileNamebval = join(dataDirectory, 'bvals.txt')
            rename(oldFileNamebval, newFileNamebval)
            
            oldFileNamebvecs = join(dataDirectory, 'phantom64.bvec')
            newFileNamebvecs = join(dataDirectory, 'bvecs_moco_norm.txt')
            rename(oldFileNamebvecs, newFileNamebvecs)
            
            oldFileNameData = join(dataDirectory, 'phantom64.nii.gz')
            newDataDirectory = join(dataDirectory, 'mri')
            makedirs(newDataDirectory)
            newFileNameData = join(newDataDirectory, 'diff_preproc.nii.gz')
            rename(oldFileNameData, newFileNameData)
            
        gtab, data = readHCP(dataDirectory)
        self.assertIsInstance(gtab, GradientTable)
        self.assertEqual(gtab.bvals.shape, np.array([64, ]))
        self.assertTrue(np.allclose(np.unique(np.round(gtab.bvals)), 
                                    np.array([0., 1500., 2500.])))
        self.assertTrue(np.array_equal(gtab.bvecs.shape, np.array([64, 3])))
        
        normOfbvecs = np.sum(gtab.bvecs ** 2,1)
        self.assertTrue(np.allclose(normOfbvecs[np.invert(gtab.b0s_mask)], 1.))
        
        smallDelta = 12.9
        bigDelta = 21.8
        tau = bigDelta-smallDelta / 3.0
        expectedqvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
        self.assertTrue(np.allclose(gtab.qvals, expectedqvals))
        
        self.assertTrue(np.array_equal(data.shape, np.array([50, 50, 50, 64])))
        self.assertTrue(np.all(data >= 0))


def main():
    unittest.main()

if __name__ == '__main__':
    main()

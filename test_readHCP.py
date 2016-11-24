# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:12:09 2016

@author: Jens Sjölund
"""

import unittest
from os import makedirs, rmdir, rename
from os.path import dirname, abspath, join, expanduser, exists
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


def main():
    unittest.main()

if __name__ == '__main__':
    main()

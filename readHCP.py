# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:50:56 2016

@author: SESjoJen
"""
from os.path import join
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs


def readHCP(directoryName):

    fileNameNifti = join(directoryName, 'mri', 'diff_preproc.nii.gz')
    fileNamebval = join(directoryName, 'bvals.txt')
    fileNamebvecs = join(directoryName, 'bvecs_moco_norm.txt')
    smallDelta = 12.9
    bigDelta = 21.8
    try:
        niftiFile = nib.load(fileNameNifti)
        data = niftiFile.get_data()
        bvals, bvecs = read_bvals_bvecs(fileNamebval, fileNamebvecs)
        gtab = gradient_table(bvals, bvecs, bigDelta, smallDelta)
    except IOError:
        print('Error when attempting to read HCP data.')
        raise

    return gtab, data

# -*- coding: utf-8 -*-


from os.path import join
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
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
        voxelSize = niftiFile.header.get_zooms()[0:3]
    except IOError:
        print('Error when attempting to read HCP data.')
        raise

    return gtab, data, voxelSize


def averageb0Volumes(data, gtab):
    maskedData = data[:, :, :, gtab.b0s_mask]
    averagedMaskedData = np.mean(maskedData, axis=3)
    return np.squeeze(averagedMaskedData)


def createBrainMaskFromb0Data(b0Data, affineMatrix=None, saveDir=None):
    """Creates a mask of the brain from a b0 volume.
    The output is written to file if affineMatrix and saveDir are provided.

    Parameters:
    ----------
    b0Data : 3D ndarray
        MRI scan of the head without diffusion weighting
    affineMatrix : 4x4 array
        Affine transformation matrix as in Nifti
    saveDir : string
        Directory where the created mask will be saved as 'brainMask.nii.gz'.

    Returns:
    --------
    mask : 3D ndarray
        The binary brain mask.
    """

    # Call median_otsu and discard first return value
    mask = median_otsu(b0Data)[1]
    mask = np.logical_and(mask == 1, b0Data > 0)

    if affineMatrix is not None and saveDir is not None:
        try:
            maskNifti = nib.Nifti1Image(mask.astype(np.float32), affineMatrix)
            nib.save(maskNifti, join(saveDir, 'brainMask.nii.gz'))
        except Exception as e:
            print('Saving of the brain mask \
                  failed with message {}'.format(e.message))
    return mask


def replaceNegativeData(data, gtab):
    """ Replace negative data points with mean if b = 0 and zero otherwise.

    Parameters:
    -------------
    data : ndarray
        4D diffusion data to process.
    gtab : Gradient table
        Gradient table that specifies the b-values used.

    Returns:
    --------
    processedData : ndarray
        Data of the same size as input data but with negative values replaced.
    """

    data[data < 0] = 0
    b0 = averageb0Volumes(data, gtab)
    zeroIdx = np.nonzero(data == 0)
    for b0Idx in np.nditer(np.nonzero(gtab.b0s_mask)):
        match = (zeroIdx[-1] == b0Idx)
        x = zeroIdx[0][match]
        y = zeroIdx[1][match]
        z = zeroIdx[2][match]
        data[x, y, z, b0Idx] = b0[x, y, z]

    return data

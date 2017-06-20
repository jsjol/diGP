# -*- coding: utf-8 -*-


from os.path import join
import numpy as np
from glob import glob
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

    gtab, data, voxelSize = _load(fileNameNifti, fileNamebval,
                                  fileNamebvecs, smallDelta, bigDelta)

    return gtab, data, voxelSize


def readSPARC(directoryName):
    fileNameNifti = glob(join(directoryName, 'G*.nii'))[0]
    fileNamebval = glob(join(directoryName, 'bval*.txt'))[0]
    fileNamebvecs = glob(join(directoryName, 'bvec*.txt'))[0]

    # The SPARC paper says that smallDelta = bigDelta = 62 ms, but according
    # to the challenge instructions tau = bigDelta - smallDelta/3 = 70 ms.
    # We will go with the latter of these conflicting statements here.
    smallDelta = 0
    bigDelta = 70

    gtab, data, _ = _load(fileNameNifti, fileNamebval,
                          fileNamebvecs, smallDelta, bigDelta)

    voxelSize = (2., 2., 7.)  # From SPARC paper

    return gtab, data, voxelSize


def _load(fileNameNifti, fileNamebval, fileNamebvecs, smallDelta, bigDelta):
    try:
        niftiFile = nib.load(fileNameNifti)
        data = niftiFile.get_data()
        bvals, bvecs = read_bvals_bvecs(fileNamebval, fileNamebvecs)
        gtab = gradient_table(bvals, bvecs, bigDelta, smallDelta)
        voxelSize = niftiFile.header.get_zooms()[0:3]
    except IOError:
        print('Error when attempting to read the data.')
        raise
    return gtab, data, voxelSize


def averageb0Volumes(data, gtab):
    return np.mean(data[:, :, :, gtab.b0s_mask], axis=3, keepdims=False)


def normalize_data(data, b0, mask):
    """ Divide data inside the mask with the corresponding b0 value. Set all
    values outside the mask to zero.
    """
    out = np.zeros_like(data)
    maskIdx = np.nonzero(mask)
    out[maskIdx[0], maskIdx[1], maskIdx[2], :] = (
        data[maskIdx[0], maskIdx[1], maskIdx[2], :] /
        b0[maskIdx[0], maskIdx[1], maskIdx[2], np.newaxis])
    return out


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

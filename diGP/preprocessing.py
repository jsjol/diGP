# -*- coding: utf-8 -*-


from os.path import join, splitext
import numpy as np
from glob import glob
import nibabel as nib
from dipy.segment.mask import median_otsu, crop, bounding_box
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs


class Loader():

    def __init__(self, filename_data, filename_bval, filename_bvecs,
                 small_delta, big_delta, voxel_size=None):
        self.filename_data = filename_data
        self.filename_bval = filename_bval
        self.filename_bvecs = filename_bvecs
        self.small_delta = small_delta
        self.big_delta = big_delta

        self.img = None
        self._data = None
        self._voxel_size = voxel_size
        self._bvals = None
        self._bvecs = None
        self._gtab = None

    def update_filename_data(self, new_filename):
        self.filename_data = new_filename
        self._data = None
        self.img = None
        return self

    @property
    def data(self):
        if self._data is None:
            if splitext(self.filename_data)[1] == '.npy':
                _data = np.load(self.filename_data)
            else:
                self.img = nib.load(self.filename_data)
                _data = self.img.get_data()

            # Convert to single precision
            self._data = _data.astype('float32')

        return self._data

    @property
    def header(self):
        if self.img is None:
            self.img = nib.load(self.filename_data)
        return self.img.header

    @property
    def voxel_size(self):
        if self._voxel_size is None:
            self._voxel_size = self.header.get_zooms()[0:3]
        return self._voxel_size

    @property
    def bvals(self):
        if self._bvals is None:
            bvals, bvecs = read_bvals_bvecs(self.filename_bval,
                                            self.filename_bvecs)
            self._bvals = bvals
            self._bvecs = bvecs
        return self._bvals

    @property
    def bvecs(self):
        if self._bvecs is None:
            bvals, bvecs = read_bvals_bvecs(self.filename_bval,
                                            self.filename_bvecs)
            self._bvals = bvals
            self._bvecs = bvecs
        return self._bvecs

    @property
    def gtab(self):
        if self._gtab is None:
            self._gtab = gradient_table(self.bvals, self.bvecs,
                                        self.big_delta, self.small_delta)
        return self._gtab


def get_HCP_loader(directoryName):
    fileNameNifti = join(directoryName, 'mri', 'diff_preproc.nii.gz')
    fileNamebval = join(directoryName, 'bvals.txt')
    fileNamebvecs = join(directoryName, 'bvecs_moco_norm.txt')
    smallDelta = 12.9
    bigDelta = 21.8
    voxel_size = (1.5, 1.5, 1.5)

    return Loader(fileNameNifti, fileNamebval, fileNamebvecs,
                  smallDelta, bigDelta, voxel_size=voxel_size)


def get_SPARC_loader(directoryName):
    fileNameNifti = glob(join(directoryName, 'G*.nii'))[0]
    fileNamebval = glob(join(directoryName, 'bval*.txt'))[0]
    fileNamebvecs = glob(join(directoryName, 'bvec*.txt'))[0]

    # The SPARC paper says that smallDelta = bigDelta = 62 ms, but according
    # to the challenge instructions tau = bigDelta - smallDelta/3 = 70 ms.
    # We will go with the latter of these conflicting statements here.
    smallDelta = 0
    bigDelta = 70

    voxelSize = (2., 2., 7.)  # From SPARC paper
    return Loader(fileNameNifti, fileNamebval, fileNamebvecs,
                  smallDelta, bigDelta, voxel_size=voxelSize)


def readHCP(directoryName):
    hcp = get_HCP_loader(directoryName)
    return hcp.gtab, hcp.data, hcp.voxel_size


def readSPARC(directoryName):
    sparc = get_SPARC_loader(directoryName)
    return sparc.gtab, sparc.data, sparc.voxel_size


def averageb0Volumes(data, gtab):
    return np.mean(data[:, :, :, gtab.b0s_mask], axis=3, keepdims=False)


def crop_using_mask(data, mask):
    mins, maxs = bounding_box(mask)
    cropped_data = crop(data, mins, maxs)
    return cropped_data


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
    masked_b0, mask = median_otsu(b0Data, median_radius=3, numpass=5,
                                  dilate=2)
    mask = np.logical_and(mask == 1, masked_b0 > 0)

    if affineMatrix is not None and saveDir is not None:
        try:
            maskNifti = nib.Nifti1Image(mask.astype(np.float32), affineMatrix)
            nib.save(maskNifti, join(saveDir, 'brainMask.nii.gz'))
        except Exception as e:
            print('Saving of the brain mask \
                  failed with message {}'.format(e.message))
    return masked_b0, mask


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

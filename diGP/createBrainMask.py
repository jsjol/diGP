# -*- coding: utf-8 -*-


from os.path import join
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu


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

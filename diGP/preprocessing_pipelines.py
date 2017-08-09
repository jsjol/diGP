# -*- coding: utf-8 -*-


import os.path
from os.path import isfile
import numpy as np
from dipy.core.gradients import gradient_table
from diGP.preprocessing import (get_SPARC_loader, get_HCP_loader,
                                replaceNegativeData, crop_using_mask,
                                averageb0Volumes, normalize_data,
                                createBrainMaskFromb0Data)


def get_filenames(filename):
    head, tail = os.path.split(filename)
    root = os.path.join(head, tail.split('.')[0])
    fnames = {}
    keys = ['nonneg', 'b0', 'cropped', 'mask',
            'normalized', 'normalized_cropped']
    for k in keys:
        fnames[k] = '{}_{}.npy'.format(root, k)

    return fnames


def attempt_loading(loader, filenames, crop=False, normalize=False):
    success = False

    if normalize and crop:
        if isfile(filenames['normalized_cropped']):
            loader = loader.update_filename_data(
                                        filenames['normalized_cropped'])
            success = True
            return success, loader

    if normalize and not crop:
        if isfile(filenames['normalized']):
            loader = loader.update_filename_data(filenames['normalized'])
            success = True
            return success, loader

    if not normalize and crop:
        if isfile(filenames['cropped']):
            loader = loader.update_filename_data(filenames['cropped'])
            success = True
            return success, loader

    return success, loader


def preprocess(loader, use_mask=False, crop=False, normalize=False):
    filenames = get_filenames(loader.filename_data)

    load_success, loader = attempt_loading(loader, filenames,
                                           crop=crop, normalize=normalize)
    if load_success:
        return loader

    if isfile(filenames['nonneg']):
        loader = loader.update_filename_data(filenames['nonneg'])
        data = loader.data
        print("Loaded existing non-negative data.")
    else:
        print("Replacing negative data.")
        data = replaceNegativeData(loader.data, loader.gtab)
        np.save(filenames['nonneg'], data)
        loader = loader.update_filename_data(filenames['nonneg'])

    if normalize or crop:
        try:
            b0 = np.load(filenames['b0'])
            print("Loaded existing b0 image.")
        except:
            print("Extracting b0 image.")
            b0 = averageb0Volumes(data, loader.gtab)
            np.save(filenames['b0'], b0)

        if use_mask:
            try:
                mask = np.load(filenames['mask'])
                print("Loaded existing mask.")
            except:
                print("Creating new mask.")
                b0, mask = createBrainMaskFromb0Data(b0)
                np.save(filenames['mask'], mask)
        else:
            mask = np.ones_like(b0)

    if crop:
        if isfile(filenames['cropped']):
            data = np.load(filenames['cropped'])
            print("Loaded existing cropped data.")
        else:
            print("Cropping data.")
            data = crop_using_mask(data, mask)
            b0 = crop_using_mask(b0, mask)
            mask = crop_using_mask(mask, mask)
            np.save(filenames['cropped'], data)
        loader = loader.update_filename_data(filenames['cropped'])

    if normalize:
        print("Normalizing data.")
        data = normalize_data(data, b0, mask)
        if crop:
            key = 'normalized_cropped'
        else:
            key = 'normalized'
        np.save(filenames[key], data)
        loader = loader.update_filename_data(filenames[key])

    return loader


def preprocess_SPARC(directory, use_mask=False, crop=False, normalize=True):
    sparc = get_SPARC_loader(directory)
    sparc = preprocess(sparc, use_mask=use_mask, crop=crop,
                       normalize=normalize)
    return sparc.gtab, sparc.data, sparc.voxel_size


def preprocess_HCP(directory, use_mask=True, crop=True, normalize=True,
                   max_normalized_signal=1.2):
    hcp = get_HCP_loader(directory)
    hcp = preprocess(hcp, use_mask=use_mask, crop=crop,
                     normalize=normalize)

    gtab = hcp.gtab
    voxel_size = hcp.voxel_size
    data = hcp.data
    if normalize:
        data[data > max_normalized_signal] = max_normalized_signal

    return gtab, data, voxel_size


def get_SPARC_train_and_test(train_dir, test_dir, q_test_path):

    gtab, data, voxelSize = preprocess_SPARC(train_dir)
    data = data[:, :, 0, :]  # Remove singleton dimension

    _, data_test, _ = preprocess_SPARC(test_dir)
    data_test = data_test[:, :, 0, 1:]  # Remove b0 and singleton dimension
    # Transpose to get same shape as training data
    data_test = data_test.transpose((1, 0, 2))

    # Would have been simple to use the gradient table from the gold standard
    # if it hadn't contained b0. Nevertheless, this is more fair since it is
    # the same information as given to the contestants in SPARC
    q_test = load_q_test(q_test_path)
    gtab_test = get_gtab_test(gtab, q_test)

    gtab_dict = {'train': gtab, 'test': gtab_test}
    data_dict = {'train': data, 'test': data_test}
    return gtab_dict, data_dict, voxelSize


def load_q_test(path):
    with open(path, 'r') as f:
        out = [line.split(' ') for line in f.readlines()]
    return np.array(out, dtype=float)


def get_gtab_test(gtab, q_test):
    small_delta = gtab.small_delta
    big_delta = gtab.big_delta
    tau = (big_delta - small_delta/3) * 1e-3
    q_magnitude = np.sqrt(np.sum(q_test ** 2, 1))
    bvals = (2*np.pi*q_magnitude) ** 2 * tau
    bvecs = q_test / q_magnitude[:, None]
    return gradient_table(bvals=bvals, bvecs=bvecs,
                          small_delta=small_delta, big_delta=big_delta)

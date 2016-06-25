import os
import copy
import warnings

import numpy as np
from scipy import io
import nibabel


def check_images(file1, file2):
    """Check that 2 images have the same affines and data shapes.

    Parameters
    ----------
    file1 : str
        Path to the first nifti image

    file2 : str
        Path to the second nifti image
    """
    img = nibabel.load(file1)
    shape1 = np.shape(img.get_data())
    affine1 = img.get_affine()
    img = nibabel.load(file2)
    shape2 = np.shape(img.get_data())
    affine2 = img.get_affine()
    if shape1 != shape2:
        raise ValueError('Images got different shapes: {0} of shape {1}, {2} '
                         'of shape {3}'.format(file1, shape1, file2, shape2))

    if np.any(affine1 != affine2):
        raise ValueError('Images got different affines: {0} has affine {1}, '
                         '{2} has affine {3}'.format(file1, affine1,
                                                     file2, affine2))


def get_voxel_dims(in_file):
    """Return the voxels resolution of a nifti image.

    Parameters
    ----------
    in_file : str
        Path to the nifti image

    Returns
    -------
    list of 3 float
        Resolutions
    """
    img = nibabel.load(in_file)
    header = img.get_header()
    voxdims = header.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def threshold(in_file, threshold_min=-1e7, threshold_max=1e7, out_file=None):
    """Put to thresholds values outside given thresholds

    Parameters
    ----------
    in_file : str
        Path to the nifti image

    threshold_min : float or None, optional
        Values less than this threshold are set to it.

    threshold_max : float or None, optional
        Values greater than this threshold are set to it.

    out_file : str or None, optional
        Path to the thresholded image

    Returns
    -------
    out_file : str
        Path to the thresholded image
    """
    img = nibabel.load(in_file)
    data = img.get_data()
    if threshold_max is not None:
        data[data > threshold_max] = threshold_max

    if threshold_min is not None:
        data[data < threshold_min] = threshold_min

    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    if out_file is None:
        out_file, _ = os.path.splitext(in_file)
        out_file += '_thresholded.nii'

    if os.path.isfile(out_file):
        warnings.warn('File {0} exits, overwriting.'.format(out_file))

    nibabel.save(img, out_file)
    return out_file


def fill_nan(in_file, fill_value=0.):
    """Replace nan values with a given value

    Parameters
    ----------
    in_file : str
        Path to image file

    fill_value : float, optional
        Value replacing nan

    Returns
    -------
    out_file : str
        Path to the filled file
    """
    img = nibabel.load(in_file)
    data = img.get_data()
    if np.any(np.isnan(data)):
        data[np.isnan(data)] = fill_value
    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    out_file, _ = os.path.splitext(in_file)
    out_file += '_no_nan.nii'
    nibabel.save(img, out_file)
    return out_file


def _compute_depth(dictionary):
    if not isinstance(dictionary, dict):
        raise ValueError('Can not compute depth of non dictionary type.'
        'Data has type {0}'.format(dictionary))

    depths = []
    for value in dictionary.itervalues():
        if isinstance(value, dict):
            depth = 1 + _compute_depth(value)
        else:
            depth = 0
        depths.append(depth)
    if depths:
        depth = np.max(depths)
    else:
        depth = 0

    return depth


def _deepest_dict_to_rec(dictionary):
    """transforms the deepest dictionary to a recarray
    """
    if not isinstance(dictionary, dict):
        raise ValueError('Can not convert non dictionary type.'
                         'Data has type {0}'.format(dictionary))

    if dictionary == {}:
        return dictionary

    dictionary_out = copy.copy(dictionary)
    depth = _compute_depth(dictionary)
    actual_depth = -1
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            actual_depth += 1
            dictionary_out[key] = _deepest_dict_to_rec(value)
            if actual_depth == depth:
                dictionary_out_keys, dictionary_out_values = \
                    zip(*dictionary_out.items())
                dtype = zip(dictionary_out_keys,
                              [type(key) for key in dictionary_out_keys])
                dtype = zip(dictionary_out_keys,
                              [type(value) for value in dictionary_out_values])
                dictionary_out = np.array([tuple(dictionary_out_values), ],
                                           dtype=dtype)
    return dictionary_out


def squeeze_data(mat_struct):
    data = copy.copy(mat_struct)
    if isinstance(data, np.void):
        for name in data.dtype.names:
            if isinstance(name, str):
                print name
            data[name] = squeeze_data(data[name])
    elif isinstance(data, dict):
        for name in data.keys():
            if isinstance(name, str):
                print name
            data[name] = squeeze_data(data[name])
    elif isinstance(data, io.matlab.mio5_params.MatlabObject) or\
            isinstance(data, np.ndarray):
        if data.shape == (1, 1):
            data = squeeze_data(data[0, 0])
        elif data.dtype.fields is not None:
            for name in data.dtype.names:
                if isinstance(name, str):
                    print name
                data[name] = squeeze_data(data[name])
        else:
            for n, _ in enumerate(data):
                data[n] = squeeze_data(data[n])

    return data


def remove_none(mat_struct, recarray=False):
    data = copy.copy(mat_struct)
    if isinstance(data, np.void):
        for name in data.dtype.names:
            data[name] = remove_none(data[name])
    elif isinstance(data, dict):
        for name in data.keys():
            data[name] = remove_none(data[name])
    elif isinstance(data, io.matlab.mio5_params.MatlabObject) or\
            isinstance(data, np.ndarray):
        if data.dtype.fields is not None:
            recarray = True
            for name in data.dtype.names:
                if isinstance(name, str):
                    print name
                data[name] = remove_none(data[name], recarray=True)
        else:
            if not recarray and np.all(data == np.array([[None]], dtype='O')):
                data = np.array([[]])
            else:
                for n, _ in enumerate(data):
                    data[n] = remove_none(data[n], recarray=recarray)
    elif data is None:
        data = []

    return data


def void_to_dict(mat_struct):
    """Transforms a nmpy.void to dictionary
    """
    data = copy.copy(mat_struct)
#    print type(data)
    if isinstance(data, np.void):
        dictionary = {}
        for name in data.dtype.names:
            dictionary[name] = void_to_dict(data[name])
    elif isinstance(data, io.matlab.mio5_params.MatlabObject):
#        if np.shape(data) == (1, 1):
#            dictionary = void_to_dict(data[0, 0])
        if data.dtype.hasobject:
            if data.dtype.fields is not None and data.size == 1:
                dictionary = {}
                for name in data.dtype.names:
                    if isinstance(name, str):
                        print name
                    dictionary[name] = void_to_dict(data[name])
            else:
                dictionary = void_to_dict(data[0])
    elif isinstance(data, np.ndarray) and np.shape(data) == (1, 1):
        dictionary = void_to_dict(data[0, 0])
    elif isinstance(data, np.ndarray) and data.dtype.hasobject:
        if data.dtype.fields is not None:
            dictionary = {}
            for name in data.dtype.names:
                if isinstance(name, str):
                    print name
                dictionary[name] = void_to_dict(data[name])
        else:
            dictionary = np.empty(shape=data.shape, dtype=data.dtype)
            for n, sub_data in enumerate(data):
                if data.shape == (164, 1) and (n == 0 or n == 100):
                    print "_" * 10
                    print n, sub_data
                x = void_to_dict(sub_data)
                dictionary[n] = x
#            dictionary = dictionary.reshape(data.shape)
    elif data is None:
        print '*' * 25
        dictionary = np.array([[]], dtype=object)
    else:
        dictionary = data

    if isinstance(dictionary, dict):
        dictionary = _deepest_dict_to_rec(dictionary)
    if isinstance(dictionary, np.ndarray):
        dictionary = np.squeeze(dictionary)

    return dictionary


def _check_structured_data_equal(data1, data2):
    if isinstance(data1, dict) and isinstance(data2, dict):
        set1 = set(data1.keys())
        set2 = set(data2.keys())
        if set1 == set2:
            checks = []
            for key in set1:
                print key
                checks.append(_check_structured_data_equal(data1[key],
                                                           data2[key]))
            check = np.all(checks)
        else:
            message = ''
            if set1 - set2:
                message.extend('Keys {0} not found in data 2. '.format(
                    set1 - set2))
            if set2 - set1:
                message.extend('Keys {0} not found in data 1. '.format(
                    set2 - set1))
            print(message)
            return False
    elif (isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray)) or\
         (isinstance(data1, np.void) and isinstance(data2, np.void)):
        if data1.shape == data2.shape:
            print data1.shape
            dtype1 = data1.dtype
            dtype2 = data2.dtype
            if set(dtype1.descr) == set(dtype2.descr):
                check = np.all(data1 == data2)
            else:
                print('data1 has dtype {0}, data2 {1}'.format(
                    data1.dtype, data2.dtype))
                return False
        else:
            print('data1 has shape {0}, data2 {1}'.format(
                data1.shape, data2.shape))
            return False
    elif type(data1) == type(data2):
        check = (data1 == data2)
    else:
        print('data& has type {1}, data2 has type {2}'.format(
            type(data1), type(data2)))
        return False
    return check

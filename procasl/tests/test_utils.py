import os
import numpy as np
from numpy.testing import assert_array_equal
from scipy.io import loadmat, savemat
from nose.tools import assert_raises, assert_equal
import nibabel

from procasl import _utils


def test_check_images():
    data = np.zeros((91, 91, 60))
    affine = np.eye(4)
    affine[:, 3] = np.ones(4)
    file1 = '/tmp/file1.nii'
    file2 = '/tmp/file2.nii'
    file3 = '/tmp/file3.nii'
    nibabel.Nifti1Image(data, affine).to_filename(file1)
    nibabel.Nifti1Image(data, affine * 3).to_filename(file2)
    nibabel.Nifti1Image(data + 1, affine * 3).to_filename(file3)
    _utils.check_images(file1, file1)
    assert_raises(ValueError, _utils.check_images, file1, file2)
    assert_raises(ValueError, _utils.check_images, file1, file3)


def test_get_voxel_dims():
    data = np.zeros((91, 91, 60))
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/file1.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    assert_array_equal(_utils.get_voxel_dims(in_file), [3., 3., 3.])


def test_threshold():
    data = np.zeros((5, 5, 6))
    data[1, 4, 1] = 1.
    data[2, 3, 2] = -1.
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/in_file.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    threshold_min = -.1
    threshold_max = .1
    out_file =_utils.threshold(in_file, threshold_min=threshold_min,
                               threshold_max=threshold_max)
    data[1, 4, 1] = threshold_max
    data[2, 3, 2] = threshold_min
    assert_array_equal(nibabel.load(out_file).get_data(), data)
    assert_array_equal(nibabel.load(out_file).get_affine(), affine)


def test_fill_nan():
    data = np.zeros((5, 5, 6))
    data[1, 4, 1] = np.nan
    data[2, 3, 2] = np.nan
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/in_file.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    fill_value = 1.
    out_file = _utils.fill_nan(in_file, fill_value=fill_value)
    data[1, 4, 1] = fill_value
    data[2, 3, 2] = fill_value
    assert_array_equal(nibabel.load(out_file).get_data(), data)


def test_compute_depth():
    dictionary1 = {'name': 'Thomas', 'age': 7.5}
    assert_equal(_utils._compute_depth(dictionary1), 0)
    dictionary2 = {'name': 'Thomas', 'age': 7.5, 'dict1': dictionary1}
    assert_equal(_utils._compute_depth(dictionary2), 1)
    dictionary2 = {'name': 'Thomas', 'dict1': dictionary1,
                   'dict2': dictionary1}
    assert_equal(_utils._compute_depth(dictionary2), 1)
    dictionary3 = {'name': 'Thomas', 'dict1': dictionary1,
                   'dict2': dictionary2}
    assert_equal(_utils._compute_depth(dictionary3), 2)


# TODO: create input correctly, as in nilearn.datasets
def test_void_to_dict():
    # Simple dict, simple values
    filename = '/tmp/test_file.mat'
    dictionary_in = {'name': 'Thomas', 'age': 7.5}
    savemat(filename, dictionary_in)
    mat_data = loadmat(filename, struct_as_record=True, squeeze_me=False)
    dictionary_out = _utils.void_to_dict(mat_data)
    assert(dictionary_out['name'] == 'Thomas')
    assert(dictionary_out['age'] == [7.5])

    # Simple dict, complex values
    dictionary_in = {'name': 'Thomas', 'age': np.array([[7.5]])}
    savemat(filename, dictionary_in)
    mat_data = loadmat(filename, struct_as_record=True, squeeze_me=False)
    dictionary_out = _utils.void_to_dict(mat_data)
    assert(dictionary_out['name'] == 'Thomas')
    assert(dictionary_out['age'] == np.array([[7.5]]))

    # Dict of dict
    dictionary_in = {'agent': {'name': 'Thomas', 'age': np.array([[7.5]])},
                     'building': ['beautiful', 'expensive']}
    savemat(filename, dictionary_in)
    mat_data = loadmat(filename, struct_as_record=True, squeeze_me=True)
    dictionary_out = _utils.void_to_dict(mat_data)
    assert(np.size(dictionary_out['building']) == 2)
    assert(dictionary_out['building'][0] == 'beautiful')
    assert(dictionary_out['building'][1] == 'expensive')
    assert(dictionary_out['agent']['name'] == 'Thomas')
    assert(dictionary_out['agent']['age'] == 7.5)

    # Dict of dict of dict
    dictionary_in = {'dict1': dictionary_in, 'depth': 2}
    savemat(filename, dictionary_in)
    mat_data = loadmat(filename, struct_as_record=True, squeeze_me=True)
    dictionary_out = _utils.void_to_dict(mat_data)
    sub_dict = dictionary_out['dict1']
    assert(str(sub_dict['building']) == "[u'beautiful' u'expensive']")
    assert(str(sub_dict['agent']) == "(7.5, u'Thomas')")
    os.remove(filename)


def test_check_structured_data_equal():
    dictionary1 = {'agent': {'name': 'Thomas', 'age': np.array([[7.5]])},
                   'building': ['beautiful', 'expensive']}
    assert(_utils._check_structured_data_equal(dictionary1,
                                               dictionary1.copy()))
    array1 = np.random.rand(34, 12, 2)
    assert(_utils._check_structured_data_equal(array1, array1.copy()))
    dictionary2 = {'agent': {'name': 'Thomas',
                             'age': np.array([[7.5]], dtype='O')},
                   'building': ['beautiful', 'expensive']}
    assert(not (_utils._check_structured_data_equal(dictionary1, dictionary2)))


def test_remove_none():
    array = np.array([[None]], dtype='O')
    nan_array = np.array([[np.nan]], dtype=float)
    assert_array_equal(_utils.remove_none(array), nan_array)

    dictionary = {'field1': 2.3, 'field2': array}
    dictionary_out = _utils.remove_none(dictionary)
    assert_equal(set(dictionary_out.keys()), {'field1', 'field2'})
    assert_equal(dictionary_out['field1'], 2.3)
    assert_array_equal(dictionary_out['field2'], nan_array)

    rec_array = np.array([(3, 2.1, array), ],
                         dtype=[('field1', np.int), ('field2', np.float),
                                ('field3', np.ndarray)])
    rec_array_out = _utils.remove_none(rec_array)
    assert_equal(rec_array_out['field1'], 3)
    assert_equal(rec_array_out['field2'], 2.1)
    assert_array_equal(rec_array['field3'][0], array)
    assert_array_equal(rec_array_out['field3'][0],
                       np.array([[np.nan]], dtype='O'))

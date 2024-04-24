import sys
import os
import math
import time
import h5py
import numpy as np

def splitDatasetSingleFile(input_file, step, file_ind, total_file_inds, output_initial):
    chunk_size = math.floor(131072) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_name = 'res/slice%05d.h5' % (file_ind) 
        os.remove(output_file_name)
    except:
        pass

    h5step = input_file["Step#%d" % step]
    num_total_rows = len(h5step["z"])

    slice_ranges = [i for i in range(0, num_total_rows, int(num_total_rows/total_file_inds))]
    if len(slice_ranges) == total_file_inds:
        slice_ranges.append(num_total_rows)
    else:
        slice_ranges[-1] = num_total_rows

    curr_start_ind = slice_ranges[file_ind]
    curr_end_ind = slice_ranges[file_ind+1]
    curr_slice_size = curr_end_ind - curr_start_ind

    output_file = h5py.File(output_initial+'%05d.h5' % (file_ind) , 'w')
    group = output_file.create_group('Step#0')
    attrs = [
        {
            'name': 'Kcour',
            'type': 'f8',
            'old_name': 'Kcour',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'stEnergyPrefac',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 5e-3
        },
        {
            'name': 'Krho',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.06
        },
        {
            'name': 'Lbox',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'alphamax',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'alphamin',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.05
        },
        {
            'name': 'anglesExp',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 2
        },
        {
            'name': 'decay_constant',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.2
        },
        {
            'name': 'eps',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.005
        },
        {
            'name': 'epsilon',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1e-15
        },
        {
            'name': 'etaAcc',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.2
        },
        {
            'name': 'gamma',
            'type': 'f8',
            'old_name': 'gamma',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'gravConstant',
            'type': 'f8',
            'old_name': 'gravConstant',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'iteration',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'kernelChoice',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'mTotal',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'minDt',
            'type': 'f8',
            'old_name': 'minDt',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'minDt_m1',
            'type': 'f8',
            'old_name': 'minDt_m1',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'mui',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.62
        },
        {
            'name': 'muiConst',
            'type': 'f8',
            'old_name': 'muiConst',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'ng0',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 100
        },
        {
            'name': 'ngmax',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 150
        },
        {
            'name': 'numParticlesGlobal',
            'type': 'f8',
            'old_name': 'numParticlesGlobal',
            'old_type': 'int64',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'powerLawExp',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1.66667
        },

        {
            'name': 'rngSeed',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 251299
        },
        {
            'name': 'sincIndex',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 6
        },
        {
            'name': 'solWeight',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.5
        },

        {
            'name': 'stMachVelocity',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 4.5
        },
        {
            'name': 'stMaxModes',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 100000
        },
        {
            'name': 'stSpectForm',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },

        {
            'name': 'time',
            'type': 'f8',
            'old_name': 'time',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },

        {
            'name': 'turbulence',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'u0',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1000
        },
    ]

    attrs_group = [
        {
            'name': 'Kcour',
            'type': 'f8',
            'old_name': 'Kcour',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'Krho',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.06
        },
        {
            'name': 'box',
            'type': 'f8',
            'old_name': 'box',
            'old_type': 'f8',
            'is_arr': True,
            'length': 6,
            'default_val': None
        },
        {
            'name': 'eps',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.005
        },
        {
            'name': 'etaAcc',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.2
        },
        {
            'name': 'gamma',
            'type': 'f8',
            'old_name': 'gamma',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'gravConstant',
            'type': 'f8',
            'old_name': 'gravConstant',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'minDt',
            'type': 'f8',
            'old_name': 'minDt',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'minDt_m1',
            'type': 'f8',
            'old_name': 'minDt_m1',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'sincIndex',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 6
        },
        {
            'name': 'time',
            'type': 'f8',
            'old_name': 'time',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'turbulence::amplitudes',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': True,
            'length': 112,
            'default_val': [1.89495e-07, 1.89495e-07, 1.89495e-07, 1.89495e-07, 2, 2, 2, 2, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 1.89495e-07, 1.89495e-07, 1.89495e-07, 1.89495e-07, 2.29234, 2.29234, 2.29234, 2.29234, 1.7383, 1.7383, 1.7383, 1.7383, 2, 2, 2, 2, 1.7383, 1.7383, 1.7383, 1.7383, 0.792097, 0.792097, 0.792097, 0.792097, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 1.89495e-07, 1.89495e-07, 1.89495e-07, 1.89495e-07, 2.29234, 2.29234, 2.29234, 2.29234, 1.7383, 1.7383, 1.7383, 1.7383, 2.29234, 2.29234, 2.29234, 2.29234, 2.22495, 2.22495, 2.22495, 2.22495, 1.45873, 1.45873, 1.45873, 1.45873, 1.7383, 1.7383, 1.7383, 1.7383, 1.45873, 1.45873, 1.45873, 1.45873, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 2, 2, 2, 2, 1.7383, 1.7383, 1.7383, 1.7383, 0.792097, 0.792097, 0.792097, 0.792097, 1.7383, 1.7383, 1.7383, 1.7383, 1.45873, 1.45873, 1.45873, 1.45873, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 0.792097, 0.792097, 0.792097, 0.792097, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08, 6.3165e-08]
        },
        {
            'name': 'turbulence::decayTime',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.111111
        },
        {
            'name': 'turbulence::modes',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': True,
            'length': 336,
            'default_val': [0, 0, 6.28319, 0, -0, 6.28319, 0, 0, -6.28319, 0, -0, -6.28319, 0, 0, 12.5664, 0, -0, 12.5664, 0, 0, -12.5664, 0, -0, -12.5664, 0, 0, 18.8496, 0, -0, 18.8496, 0, 0, -18.8496, 0, -0, -18.8496, 0, 6.28319, 0, 0, -6.28319, 0, 0, 6.28319, -0, 0, -6.28319, -0, 0, 6.28319, 6.28319, 0, -6.28319, 6.28319, 0, 6.28319, -6.28319, 0, -6.28319, -6.28319, 0, 6.28319, 12.5664, 0, -6.28319, 12.5664, 0, 6.28319, -12.5664, 0, -6.28319, -12.5664, 0, 12.5664, 0, 0, -12.5664, 0, 0, 12.5664, -0, 0, -12.5664, -0, 0, 12.5664, 6.28319, 0, -12.5664, 6.28319, 0, 12.5664, -6.28319, 0, -12.5664, -6.28319, 0, 12.5664, 12.5664, 0, -12.5664, 12.5664, 0, 12.5664, -12.5664, 0, -12.5664, -12.5664, 0, 18.8496, 0, 0, -18.8496, 0, 0, 18.8496, -0, 0, -18.8496, -0, 6.28319, 0, 0, 6.28319, -0, 0, 6.28319, 0, -0, 6.28319, -0, -0, 6.28319, 0, 6.28319, 6.28319, -0, 6.28319, 6.28319, 0, -6.28319, 6.28319, -0, -6.28319, 6.28319, 0, 12.5664, 6.28319, -0, 12.5664, 6.28319, 0, -12.5664, 6.28319, -0, -12.5664, 6.28319, 6.28319, 0, 6.28319, -6.28319, 0, 6.28319, 6.28319, -0, 6.28319, -6.28319, -0, 6.28319, 6.28319, 6.28319, 6.28319, -6.28319, 6.28319, 6.28319, 6.28319, -6.28319, 6.28319, -6.28319, -6.28319, 6.28319, 6.28319, 12.5664, 6.28319, -6.28319, 12.5664, 6.28319, 6.28319, -12.5664, 6.28319, -6.28319, -12.5664, 6.28319, 12.5664, 0, 6.28319, -12.5664, 0, 6.28319, 12.5664, -0, 6.28319, -12.5664, -0, 6.28319, 12.5664, 6.28319, 6.28319, -12.5664, 6.28319, 6.28319, 12.5664, -6.28319, 6.28319, -12.5664, -6.28319, 6.28319, 12.5664, 12.5664, 6.28319, -12.5664, 12.5664, 6.28319, 12.5664, -12.5664, 6.28319, -12.5664, -12.5664, 12.5664, 0, 0, 12.5664, -0, 0, 12.5664, 0, -0, 12.5664, -0, -0, 12.5664, 0, 6.28319, 12.5664, -0, 6.28319, 12.5664, 0, -6.28319, 12.5664, -0, -6.28319, 12.5664, 0, 12.5664, 12.5664, -0, 12.5664, 12.5664, 0, -12.5664, 12.5664, -0, -12.5664, 12.5664, 6.28319, 0, 12.5664, -6.28319, 0, 12.5664, 6.28319, -0, 12.5664, -6.28319, -0, 12.5664, 6.28319, 6.28319, 12.5664, -6.28319, 6.28319, 12.5664, 6.28319, -6.28319, 12.5664, -6.28319, -6.28319, 12.5664, 6.28319, 12.5664, 12.5664, -6.28319, 12.5664, 12.5664, 6.28319, -12.5664, 12.5664, -6.28319, -12.5664, 12.5664, 12.5664, 0, 12.5664, -12.5664, 0, 12.5664, 12.5664, -0, 12.5664, -12.5664, -0, 12.5664, 12.5664, 6.28319, 12.5664, -12.5664, 6.28319, 12.5664, 12.5664, -6.28319, 12.5664, -12.5664, -6.28319, 18.8496, 0, 0, 18.8496, -0, 0, 18.8496, 0, -0, 18.8496, -0, -0]
        },
        {
            'name': 'turbulence::phases',
            'type': 'f8',
            'old_name': 'turbulencePhases',
            'old_type': 'f8',
            'is_arr': True,
            'length': 672,
            'default_val': None
        },
        {
            'name': 'turbulence::solWeight',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.5
        },
        {
            'name': 'turbulence::solWeightNorm',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 2
        },
        {
            'name': 'turbulence::variance',
            'type': 'f8',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 2.025
        },
        {
            'name': 'alphamax',
            'type': 'f4',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'alphamin',
            'type': 'f4',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.05
        },
        {
            'name': 'decay_constant',
            'type': 'f4',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 0.2
        },
        {
            'name': 'muiConst',
            'type': 'f4',
            'old_name': 'muiConst',
            'old_type': 'f8',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'boundaryType',
            'type': 'int8',
            'old_name': 'boundaryType',
            'old_type': 'int8',
            'is_arr': True,
            'length': 3,
            'default_val': None
        },
        # {
        #     'name': 'rngEngineState',
        #     'type': 'int8',
        #     'old_name': 'rngEngineState',
        #     'old_type': 'int8',
        #     'is_arr': True,
        #     'length': 6713,
        #     'default_val': None
        # },
        {
            'name': 'iteration',
            'type': 'int64',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'numParticlesGlobal',
            'type': 'int64',
            'old_name': 'numParticlesGlobal',
            'old_type': 'int64',
            'is_arr': False,
            'length': 1,
            'default_val': None
        },
        {
            'name': 'turbulence::numModes',
            'type': 'int64',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 112
        },
        {
            'name': 'kernelChoice',
            'type': 'int32',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 1
        },
        {
            'name': 'ng0',
            'type': 'int32',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 100
        },
        {
            'name': 'ngmax',
            'type': 'int32',
            'old_name': None,
            'old_type': None,
            'is_arr': False,
            'length': 1,
            'default_val': 150
        },
    ]

    # Copy all attrs in group
    for attr in attrs_group:
        data_type = attr['type']
        try:
            if attr['old_name'] is not None:
                # Copy from original
                # print(h5step.attrs.get(attr['old_name']))
                if attr['is_arr'] is True:
                    group.attrs[attr['name']] = np.array(h5step.attrs.get(attr['old_name']), dtype=data_type)
                else:
                    group.attrs[attr['name']] = np.array([h5step.attrs.get(attr['old_name'])], dtype=data_type)
            else:
                if attr['is_arr'] is True:
                    group.attrs[attr['name']] = np.array(attr['default_val'], dtype=data_type)
                else:
                    group.attrs[attr['name']] = np.array([attr['default_val']], dtype=data_type)
        except Exception as e:
            print(e)
            print(h5step.attrs.get(attr['old_name']))
            print(attr['name'], attr['old_name'])
    # Copy all attrs in file
    for attr in attrs:
        data_type = attr['type']
        try:
            if attr['old_name'] is not None:
                # Copy from original
                if attr['is_arr'] is True:
                    output_file.attrs[attr['name']] = np.array(h5step.attrs.get(attr['old_name']), dtype=data_type)
                else:
                    output_file.attrs[attr['name']] = np.array([h5step.attrs.get(attr['old_name'])], dtype=data_type)
            else:
                if attr['is_arr'] is True:
                    output_file.attrs[attr['name']] = np.array(attr['default_val'], dtype=data_type)
                else:
                    output_file.attrs[attr['name']] = np.array([attr['default_val']], dtype=data_type)
        except Exception as e:
            print(e)
            print(h5step.attrs.get(attr['old_name']))
            print(attr['name'], attr['old_name'])

    namelist_32 = ['alpha', 'du_m1', 'h', 'm', 'vx', 'vy', 'vz', 'x_m1',  'y_m1', 'z_m1']

    namelist_64 = ['temp', 'x','y', 'z']


    for name in namelist_32:
        dset = group.create_dataset(name, (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')

        for chunk in dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            dset[chunk] = np.array([h5step[name][curr_start_ind+start:curr_start_ind+stop]])
    
    for name in namelist_64:
        dset = group.create_dataset(name, (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f8')

        for chunk in dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            dset[chunk] = np.array([h5step[name][curr_start_ind+start:curr_start_ind+stop]])


    output_file.close()

# With current performance maybe we don't need MPI at all
# When calling, num of ranks == num of files
def splitDatasetParallel(input_file_path, step, output_initial):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    total_ranks = MPI.COMM_WORLD.Get_size()
    input_file = h5py.File(input_file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    splitDatasetSingleFile(input_file, step, rank, total_ranks, output_initial)

def splitDatasetSerial(input_file_path, step, output_initial):
    total_file_inds = 1
    input_file = h5py.File(input_file_path, 'r')
    for i in range(total_file_inds):
        splitDatasetSingleFile(input_file, step, i, total_file_inds, output_initial)
    input_file.close()

if __name__ == "__main__":
    # first cmdline argument: hdf5 file name
    input_path = sys.argv[1]
    # second cmdline argument: path and filename of output files
    output_initial = sys.argv[2]
    # third cmdline argument: hdf5 step number to extract and split data
    step = int(sys.argv[3])
    # third cmdline argument: split serially or parallelly -- serial by default
    mode = sys.argv[4]

    # python ./fill_attr.py ./3000c.h5 "/scratch/snx3000/yzhu/filled" 0 serial

    # Test data
    # input_path = "/home/appcell/unibas-sphexa/sphexa-vis/output/develop/res.h5"
    # output_initial = "/home/appcell/unibas-sphexa/sphexa-vis/output/develop/rescopy"
    # step = 0
    # mode = "serial"

    if mode == "serial":
        splitDatasetSerial(input_path, step, output_initial)
    elif mode == "parallel":
        splitDatasetParallel(input_path, step, output_initial)
    else:
        splitDatasetSerial(input_path, step, output_initial)
    sys.exit(0)
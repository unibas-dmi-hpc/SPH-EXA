import sys
import os
import math
import time
import h5py
from mpi4py import MPI
import numpy as np



# With current performance maybe we don't need MPI at all
def splitDataset(input_file_path, step):
    rank = MPI.COMM_WORLD.rank
    total_ranks = MPI.COMM_WORLD.Get_size()
    num_columns = 1
    chunk_size = math.floor(131072) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.

    # chunk_size = math.floor(131072/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_name = 'res_2/slice%05d.h5' % (rank) 
        os.remove(output_file_name)
    except:
        pass

    input_file = h5py.File(input_file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    h5step = input_file["Step#%s" % step]
    num_total_rows = len(h5step["z"])
    num_max_slice =  num_total_rows / total_ranks

    slice_ranges = [i for i in range(0, num_total_rows, int(num_total_rows/total_ranks))]
    if len(slice_ranges) == total_ranks:
        slice_ranges.append(num_total_rows)
    else:
        slice_ranges[-1] = num_total_rows

    curr_start_ind = slice_ranges[rank]
    curr_end_ind = slice_ranges[rank+1]

    output_file = h5py.File('res/slice%05d.h5' % (rank) , 'w')
    group = output_file.create_group('Step#0')
    x_dset = group.create_dataset('x', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    y_dset = group.create_dataset('y', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    z_dset = group.create_dataset('z', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    v_dset = group.create_dataset('v', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')

    for chunk in x_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        x_dset[chunk] = np.array([h5step["x"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in y_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        y_dset[chunk] = np.array([h5step["y"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in z_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        z_dset[chunk] = np.array([h5step["z"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in v_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        vx_arr = np.array([h5step["vx"][curr_start_ind+start:curr_start_ind+stop]])
        vy_arr = np.array([h5step["vy"][curr_start_ind+start:curr_start_ind+stop]])
        vz_arr = np.array([h5step["vz"][curr_start_ind+start:curr_start_ind+stop]])
        v_arr = np.sqrt(vx_arr*vx_arr+vy_arr*vy_arr+vz_arr*vz_arr)
        v_dset[chunk] = v_arr

    output_file.close()
    input_file.close()

def splitDatasetSingleFile(input_file, step, file_ind, total_file_inds):
    
    num_columns = 1
    chunk_size = math.floor(131072) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.

    # chunk_size = math.floor(131072/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_name = 'res/slice%05d.h5' % (file_ind) 
        os.remove(output_file_name)
    except:
        pass

    h5step = input_file["Step#%s" % step]
    num_total_rows = len(h5step["z"])
    num_max_slice =  num_total_rows / total_file_inds

    slice_ranges = [i for i in range(0, num_total_rows, int(num_total_rows/total_file_inds))]
    if len(slice_ranges) == total_file_inds:
        slice_ranges.append(num_total_rows)
    else:
        slice_ranges[-1] = num_total_rows

    curr_start_ind = slice_ranges[file_ind]
    curr_end_ind = slice_ranges[file_ind+1]

    output_file = h5py.File('res/slice%05d.h5' % (file_ind) , 'w')
    group = output_file.create_group('Step#0')
    x_dset = group.create_dataset('x', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    y_dset = group.create_dataset('y', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    z_dset = group.create_dataset('z', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    v_dset = group.create_dataset('v', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')

    for chunk in x_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        x_dset[chunk] = np.array([h5step["x"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in y_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        y_dset[chunk] = np.array([h5step["y"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in z_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        z_dset[chunk] = np.array([h5step["z"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in v_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        vx_arr = np.array([h5step["vx"][curr_start_ind+start:curr_start_ind+stop]])
        vy_arr = np.array([h5step["vy"][curr_start_ind+start:curr_start_ind+stop]])
        vz_arr = np.array([h5step["vz"][curr_start_ind+start:curr_start_ind+stop]])
        v_arr = np.sqrt(vx_arr*vx_arr+vy_arr*vy_arr+vz_arr*vz_arr)
        v_dset[chunk] = v_arr

    output_file.close()

def splitDatasetSingleThread(input_file_path, step):
    total_file_inds = 90
    input_file = h5py.File(input_file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    for i in range(total_file_inds):
        splitDatasetSingleFile(input_file, step, i, total_file_inds)
    input_file.close()

if __name__ == "__main__":
    # first cmdline argument: hdf5 file name
    fname = sys.argv[1]
    # second cmdline argument: hdf5 step number to extract and split data
    step = sys.argv[2]
    # third cmdline argument: split serially or parallelly -- serial by default
    options = sys.argv[3]
    if options == "serial":
        splitDatasetSingleThread(fname, step)
    elif options == "parallel":
        splitDataset(fname, step)
    else:
        splitDatasetSingleThread(fname, step)

    splitDatasetSingleThread(fname, step)
    sys.exit(0)

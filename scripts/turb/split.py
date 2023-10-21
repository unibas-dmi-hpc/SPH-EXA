import h5py
from mpi4py import MPI
import numpy as np
import sys
import os
import math
import time

rank = MPI.COMM_WORLD.rank
total_ranks = MPI.COMM_WORLD.Get_size()

def splitDataset(input_file_path, step):
    total_ranks = 20
    num_columns = 1
    chunk_size = math.floor(131072) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.

    # chunk_size = math.floor(131072/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_name = 'res/slice%05d.h5' % (rank) 
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
    vx_dset = group.create_dataset('vx', (num_max_slice,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')

    for chunk in x_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        # x_dset.resize(stop, axis=0)
        x_dset[chunk] = np.array([h5step["x"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in y_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        # x_dset.resize(stop, axis=0)
        y_dset[chunk] = np.array([h5step["y"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in z_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        # x_dset.resize(stop, axis=0)
        z_dset[chunk] = np.array([h5step["z"][curr_start_ind+start:curr_start_ind+stop]])

    for chunk in vx_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        # x_dset.resize(stop, axis=0)
        vx_dset[chunk] = np.array([h5step["vx"][curr_start_ind+start:curr_start_ind+stop]])

    output_file.close()
    input_file.close()

if __name__ == "__main__":
    # first cmdline argument: hdf5 file name
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]
    if step == "-p":
        printSteps(fname)
        sys.exit(1)

    splitDataset(fname, step)
    sys.exit(0)

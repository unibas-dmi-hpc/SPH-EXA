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
    x_dset = group.create_dataset('x', (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    y_dset = group.create_dataset('y', (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    z_dset = group.create_dataset('z', (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')
    v_dset = group.create_dataset('v', (curr_slice_size,), chunks=(chunk_size,), maxshape=(None,), dtype='f4')

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

# With current performance maybe we don't need MPI at all
# When calling, num of ranks == num of files
def splitDatasetParallel(input_file_path, step, output_initial):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    total_ranks = MPI.COMM_WORLD.Get_size()
    input_file = h5py.File(input_file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    splitDatasetSingleFile(input_file, step, rank, total_ranks, output_initial)

def splitDatasetSerial(input_file_path, step, output_initial):
    total_file_inds = 2
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

    # # Test data
    # input_path = "/home/appcell/demo_turbulence_151.h5"
    # output_initial = "/home/appcell/unibas/test_temp_res/extracted/res"
    # step = 1
    # mode = "serial"

    if mode == "serial":
        splitDatasetSerial(input_path, step, output_initial)
    elif mode == "parallel":
        splitDatasetParallel(input_path, step, output_initial)
    else:
        splitDatasetSerial(input_path, step, output_initial)
    sys.exit(0)

import sys
import os
import math
import time
import h5py
import numpy as np

def splitDatasetSingleFile(input_file_path, step, file_ind, total_file_inds, output_initial):
    chunk_size = math.floor(131072) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_name = output_initial+'%05d.txt' % (file_ind)
        os.remove(output_file_name)
    except:
        pass

    input_file = h5py.File(input_file_path+"%05d.h5" % (file_ind) , 'r')
    h5step = input_file["Step#%d" % step]
    num_total_rows = len(h5step["z"])

    output_file = open(output_file_name, 'w+')
    x_dset = h5step["x"]
    y_dset = h5step["y"]
    z_dset = h5step["z"]
    v_dset = h5step["v"]
    chunks = np.linspace(0, num_total_rows, 500, dtype="int")
    for i in range(499):
        x_chunk = np.array([x_dset[chunks[i]:chunks[i+1]]], dtype='f4')
        y_chunk = np.array([y_dset[chunks[i]:chunks[i+1]]], dtype='f4')
        z_chunk = np.array([z_dset[chunks[i]:chunks[i+1]]], dtype='f4')
        v_chunk = np.array([v_dset[chunks[i]:chunks[i+1]]], dtype='f4')
        result = np.vstack((x_chunk, y_chunk, z_chunk, v_chunk)).T
        np.savetxt(output_file, result, fmt='%.6f', delimiter=' ')
    output_file.close()
    input_file.close()

# With current performance maybe we don't need MPI at all
# When calling, num of ranks == num of files
def splitDatasetParallel(input_file_path, step, output_initial):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    total_ranks = MPI.COMM_WORLD.Get_size()

    splitDatasetSingleFile(input_file_path, step, rank, total_ranks, output_initial)

def splitDatasetSerial(input_file_path, step, output_initial):
    total_file_inds = 90
    for i in range(total_file_inds):
        splitDatasetSingleFile(input_file_path, step, i, total_file_inds, output_initial)

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
    # input_path = "/Users/zhu0002-adm/slice"
    # output_initial = "/Users/zhu0002-adm/unibas/res/slice"
    # step = 0
    # mode = "serial"

    if mode == "serial":
        splitDatasetSerial(input_path, step, output_initial)
    elif mode == "parallel":
        splitDatasetParallel(input_path, step, output_initial)
    else:
        splitDatasetSerial(input_path, step, output_initial)
    sys.exit(0)
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
    num_slices = 1000
    num_columns = 1
    chunk_size = math.floor(100000) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.

    # chunk_size = math.floor(131072/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    try:
        output_file_names = ['res/slice%05d_rank%02d.h5' % (slice_ind, rank) for slice_ind in range(num_slices)]
        for ofn in output_file_names:
            os.remove(ofn)
    except:
        pass

    input_file = h5py.File(input_file_path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    h5step = input_file["Step#%s" % step]
    num_total_rows = len(h5step["z"])
    num_max_slice =  2 * num_total_rows / (total_ranks * num_slices)

    output_files = [h5py.File('res/slice%05d_rank%02d.h5' % (slice_ind, rank) , 'w') for slice_ind in range(num_slices)]

    print("Created output files!")
    groups = [output_file.create_group('Step#0') for output_file in output_files]
    print("Created groups!")
    x_datasets = [group.create_dataset('x', (0,num_columns), maxshape=(None,num_columns), dtype='f4') for group in groups]
    y_datasets = [group.create_dataset('y', (0,num_columns), maxshape=(None,num_columns), dtype='f4') for group in groups]
    z_datasets = [group.create_dataset('z', (0,num_columns), maxshape=(None,num_columns), dtype='f4') for group in groups]
    vx_datasets = [group.create_dataset('vx', (0,num_columns), maxshape=(None,num_columns), dtype='f4') for group in groups]
    print("Created datasets!")


    rank_ranges = [i for i in range(0, num_total_rows, int(num_total_rows/total_ranks))]
    if len(rank_ranges) == total_ranks:
        rank_ranges.append(num_total_rows)
    else:
        rank_ranges[-1] = num_total_rows

    x_chunks = [np.zeros(chunk_size) for slice_ind in range(num_slices)]
    y_chunks = [np.zeros(chunk_size) for slice_ind in range(num_slices)]
    z_chunks = [np.zeros(chunk_size) for slice_ind in range(num_slices)]
    vx_chunks = [np.zeros(chunk_size) for slice_ind in range(num_slices)]
    print("Created chunks!")
    flag_chunks = [0 for slice_ind in range(num_slices)]
    ind_curr_chunks = [0 for slice_ind in range(num_slices)]

    start_range = np.linspace(-0.5, 0.49, num_slices)
    end_range = start_range + 0.01
    bin_edges = np.concatenate((start_range, [end_range[-1]]))
    print(f"Initialized!")

    def writeData(i):
        x_datasets[i].resize(ind_curr_chunks[i] + flag_chunks[i], axis=0)
        y_datasets[i].resize(ind_curr_chunks[i] + flag_chunks[i], axis=0)
        z_datasets[i].resize(ind_curr_chunks[i] + flag_chunks[i], axis=0)
        vx_datasets[i].resize(ind_curr_chunks[i] + flag_chunks[i], axis=0)

        x_datasets[i][ind_curr_chunks[i]:(ind_curr_chunks[i] + flag_chunks[i])] = x_chunks[i][:flag_chunks[i]].reshape((flag_chunks[i], 1))
        y_datasets[i][ind_curr_chunks[i]:(ind_curr_chunks[i] + flag_chunks[i])] = y_chunks[i][:flag_chunks[i]].reshape((flag_chunks[i], 1))
        z_datasets[i][ind_curr_chunks[i]:(ind_curr_chunks[i] + flag_chunks[i])] = z_chunks[i][:flag_chunks[i]].reshape((flag_chunks[i], 1))
        vx_datasets[i][ind_curr_chunks[i]:(ind_curr_chunks[i] + flag_chunks[i])] = vx_chunks[i][:flag_chunks[i]].reshape((flag_chunks[i], 1))
        flag_chunks[i] = 0
        ind_curr_chunks[i] = ind_curr_chunks[i] + chunk_size

        # output_files[i].flush()

    start_time = time.time()
    for i in range(rank_ranges[rank], rank_ranges[rank+1]):
        if (i % chunk_size == 0):
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Rank {rank} finished: {i - rank_ranges[rank]}/{rank_ranges[rank+1] - rank_ranges[rank]}, using {elapsed_time} seconds.")
        bin_index = np.digitize(h5step['z'][i], bin_edges) - 1
        bin_index = bin_index[0]
        if bin_index >= len(flag_chunks):
            bin_index = len(flag_chunks) - 1
        if flag_chunks[bin_index] == chunk_size:
            writeData(bin_index)
        x_chunks[bin_index][flag_chunks[bin_index]] = h5step['x'][i]
        y_chunks[bin_index][flag_chunks[bin_index]] = h5step['y'][i]
        z_chunks[bin_index][flag_chunks[bin_index]] = h5step['z'][i]
        vx_chunks[bin_index][flag_chunks[bin_index]] = h5step['vx'][i]
        flag_chunks[bin_index] = flag_chunks[bin_index] +1

    for bin_index in range(num_slices):
        writeData(bin_index)

    for f in output_files:
        f.close()
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

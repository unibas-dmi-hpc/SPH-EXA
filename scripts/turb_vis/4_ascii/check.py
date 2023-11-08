import sys
import os
import math
import time
import h5py
import numpy as np


def getlen(input_file):
    input_file = open(input_file, 'r')
    line_count = sum(1 for line in input_file)
    input_file.close()
    return line_count

def splitDatasetSingleFile(input_file_path, step, file_ind, output_initial):
    try:
        output_file_name = output_initial+'%05d.txt' % (file_ind)
    except:
        pass

    input_file = h5py.File(input_file_path+"%05d.h5" % (file_ind) , 'r')
    h5step = input_file["Step#%d" % step]
    num_total_rows = len(h5step["z"])

    
    x_dset = h5step["x"]
    y_dset = h5step["y"]
    z_dset = h5step["z"]
    v_dset = h5step["v"]
    start = getlen(output_file_name)
    output_file = open(output_file_name, 'a+')
    print(f"Start: {start}, end: {num_total_rows} in file {file_ind}")
    end = num_total_rows
    x_chunk = np.array([x_dset[start:end]], dtype='f4')
    y_chunk = np.array([y_dset[start:end]], dtype='f4')
    z_chunk = np.array([z_dset[start:end]], dtype='f4')
    v_chunk = np.array([v_dset[start:end]], dtype='f4')
    result = np.vstack((x_chunk, y_chunk, z_chunk, v_chunk)).T
    np.savetxt(output_file, result, fmt='%.6f', delimiter=' ')
    output_file.close()
    input_file.close()

def splitDatasetSerial(input_file_path, step, output_initial):
    inds = [33, 39, 41, 49, 53, 54,55,56,57,58,59,60]
    for i in inds:
        splitDatasetSingleFile(input_file_path, step, i, output_initial)

if __name__ == "__main__":
    # first cmdline argument: hdf5 file name
    input_path = "/users/staff/uniadm/zhu0002/extract/slice"
    # second cmdline argument: path and filename of output files
    output_initial = "/storage/shared/projects/sph-exa/data/slice"
    splitDatasetSerial(input_path, 0, output_initial)

#!/usr/bin/env python3

import h5py
from mpi4py import MPI
import numpy as np
import os
import math
import sys

rank = MPI.COMM_WORLD.rank


def printSteps(fname):
    """ Display contents of HDF5 file: step, iteration and time """
    ifile = h5py.File(fname, "r")
    print(fname, "contains the following steps:")
    print("hdf5 step number".rjust(15), "sph iteration".rjust(15), "time".rjust(15))
    for i in range(len(list(ifile["/"]))):
        h5step = ifile["Step#%d" % i]
        print("%5d".rjust(14) % i, "%5d".rjust(14) % h5step.attrs["step"][0],
              "%5f".rjust(14) % h5step.attrs["time"][0])


def readStep(fname, step):
    ifile = h5py.File(fname, "r", driver='mpio', comm=MPI.COMM_WORLD)
    print("Rank: ", rank)
    try:
        h5step = ifile["Step#%s" % step]
        return h5step, ifile
    except KeyError:
        print(fname, "step %s not found" % step)
        printSteps(fname)
        sys.exit(1)

def plotSlice(fname, step):
    """ Plot a 2D xy-cross section with particles e.g. abs(z) < 0.1, using density as color """
    outFilename = "turb_fields_app.h5"
    h5step, ifile = readStep(fname, step)
    try:
        os.remove(outFilename)
    except:
        pass
    
    outFile = h5py.File(outFilename, "a", driver='mpio', comm=MPI.COMM_WORLD)
    grp = outFile.create_group('Step#0')

    num_total_rows = len(h5step["z"])
    # For position
    num_columns = 1
    chunk_size = math.floor(131072*2/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    x_dset = grp.create_dataset('x', (num_total_rows, ), chunks=(chunk_size, ), maxshape=(num_total_rows, ), dtype='f4')
    y_dset = grp.create_dataset('y', (num_total_rows, ), chunks=(chunk_size, ), maxshape=(num_total_rows, ), dtype='f4')
    z_dset = grp.create_dataset('z', (num_total_rows, ), chunks=(chunk_size, ), maxshape=(num_total_rows, ), dtype='f4')
    vx_dset = grp.create_dataset('vx', (num_total_rows, ), chunks=(chunk_size, ), maxshape=(num_total_rows, ), dtype='f4')
    if rank == 0:
        for chunk in x_dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            print(f"Current progress: {stop}/27 billion")
            # x_dset.resize(stop, axis=0)
            x_dset[chunk] = np.array([h5step["x"][start:stop]])

    if rank == 1:
        for chunk in y_dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            # y_dset.resize(stop, axis=0)
            y_dset[chunk] = np.array([h5step["y"][start:stop]])

    if rank == 2:
        for chunk in z_dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            # z_dset.resize(stop, axis=0)
            z_dset[chunk] = np.array([h5step["z"][start:stop]])

    if rank == 3:
        for chunk in vx_dset.iter_chunks():
            start = chunk[0].start
            stop = chunk[0].stop
            # vx_dset.resize(stop, axis=0)
            vx_dset[chunk] = np.array([h5step["vx"][start:stop]])
    
    outFile.flush()
    outFile.close()
    ifile.close()


if __name__ == "__main__":
    # first cmdline argument: hdf5 file name to plot
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]
    if step == "-p":
        printSteps(fname)
        sys.exit(1)

    plotSlice(fname, step)
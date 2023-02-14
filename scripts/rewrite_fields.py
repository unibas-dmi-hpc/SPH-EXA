#!/usr/bin/env python3

import h5py
import numpy as np
import os
import math
import sys


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
    ifile = h5py.File(fname, "r")
    try:
        h5step = ifile["Step#%s" % step]
        return h5step
    except KeyError:
        print(fname, "step %s not found" % step)
        printSteps(fname)
        sys.exit(1)

def plotSlice(fname, step):
    """ Plot a 2D xy-cross section with particles e.g. abs(z) < 0.1, using density as color """

    h5step = readStep(fname, step)
    try:
        os.remove("turb_fields_app.h5")
    except:
        pass
    outFilename = "turb_fields_app.h5"
    outFile = h5py.File(outFilename, "a")
    grp = outFile.create_group('Step#0')

    num_total_rows = len(h5step["z"])
    # For position
    num_columns = 3
    chunk_size = math.floor(131072/num_columns) # An optimal chunk size is ~1MB, i.e. 131072 64bit float.
    pos_dset = grp.create_dataset('pos', (num_total_rows, 3), chunks=(chunk_size, 3), maxshape=(num_total_rows, 3), dtype='f8')
    for chunk in pos_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        pos_dset.resize(stop, axis=0)
        pos_dset[chunk] = np.array([h5step["x"][start:stop], h5step["y"][start:stop], h5step["z"][start:stop]]).transpose()

    # For velocity -- auto chunking because it's scalar data
    vel_norm_dset = grp.create_dataset('v', (num_total_rows, 1), chunks=True, maxshape=(num_total_rows, 1), dtype='f8')
    for chunk in vel_norm_dset.iter_chunks():
        start = chunk[0].start
        stop = chunk[0].stop
        vx = np.array(h5step["vx"][start:stop])
        vy = np.array(h5step["vy"][start:stop])
        vz = np.array(h5step["vz"][start:stop])
        vel_norm_dset.resize(stop, axis=0)
        # Here the output of np.sqrt is a scalar array. It has to be reshaped into a nx1 matrix before broadcasting
        vel_norm_dset[chunk] = np.atleast_1d(np.sqrt(vx*vx + vy*vy + vz*vz).reshape(-1, 1))
    
    outFile.flush()
    outFile.close()


if __name__ == "__main__":
    # first cmdline argument: hdf5 file name to plot
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]
    if step == "-p":
        printSteps(fname)
        sys.exit(1)

    plotSlice(fname, step)
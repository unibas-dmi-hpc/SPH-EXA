#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt

import sys


def printSteps(fname):
    """ Display contents of HDF5 file: step, iteration and time """
    ifile = h5py.File(fname, "r")
    print(fname, "contains the following steps:")
    print("hdf5 step number".rjust(15), "sph iteration".rjust(15), "time".rjust(15))
    for i in range(len(list(ifile["/"]))):
        h5step = ifile["Step#%d" % i]
        print("%5d".rjust(14) % i, "%5d".rjust(14) % h5step.attrs["iteration"][0],
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

    x = np.array(h5step["x"])
    y = np.array(h5step["y"])
    z = np.array(h5step["z"])

    rho = np.array(h5step["rho"])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')

    mask = abs(z) < 0.1
    # mask = (0.27 < z) & (z < 0.35)

    cm = plt.cm.get_cmap('plasma')

    plabel = fname + ", time = %3f" % h5step.attrs["time"][0] + " N = %d" % len(x)

    im = ax.scatter(x[mask], y[mask], c=rho[mask], s=10.0, cmap=cm, vmin=0, vmax=8, label=plabel)
    fig.colorbar(im)

    plt.legend(loc="lower left")
    # plt.savefig("slice_%s_%3f.png" % (fname, h5step.attrs["time"][0]), format="png")
    plt.show()


if __name__ == "__main__":
    # first cmdline argument: hdf5 file name to plot
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]
    if step == "-p":
        printSteps(fname)
        sys.exit(1)

    plotSlice(fname, step)

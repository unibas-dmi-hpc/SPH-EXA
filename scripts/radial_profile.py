#!/usr/bin/env python3

from argparse import ArgumentParser
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


def radialProfile(h5step, what):
    """ Plot particle radii against some other quantity """
    x = np.array(h5step["x"])
    y = np.array(h5step["y"])
    z = np.array(h5step["z"])
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    quantity = 0
    if what == "v":
        vx = np.array(h5step["vx"])
        vy = np.array(h5step["vy"])
        vz = np.array(h5step["vz"])
        # radial projection of velocity
        quantity = (vx * x + vy * y + vz * z) / radius
    else:
        quantity = np.array(h5step[what])

    return radius, quantity


def plotProfile(fnames, step, what):
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    plt.ylabel(what)
    plt.xlabel("radius")

    for fn in fnames:
        h5step = readStep(fn, step)
        r1, q1 = radialProfile(h5step, what)
        ax.scatter(r1, q1, s=0.1, label=fn + " time = %3f" % h5step.attrs["time"][0])

    plt.legend(loc="lower left")
    # plt.savefig("rad_%s.png" % what, format="png")
    plt.show()


if __name__ == "__main__":
    # all cmdline arguments except the last are hdf5 files to plot
    files = sys.argv[1:-1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[-1]
    if step == "-p":
        for fname in files:
            printSteps(fname)
        sys.exit(1)

    # the thing to plot against the particle radii, e.g. rho, p, c, v, etc.
    quantity = "rho"

    plotProfile(files, step, quantity)

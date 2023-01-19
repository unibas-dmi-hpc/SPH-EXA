import h5py
import numpy as np
import sys

# usage: python add_m1.py <hdf5-output-file>

inputFilename = sys.argv[1]
h5File = h5py.File(inputFilename, "r+")

latestH5Step = list(h5File["/"])[-1]
stepHandle = h5File[latestH5Step]

if "x_m1" not in stepHandle:
    print("Adding x_m1 to SPH iteration %d" % stepHandle.attrs["iteration"])
    stepHandle["x_m1"] = np.array(stepHandle["vx"] * stepHandle.attrs["minDt"], dtype=np.float32)

if "y_m1" not in stepHandle:
    print("Adding y_m1 to SPH iteration %d" % stepHandle.attrs["iteration"])
    stepHandle["y_m1"] = np.array(stepHandle["vy"] * stepHandle.attrs["minDt"], dtype=np.float32)

if "z_m1" not in stepHandle:
    print("Adding z_m1 to SPH iteration %d" % stepHandle.attrs["iteration"])
    stepHandle["z_m1"] = np.array(stepHandle["vz"] * stepHandle.attrs["minDt"], dtype=np.float32)

if "du_m1" not in stepHandle:
    print("Adding du_m1 to SPH iteration %d" % stepHandle.attrs["iteration"])
    stepHandle["du_m1"] = np.zeros(stepHandle.attrs["numParticlesGlobal"], dtype=np.float32)

h5File.close()

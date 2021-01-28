import sys
import numpy as np

if len(sys.argv) != 2:
    print("Incorrect number of parameters. Correct usage: ./{} dat_filename_without_extension".format(sys.argv[0]))
    exit()

filename = sys.argv[1]
data = np.loadtxt(filename + '.dat')
data.T.tofile(filename + '.bin')

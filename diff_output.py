import sys
import numpy as np

if len(sys.argv) != 3:
    print("Incorrect number of parameters. Correct usage: ./{} dat_filename_without_extension".format(sys.argv[0]))
    exit()

file1 = sys.argv[1]
file2 = sys.argv[2]
data1 = np.fromfile(file1)
data2 = np.fromfile(file1)

if np.allclose(data1, data2):
    print("Outputs are similar! Test passed.")
else:
    print("Outputs difference is too great. Test failed.")

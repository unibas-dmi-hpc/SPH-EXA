import numpy as np
import h5py
import time, math
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Calculate volume-weighted PDF from density data in an HDF5 file.')
parser.add_argument('filename', type=str, help='Path to the HDF5 file containing density data.')
parser.add_argument('num_bins', type=int, help='Number of bins for the PDF calculation.')
parser.add_argument('outfile', type=str, help='Path to the output file')

# Parse arguments
args = parser.parse_args()

#file_name='200.h5'
# Record the start time
start_time = time.time()
print("Reading")

input_file = h5py.File(args.filename, 'r')
h5step = input_file["Step#0"]
N = len(h5step["rho"])
chunk_size = math.floor(131072)
min_rho = 1
max_rho = -1

total_file_inds = 1
slice_ranges = [i for i in range(0, N, chunk_size)]
curr_cnt = 0

for (ind, slice_start) in enumerate(slice_ranges):
    if ind + 1 < len(slice_ranges):
        curr_slice = np.array([h5step['rho'][slice_ranges[ind]:slice_ranges[ind+1]]])
    else:
        curr_slice = np.array([h5step['rho'][slice_ranges[ind]:N]])
    if int(slice_ranges[ind] / 1000000000) > curr_cnt:
        print(f"calculating limits at {slice_ranges[ind]}")
        curr_cnt = curr_cnt + 1
    if curr_slice.min() < min_rho:
        min_rho = curr_slice.min()
    if curr_slice.max() > max_rho:
        max_rho = curr_slice.max()
print(f"calculating limits finished, min: {min_rho}, max: {max_rho}")


# Define the bins for density
bins = np.linspace(min_rho, max_rho, num=args.num_bins)
res_arr = []
bin_edges = []
curr_cnt = 0
for (ind, slice_start) in enumerate(slice_ranges):
    if ind + 1 < len(slice_ranges):
        curr_slice = np.array([h5step['rho'][slice_ranges[ind]:slice_ranges[ind+1]]])
    else:
        curr_slice = np.array([h5step['rho'][slice_ranges[ind]:N]])
    if int(slice_ranges[ind] / 1000000000) > curr_cnt:
        print(f"calculating hists at {slice_ranges[ind]}")
        curr_cnt = curr_cnt + 1
    
    hist, bin_edges = np.histogram(curr_slice, bins=bins, density=False)
    if len(res_arr) is 0:
        res_arr = hist
    else:
        res_arr += hist
print(f"calculating hists finished")

# Normalize the histogram to form a PDF
pdf = res_arr #/ hist.sum()

print(f'{res_arr.sum()}')

pdf.astype('f8')

# bin_centers for plotting or analysis
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Output results to a text file
with open(args.outfile, 'w') as f:
    f.write('bin_centers\tPDF\n')  # Writing header
    for center, value in zip(bin_centers, pdf):
        f.write(f'{center}\t{value}\n')  # Writing data

print(f'Results saved to {args.outfile}')
import numpy as np
import h5py
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys
import threading
import scipy.fft


def readStep(fname, step):
    ifile = h5py.File(fname, "r")
    try:
        h5step = ifile["Step#%s" % step]
        return h5step
    except KeyError:
        print(fname, "step %s not found" % step)
        sys.exit(1)


def spherical_average(k_values, k_1d, power_spectrum, power_spectrum_radial, count, numThreads, thread_index):
    outer_loop_step = int(len(k_1d) / numThreads)
    for ind in range(outer_loop_step):
        i = ind + thread_index * outer_loop_step
        for j in range(len(k_1d)):
            for l in range(len(k_1d)):
                k = np.sqrt(k_values[i] ** 2 + k_values[j]
                            ** 2 + k_values[l] ** 2)
                k_index = np.argmin(np.abs(k_1d - k))
                count[thread_index, k_index] += 1.0
                power_spectrum_radial[thread_index,
                                      k_index] += power_spectrum[i, j, l]


def rasterAndSpectra(fname, step, gridSize, numT):
    # Record the start time
    start_time = time.time()

    file_name = fname
    numThreads = int(numT)
    print("Number of threads = ", numThreads)

    print("Reading")
    h5step = readStep(file_name, step)

    x = np.array(h5step["x"])
    y = np.array(h5step["y"])
    z = np.array(h5step["z"])
    vx = np.array(h5step["vx"])
    vy = np.array(h5step["vy"])
    vz = np.array(h5step["vz"])

    end_time1 = time.time()
    elapsed = end_time1 - start_time
    print(f"Reading done. Time: {elapsed} s")

    # Create a 3D Cartesian mesh
    mesh_resolution = int(gridSize)
    mesh_x, mesh_y, mesh_z = np.meshgrid(np.linspace(-0.5, 0.5, mesh_resolution),
                                         np.linspace(-0.5, 0.5,
                                                     mesh_resolution),
                                         np.linspace(-0.5, 0.5, mesh_resolution))

    # Convert mesh coordinates to a flat array
    target_grid_points = np.column_stack(
        (mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()))
    end_time2 = time.time()
    elapsed = end_time2 - end_time1
    print(f"Creating Cartesian mesh. Time: {elapsed} s")

    # Create KD-Tree from scattered points
    scattered_points = np.column_stack((x, y, z))
    kdtree = cKDTree(scattered_points)
    end_time3 = time.time()
    elapsed = end_time3 - end_time2
    print(f"Creating KD-tree. Time: {elapsed} s")

    # Query the KD-Tree for nearest neighbors
    distances, indices = kdtree.query(
        target_grid_points, k=1, workers=numThreads)
    end_time4 = time.time()
    elapsed = end_time4 - end_time3
    print(f"Querying KD-tree. Time: {elapsed} s")

    # Interpolate values based on the nearest neighbors
    interpolated_values = np.column_stack(
        (vx[indices], vy[indices], vz[indices]))

    # Reshape the interpolated values to match the shape of the target grid
    interpolated_values_reshaped = interpolated_values.reshape(
        (mesh_resolution, mesh_resolution, mesh_resolution, 3))
    end_time5 = time.time()
    elapsed = end_time5 - end_time4
    print(f"Interpolating. Time: {elapsed} s")

    # Number of grid points in each dimension
    grid_size = mesh_resolution
    full_grid = grid_size**3

    # Compute the velocity field
    vx_field = interpolated_values[:, 0].reshape(
        (grid_size, grid_size, grid_size))
    vy_field = interpolated_values[:, 1].reshape(
        (grid_size, grid_size, grid_size))
    vz_field = interpolated_values[:, 2].reshape(
        (grid_size, grid_size, grid_size))
    print("Calculating means")

    # Calculate the mean along each spatial dimension
    mean_vx = np.mean(vx_field, axis=(0, 1, 2))
    mean_vy = np.mean(vy_field, axis=(0, 1, 2))
    mean_vz = np.mean(vz_field, axis=(0, 1, 2))
    print(mean_vx)
    print(mean_vy)
    print(mean_vz)

    print("Calculating FFTs")
    vx_fft = scipy.fft.fftn(vx_field, workers=numThreads)/full_grid
    vy_fft = scipy.fft.fftn(vy_field, workers=numThreads)/full_grid
    vz_fft = scipy.fft.fftn(vz_field, workers=numThreads)/full_grid

    # write values of the FFTs to a file
    # output_file_path = 'fft_data_200.txt'
    # with open(output_file_path, 'w') as output_file:
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             for k in range(grid_size):
    #                 output_file.write(
    #                     f"{vx_fft[i, j, k]}, {vy_fft[i, j, k]}, {vz_fft[i, j, k]}\n")

    print("Calculating Power Spectra")
    power_spectrum = (np.abs(vx_fft) ** 2 + np.abs(vy_fft)
                      ** 2 + np.abs(vz_fft) ** 2)
    end_time6 = time.time()
    elapsed = end_time6 - end_time5
    print(f"Calculating Power Spectra. Time: {elapsed} s")

    # Compute the 1D wavenumber array
    k_values = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    k_1d = np.abs(k_values)

    # Perform spherical averaging to get 1D power spectrum
    print("Spherical averaging")
    threads = [None] * numThreads
    power_spectrum_th = np.zeros((numThreads, len(k_1d)))
    counts_th = np.zeros((numThreads, len(k_1d)))
    for thread_index in range(len(threads)):
        threads[thread_index] = threading.Thread(target=spherical_average, args=(
            k_values, k_1d, power_spectrum, power_spectrum_th, counts_th, numThreads, thread_index))
        threads[thread_index].start()

    for i in range(len(threads)):
        threads[i].join()

    power_spectrum_radial = np.zeros_like(k_1d)
    count = np.zeros_like(k_1d)
    for i in range(len(threads)):
        power_spectrum_radial += power_spectrum_th[i]
        count += counts_th[i]

    for i in range(len(k_1d)):
        power_spectrum_radial[i] = (
            power_spectrum_radial[i] * 4.0 * np.pi * k_1d[i]**2) / count[i]

    end_time7 = time.time()
    elapsed = end_time7 - end_time6
    print(f"Spherical averaging. Time: {elapsed} s")

    # Total energy of the 3D FFT
    sum_PS = np.sum(power_spectrum)
    print(sum_PS)
    # Total energy of the power spectra
    print(np.sum(power_spectrum_radial[count > 0]))

    print("Outputing...")

    # Save 1D spectra and k to a file
    output_file_path = "power_spectrum_data_analytical_%s.txt" % gridSize
    np.savetxt(output_file_path, np.column_stack(
        (k_1d[k_values > 0], power_spectrum_radial[k_values > 0])))

    end_time8 = time.time()
    elapsed = end_time8 - end_time7
    print(f"Outputing. Time: {elapsed} s")

    # Plot the 1D power spectrum
    print("Plotting")
    plt.plot(k_1d[1:], power_spectrum_radial[1:])
    plt.xscale('log')
    plt.yscale('log')

    # Set x-axis limit to start from 6
    plt.xlim(6, k_1d.max())

    # Add a line for the expected Kolmogorov slope of -5/3
    kolmogorov_line = k_1d[1:]**(-5.0/3.0)
    plt.plot(k_1d[1:], kolmogorov_line,
             label='Kolmogorov slope (-5/3)', linestyle='--')

    plt.xlabel('Wavenumber (k)')
    plt.ylabel('E_k')
    plt.title('Power Spectrum 100^3')

    # Add legend
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig("power_spectrum_%s.png" % gridSize)

    print("Done!")
    elapsed = end_time8 - start_time
    print(f"Total elapsed time: {elapsed} s")


if __name__ == "__main__":
    # first cmdline argument: hdf5 file name to plot
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]

    # third cmdline argument: size of grid to interpolate to
    gridSize = sys.argv[3]

    # fourth cmdline argument: number of threads to use
    numT = sys.argv[4]

    rasterAndSpectra(fname, step, gridSize, numT)
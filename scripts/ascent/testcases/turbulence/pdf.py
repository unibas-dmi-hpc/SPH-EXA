import yaml #pip install --user pyyaml
import numpy as np
import matplotlib.pyplot as plt
import os


file_path = "/home/appcell/unibas-sphexa/sphexa-vis/output/develop/turbulence_pdf_session.yaml"
def file_exists(file_path):
    return os.path.exists(file_path)

def plot_binning(curr_key, binning):
    curr_data = binning[curr_key]['attrs']['value']['value']

    # create the coordinate axis using bin centers
    x_axis = binning[curr_key]['attrs']['bin_axes']['value']['X']
    y_axis = binning[curr_key]['attrs']['bin_axes']['value']['Y']
    x_min = x_axis['min_val']
    x_max = x_axis['max_val']
    y_min = y_axis['min_val']
    y_max = y_axis['max_val']
    x_bins = x_axis['num_bins']
    y_bins = y_axis['num_bins']

    x_values = np.linspace(x_min, x_max, x_bins)  # Example x-axis values
    y_values = np.linspace(y_min, y_max, y_bins)  # Example y-axis values

    curr_data = np.array(curr_data).reshape((x_bins, y_bins))
    plt.imshow(curr_data, extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()], origin='lower', cmap='hot')
    plt.colorbar()  # Add color bar
    plt.savefig("sedov_step{:05d}.png".format(curr_key))
    plt.clf()

def plot_radial_density(curr_key, radial_density):
    print(curr_key)
    curr_data = radial_density[curr_key]['attrs']['value']['value']

    # create the coordinate axis using bin centers
    x_axis = radial_density[curr_key]['attrs']['bin_axes']['value']['Radius']
    x_min = x_axis['min_val']
    x_max = x_axis['max_val']
    x_bins = x_axis['num_bins']

    x_delta = (x_max - x_min) / float(x_bins)
    x_start = x_min + 0.5 * x_delta
    x_vals = []
    for b in range(0,x_bins):
        x_vals.append(b * x_delta + x_start)

    # plot the curve from the last cycle
    plt.plot(x_vals, curr_data)
    plt.xlabel('Radial distance')
    plt.ylabel('Density')
    plt.savefig("turbulence_pdf_step{:05d}.png".format(curr_key))
    plt.clf()

if file_exists(file_path):
    session = []
    file = open(file_path)
    session = yaml.load(file, Loader=yaml.FullLoader)
    radial_density = session['RadialDensity']
    sorted_keys = sorted(radial_density.keys())
    for curr_key in sorted_keys:
        plot_radial_density(curr_key, radial_density)
else:
    print(os.getcwd())
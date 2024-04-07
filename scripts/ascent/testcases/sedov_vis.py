


file_path = "sedov_session.yaml"

try:
    count = count + 1
except NameError:
    import os
    def file_exists(file_path):
        return os.path.exists(file_path)
    if file_exists(file_path):
        os.remove(file_path)
    count = 0

if count == 0:
    import conduit
    import yaml #pip install --user pyyaml
    import numpy as np
    import matplotlib.pyplot as plt


if file_exists(file_path):
    session = []
    file = open(r'sedov_session.yaml')
    session = yaml.load(file, Loader=yaml.FullLoader)
    binning = session['Pressure_binned']
    sorted_keys = sorted(binning.keys())
    curr_key = sorted_keys[int(count) - 1]
    curr_data = binning[curr_key]['attrs']['value']['value']

    # create the coordinate axis using bin centers
    x_axis = binning[curr_key]['attrs']['bin_axes']['value']['X']
    y_axis = binning[curr_key]['attrs']['bin_axes']['value']['Y']
    x_min = x_axis['min_val']
    x_max = x_axis['max_val']
    y_min = y_axis['min_val']
    y_max = y_axis['max_val']
    print(x_min, x_max)
    # x_min = -0.5
    # x_max = 0.5
    # y_min = -0.5
    # y_max = 0.5
    x_bins = x_axis['num_bins']
    y_bins = y_axis['num_bins']

    x_values = np.linspace(x_min, x_max, x_bins)  # Example x-axis values
    y_values = np.linspace(y_min, y_max, y_bins)  # Example y-axis values

    curr_data = np.array(curr_data).reshape((x_bins, y_bins))
    plt.imshow(curr_data, extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()], origin='lower', cmap='hot')
    plt.colorbar()  # Add color bar
    plt.savefig("sedov_step{:05d}.png".format(curr_key))
    plt.clf()
else:
    pass
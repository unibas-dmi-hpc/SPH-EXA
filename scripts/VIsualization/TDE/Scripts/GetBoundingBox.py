import h5py
import numpy as np

def calculate_averages(hdf5_file_path):
    res_center = {}
    res_min = {}
    res_max = {}
    res_scale = {}
    with h5py.File(hdf5_file_path, 'r') as file:
        # Iterate through each group in the file
        for group_name in file:
            group = file[group_name]

            # Check if the datasets "x", "y", "z" exist in the group
            if all(dataset in group for dataset in ['x', 'y', 'z']):
                # Retrieve datasets "x", "y", "z"
                x = group['x'][:]
                y = group['y'][:]
                z = group['z'][:]

                # Calculate the average value of x, y, z
                avg_x = np.mean(x)
                avg_y = np.mean(y)
                avg_z = np.mean(z)

                min_x = np.min(x)
                min_y = np.min(y)
                min_z = np.min(z)

                max_x = np.max(x)
                max_y = np.max(y)
                max_z = np.max(z)


                # Print the results
                # print(f"{group_name}: [{avg_x}, {avg_y}, {avg_z}]\n")
                group_id = int(group_name[5:])
                res_center[group_id] = [avg_x, avg_y, avg_z]
                res_min[group_id] = [min_x, min_y, min_z]
                res_max[group_id] = [max_x, max_y, max_z]
                res_scale[group_id] = [max_x - min_x, max_y - min_y, max_z - min_z]
            else:
                print(f"Group: {group_name} does not contain datasets 'x', 'y', 'z'\n")
    return res_center, res_min, res_max, res_scale


if __name__ == "__main__":
    hdf5_file_path = '/home/appcell/tde_snapshot00000.h5'  # Replace with your HDF5 file path
    calculate_averages(hdf5_file_path)

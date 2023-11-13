import h5py
import numpy as np
import sys

def prune(input_path, step, output_path, len_after_prune):
    original_file = h5py.File(input_path, 'r')
    h5step = original_file["Step#%d" % step]
    x_dataset = h5step['x']
    # y_dataset = original_file['y']
    # z_dataset = original_file['z']
    # vx_dataset = original_file['vx']
    total_entries = len(x_dataset)
    total_entries_after_prune = len_after_prune*len_after_prune*len_after_prune
    # ratio = total_entries_after_prune / total_entries
    # ratio_inverse = total_entries / total_entries_after_prune
    chunks = np.linspace(0, total_entries, 2701, dtype="int")
    kept_per_chunk = np.linspace(0, total_entries_after_prune, 2701, dtype="int")
    # chunk_size = kept_per_chunk[1] - kept_per_chunk[0]

    output_file = h5py.File(output_path, 'w')
    group = output_file.create_group('Step#0')
    x_dset = group.create_dataset('x', (total_entries_after_prune,), maxshape=(None,), dtype='f4')
    y_dset = group.create_dataset('y', (total_entries_after_prune,), maxshape=(None,), dtype='f4')
    z_dset = group.create_dataset('z', (total_entries_after_prune,), maxshape=(None,), dtype='f4')
    vx_dset = group.create_dataset('vx', (total_entries_after_prune,), maxshape=(None,), dtype='f4')

    for i in range(len(chunks) - 1):
        curr_len = chunks[i+1] - chunks[i]
        indices = np.random.choice(curr_len, size=kept_per_chunk[i+1]-kept_per_chunk[i], replace=False)
        xs = np.array([h5step["x"][chunks[i]:chunks[i+1]]]).T
        ys = np.array([h5step["y"][chunks[i]:chunks[i+1]]]).T
        zs = np.array([h5step["z"][chunks[i]:chunks[i+1]]]).T
        vxs = np.array([h5step["vx"][chunks[i]:chunks[i+1]]]).T
        selected_x = xs[indices].reshape(-1)
        selected_y = ys[indices].reshape(-1)
        selected_z = zs[indices].reshape(-1)
        selected_vx = vxs[indices].reshape(-1)
        x_dset[kept_per_chunk[i]:kept_per_chunk[i+1]] = selected_x
        y_dset[kept_per_chunk[i]:kept_per_chunk[i+1]] = selected_y
        z_dset[kept_per_chunk[i]:kept_per_chunk[i+1]] = selected_z
        vx_dset[kept_per_chunk[i]:kept_per_chunk[i+1]] = selected_vx

    output_file.close()
    original_file.close()


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    len_after_prune = int(sys.argv[3])

    # input_path = "/home/appcell/demo_update_400.h5"
    # output_path = "/home/appcell/res.h5"
    # len_after_prune = 150

    prune(input_path, 0, output_path, len_after_prune)
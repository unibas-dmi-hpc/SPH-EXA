import numpy as np
import sys
import h5py
import time
from mpi4py import MPI


# Storage order: z, y, x
num_slices = 3000
num_slices_z = 1000

grid = np.linspace(-0.5, 0.5, num_slices)
grid_z = np.linspace(-0.5, 0.5, num_slices_z)
grid_limit = np.array([num_slices_z, num_slices, num_slices])
alpha = 1.0/(num_slices-1)
alpha_z = 1.0/(num_slices_z-1)
domain_min = -0.5
domain_max = 0.5

def get_weights(z, y, x):
    w000 = (1-x)*(1-y)*(1-z)
    w001 = (1-x)*(1-y)*z
    w010 = (1-x)*y*(1-z)
    w011 = (1-x)*y*z

    w100 = x*(1-y)*(1-z)
    w101 = x*(1-y)*z
    w110 = x*y*(1-z)
    w111 = x*y*z
  
    res = np.array([[[w000, w100],[w010, w110]], [[w001, w101],[w011, w111]]])
    return res

def get_grids(z, y, x):
    ind_z = np.floor((z - domain_min) / alpha_z)
    ind_y = np.floor((y - domain_min) / alpha)
    ind_x = np.floor((x - domain_min) / alpha)
    return [ind_z, ind_y, ind_x]

# Convert (z, y, x) into unit coord
def normalize(z, y, x, coords):
    grid_z = coords[2]*alpha_z + domain_min
    grid_y = coords[1]*alpha + domain_min
    grid_x = coords[0]*alpha + domain_min
    return [z-grid_z, y-grid_y, x-grid_x]

def assign(container, base_grid_coords, weight_arr, value):
    # Using Numpy array ops is slower -- but why?
    grid_z = int(base_grid_coords[0])
    grid_y = int(base_grid_coords[1])
    grid_x = int(base_grid_coords[2])
    grid_z_max = min(grid_z+2, num_slices_z)
    grid_y_max = min(grid_y+2, num_slices)
    grid_x_max = min(grid_x+2, num_slices)
    container[grid_z:grid_z_max, grid_y:grid_y_max, grid_x:grid_x_max] += weight_arr[:(grid_z_max-grid_z), :(grid_y_max-grid_y), :(grid_x_max-grid_x)]*value
    return

def interpolate(z, y, x, value, container):
    coords = get_grids(z, y, x)
    normalized_z, normalized_y, normalized_x = normalize(z, y, x, coords)
    weights = get_weights(normalized_z, normalized_y, normalized_x)
    assign(container, coords, weights, value)

def generateFile(input_initial, file_ind, output_initial, chunk_size = 100000, checkpoint_size = 40000000):
    # Would be way faster(100x) if it's read in chunks
    # Writing order: z, y, x
    container = np.zeros((num_slices_z, num_slices, num_slices), dtype="f4")
    original_dataset = h5py.File(input_initial+"%05d.h5" % file_ind, 'r')
    group = original_dataset["Step#0"]
    num_total_particles = len(original_dataset["Step#0"]["z"])
    start_time = time.time()
    for chunk_ind in range(int(num_total_particles/chunk_size)):
        end = (chunk_ind+1)*chunk_size
        if end > num_total_particles:
            end = num_total_particles
        z_data = np.array(group['z'][chunk_ind*chunk_size:end])
        y_data = np.array(group['y'][chunk_ind*chunk_size:end])
        x_data = np.array(group['x'][chunk_ind*chunk_size:end])
        v_data = np.array(group['v'][chunk_ind*chunk_size:end])
        for particle_ind in range(chunk_size):
            interpolate(z_data[particle_ind], y_data[particle_ind], x_data[particle_ind], v_data[particle_ind], container)
        end_time = time.time()
        print(f"Finished {end} in {end_time - start_time} seconds.")
        if end % checkpoint_size == 0:
            container.tofile(output_initial+"%05d" % file_ind) # save every 5min!!!
            MPI.COMM_WORLD.Barrier()
    end_time = time.time()
    print(f"Rank {file_ind} finished! in {end_time - start_time} seconds.")
    
    container.tofile(output_initial+"%05d" % file_ind)
    

def generateParallel(input_initial, output_initial):
    rank = MPI.COMM_WORLD.rank
    generateFile(input_initial, rank, output_initial)
    MPI.COMM_WORLD.Barrier()

if __name__ == "__main__":
    mode = sys.argv[1]
    input_initial = sys.argv[2]
    num_files = int(sys.argv[3])
    output_initial = sys.argv[4]

    # Test data
    # mode = "serial"
    # input_initial = "/home/appcell/unibas/test_temp_res/extracted/res"
    # num_files = 2
    # output_initial = "/home/appcell/unibas/test_temp_res/interpolated/res"

    if mode == "serial":
        generateFile(input_initial, num_files, output_initial)
    elif mode == "parallel":
        # In parallel mode, num of ranks should be equal to num of files
        generateParallel(input_initial, output_initial)
    else:
        print("please specify mode!")
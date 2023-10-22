import numpy as np
import sys
import h5py
import time


num_slices = 100
num_slices_z = 100

grid = np.linspace(-0.5, 0.5, num_slices)
grid_z = np.linspace(-0.5, 0.5, num_slices_z)
grid_limit = np.array([num_slices, num_slices, num_slices_z])
alpha = 1.0/(num_slices-1)
alpha_z = 1.0/(num_slices_z-1)
domain_min = -0.5
domain_max = 0.5

def get_weights(x, y, z):
    # w000 = (1-x)*(1-y)*(1-z)
    # w001 = (1-x)*(1-y)*z
    # w010 = (1-x)*y*(1-z)
    # w011 = (1-x)*y*z

    # w100 = x*(1-y)*(1-z)
    # w101 = x*(1-y)*z
    # w110 = x*y*(1-z)
    # w111 = x*y*z

    # First 4
    res_first = np.array([[(1-y)*(1-z), (1-y)*z], [y*(1-z), y*z]])
    res = np.array([res_first*(1-x), res_first*x])

    # resarr = np.array([w000, w001, w010, w011, w100, w101, w110, w111])
    # if len(resarr[resarr < 0]) > 0:
    #     print("Not possible!")
    # if len(resarr[resarr > 1.0]) > 0:
    #     print("Not possible either!")    
    # res = np.array([[[w000, w001],[w010, w011]], [[w100, w101],[w110, w111]]])
    return res

def get_grids(x, y, z):
    ind_x = np.floor((x - domain_min) / alpha)
    ind_y = np.floor((y - domain_min) / alpha)
    ind_z = np.floor((z - domain_min) / alpha_z)
    return [ind_x, ind_y, ind_z]

# Convert (x, y, z) into unit coord
def normalize(x, y, z, coords):
    grid_x = coords[0]*alpha + domain_min
    grid_y = coords[1]*alpha + domain_min
    grid_z = coords[2]*alpha_z + domain_min
    return [x-grid_x, y-grid_y, z-grid_z]

def assign(container, base_grid_coords, weight_arr, value):
    # Using Numpy array ops is slower -- but why?
    # curr_grid = np.array(base_grid_coords).astype("int8")
    # curr_grid_max = np.minimum(curr_grid+2, grid_limit)
    # container[curr_grid[0]:curr_grid_max[0], curr_grid[1]:curr_grid_max[1], curr_grid[2]:curr_grid_max[2]] += weight_arr[:(curr_grid_max[0]-curr_grid[0]), :(curr_grid_max[1]-curr_grid[1]), :(curr_grid_max[2]-curr_grid[2])]*value
    grid_x = int(base_grid_coords[0])
    grid_y = int(base_grid_coords[1])
    grid_z = int(base_grid_coords[2])
    grid_x_max = min(grid_x+2, num_slices)
    grid_y_max = min(grid_y+2, num_slices)
    grid_z_max = min(grid_z+2, num_slices_z)
    container[grid_x:grid_x_max, grid_y:grid_y_max, grid_z:grid_z_max] += weight_arr[:(grid_x_max-grid_x), :(grid_y_max-grid_y), :(grid_z_max-grid_z)]*value
    return

def interpolate(x, y, z, value, container):
    coords = get_grids(x, y, z)
    normalized_x, normalized_y, normalized_z = normalize(x, y, z, coords)
    weights = get_weights(normalized_x, normalized_y, normalized_z)
    assign(container, coords, weights, value)

def generateFile(input_initial, file_ind, output_initial, chunk_size = 100000, checkpoint_size = 40000000):
    # Would be way faster(100x) if it's read in chunks
    # Writing order: x, y, z
    container = np.zeros((num_slices, num_slices, num_slices_z), dtype="f4")
    original_dataset = h5py.File(input_initial+"%05d.h5" % file_ind, 'r')
    group = original_dataset["Step#0"]
    num_total_particles = len(original_dataset["Step#0"]["x"])
    start_time = time.time()
    for chunk_ind in range(int(num_total_particles/chunk_size)):
        end = (chunk_ind+1)*chunk_size
        if end > num_total_particles:
            end = num_total_particles
        x_data = np.array(group['x'][chunk_ind*chunk_size:end])
        y_data = np.array(group['y'][chunk_ind*chunk_size:end])
        z_data = np.array(group['z'][chunk_ind*chunk_size:end])
        v_data = np.array(group['v'][chunk_ind*chunk_size:end])
        for particle_ind in range(chunk_size):
            interpolate(x_data[particle_ind], y_data[particle_ind], z_data[particle_ind], v_data[particle_ind], container)
        end_time = time.time()
        print(f"Finished {end} in {end_time - start_time} seconds.")
        if end % checkpoint_size == 0:
            container.tofile(output_initial+"%05d" % file_ind) # save every 5min!!!
    end_time = time.time()
    print(f"Finished! in {end_time - start_time} seconds.")
    container.tofile(output_initial+"%05d" % file_ind)

def generateParallel(input_initial, output_initial):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    generateFile(input_initial, rank, output_initial)

if __name__ == "__main__":
    mode = sys.argv[1]
    input_initial = sys.argv[2]
    num_files = int(sys.argv[3])
    output_initial = sys.argv[4]

    # # Test data
    # mode = "serial"
    # input_initial = "/home/appcell/unibas/test_temp_res/extracted/res"
    # num_files = 2
    # output_initial = "/home/appcell/unibas/test_temp_res/interpolated/res"

    if mode == "serial":
        for i in range(num_files):
            generateFile(input_initial, i, output_initial)
    elif mode == "parallel":
        # In parallel mode, num of ranks should be equal to num of files
        generateParallel(input_initial, output_initial)
    else:
        print("please specify mode!")
    sys.exit(0)

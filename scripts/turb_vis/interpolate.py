import numpy as np
from PIL import Image
import sys
import h5py


# image_path = '/Users/zhu0002-adm/output_0059.png'

# # Open the image using Pillow
# image = Image.open(image_path)

# # Convert the image to a NumPy array
# image_array = np.array(image)

# # Display the shape of the array (height, width, and channels for color images)
# print("Image shape:", image_array.shape)

container = np.fromfile("/Users/zhu0002-adm/nparray_03", dtype="f4")

original = h5py.File("/Users/zhu0002-adm/slice00003.h5", 'r')

group = original["Step#0"]

num_total_rows = len(original["Step#0"]["x"])

grid = np.linspace(-0.5, 0.5, 1500)
alpha = 1.0/(1500-1)
domain_min = -0.5
domain_max = 0.5

def get_weights(x, y, z):
    w000 = (1-x)*(1-y)*(1-z)
    w100 = x*(1-y)*(1-z)
    w010 = (1-x)*y*(1-z)
    w001 = (1-x)*(1-y)*z
    w101 = x*(1-y)*z
    w011 = (1-x)*y*z
    w110 = x*y*(1-z)
    w111 = x*y*z

    resarr = np.array([w000, w001, w010, w011, w100, w101, w110, w111])
    if len(resarr[resarr < 0]) > 0:
        print("!!!")
    res = np.array([[[w000, w001],[w010, w011]], [[w100, w101],[w110, w111]]])
    return res

def get_grids(x, y, z):
    ind_x = np.floor((x - domain_min) / alpha)
    ind_y = np.floor((y - domain_min) / alpha)
    ind_z = np.floor((z - domain_min) / alpha)
    return (ind_x, ind_y, ind_z)

def normalize(x, y, z, coords):
    grid_x = coords[0]*alpha - domain_min
    grid_y = coords[1]*alpha - domain_min
    grid_z = coords[2]*alpha - domain_min
    return [x-grid_x, y-grid_y, z-grid_z]

# container = container.reshape(1500, 1500, 1500)

for i in range(num_total_rows):
    x = group['x'][i]
    y = group['y'][i]
    z = group['z'][i]
    vx = group['vx'][i]
    coords = get_grids(x, y, z)
    normalized_pos = normalize(x, y, z, coords)
    weights = get_weights(x, y, z)

    


sys.exit(0)





# print(container[(59, 302, 671)])




# x: 302, y: 671, z: 59
# x: 308, y: 671, z: 59
# 59*(3000/1500)*(3000*3000) + 671*(3000/1500)*3000 + 302*(3000/1500)
# 1066026604 - int(1066026604/(27000000000/88))*int(27000000000/88) = 1066026604.0 <- index of particle

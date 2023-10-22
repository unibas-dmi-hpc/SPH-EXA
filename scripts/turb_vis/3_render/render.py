import numpy as np
import time
import sys
from PIL import Image
import matplotlib.pyplot as plt


total_slices = 100
total_slices_z = 100
colormap = plt.get_cmap("bone")

def getcolor(val):
    color = colormap(val)
    return color

def save(data, i, output_initial):
    scaled_data = (data * 255).astype(np.uint8)
    image = Image.fromarray(scaled_data)
    image.save(output_initial + "%04d.png" % i)

def render(input_path, output_initial):
    container = np.fromfile(input_path, dtype='f4')
    max_value = np.amax(container)
    min_value = np.amin(container)
    container = (container - min_value) / (max_value - min_value)
    container = container.reshape(total_slices,total_slices,total_slices_z)
    all_colors = [getcolor(x)[0:3] for x in np.linspace(0, 1, 255)]
    start = time.time()
    for i in range(total_slices_z):
        curr_frame = container[:, :, i]
        converted = (curr_frame * 255).astype(np.uint8)
        final_image = np.zeros((total_slices, total_slices, 3), dtype="f4")
        for x in range(total_slices):
            for y in range(total_slices):
                try:
                    final_image[x, y, :] = all_colors[converted[x][y]]
                except:
                    print("??")
        image = Image.fromarray((final_image * 255).astype(np.uint8))
        image.save(output_initial + "%04d.png" % i)
        end = time.time()
        print(f"{start - end} seconds with {i} pics." )

if __name__ == "__main__":
    # input_path = sys.argv[1]
    # output_initial = sys.argv[2]
    input_path = "/home/appcell/unibas/test_temp_res/merged/res"
    output_initial = "/home/appcell/unibas/test_temp_res/rendered/res"
    render(input_path, output_initial)
    sys.exit(0)
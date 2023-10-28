import numpy as np
import time
import sys
import gc
from PIL import Image
import matplotlib.pyplot as plt
# import cv2

total_slices = 3000
total_slices_z = 1000
colormap = plt.get_cmap("bone")

def getColormapColor(val):
    color = colormap(val)
    return color[:3]

all_colors = [getColormapColor(x)[0:3] for x in np.linspace(0, 1, 255)]

def getMinMax(input_path):
    num_parts = 16
    total_length = total_slices*total_slices*total_slices_z
    record_length = int(total_length / num_parts)
    record_size = record_length * 4 # size of a record in bytes
    min_val = sys.float_info.max
    max_val = sys.float_info.min
    input_file = open(input_path, 'rb')
    for part_ind in range(num_parts):
        input_file.seek(record_size * part_ind)
        mem_arr = input_file.read(record_size)
        curr_arr = np.frombuffer(mem_arr, dtype="f4")
        if np.amin(curr_arr) < min_val:
            min_val = np.amin(curr_arr)
        if np.amax(curr_arr) > max_val:
            max_val = np.amax(curr_arr)
        del(mem_arr)
        del(curr_arr)
        gc.collect()
    input_file.close()
    return min_val, max_val

def getColor(val):
    val = np.float_power(val, 0.55)
    if val >= 1.0:
        return all_colors[254]
    return all_colors[int(val*255)]

def filter(image):
    return image
    # brightness_factor = 0.85
    # brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    # contrast_factor = 2.0
    # res = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=-80)
    # blueish_tinted_image = np.zeros_like(res)
    # blue_tint = (40, 60, 0)
    # blueish_tinted_image[:, :] = blue_tint
    # res = cv2.addWeighted(res, 1.0, blueish_tinted_image, 0.4, 0)

    # # Convert the image to the HSV color space
    # hsv_image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # # Lower the saturation
    # s_factor = 0.9  # You can adjust this factor to control the saturation level
    # hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * s_factor, 0, 255).astype(np.uint8)

    # # Convert the modified HSV image back to BGR color space
    # res = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return res

def save(data, i, output_initial):
    scaled_data = (data * 255).astype(np.uint8)
    image = Image.fromarray(scaled_data)
    image.save(output_initial + "%04d.png" % i)

def render(input_path, output_initial, start_img, end_img):
    min_value, max_value = getMinMax(input_path)
    input_file = open(input_path, 'rb')
    image_size = total_slices*total_slices_z*4
    start = time.time()
    print(min_value, max_value)
    for image_ind in range(start_img, end_img):
        input_file.seek(image_size * image_ind)
        mem_arr = input_file.read(image_size)
        curr_frame = np.frombuffer(mem_arr, dtype="f4")
        print(np.amin(curr_frame), np.amax(curr_frame))
        curr_frame = (curr_frame - min_value) / (max_value - min_value)
        curr_frame = curr_frame.reshape((total_slices, total_slices_z))
        final_image = np.zeros((total_slices, total_slices_z, 3), dtype="f4")
        # Normalize by picture
        for y in range(total_slices):
            for z in range(total_slices_z):
                final_image[y, z, :] = getColor(curr_frame[y][z])
        
        final_image = filter((final_image * 255).astype(np.uint8))
        image = Image.fromarray(final_image)
        image.save(output_initial + "%04d.png" % image_ind)
        end = time.time()
        if image_ind % 100 == 0:
            del(mem_arr)
            del(curr_frame)
            del(final_image)
            del(image)
            gc.collect()
        print(f"{start - end} seconds with {image_ind - start_img} pics." )

def renderSerial(input_path, output_initial):
    render(input_path, output_initial, start_img=0, end_img=total_slices_z)

def renderParallel(input_path, output_initial):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    total_ranks = MPI.COMM_WORLD.Get_size()
    ranges = np.linspace(0, total_slices, total_ranks+1, dtype=int)
    render(input_path, output_initial, ranges[rank], ranges[rank+1])
    

if __name__ == "__main__":
    # mode = sys.argv[1]
    # input_path = sys.argv[2]
    # output_initial = sys.argv[3]
    mode = "parallel"
    input_path = "/home/appcell/unibas/test_temp_res/merged/merged_3000_"
    output_initial = "/home/appcell/unibas/test_temp_res/rendered/res"

    if mode == "serial":
        renderSerial(input_path, output_initial)
    elif mode == "parallel":
        renderParallel(input_path, output_initial)
    else:
        print("please specify mode!")
    sys.exit(0)
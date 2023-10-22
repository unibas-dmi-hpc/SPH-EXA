import numpy as np
import sys
import gc
import time

def mergeSerial(input_initial, num_files, output_path):
    start = time.time()
    container = np.fromfile(input_initial+"%05d" % (0), dtype='f4')
    container.tofile(output_path)
    del(container)
    gc.collect()
    
    for i in range(1, num_files):
        container = np.fromfile(output_path, dtype='f4')
        container += np.fromfile(input_initial+"%05d" % (i), dtype='f4')
        container.tofile(output_path)
        del(container)
        gc.collect()
        end = time.time()
        print(f"Processed {i} files in {end - start} seconds.")

if __name__ == "__main__":
    input_initial = sys.argv[1]
    output_path = sys.argv[2]
    num_files = int(sys.argv[3])

    # # Test data
    # input_initial = "/home/appcell/unibas/test_temp_res/interpolated/res"
    # output_path = "/home/appcell/unibas/test_temp_res/merged/res"
    # num_files = 2

    mergeSerial(input_initial, num_files, output_path)
    sys.exit(0)
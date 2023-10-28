import numpy as np
import sys, os
import gc
import time

# Tip for using this script: the initial file should not be more than 32GB
num_parts = 16
total_slices = 3000
total_slices_z = 1000
total_length = total_slices_z * total_slices * total_slices
record_length = int(total_length / num_parts)
record_size = record_length * 4 # size of a record in bytes

def mergeSerial(input_initial, num_files, output_path):
    output_file = open(output_path, 'w+b')
    start_time = time.time()
    for part_ind in range(num_parts):
        mem_arr = np.zeros(record_length, dtype="f4")
        for file_ind in range(num_files):
            input_file = open(input_initial+"%05d" % file_ind, 'rb')
            input_file.seek(record_size * part_ind)
            bytes = input_file.read(record_size)
            mem_arr += np.frombuffer(bytes, dtype="f4")
            del(bytes)
            input_file.close()
            gc.collect()
            print("Finished part %d in file %d, in %f seconds." % (part_ind, file_ind, time.time()-start_time))
        
        bytes = mem_arr.tobytes()
        output_file.seek(record_size * part_ind)
        output_file.write(bytes)
        del(bytes)
        gc.collect()
        print("Finished part %d in all files, in %f seconds." % (part_ind, time.time()-start_time))

if __name__ == "__main__":
    input_initial = sys.argv[1]
    output_path = sys.argv[2]
    num_files = int(sys.argv[3])

    # Test data
    # input_initial = "/home/appcell/unibas/test_temp_res/interpolated/res"
    # output_path = "/home/appcell/unibas/test_temp_res/merged/res"
    # num_files = 2

    mergeSerial(input_initial, num_files, output_path)
    sys.exit(0)
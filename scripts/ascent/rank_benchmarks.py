import numpy as np
from mpi4py import MPI
import json
import os
import time



# obtain a mpi4py mpi comm object
comm = MPI.Comm.f2py(ascent_mpi_comm_id())
rank = comm.Get_rank()
total_ranks = comm.Get_size()

# get this MPI task's published blueprint data
mesh_data = ascent_data().child(0)
output_path = '../output/ascent/session_benchmarks.json'

e_vals = mesh_data["fields/rank_time/values"]
e_all = np.zeros(total_ranks, dtype=np.float32)
comm.Gather(e_vals, e_all)
if rank == 0:
    step = int(mesh_data["state/cycle"])
    data = {
        'step': step,
        'time': e_all.tolist()
    }
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, 'r+') as file:
            file.seek(0, os.SEEK_END)  # Move the file cursor to the end
            file.seek(file.tell() - 1, os.SEEK_SET)  # Move the cursor to the last character
            file.truncate()  # Remove the last character (']')
            file.write(',')  # Add a comma before appending new data
            json.dump(data, file)
            file.write(']')
    else:
        with open(output_path, 'w') as file:
            json.dump([data], file)
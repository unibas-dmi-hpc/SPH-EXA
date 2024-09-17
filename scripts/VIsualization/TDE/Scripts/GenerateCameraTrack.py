# Usage: Copy the output of this script into ParaView Python cmdline.
import numpy as np
from GetBoundingBox import calculate_averages

def expand_vector_length(vector, length_increase):
    # Calculate the current length of the vector
    current_length = np.linalg.norm(vector)
    
    # Calculate the scaling factor to increase the length
    scale_factor = (current_length + length_increase) / current_length
    
    # Scale the vector to the new length
    expanded_vector = vector * scale_factor
    
    return expanded_vector

# Setup z-values
zx_points = np.array([0, 900, 1200, 1481])
zy_points = np.array([0, 20, 0, 0])

def interpolate_z(x):
    z = np.interp(x, zx_points, zy_points)
    return z

# Setup z-values
tdx_points = np.array([0, 300, 639, 800, 1481])
tdy_points = np.array([0, -50, 0, 50, 0])

def interpolate_td(x):
    td = np.interp(x, tdx_points, tdy_points)
    return td

def calc_camera(frame_ind, center_coords_dict):
    # For first 500 frames, position moves gradually
    origin = np.array([0,0,0])
    fixed_pose = expand_vector_length(np.array(center_coords_dict[1480]), 5)
    if frame_ind <= 500:
        pos_vec = np.array(center_coords_dict[frame_ind]) * (500 - frame_ind)/500.0 + origin * frame_ind / 500.0
    elif frame_ind < 900:
        pos_vec = origin  * ((400 - (frame_ind-500))/400.0) + fixed_pose *  ((frame_ind-500) / 400.0)
    else:
        pos_vec = fixed_pose
    pos_vec[2] = interpolate_z(frame_ind)
    if frame_ind <= 500:
        focal_vec = np.array(center_coords_dict[frame_ind]) * (1 - frame_ind * 0.7 / 500.0)
    else:
        focal_vec = np.array([0, 0, 0]) + np.array(center_coords_dict[frame_ind]) *0.3
    

    print(
        f"""
keyframe_{frame_ind} = CameraKeyFrame()
keyframe_{frame_ind}.KeyTime = {float(frame_ind)/1481.0}
keyframe_{frame_ind}.Position = [{pos_vec[0]}, {pos_vec[1]}, {pos_vec[2]}]
keyframe_{frame_ind}.FocalPoint = [{focal_vec[0]}, {focal_vec[1]}, {focal_vec[2]}]
keyframe_{frame_ind}.ViewUp = [-0.106555,0.150353,0.982873]
keyframe_{frame_ind}.ViewAngle = 30
keyframe_{frame_ind}.ParallelScale = 225.768
track.KeyFrames.append(keyframe_{frame_ind})
"""
    )


if __name__ == "__main__":
    center_coords, min_coords, max_coords, scales = calculate_averages('/home/appcell/Visualizations/TidalDisruptionEvent/Datasets/tde_snapshot00000.h5')

    print("""
track = GetCameraTrack()
track.KeyFrames = []
    """)

    for i in range(0, 1482, 50):
        calc_camera(i, center_coords)
    i = 1481
    calc_camera(i, center_coords)

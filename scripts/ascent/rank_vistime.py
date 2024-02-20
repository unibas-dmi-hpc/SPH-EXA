import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.backends.backend_agg import FigureCanvasAgg

file_path = '../output/ascent/session_benchmarks.json'
with open(file_path, 'r') as file:
    data = json.load(file)
N = len(data)
N_ranks = len(data[0]["time"])

# look at the last timestep when actions were triggered
last_cycle = data[N-1]["step"]
x_vals = []
all_times = []
for item in data:
    x_vals.append(item["step"])
    all_times.append(item["time"])
x_size = N
y_vals = np.linspace(0, 7, 8)
y_size = len(y_vals)
x, y = np.meshgrid(x_vals, y_vals)

cycles = x_vals

timesteps = [int(cycle_id) for cycle_id in cycles]
total_exec_times = np.array(all_times)
min_exec_time = np.min(total_exec_times)
max_exec_time = np.max(total_exec_times)
exec_times = []
for item in data:
    cycle_id = item["step"]
    exec_times.append(item["time"])
    exec_times_plot = np.concatenate((exec_times, np.full((len(timesteps)-len(exec_times), N_ranks), np.nan)), axis=0)
    exec_times_all = [(exec_times_plot[:, i]).flatten() for i in range(N_ranks)]
    color_values = []
    for i in range(N_ranks):
        color_values.append(np.full(len(timesteps), i))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 3), dpi=100, facecolor='black')
    ax1 = axs


    jet = cm = plt.get_cmap('coolwarm') 
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=N_ranks-1)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    values = range(N_ranks)
    lines = []
    for i in range(N_ranks):
        colorVal = scalarMap.to_rgba(values[i])
        lines.append(
            ax1.plot(timesteps, exec_times_all[i], label=f'Rank {i}', color=colorVal)
            )
    ax1.set_ylabel('Exec time(s)')
    ax1.set_xlabel('Timestep')
    ax1.set_xlim(min(timesteps), max(timesteps))  # Set x-axis limits
    ax1.set_ylim(min_exec_time, max_exec_time)  # Set y-axis limits
    ax1.set_facecolor('black')

    # Set the color of other plot elements to white
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='white', edgecolor='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')

    # Render the plot onto a canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Access the rendered plot buffer as a numpy array
    plot_buffer = np.array(canvas.renderer.buffer_rgba())

    file = "./output/step_.%05d.png" % int(cycle_id)
    particles = image.imread(file)
    particles = particles.astype(np.float32) * 255.0
    if particles.shape[2] == 3:
        particles = np.dstack((particles, np.ones((particles.shape[0], particles.shape[1]), dtype=np.uint8) * 255))
    # Determine the dimensions of the image and plot
    image_height, image_width, _ = particles.shape
    plot_height, plot_width, _ = plot_buffer.shape

    # Calculate the padding needed to match the dimensions
    height_diff = abs(image_height - plot_height)
    width_diff = abs(image_width - plot_width)
    top_pad = height_diff // 2
    bottom_pad = height_diff - top_pad
    left_pad = width_diff // 2
    right_pad = width_diff - left_pad

    # Pad the smaller dimension with black to match the dimensions
    if image_width < plot_width:
        particles = np.pad(particles, ((0, 0), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
    else:
        plot_buffer = np.pad(plot_buffer, ((0, 0), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
    particles = particles[:-120, :, :]
    plot_buffer = plot_buffer[30:, :, :]
    concatenated = np.concatenate((particles, plot_buffer), axis=0)
    concatenated = concatenated[:, :, :3]
    concatenated = concatenated.astype(np.float32) / 255.0
    plt.imsave("./output_tmp/step_.%05d.png" % int(cycle_id), concatenated)
    plt.close()
# support file for the actions programmed in {trigger_binning_actions.yaml,
#                                             binning_actions.yaml}
# created by Jean M. Favre, tested Tue Jun  7 03:32:58 PM CEST 2022
# the file "ascent_session.yaml" is written upon successful conclusion of
# a run of SPH-EXA instrumented with Ascent

import conduit

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.backends.backend_agg import FigureCanvasAgg

results = conduit.Node()
conduit.relay.io.load(results, "ascent_session.yaml")
N = results["my_cycle"].number_of_children()


# plot 2D probability density function Density versus radius)
def bin_values(axis_name):
    # create the coordinate axis using bin centers
    a_min = axis_name['min_val']
    a_max = axis_name['max_val']
    a_bins = axis_name['num_bins']

    a_delta = (a_max - a_min) / float(a_bins)
    a_start = a_min + 0.5 * a_delta

    axis_vals = []
    for b in range(0, a_bins):
        axis_vals.append(b * a_delta + a_start)
    return axis_vals, a_bins


# look at the last timestep when actions were triggered
last_cycle = int(results["my_cycle"].child_names()[-1])
# x_vals, x_size = bin_values(results["my_cycle/" + cycle +
#                                     "/value"])
x_vals = results["my_cycle"].child_names()
x_size = len(results["my_cycle"].child_names())
# y_vals, y_size = bin_values(results["my_cycle/" + cycle +
                                    # "/time"])
y_vals = np.linspace(0, 7, 8)
y_size = len(y_vals)
x, y = np.meshgrid(x_vals, y_vals)

cycles = results["my_cycle"].child_names()

timesteps = [int(cycle_id) for cycle_id in cycles]
total_exec_times = [results["my_cycle/" + cycle_id + "/time"] for cycle_id in cycles]
min_exec_time = min(total_exec_times)
max_exec_time = max(total_exec_times)
exec_times = []
for cycle_id in cycles:
    exec_times.append(results["my_cycle/" + cycle_id + "/time"])
    exec_times_plot = exec_times + [np.NAN]*(len(timesteps)-len(exec_times))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 3), dpi=100, facecolor='black')
    ax1 = axs
    line = ax1.plot(timesteps, exec_times_plot, label='Exec time(s)')
    ax1.set_ylabel('Exec time(s)')
    ax1.set_xlabel('Timestep')
    ax1.set_xlim(min(timesteps), max(timesteps))  # Set x-axis limits
    ax1.set_ylim(min_exec_time, max_exec_time)  # Set y-axis limits
    ax1.set_facecolor('black')

    plt.setp(line, color='white')  # Set line color to white

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





# plot 1D reduction functions min(Density) versus radius,
# max(Density) versus radius, and average(Density) versus radius
# x_vals, x_size = bin_values(results["min_density/" + cycle +
#                             "/attrs/bin_axes/value/radius"])
# plt.xlabel('Timestep')
# plt.ylabel('Exec time(s)')
# plt.plot(x_vals, results["min_density/" + cycle + "/attrs/value/value"],
#          label='min(Density)')
# plt.plot(x_vals, results["max_density/" + cycle + "/attrs/value/value"],
#          label='max(Density)')
# plt.plot(x_vals, results["avg_density/" + cycle + "/attrs/value/value"],
#          label='avg(Density)')
# plt.savefig("min_max_avg.png")
# plt.show()

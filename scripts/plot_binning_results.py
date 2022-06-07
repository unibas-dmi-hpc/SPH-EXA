# support file for the actions programmed in {trigger_binning_actions.yaml,binning_actions.yaml}
# created by Jean M. Favre, tested Tue Jun  7 03:32:58 PM CEST 2022
# the file "ascent_session.yaml" is written upon successful conclusion of
# a run of SPH-EXA instrumented with Ascent

import conduit

import numpy as np
import matplotlib.pyplot as plt

results = conduit.Node()
conduit.relay.io.load(results, "ascent_session.yaml")
N = results["pdf_density"].number_of_children()

# plot 2D  probability density function Density versus radius)
def bin_values(axis_name):
  # create the coordinate axis using bin centers
  a_min = axis_name['min_val']
  a_max = axis_name['max_val']
  a_bins = axis_name['num_bins']

  a_delta = (a_max - a_min) / float(a_bins)
  a_start = a_min + 0.5 * a_delta

  axis_vals = []
  for b in range(0,a_bins):
    axis_vals.append(b * a_delta + a_start)
  return axis_vals, a_bins

# look at the last timestep when actions were triggered
cycle = results["pdf_density"].child_names()[-1]
x_vals, x_size = bin_values(results["pdf_density/" + cycle + "/attrs/bin_axes/value/radius"])
y_vals, y_size = bin_values(results["pdf_density/" + cycle + "/attrs/bin_axes/value/Density"])
x, y = np.meshgrid(x_vals, y_vals)

c = plt.pcolormesh(x, y, results["pdf_density/" + cycle + "/attrs/value/value"].reshape(x_size, y_size), shading="auto", cmap="viridis")
    
plt.ylabel('Density')
plt.xlabel('radius')
plt.colorbar(c)
plt.savefig("pdf.png")
plt.close()
#plt.show()

# plot 1D reduction functions min(Density) versus radius,
# max(Density) versus radius, and average(Density) versus radius
x_vals, x_size = bin_values(results["min_density/" + cycle + "/attrs/bin_axes/value/radius"])
plt.xlabel('radius')
plt.ylabel('Density')
plt.plot(x_vals, results["min_density/" + cycle + "/attrs/value/value"], label='min(Density)')
plt.plot(x_vals, results["max_density/" + cycle + "/attrs/value/value"], label='max(Density)')
plt.plot(x_vals, results["avg_density/" + cycle + "/attrs/value/value"], label='avg(Density)')
plt.savefig("min_max_avg.png")
#plt.show()

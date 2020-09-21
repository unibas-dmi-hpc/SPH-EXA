import os
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 1000000

d = np.fromfile('dump_Sedov200.bin')

x = d[0:n]
y = d[1*n:2*n]
z = d[2*n:3*n]
vx = d[3*n:4*n]
vy = d[4*n:5*n]
vz = d[5*n:6*n]
h = d[6*n:7*n]
ro = d[7*n:8*n]
u = d[8*n:9*n]
p = d[9*n:10*n]
c = d[10*n:11*n]
grad_P_x = d[11*n:12*n]
grad_P_y = d[12*n:13*n]
grad_P_z = d[13*n:14*n]

mask = abs(z / h) < 1.0

cm = plt.cm.get_cmap('RdYlBu')

plt.style.use('dark_background')

# Create figure
plt.figure(figsize=(10,8))

# Plot 2D projection a middle cut
sc = plt.scatter(x[mask], y[mask], c=ro[mask], s=10.0, label="Sedov", vmin=min(ro[mask]), vmax=max(ro[mask]), cmap=cm)
plt.colorbar(sc)

plt.axis('square')
#plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.title('Density')

# fig = plt.figure()
# ax = Axes3D(fig)

# mask = abs(ro) < 0.25

# ax.scatter(x[mask], y[mask], z[mask], c=ro[mask], s=10.0, label="Sedov", vmin=min(ro[mask]), vmax=max(ro[mask]), cmap=cm)

plt.show()

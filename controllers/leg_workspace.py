import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import radians
from kinematic import Controller
from kinematic import tripod_gait
from scipy.spatial import ConvexHull
import matplotlib

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.unicode_minus': False
})

num = 150

theta_1 = np.linspace(-radians(100), radians(100), num)
theta_2 = np.linspace(-radians(150), radians(130), num)
theta_3 = np.linspace(-radians(170), radians(130), num)
joint_angles = np.array(np.meshgrid(theta_1, theta_2, theta_3)).reshape(3,-1)

controller = Controller(tripod_gait, body_height=0.14, velocity=0.2, period=0.5, crab_angle=-np.pi/6)
xyz = controller.forward_kinematics(joint_angles)*1000

# get only outer points
# xyz = xyz[:,xyz[0,:] > 0]
# xyz = xyz.T[ConvexHull(xyz.T).vertices].T

# fig1 = plt.figure(1)
# fig1.set_size_inches(w=4.7747, h=3.5)

# 3D scatter plot
# ax = Axes3D(fig1)

# radius = np.sqrt(xyz[0,:]**2 + xyz[1,:]**2 + xyz[2,:]**2)
# x, y, z = xyz[:,np.argsort(radius)]
# radius = np.flip(np.flip(np.sort(radius)))

# ax.scatter(x, y, z, marker='o', c=radius)

# # 2D scatter plot
# x, y, z = xyz[:,np.argsort(xyz[2])]
# plt.scatter(x, y, c=z)
# plt.xlabel("X axis (mm)")
# plt.ylabel("Y axis (mm)")
# plt.title("Top view of foot workspace")
# clb = plt.colorbar()
# clb.ax.set_ylabel("Z axis (mm)")

# x, y, z = xyz[:,np.argsort(xyz[0])]
# fig2 = plt.figure(2)
# fig2.set_size_inches(w=4.7747, h=3.5)
# plt.scatter(y, z, c=x)
# plt.title("Front view of foot workspace")
# plt.xlabel("Y axis (mm)")
# plt.ylabel("Z axis (mm)")
# clb = plt.colorbar()
# clb.ax.set_ylabel("X axis (mm)")
xyz = xyz[:,xyz[1,:] > 0]
x, y, z = xyz[:,np.argsort(xyz[1])]
fig = plt.figure()
fig.set_size_inches(w=4.7747, h=3.5)
plt.scatter(x, z, c=y, vmin=0)
plt.title("Side view of leg workspace")
plt.xlabel("$x$-axis ($mm$)")
plt.ylabel("$z$-axis ($mm$)")
clb = plt.colorbar()
clb.ax.set_ylabel("$y$-axis ($mm$)")

# fig1.tight_layout()
# fig2.tight_layout()
fig.tight_layout()
plt.savefig('../../Final Report/figures/leg_workspace_side_view.png', dpi=800)

# plt.show()

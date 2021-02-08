import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kinematic import Controller
from kinematic import tripod_gait
import matplotlib

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

leg_n = 2

fig = plt.figure()
fig.set_size_inches(w=4.7747, h=3.5)

ax = fig.add_subplot(111, projection='3d')

controller = Controller(tripod_gait, body_height=0.14, velocity=0.15, period=0.5, crab_angle=0)
joint_angles = controller.angles[(leg_n-1)*3:leg_n*3,:] # leg 3
x, y, z = controller.forward_kinematics(joint_angles) * 1000

ax.scatter(x[:60], y[:60], z[:60], label="Swing phase")
ax.scatter(x[60:], y[60:], z[60:], label="Support phase")

ax.set_title("Foot trajectory")
ax.set_xlabel('$X (mm)$')
ax.set_ylabel('$Y (mm)$')
ax.set_zlabel('$Z (mm)$')

# ax.set_xlim([-300,300])
# ax.set_ylim([-300,300])
# ax.set_zlim([-300,300])

plt.legend()

fig.tight_layout()

# plt.savefig('histogram.pgf')

plt.show()

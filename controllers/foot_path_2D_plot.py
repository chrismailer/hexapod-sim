import numpy as np
import matplotlib.pyplot as plt
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
period = 0.5
duty_factor = 0.5

# radius, offset, step_height, phase, duty_cycle
tripod_gait = np.array([[0.15, 0, 0.06, 0, duty_factor],]*6)

fig, ax = plt.subplots()
fig.set_size_inches(w=4.7747, h=3.5)

controller = Controller(tripod_gait, body_height=0.14, velocity=0.3, period=period, crab_angle=-np.pi/6)
x, y, z = controller.positions[(leg_n-1)*3:leg_n*3,:] * 1000
dx, dy, dz = controller.velocities[(leg_n-1)*3:leg_n*3,:] * 1000

shift = -1
dx = np.roll(dx, shift)
dy = np.roll(dy, shift)
dz = np.roll(dz, shift)

mid = int(period*240*duty_factor)
ax.scatter(y[mid:], z[mid:], label="Swing phase", marker='.')
ax.scatter(y[:mid], z[:mid], label="Support phase", marker='.')

# show velocity arrows
step = 12
q = ax.quiver(y[::step], z[::step], dy[::step], dz[::step], units="xy", angles="xy", pivot='tail')
# ax.quiverkey(q, X=0.3, Y=0.85, U=500, label="0.5 m/s foot velocity", labelpos='E', coordinates='figure')

ax.set_title("Foot Trajectory")
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')

ax.set_ylim([-150,-40])
# ax.set_xlim([-150,30])

plt.legend()

fig.tight_layout()

plt.savefig('../../Final Report/figures/foot_traj_plot.pdf')

# plt.show()

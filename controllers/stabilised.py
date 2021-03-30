import numpy as np


# TODO:
# - Make 0.014 more explicit
# - Make PID use numpy

# radius, offset, step_height, phase, duty_factor
tripod_gait = [	0.15, 0, 0.05, 0.5, 0.5, # leg 1
				0.15, 0, 0.05, 0.0, 0.5, # leg 2
				0.15, 0, 0.05, 0.5, 0.5, # leg 3
				0.15, 0, 0.05, 0.0, 0.5, # leg 4
				0.15, 0, 0.05, 0.5, 0.5, # leg 5
				0.15, 0, 0.05, 0.0, 0.5] # leg 6

wave_gait = [	0.15, 0, 0.05, 0/6, 5/6, # leg 1
				0.15, 0, 0.05, 1/6, 5/6, # leg 2
				0.15, 0, 0.05, 2/6, 5/6, # leg 3
				0.15, 0, 0.05, 3/6, 5/6, # leg 4
				0.15, 0, 0.05, 4/6, 5/6, # leg 5
				0.15, 0, 0.05, 5/6, 5/6] # leg 6

quadruped_gait = [	0.15, 0, 0.05, 0/3, 2/3, # leg 1
					0.15, 0, 0.05, 1/3, 2/3, # leg 2
					0.15, 0, 0.05, 2/3, 2/3, # leg 3
					0.15, 0, 0.05, 0/3, 2/3, # leg 4
					0.15, 0, 0.05, 1/3, 2/3, # leg 5
					0.15, 0, 0.05, 2/3, 2/3] # leg 6

stationary = [0.15, 0, 0, 0, 0] * 6


class Controller:
	def __init__(self, params, crab_angle=0.0, body_height=0.1, period=1.0, velocity=0.1, dt=1/240):
		self.dt = dt

		self.period = period # gait period is fixed
		self.velocity = velocity
		self.crab_angle = crab_angle
		self.body_height = body_height
		self.angle_limit = 0.4
		# link lengths
		self.l_1 = 0.05317
		self.l_2 = 0.10188
		self.l_3 = 0.14735
		self.body_radius = 0.12529

		self.base_attitude = np.array([0.0, 0.0]) # [ roll, pitch ]
		self.slope = np.array([0.0, 0.0]) # [ roll, pitch ]

		self.array_dim = int(np.around(period / dt))

		self.PID_pitch = PID()
		self.PID_roll = PID()

		self.angles = np.zeros((18, self.array_dim))
		self.speeds = np.zeros((18, self.array_dim))

		self.params = np.array(params).reshape(6, 5)
		self.__generate_traj()


	# returns the joint angles and speeds for each of the legs at time t
	def joint_angles(self, t):
		k = int(((t % self.period) / self.period) * self.array_dim)
		return self.angles[:, k]


	def joint_speeds(self, t):
		k = int(((t % self.period) / self.period) * self.array_dim)
		return self.speeds[:, k]


	# Generate trajectories for each leg
	def __generate_traj(self):
		for leg_index in range(6):
			foot_positions, foot_velocities = self.__leg_traj(leg_index)
			joint_angles, joint_speeds = self.__inverse_kinematics(foot_positions, foot_velocities)
			self.angles[leg_index*3:leg_index*3+3,:] = joint_angles
			self.speeds[leg_index*3:leg_index*3+3,:] = joint_speeds

			achieved_positions = self.forward_kinematics(joint_angles)
			valid = np.all(np.isclose(foot_positions, achieved_positions))
			if not valid:
				raise RuntimeError('Desired foot trajectory not achieveable')


	# leg index starts at 0
	# calculates a sequence of points which make up the leg path
	def __leg_traj(self, leg_index):
		leg_angle = (np.pi / 3.0) * (leg_index)
		radius, offset, step_height, phase, duty_factor = self.params[leg_index]
		stride = self.velocity * duty_factor * self.period
		roll, pitch = self.slope
		# start in the body coordinate frame
		# mid / top point
		mid = np.zeros(3)
		mid[0] = self.body_radius * np.cos(leg_angle) + radius * np.cos(leg_angle + offset)
		mid[1] = self.body_radius * np.sin(leg_angle) + radius * np.sin(leg_angle + offset)
		mid[2] = -self.body_height + 0.014 + step_height
		# start point
		start = np.zeros(3)
		start[0] = mid[0] + (stride/2) * np.cos(self.crab_angle)
		start[1] = mid[1] + (stride/2) * np.sin(self.crab_angle)
		start[2] = -self.body_height + 0.014 + (stride/2)*(np.tan(pitch)*np.cos(self.crab_angle) + np.tan(roll)*np.sin(self.crab_angle))
		# end point	
		end = np.zeros(3)
		end[0] = mid[0] - (stride/2) * np.cos(self.crab_angle)
		end[1] = mid[1] - (stride/2) * np.sin(self.crab_angle)
		end[2] = -self.body_height + 0.014 - (stride/2)*(np.tan(pitch)*np.cos(self.crab_angle) + np.tan(roll)*np.sin(self.crab_angle))
		# compute support path
		support_dim = int(np.around(self.array_dim * duty_factor))
		support_positions, support_velocities = self.__support_traj(start, end, support_dim)
		# compute swing path
		swing_dim = int(np.around(self.array_dim * (1.0 - duty_factor)))
		swing_positions, swing_velocities = self.__swing_traj(end, mid, start, swing_dim)
		# combine support and swing paths
		positions = np.append(support_positions, swing_positions, axis=1)
		velocities = np.append(support_velocities, swing_velocities, axis=1)
		# phase shift trajectory
		phase_shift = int(np.around(phase * self.array_dim))
		positions = np.roll(positions, phase_shift, axis=1)
		velocities = np.roll(velocities, phase_shift, axis=1)
		# apply height offset
		z_offset = positions[0] * np.tan(pitch) + positions[1] * np.tan(roll)
		positions[2] = positions[2] + z_offset
		# transform from body frame to leg frame
		R = np.array([[np.cos(-leg_angle), -np.sin(-leg_angle), 0], [np.sin(-leg_angle), np.cos(-leg_angle), 0], [0, 0, 1]])
		positions = R.dot(positions)
		positions[0] -= self.body_radius

		return positions, velocities


	# Generates a smooth trajectory from start point to end point through the via point
	def __swing_traj(self, start, via, end, num):
		t = np.ones((7, num))
		tf = num * self.dt
		time = np.linspace(0, tf, num)

		for i in range(7):
			t[i,:] = np.power(time, i)

		a_0 = start
		a_1 = np.zeros(3)
		a_2 = np.zeros(3)
		a_3 = (2 / (tf ** 3)) * (32 * (via - start) - 11 * (end - start))
		a_4 = -(3 / (tf ** 4)) * (64 * (via - start) - 27 * (end - start))
		a_5 = (3 / (tf ** 5)) * (64 * (via - start) - 30 * (end - start))
		a_6 = -(32 / (tf ** 6)) * (2 * (via - start) - (end - start))

		positions = np.stack([a_0, a_1, a_2, a_3, a_4, a_5, a_6], axis=-1).dot(t)
		velocities = np.stack([a_1, 2*a_2, 3*a_3, 4*a_4, 5*a_5, 6*a_6, np.zeros(3)], axis=-1).dot(t)

		return positions, velocities


	# straight line trajectory for support phase
	# returns a point and velocity
	def __support_traj(self, start, end, num):
		# Straight line from start to end point
		positions = np.linspace(start, end, num, axis=1)
		# the servo velocity is the same for each position
		duration = num * self.dt
		# duration can sometimes be 0
		with np.errstate(divide='ignore', invalid='ignore'):
			velocity = ((end - start) / duration).reshape(3,1)
		# velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
		velocities = np.tile(velocity, num)

		return positions, velocities
			

	# position and speed need to be in the leg coordinate frame
	def __inverse_kinematics(self, foot_position, foot_speed):
		l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3

		x, y, z = foot_position
		dx, dy, dz = foot_speed

		theta_1 = np.arctan2(y, x)

		c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
		c_3 = ((x - l_1 * c_1)**2 + (y - l_1 * s_1)**2 + z**2 - l_2**2 - l_3**2) / (2 * l_2 * l_3)
		s_3 = -np.sqrt(np.maximum(1 - c_3**2, 0)) # maximum ensures not negative

		theta_2 = np.arctan2(z, (np.sqrt((x - l_1 * c_1)**2 + (y - l_1 * s_1)**2))) - np.arctan2((l_3 * s_3), (l_2 + l_3 * c_3))
		theta_3 = np.arctan2(s_3, c_3)

		c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
		c_23 = np.cos(theta_2 + theta_3)

		with np.errstate(all='ignore'):
			theta_dot_1 = (dy*c_1 - dx*s_1) / (l_1 + l_3*c_23 + l_2*c_2)
			theta_dot_2 = (1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + (c_3 / s_3)*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))
			theta_dot_3 = -(1/l_2)*(dz*c_2 - dx*c_1*s_2 - dy*s_1*s_2 + ((l_2 + l_3*c_3)/(l_3*s_3))*(dz*s_2 + dx*c_1*c_2 + dy*c_2*s_1))

		theta_dot_1 = np.nan_to_num(theta_dot_1, nan=0.0, posinf=0.0, neginf=0.0)
		theta_dot_2 = np.nan_to_num(theta_dot_2, nan=0.0, posinf=0.0, neginf=0.0)
		theta_dot_3 = np.nan_to_num(theta_dot_3, nan=0.0, posinf=0.0, neginf=0.0)

		joint_angles = np.array([theta_1, theta_2, theta_3])
		joint_speeds = np.array([theta_dot_1, theta_dot_2, theta_dot_3])

		return joint_angles, joint_speeds


	def forward_kinematics(self, joint_angles):
		l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3
		theta_1, theta_2, theta_3 = joint_angles
		
		# Compute point from joint angles
		x = np.cos(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
		y = np.sin(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
		z = l_3 * np.sin(theta_2 + theta_3) + l_2 * np.sin(theta_2)

		return np.array([x, y, z])


	# feedback for controller
	def IMU_feedback(self, measured_attitude):
		self.slope[0] += self.PID_roll.output(self.base_attitude[0], measured_attitude[0])
		self.slope[1] -= self.PID_pitch.output(self.base_attitude[1], measured_attitude[1])

		self.slope[0] = min(max(-self.angle_limit, self.slope[0]), self.angle_limit)
		self.slope[1] = min(max(-self.angle_limit, self.slope[1]), self.angle_limit)
		# regenerate trajectory
		self.__generate_traj()


# reshapes a 32 length array of floats range 0.0 - 1.0 into the range expected by the controller
def reshape(x):
	x = np.array(x)
	# get body height and velocity
	height = x[0] * 0.2
	velocity = x[1] * 0.5
	leg_params = x[2:].reshape((6,5))
	# radius, offset, step_height, phase, duty_cycle
	param_min = np.array([0.0, -1.745, 0.01, 0.0, 0.0])
	param_max = np.array([0.3, 1.745, 0.2, 1.0, 1.0])
	# scale and shifted params into the ranges expected by controller
	leg_params = leg_params * (param_max - param_min) + param_min

	return height, velocity, leg_params


# single PID controller
class PID:
	def __init__(self, dt=0.1, Kp=1.10, Ki=2.21, Kd=0.032):
		self.dt = dt
		self.Kp = Kp
		self.Ki = Ki
		self.Kd = Kd
		self.integral_prior = 0
		self.error_prior = 0


	def output(self, setpoint, measured):
		error = measured - setpoint
		integral = self.integral_prior + error * self.dt
		derivative = (error - self.error_prior) / self.dt

		self.integral_prior = integral
		self.error_prior = error

		return self.Kp * error + self.Ki * integral + self.Kd * derivative


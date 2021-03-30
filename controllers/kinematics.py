import numpy as np

l_1 = 0.05317 # coxa link
l_2 = 0.10188 # femur link
l_3 = 0.14735 # tibia link


# transforms foot positions and speeds into joint angles and speeds in the leg coordinate frame
def inverse(foot_position, foot_speed):

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


# transforms leg joint angles into foot positions in leg coordinate frame
def forward(joint_angles):
	l_1, l_2, l_3 = self.l_1, self.l_2, self.l_3
	theta_1, theta_2, theta_3 = joint_angles
	
	# Compute point from joint angles
	x = np.cos(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
	y = np.sin(theta_1) * (l_1 + l_3 * np.cos(theta_2 + theta_3) + l_2 * np.cos(theta_2))
	z = l_3 * np.sin(theta_2 + theta_3) + l_2 * np.sin(theta_2)

	return np.array([x, y, z])


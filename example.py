from simulator import Simulator
from controllers.kinematic import Controller, tripod_gait

controller = Controller(tripod_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[])

# run indefinitely
while True:
	simulator.step()

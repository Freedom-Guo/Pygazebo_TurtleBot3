import random
import numpy as np
import time
import math

import _thread

from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

import pygazebo as gazebo

class MultiEnv(object):
	def __init__(self):
		gazebo.initialize()
		self.world = gazebo.new_world_from_file("/home/sust-gtw/3D_collisionavoidance/world/turtlebot3_stage_2.world")
		self.agents = self.world.get_agents()
		
		action_low = np.array([-0.5, -1.5707])
		action_high = np.array([0.5, 1.5707])
		self.action_space = spaces.Box(low = action_low, high = action_high)
		
		observation_low = [-1.9, -1.9, -0.5, -1.5707, 0.2, 0.0, 0.2, 0.0]
		for i in range(360):
			observation_low.append(0.0)
		observation_low = np.array(observation_low)
			
		observation_high = [1.9, 1.9, 0.5, 1.5707, 1.9*(2**0.5), 2*np.pi, 1.9*(2**0.5), 2*np.pi]
		for i in range(360):
			observation_high.append(0.0)
		observation_high = np.array(observation_high)
		
		self.observation_space = spaces.Box(low = observation_low, high = observation_high)
		
		self.lidar_add = ["default::turtlebot3_waffle::turtlebot3_waffle::lidar::hls_lfcd_lds", "default::turtlebot3_waffle1::turtlebot3_waffle::lidar::hls_lfcd_lds", "default::turtlebot3_waffle2::turtlebot3_waffle::lidar::hls_lfcd_lds"]
		self.state = [[0] * (8) for _ in range(3)]
		self.observation = [[0] * (368) for _ in range(3)]
		self.capture_flag = False
		self.seed()
		self.steps_beyond_done = None

	def seed(self, seed=None):
        	self.np_random, seed = seeding.np_random(seed)
        	return [seed]
	
	def reset(self):
		self.state[0][0] = 0.0
		self.state[0][1] = -5.0
		self.state[1][0] = -2.0
		self.state[1][1] = 5.0 
		self.state[2][0] = 2.0
		self.state[2][1] = 5.0
		for i in range(3):
			self.state[i][2] = random.uniform(-0.5, 0.5)
			self.state[i][3] = random.uniform(-1.5707, 1.5707)
			d = np.array([self.state[(i+1)%3][0], self.state[(i+1)%3][1]])-np.array([self.state[i][0], self.state[i][1]])
			self.state[i][4] = np.sqrt(np.sum(np.square(d)))
			if d[0] > 0 and d[1] >= 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0]))
			elif d[0] > 0 and d[1] < 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
			elif d[0] < 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0])) + np.pi
			elif d[0] == 0:
				if d[1] > 0:
					self.state[i][5] = 1/2*np.pi
				else:
					self.state[i][5] = 3/2*np.pi
			d = np.array([self.state[(i+2)%3][0], self.state[(i+2)%3][1]])-np.array([self.state[i][0], self.state[i][1]])
			self.state[i][6] = np.sqrt(np.sum(np.square(d)))
			if d[0] > 0 and d[1] >= 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0]))
			elif d[0] > 0 and d[1] < 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
			elif d[0] < 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0])) + np.pi
			elif d[0] == 0:
				if d[1] > 0:
					self.state[i][7] = 1/2*np.pi
				else:
					self.state[i][7] = 3/2*np.pi
			self.agents[i].set_pose(((self.state[i][0], self.state[i][1], 0.0), (0.0, 0.0, 0.0)))
			obs = self.agents[i].get_ray_observation(self.lidar_add[i])
			npdate = np.array(obs, copy=False)
			for j in range(8):
				self.observation[i][j] = self.state[i][j]
			self.observation[i][8:] = npdate[:]
		return np.array(self.state)
	
	def step(self, action):
		for i in range(3):
			self.agents[i].set_twist(((action[i][0], 0.0, 0.0), (0.0, 0.0, action[i][1])))
		self.world.step(200)
		for i in range(3):
			self.state[i][0] = self.agents[i].get_pose()[0][0]
			self.state[i][1] = self.agents[i].get_pose()[0][1]
			self.state[i][2] = self.agents[i].get_twist()[0][0]
			self.state[i][3] = self.agents[i].get_twist()[0][1]
		for i in range(3):
			d = np.array([self.state[(i+1)%3][0], self.state[(i+1)%3][1]])-np.array([self.state[i][0], self.state[i][1]])
			self.state[i][4] = np.sqrt(np.sum(np.square(d)))
			if d[0] > 0 and d[1] >= 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0]))
			elif d[0] > 0 and d[1] < 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
			elif d[0] < 0:
				self.state[i][5] = math.atan(math.tan(d[1]/d[0])) + np.pi
			elif d[0] == 0:
				if d[1] > 0:
					self.state[i][5] = 1/2*np.pi
				else:
					self.state[i][5] = 3/2*np.pi
			d = np.array([self.state[(i+2)%3][0], self.state[(i+2)%3][1]])-np.array([self.state[i][0], self.state[i][1]])
			self.state[i][6] = np.sqrt(np.sum(np.square(d)))
			if d[0] > 0 and d[1] >= 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0]))
			elif d[0] > 0 and d[1] < 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0])) + 2*np.pi
			elif d[0] < 0:
				self.state[i][7] = math.atan(math.tan(d[1]/d[0])) + np.pi
			elif d[0] == 0:
				if d[1] > 0:
					self.state[i][7] = 1/2*np.pi
				else:
					self.state[i][7] = 3/2*np.pi
			obs = self.agents[i].get_ray_observation(self.lidar_add[i])
			npdate = np.array(obs, copy=False)
			for j in range(8):
				self.observation[i][j] = self.state[i][j]
			self.observation[i][8:] = npdate[:]
		
		if self.state[2][4] <= 0.05 or self.state[2][6] <= 0.05:
			self.capture_flag = True
		
		done = self.capture_flag
		
		if done == 0:
			evader_reward = 1.0
			pursuer_reward = 0.0
		elif self.steps_beyond_done is None:
			self.steps_beyond_done = 0
			pursuer_reward = 10.0
			evader_reward = -10.0
		else:
			self.steps_beyond_done += 1
			pursuer_reward = 10.0
			evader_reward = -10.0
		reward = np.array([evader_reward, pursuer_reward])

		obs_all = self.agents[0].get_camera_observation("default::camera::camera_link::camera")
		npdata_all = np.array(obs_all, copy=False)
		
		return self.observation, reward, done, npdata_all

if __name__ == "__main__":
	env = MultiEnv()
	while True:
		env.reset()
		while True:
			obs, reward, dones, _ = env.step([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
			plt.imshow(_)
			plt.show()
			print(obs)
			if dones:
				break

# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import random
import pygazebo as gazebo
import numpy as np
import matplotlib.pyplot as plt
import time
import math

gazebo.initialize()
world = gazebo.new_world_from_file("/home/freedomguo/3D_collisionavoidance/world/turtlebot3_stage_2.world")
# world = gazebo.new_world_from_file("../worlds/pioneer_laser.world")
agents = world.get_agents()
agents[0].get_joint_names()
agents[1].get_joint_names()
agents[2].get_joint_names()

world.info()
agents[0].set_pose(((-1, -1, 0), (0, 0, 0)))
agents[1].set_pose(((1, 1, 0), (0, 0, 0)))
agents[2].set_pose(((0, 0, 0), (0, 0, 0)))
agents[3].set_pose(((1, -1, 0), (0, 0, 0)))
agents[4].set_pose(((-1, 1, 0), (0, 0, 0)))


# class evader_observation


for i in range(10000000):
    # observation = agent.sense()
    # add reward and text to observation
    # action = model.compute_action(observation)
    # agent.take_action(action)
    # len = random.randint(10, 20)
    
    len = 200
    print(len)
    # agent.set_pose(((random.random() * (-1.9), random.random() * 1.9, 0.00), (0.00, 0.00, 0.00)))
    print("start sim time:")
    print(world.get_sim_time())
    print("start wall time:")
    print(world.get_wall_time())
    time_start = time.time()
    for j in range(len):
        agents[0].set_twist(((0.4, 0, 0),(0, 0, 0)))
        # print(world.get_sim_time())
        # time_start=world.get_sim_time()
        world.step()
        # print(world.get_sim_time())
        # time_end=world.get_sim_time()
        # timer = time_end - time_start
        # print(timer)
    time_end = time.time()
    print(time_end - time_start)
    pose = []
    pose.append(agents[0].get_pose())
    pose.append(agents[1].get_pose())
        # print(agent.get_twist())
    print("end sim time:")
    print(world.get_sim_time())
    print("end wall time:")
    print(world.get_wall_time())
    print(pose)
    # world.info()
    if i % 2 == 1:
        obs1 = agents[0].get_ray_observation(
            "default::pursuer0::turtlebot3_waffle::lidar::hls_lfcd_lds")
        obs2 = agents[0].get_ray_observation(
            "default::pursuer1::turtlebot3_waffle::lidar::hls_lfcd_lds")
        obs3 = agents[0].get_camera_observation("default::camera::camera_link::camera")
        # print(obs1)
        npdata1 = np.array(obs1, copy=False)
        npdata2 = np.array(obs2, copy=False)
        npdata3 = np.array(obs3, copy=False)
        plt.imshow(npdata3)
        plt.show()
        print(npdata1)

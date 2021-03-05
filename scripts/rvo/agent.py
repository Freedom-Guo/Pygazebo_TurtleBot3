import rvo23d

import numpy as np


class RVO(object):
    def __init__(self,
                 num_agents,
                 time_step=0.01,
                 safety_space=0.1,
                 neighbor_dist=10.0,
                 max_neighbors=5,
                 time_horizon=5.0,
                 radius=0.6,
                 max_speed=np.sqrt(3)):

        self.num_agents = num_agents
        
        # RVO parameters
        self.time_step = time_step
        self.safety_space = safety_space
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.radius = radius
        self.max_speed = max_speed

        self.sim = None

    def step(self, poses, vels, goals, terminates):
        if self.sim is not None and \
           self.sim.getNumAgents() != self.num_agents:

           del self.sim
           self.sim = None

        if self.sim is None:
            self.sim = rvo23d.PyRVOSimulator(self.time_step,
                                             self.neighbor_dist,
                                             self.max_neighbors,
                                             self.time_horizon,
                                             self.radius,
                                             self.max_speed,
                                             (0, 0, 0))

            for pose, vel in zip(poses, vels):
                self.sim.addAgent(tuple(pose),
                                  self.neighbor_dist,
                                  self.max_neighbors,
                                  self.time_horizon,
                                  self.radius + 0.01 + self.safety_space,
                                  self.max_speed,
                                  vel)
        else:
            for i, (pose, vel, goal, done) in enumerate(zip(poses,
                                                            vels,
                                                            goals,
                                                            terminates)):
                self.sim.setAgentPosition(i, tuple(pose))
                self.sim.setAgentVelocity(i, tuple(vel))
                if done:
                    pref_vel = tuple(np.zeros(3))
                else:
                    velocity = goal - pose
                    speed = np.linalg.norm(velocity)
                    pref_vel = (velocity) / (speed) if speed > 1.0 else (velocity)
                
                self.sim.setAgentPrefVelocity(i, tuple(pref_vel))

        for _ in range(int(0.5 / self.time_step)):
            self.sim.doStep()

        acts = []
        for i, done in enumerate(terminates):
            if done:
                acts.append(np.zeros(3))
            else:
                acts.append(self.sim.getAgentVelocity(i))

        return acts
        
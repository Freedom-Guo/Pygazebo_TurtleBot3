import time 
import _thread

import numpy as np
import pygazebo as gazebo

from scenarios import Scenarios
from utils import timed


class GazeboEnv(object):
    def __init__(self,
                 num_agents, 
                 agent_size=0.4, 
                 env_size=8.0,
				 neighbor_dist=10.0, 
                 neighbor_num=5,  
                 noise_level=1.0):

        self.num_agents = num_agents
        self.agent_size = agent_size
        self.env_size = env_size
        self.arrived_distance = 0.5
        self.noise_level = noise_level

        self.neighbor_dist = neighbor_dist
        self.neighbor_num = neighbor_num

        self.distance_matrix = np.zeros(
            (self.num_agents, self.num_agents)
            )

        self.robot_pos = np.zeros((self.num_agents, 3))
        self.robot_noise_pos = np.zeros((self.num_agents, 3))
        self.robot_last_pos = np.zeros((self.num_agents, 3))
        self.robot_vel = np.zeros((self.num_agents, 3))
        self.robot_noise_vel = np.zeros((self.num_agents, 3))

        self.obs = []

        self.scenarios = Scenarios(
            self.num_agents,  
            self.agent_size, self.env_size
            )

        gazebo.initialize()
        self._world = gazebo.new_world_from_file(
            "../world/multi_robots.world"
        )

        self._agents = []
        for i in range(self.num_agents):
            name = "Robot_%03d" % (i)
            self._agents.append(
                self._world.get_agent(name)
            )

        self._goals = []
        for i in range(self.num_agents):
            name = "Goal_%03d" % (i)
            self._goals.append(
                self._world.get_model(name)
            )

    def _get_pose(self, agents):
        poses = []
        for agent in agents:
            pose, _ = agent.get_pose()
            poses.append(np.array(pose))
        return np.array(poses)

    def _get_twist(self, agents):
        twists = []
        for agent in agents:
            linear, _ = agent.get_twist()
            twists.append(np.array(linear))
        return np.array(twists)

    def _set_pose(self, agents, poses):
        for agent, pos in zip(agents, poses):
            agent.set_pose(
                ((pos[0], pos[1], pos[2]), (0, 0, 0))
                )

    def _set_twist(self, agents, twists):
        for agent, twist in zip(agents, twists):
            agent.set_twist(
                ((twist[0], twist[1], twist[2]), (0, 0, 0))
            )

    def _collision_thread(self, ind_a): 
        def angle(ind_a, ind_b):
            ab = self.robot_pos[ind_a] - self.robot_pos[ind_b]
            ba = -ab
            angle_a = ab.dot(self.robot_vel[ind_a])/(np.sqrt(ab.dot(ab))*(np.sqrt(self.robot_vel[ind_a].dot(self.robot_vel[ind_a]))))
            angle_b = ba.dot(self.robot_vel[ind_b])/(np.sqrt(ba.dot(ba))*(np.sqrt(self.robot_vel[ind_b].dot(self.robot_vel[ind_b]))))
            return angle_a, angle_b

        for ind_b in range(ind_a+1, self.num_agents):  # Do not campare with self
            self.distance_matrix[ind_a,ind_b] = self.distance_matrix[ind_b, ind_a] = np.linalg.norm(self.robot_pos[ind_a] - self.robot_pos[ind_b])
            if self.distance_matrix[ind_a, ind_b] < self.agent_size:
                self.collision_flag[ind_a] = True
                self.collision_flag[ind_b] = True

        if ind_a == self.num_agents -1:
            self.thread_lock_collision = 0
        _thread.exit()

    #collision detection main function 
    def _collision_detection(self):
        for ind_a in range(self.num_agents):
            _thread.start_new_thread(self._collision_thread, (ind_a, ))
            
    def _arrived_check(self):
        for i, (robot, goal) in enumerate(zip(self.robot_pos, self.goals)):
            if np.linalg.norm(robot - goal) < self.arrived_distance:
                self.arrived_flag[i] = True
        self.thread_lock_arrive = 0
        _thread.exit()

    def _get_obses(self, poses, vels, goals):
        def get_neighbors(i, poses):
            inds = np.delete(np.arange(self.num_agents),i)
            ds = np.delete(self.distance_matrix[:,i],i)
            return inds, ds

        def sort_neighbors(i, inds, ds, poses, vels):
            ob = []
            pos, vel = poses[i], vels[i]
            sorted_inds = inds[np.lexsort((inds, ds))]
            if len(inds) < self.neighbor_num:
                for ind in sorted_inds:
                    ob.extend((poses[ind] - pos) / self.neighbor_dist)
                    ob.extend(vels[ind] - vel)
                for _ in range(self.neighbor_num - len(inds)):
                    ob.extend(np.ones(6, dtype=np.float32))
            else:
                for i, ind in enumerate(sorted_inds):
                    if i < self.neighbor_num:
                        ob.extend((poses[ind] - pos) / self.neighbor_dist)
                        ob.extend(vels[ind] - vel)
            return ob

        self.obs = []
        for i in range(len(poses)):
            inds, ds = get_neighbors(i, poses)
            ob = sort_neighbors(i, inds, ds, poses, vels)
            velocity = goals[i] - poses[i]
            speed = np.linalg.norm(velocity)
            pref_vel = (velocity) / (speed) if speed > 1.0 else (velocity)
            ob.extend(pref_vel)
            self.obs.append(ob)

        if self.thread_lock_ob:
            self.thread_lock_ob = 0
            _thread.exit()

    def reset(self):
        self.thread_lock_ob = 0
        self.starts, self.goals = self.scenarios.random_scene()
        self.perfect_distance = [np.linalg.norm(s - g) for s, g in zip(self.starts, self.goals)]
        
        self._set_pose(self._agents, self.starts)
        self._set_pose(self._goals, self.goals)

        self.collision_flag = [False for _ in range(self.num_agents)]
        self.arrived_flag = [False for _ in range(self.num_agents)]

        self.robot_pos = self._get_pose(self._agents)
        self.robot_vel = self._get_twist(self._agents)
        self.robot_noise_pos = self.robot_pos.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_pos.shape)
        self.robot_noise_vel = self.robot_vel.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_vel.shape)

        self._get_obses(
            self.robot_noise_pos, 
            self.robot_noise_vel,
            self.goals)

        self.robot_last_pos = self.robot_pos.copy()

        self.trajectory = np.zeros(self.num_agents)

        return self.obs
        
    def step(self, acts):
        def wait_thread():
            while True:
                if self.thread_lock_arrive == 0 and self.thread_lock_collision == 0:
                    break
        def wait_ob_thread():
            while True:
                if self.thread_lock_ob == 0:
                    break

        self.thread_lock_arrive = 1
        self.thread_lock_collision = 1
        self.thread_lock_ob = 1

        # with timed("gazebo"):
        self.robot_pos = self._get_pose(self._agents)
        self.robot_vel = self._get_twist(self._agents)
        self.robot_noise_pos = self.robot_pos.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_pos.shape)
        self.robot_noise_vel = self.robot_vel.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_vel.shape)
        # acts = self.get_action()
        self.take_acts(acts)
        # with timed("check"):
        self._collision_detection()
        _thread.start_new_thread(self._arrived_check, ())
        _thread.start_new_thread(
            self._get_obses, 
            (self.robot_noise_pos,
            self.robot_noise_vel,
            self.goals))
        wait_thread() # wait until collision detection thread finished
        rewards, dones = self.reward_function(acts)
        self.get_trajectory()
        self.robot_last_pos = self.robot_pos.copy()
        wait_ob_thread() # wait until obses get

        # with timed("step"):
        self._world.step(10)

        return self.obs, rewards, dones

    def take_acts(self, acts):
        self._set_twist(self._agents, acts)

    def reward_function(self, acts):
        rewards, dones = [], []
        for act, r_last_pos, r_pos, g_pos, arrive, collision in \
            zip(acts, self.robot_last_pos, self.robot_pos, self.goals,
                self.arrived_flag, self.collision_flag):
            if arrive:
                dones.append(True)
                rewards.append(20.0)
            elif collision:
                dones.append(True)
                rewards.append(-20.0)
            else:
                dones.append(False)
                last_d = np.linalg.norm(r_last_pos - g_pos)
                curr_d = np.linalg.norm(r_pos - g_pos)
                approaching_reward = 2.5 * (last_d - curr_d)
                # energy_reward = -0.5 * np.linalg.norm(act)
                reward = approaching_reward
                rewards.append(reward)

        rewards = np.array(rewards)
        return rewards, dones  

    def get_action(self):
        acts = []
        for goal, pos in zip(self.goals, self.robot_pos):
            velocity = goal - pos
            speed = np.linalg.norm(velocity)
            pref_vel = (velocity) / (speed) if speed > 1.0 else (velocity)
            acts.append(pref_vel)
        return acts
    
    def get_trajectory(self):
        robot_pos = np.asarray(self.robot_pos)
        robot_last_pos = np.asarray(self.robot_last_pos)
        distance = np.linalg.norm(robot_pos-robot_last_pos, axis=1)
        self.trajectory += distance
    
if __name__ == "__main__":
    env = GazeboEnv(num_agents=10)
    while True:
        env.reset()
        while True:
            _, _, dones = env.step()
            if all(dones):
                break
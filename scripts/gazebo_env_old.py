import rospy
import math
import random
import sys
import copy
import thread
import time

import numpy as np

from scenarios import Scenarios
from visualizer import RvizVisulizer
from utils import timed


class GazeboEnv(object):
    def __init__(self,
                 num_agents, 
                 num_input_seq=5, 
                 agent_size=0.5, 
                 env_size=20.0,
                 neighbor_dist=10.0, 
                 neighbor_num=5,  
                 noise_level=1.0):

        rospy.init_node('gazebo_env')

        self.visulizer = RvizVisulizer(num_agents, agent_size)

        self.pub_rate = rospy.Rate(10)
        self._build_ros()

        self.num_agents = num_agents
        self.num_input_seq = num_input_seq
        self.agent_size = agent_size
        self.env_size = env_size
        self.arrived_distance = 0.5
        self.noise_level = noise_level

        self.neighbor_dist = neighbor_dist
        self.neighbor_num = neighbor_num

        self.scenarios = Scenarios(
            self.num_agents,  
            self.agent_size, self.env_size)

        self.robot_pos = np.zeros((self.num_agents, 3))
        self.robot_noise_pos = np.zeros((self.num_agents, 3))
        self.robot_last_pos = np.zeros((self.num_agents, 3))
        self.robot_vel = np.zeros((self.num_agents, 3))
        self.robot_noise_vel = np.zeros((self.num_agents, 3))

        #save distance between each agent in this matrix
        self.distance_matrix = np.zeros((self.num_agents, self.num_agents))
        #using global variance to avoid using return in child-thread
        self.obses = []

        self.collision_flag = [False for _ in range(self.num_agents)]
        self.arrived_flag = [False for _ in range(self.num_agents)]
        self.last_distance = np.zeros([self.num_agents])

        #self.acts = np.zeros((self.num_agents, 3))

        self.start()

    def _build_ros(self):
        from std_srvs.srv import Empty
        from gazebo_msgs.srv import SetModelState
        from gazebo_msgs.msg import ModelStates
        from gazebo_msgs.msg import ModelState
        # ROS Subscriber
        rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self._models_cb
            )
        # ROS Service
        self.unpause = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty
            )
        self.pause = rospy.ServiceProxy(
            '/gazebo/pause_physics', Empty
            )
        self.set_model_srv = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState
            )
        self.set_model_pub = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=0
            )

    def start(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException:
            print "/gazebo/unpause_physics service call failed"

    def stop(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException:
            print "/gazebo/pause_physics service call failed"

    #collision detection thread for ind_a round
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
        thread.exit()

    #collision detection main function 
    def _collision_detection(self):
        for ind_a in range(self.num_agents):
            thread.start_new_thread(self._collision_thread, (ind_a, ))
            
    def _arrived_check(self):
        for i, (robot, goal) in enumerate(zip(self.robot_pos, self.goals)):
            if np.linalg.norm(robot - goal) < self.arrived_distance:
                self.arrived_flag[i] = True
        self.thread_lock_arrive = 0
        thread.exit()
        
    def _models_cb(self, msg):
        msg_count = 0
        for name in msg.name:
            if name[:6] == "Robot_":
                ind = int(name[-3:])
                self.robot_pos[ind, 0] = msg.pose[msg_count].position.x
                self.robot_pos[ind, 1] = msg.pose[msg_count].position.y
                self.robot_pos[ind, 2] = msg.pose[msg_count].position.z
                self.robot_vel[ind, 0] = msg.twist[msg_count].linear.x
                self.robot_vel[ind, 1] = msg.twist[msg_count].linear.y
                self.robot_vel[ind, 2] = msg.twist[msg_count].linear.z

            msg_count = msg_count + 1

        self.robot_noise_pos = self.robot_pos.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_pos.shape)
        self.robot_noise_vel = self.robot_vel.copy() + np.random.normal(
            scale=self.noise_level, size=self.robot_vel.shape)

    def _set_model(self, name, pos, vel):
        from gazebo_msgs.msg import ModelState
        state = ModelState()
        state.model_name = name
        state.pose.position.x = pos[0]
        state.pose.position.y = pos[1]
        state.pose.position.z = pos[2]
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1
        state.twist.linear.x = vel[0]
        state.twist.linear.y = vel[1]
        state.twist.linear.z = vel[2]
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        state.reference_frame = 'world'
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_srv.call(state)
        except rospy.ServiceException:
            print "/gazebo/set_model_state service call failed"
        #self.set_model_pub.publish(state)

    def _get_obses(self, poses, vels, goals):
        def get_neighbors(i, poses):
            inds = np.delete(np.arange(self.num_agents),i)
            ds = np.delete(self.distance_matrix[:,i],i)
            return inds, ds

        def sort_neighbors(i, inds, ds, poses, vels):
            obs = []
            pos, vel = poses[i], vels[i]
            sorted_inds = inds[np.lexsort((inds, ds))]
            if len(inds) < self.neighbor_num:
                for ind in sorted_inds:
                    obs.extend((poses[ind] - pos) / self.neighbor_dist)
                    obs.extend(vels[ind] - vel)
                for _ in range(self.neighbor_num - len(inds)):
                    obs.extend(np.ones(6, dtype=np.float32))
            else:
                for i, ind in enumerate(sorted_inds):
                    if i < self.neighbor_num:
                        obs.extend((poses[ind] - pos) / self.neighbor_dist)
                        obs.extend(vels[ind] - vel)
            return obs

        self.obses = []
        for i in range(len(poses)):
            inds, ds = get_neighbors(i, poses)
            obs = sort_neighbors(i, inds, ds, poses, vels)
            velocity = goals[i] - poses[i]
            speed = np.linalg.norm(velocity)
            pref_vel = (velocity) / (speed) if speed > 1.0 else (velocity)
            obs.extend(pref_vel)
            self.obses.append(obs)

        if self.thread_lock_obse:
            self.thread_lock_obse = 0
            thread.exit()

    def reset(self):
        self.stop()
        self.thread_lock_obse = 0
        # temp = np.random.random()
        # if temp > 0.5:
        # 	self.starts, self.goals = self.scenarios.random_scene()
        # elif temp < 0.2:
        # 	self.starts, self.goals = self.scenarios.circle_scene()
        # else:
        # self.starts, self.goals = self.scenarios.random_ball_scene()
        self.starts, self.goals = self.scenarios.multi_scenes()


        self.perfect_distance = [np.linalg.norm(s - g) for s, g in zip(self.starts, self.goals)]
        self.start()
        for i, (s, g) in enumerate(zip(self.starts, self.goals)):
            self._set_model("Robot_%03d" % i, s, np.zeros(3))
            self._set_model("Goal_%03d" % i, g, np.zeros(3))
            self.collision_flag[i] = False
            self.arrived_flag[i] = False

        self.visulizer.reset(self.starts, self.goals)

        self._get_obses(
            self.starts, 
            np.zeros((self.num_agents, 3)),
            self.goals)
        
        self.robot_last_pos = self.robot_pos.copy()
        time.sleep(1)
        return self.obses

    def render(self, acts):
        self.visulizer.display(
            self.robot_pos, self.robot_last_pos, acts
            )

    def step(self, acts):
        def wait_thread():
            while True:
                if self.thread_lock_arrive == 0 and self.thread_lock_collision == 0:
                    break
        def wait_obse_thread():
            while True:
                if self.thread_lock_obse == 0:
                    break

        self.thread_lock_arrive = 1
        self.thread_lock_collision = 1
        self.thread_lock_obse = 1

        self.take_acts(acts)
        
        self._collision_detection()
        thread.start_new_thread(self._arrived_check, ())
        thread.start_new_thread(
            self._get_obses, 
            (self.robot_noise_pos,
             self.robot_noise_vel,
             self.goals))
        wait_thread() # wait until collision detection thread finished
        rewards, dones = self.reward_function(acts)
        self.robot_last_pos = self.robot_pos.copy()
        wait_obse_thread() # wait until obses get
        self.pub_rate.sleep()
        return self.obses, rewards, dones

    def _pub_act(self, i, act):
        self._set_model("Robot_%03d" % i, self.robot_pos[i], act)

    def take_acts(self, acts):
        for i, (act, done) in enumerate(zip(acts, self.collision_flag)):
            if done:
                self._pub_act(i, np.zeros(3))
            else:
                self._pub_act(i, act)   
        
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
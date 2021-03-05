import argparse
import time
import csv
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from env_creator import Creator
from gazebo_env import GazeboEnv

from ddpg.agent import Agent

parser = argparse.ArgumentParser(
    description='Multi-Robot Collision Avoidance with Local Sensing via'
    'Deep Reinforcement Learning')

parser.add_argument(
    '--train', 
    action="store_true")
parser.add_argument(
    '--num_agents', 
    default=10, 
    type=int)
parser.add_argument(
    '--env_size', 
    default=4.0, 
    type=float)
parser.add_argument(
    '--buffer_size', 
    default=500000,
    type=int
)
parser.add_argument(
    '--batch_size', 
    default=1024,
    type=int
)
parser.add_argument(
    '--gamma',
    default=0.99,
    type=float
)
parser.add_argument(
    '--sigma',
    default=0.2,
    type=float
)
parser.add_argument(
    '--theta',
    default=0.15,
    type=float
)
parser.add_argument(
    '--beta',
    default=0.4,
    type=float
)
parser.add_argument(
    '--alpha',
    default=0.6,
    type=float
)
parser.add_argument(
    '--neighbor_num',
    default=10, 
    type=int)
parser.add_argument(
    '--point', 
    default=200, 
    type=int)
parser.add_argument(
    '--write', 
    action='store_true')
parser.add_argument(
    '--episode_max_steps',
    default=600,
    type=int
)
parser.add_argument(
    '--train_max_iters',
    default=40000,
    type=int
)

args = parser.parse_args()

class Runner(object):
    def __init__(self, env, agent, write=False):
        self.env = env
        self.agent = agent

        self.num_agents = args.num_agents
        self.episodes_counter = 0

        self.write_flag = write
        if self.write_flag:
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            outfile = open("./ddpg/{}.csv".format(
                time_stamp), 'w')
            self.csvwriter = csv.writer(outfile)

    def _rollout(self):
        # use multiple agents to collect paths in one episode
        # n: number of agents
        terminateds = [False for _ in range(args.num_agents)]
        terminate_idxs = [0 for _ in range(args.num_agents)]
        terminate_flag = [False for _ in range(args.num_agents)]
        paths = defaultdict(list)

        obs = self.env.reset()
        self.agent.actor.reset_noise()

        dones_id = np.zeros(self.num_agents)

        for _ in range(args.episode_max_steps):
            obs = self.agent.obfilt(obs)

            acts = self.agent.actor.act(
                obs, terminateds
            )

            paths["action"].append(acts)

            next_obs, rews, dones = self.env.step(acts)

            paths["reward"].append(rews)
            # self.plot_reward(np.asarray(reward_agents))
            next_obs = self.agent.obfilt(next_obs)

            for i, done in enumerate(dones):
                if done:
                    dones_id[i] += 1

            self.agent.perceive(
                obs,
                acts, 
                rews, 
                next_obs, 
                dones_id
                )

            obs = next_obs

            for i, d in enumerate(dones):
                if d:
                    terminateds[i] = True
                    if terminate_flag[i] is False:
                        terminate_idxs[i] += 1
                    terminate_flag[i] = True
                else:
                    terminate_idxs[i] += 1
            if all(terminateds):
                break

        path_agents = []
        for i in range(args.num_agents):
            path = defaultdict(list)
            for k, v in paths.items():
                v = np.asarray(v)
                # print 'k: ', k, '   v: ', v[:terminate_idxs[i], i]
                path[k] = np.array(v[:terminate_idxs[i], i])
                path["terminated"] = terminateds[i]
            path["done_id"] = terminate_idxs[i]
            path_agents.append(path)

        return path_agents

    def _get_paths(self):
        paths_batch = []
        timesteps_counter = 0
        while True:
            self.episodes_counter += 1
            print("***** Episode {} *****".format(self.episodes_counter))
            path_agents = self._rollout()
            for path in path_agents:
                paths_batch.append(path)
                timesteps_counter += len(path["reward"])
            break            

        return paths_batch

    def _print_statistics(self, stats):
        print(
            "*********** Iteration {} ************".format(stats["Iteration"]))
        print(tabulate(
            filter(lambda k_v: np.asarray(k_v[1]).size == 1, stats.items()),
            tablefmt="grid"))

    def run(self):
        iterCounter = 0
        while iterCounter < args.train_max_iters:
            iterCounter += 1
            tstart = time.time()
            paths = self._get_paths()
            stats = self.agent.visualize(
                paths, iterCounter, 
                self.env.perfect_distance, self.env.trajectory,
                self.env.arrived_flag, self.env.collision_flag
                )
            stats["Iteration"] = iterCounter
            if args.train:
                # stats["TimeElapsed"] = time.time() - tstart
                stats["Fps"] = stats["EpPathLengthsSum"] / (time.time() - tstart)            
            self._print_statistics(stats)


if __name__ == "__main__":
    creator = Creator(
    robot_num=args.num_agents,
    env_size=args.env_size)
    creator.run()

    # set tf graph and session
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(graph=graph, config=config)

    # initialize env, agent and algorithm
    env = GazeboEnv(num_agents=args.num_agents,
                    env_size=args.env_size,
                    neighbor_num=args.neighbor_num)

    agent = Agent(args, sess)

    runner = Runner(env, agent)
    runner.run()

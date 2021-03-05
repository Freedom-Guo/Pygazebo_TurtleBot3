import time
import argparse
import csv

import numpy as np
import tensorflow as tf

from tabulate import tabulate
from env_creator import Creator
from gazebo_env import GazeboEnv
from ppo.agent import Agent
from ppo.ppo import PPO
from collections import defaultdict, OrderedDict
from utils import timed


class Runner(object):
    def __init__(self, env, agent, alg, write, is_training=False, point=300):
        self.env = env
        self.agent = agent
        self.alg = alg

        self.episodes_counter = 0
        self.episode_max_steps = 300
        self.batch_max_steps = 20000
        self.train_max_iters = 10000
        self.is_training = is_training

        if not is_training:
            self.agent.actor.load_net("./log/model/actor_{}".format(
                point
            ))
            self.agent.critic.load_net("./log/model/critic_{}".format(
                point
            ))
            self.agent.load_ms("./log/model/{}".format(
                point
            ))

        self.write_flag = write
        if self.write_flag:
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            outfile = open("./ppo/{}.csv".format(
                time_stamp), 'w')
            self.csvwriter = csv.writer(outfile)

    def _rollout(self):
        terminateds = [False for _ in range(self.env.num_agents)]
        terminate_idxs = [0 for _ in range(self.env.num_agents)]
        terminate_flag = [False for _ in range(self.env.num_agents)]
        paths = defaultdict(list)
        obses = self.env.reset()
        for _ in range(self.episode_max_steps):
            # with timed("step"):
            obses = self.agent.obfilt(obses)

            acts, values = self.agent.step(
                obses, terminateds, self.is_training)

            # self.env.render(acts)

            paths["obses"].append(obses)
            paths["acts"].append(acts)
            paths["values"].append(values)

            obses, rews, dones = self.env.step(acts)
            paths["reward"].append(rews)

            for i, done in enumerate(dones):
                if done:
                    terminateds[i] = True
                    if not terminate_flag[i]:
                        terminate_idxs[i] += 1
                    terminate_flag[i] = True
                else:
                    terminate_idxs[i] += 1

            if all(terminateds):
                break

        path_agents = []
        for i in range(self.env.num_agents):
            path = defaultdict(list)
            for k, v in paths.items():
                v = np.asarray(v)
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

            if timesteps_counter > self.batch_max_steps or (not self.is_training):
                break

        return paths_batch

    # def _print_statistics(self, stats):
    #     print(
    #         "*********** Iteration {} ************".format(stats["Iteration"]))
    #     print(tabulate(
    #         filter(lambda (k, v): np.asarray(v).size == 1, stats.items()),
    #         tablefmt="grid"))

    def _get_stats(self, paths):
        stats = OrderedDict()
        rew_agent = np.array([path["reward"].sum() for path in paths])
        path_lengths = np.array([len(path["reward"]) for path in paths])
        succ_agent = np.zeros(len(rew_agent))
        stuck_agent = np.zeros(len(rew_agent))
        for i, (arrive, collision) in enumerate(zip(self.env.arrived_flag,
                                                    self.env.collision_flag)):
            if arrive:
                succ_agent[i] = 1
            elif not collision:
                stuck_agent[i] = 1

        stats["SuccessNum"] = succ_agent.sum()
        stats["StuckNum"] = stuck_agent.sum()
        stats["SuccessRate"] = succ_agent.sum() / succ_agent.shape[0]
        stats["StuckRate"] = stuck_agent.sum() / succ_agent.shape[0]
        stats["RewardsMean"] = rew_agent.mean()
        stats["PathLengthMean"] = (path_lengths * succ_agent).sum() / succ_agent.sum()
        stats["MeanDistance"] = (self.env.perfect_distance * succ_agent).sum() / succ_agent.sum()
        stats["MeanVelocity"] = stats["MeanDistance"] / (stats["PathLengthMean"] / 10.)
        return stats

    def run(self):
        iter_counter = 0
        while True:
            iter_counter += 1
            tstart = time.time()
            paths = self._get_paths()
            if self.is_training:
                stats = self.alg.update(
                    paths, self.env.arrived_flag, self.env.collision_flag
                    )
                stats["Fps"] = stats["EpPathLengthsSum"] / (time.time() - tstart)
            else:
                stats = self._get_stats(paths)

            stats["Iteration"] = iter_counter
            if self.write_flag:
                keys, values = [], []
                for key, value in stats.items():
                    keys.append(key)
                    values.append(value)

                if iter_counter == 1:
                    self.csvwriter.writerow(keys)
                self.csvwriter.writerow(values)
            # self._print_statistics(stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true'
        )
    parser.add_argument(
        '--num_agents', default=200, type=int
        )
    parser.add_argument(
        '--env_size', default=50, type=float
        )
    parser.add_argument(
        '--neighbor_num', default=10, type=int
        )
    parser.add_argument(
        '--point', default=200, type=int
        )
    parser.add_argument(
        '--write', action='store_true'
    )

    args = parser.parse_args()

    creator = Creator(
        robot_num=args.num_agents,
        env_size=args.env_size)
    creator.run()

    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    session = tf.Session(graph=graph, config=config)

    env = GazeboEnv(num_agents=args.num_agents,
                    env_size=args.env_size,
                    neighbor_num=args.neighbor_num)
    agent = Agent(session,
                  neighbor_num=args.neighbor_num)

    alg = PPO(agent)
    session.run(tf.global_variables_initializer())
    runner = Runner(env, agent, alg, args.write,
        is_training=args.train, point=args.point)

    runner.run()


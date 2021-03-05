import rospy
import time
import argparse
import csv

import numpy as np

from tabulate import tabulate
from gazebo_env import GazeboEnv
from rvo.agent import RVO
from collections import defaultdict, OrderedDict


class Runner(object):
    def __init__(self, env, agent, write):
        self.env = env
        self.agent = agent

        self.episodes_counter = 0
        self.episode_max_steps = 300
        self.batch_max_steps = 8000
        self.train_max_iters = 10000

        self.write_flag = write
        if self.write_flag:
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            outfile = open("./rvo/{}.csv".format(
                time_stamp), 'w')
            self.csvwriter = csv.writer(outfile)

    def _rollout(self):
        terminateds = [False for _ in range(self.env.num_agents)]
        terminate_idxs = [0 for _ in range(self.env.num_agents)]
        terminate_flag = [False for _ in range(self.env.num_agents)]
        paths = defaultdict(list)
        self.env.start()
        self.env.reset()
        for _ in range(self.episode_max_steps):
            acts = self.agent.step(
                self.env.robot_noise_pos,
                self.env.robot_noise_vel,
                self.env.goals,
                terminateds
                )

            self.env.render(acts)

            paths["acts"].append(acts)

            _, rews, dones = self.env.step(acts)
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

        self.env.stop()
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
    
        self.episodes_counter += 1
        print("***** Episode {} *****".format(self.episodes_counter))
        path_agents = self._rollout()

        for path in path_agents:
            paths_batch.append(path)

        return paths_batch

    def _print_statistics(self, stats):
        print(
            "*********** Iteration {} ************".format(stats["Iteration"]))
        print(tabulate(
            filter(lambda (k, v): np.asarray(v).size == 1, stats.items()),
            tablefmt="grid"))

    def run(self):
        iter_counter = 0
        while not rospy.is_shutdown():
            iter_counter += 1
            paths = self._get_paths()
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

            self._print_statistics(stats)

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
        stats["MeanVelocity"] = stats["MeanDistance"] / (stats["PathLengthMean"] / 2.)
        return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_agents', default=10, type=int
        )
    parser.add_argument(
        '--env_size', default=6.0, type=float
        )
    parser.add_argument(
        '--neighbor_num', default=10, type=int
        )
    parser.add_argument(
        '--write', action='store_true'
    )
    args = parser.parse_args()

    env = GazeboEnv(num_agents=args.num_agents,
                    env_size=args.env_size,
                    neighbor_num=args.neighbor_num)

    agent = RVO(num_agents=args.num_agents,
                max_neighbors=args.neighbor_num)

    runner = Runner(env, agent, args.write)
    runner.run()


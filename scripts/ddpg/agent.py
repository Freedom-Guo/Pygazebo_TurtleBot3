import os
import random
import time
import joblib

import numpy as np
import tensorflow as tf
import ddpg.utils as U

from ddpg.vel_smoother import VelocitySmoother
from collections import OrderedDict
from ddpg.utils import RunningMeanStd

class Agent(object):
    def __init__(self, args, sess):
        self.sess = sess
        if not os.path.exists("./ddpg/models"):
            os.makedirs("./ddpg/models")

        if not os.path.exists("./ddpg/summary"):
            os.makedirs("./ddpg/summary")

        self.time_step = 0

        self.args = args
        self.neighbor_num = args.neighbor_num

        self._build_ph()

        self.actor = Actor(
            sess,
            self.obs_ph,
            args)

        self.critic = Critic(
            sess, 
            self.obs_ph, self.act_ph
            )

        self._build_training()

        self._build_tensorboard()
        self.buffer = U.PrioritizedReplayBuffer(
            self.args.buffer_size, 
            alpha=self.args.alpha
        )
        self.counter = 0
        self.max_steps = 1000000
        self.speed_beta = (1. - self.args.beta) / self.max_steps
        self.beta = self.args.beta

        self.means_loss = []

        if self.args.train:
            timeString = time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime())
            self.merge_all = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(
                './ddpg/summary/{}'.format(timeString), sess.graph)

        sess.run(tf.global_variables_initializer())

        self.ob_rms = RunningMeanStd(shape=(self.neighbor_num*6+3, ))
        self.clipob = 10.
        self.epsilon = 1e-8

    def obfilt(self, obs):
        new_obs = []
        for ob in obs:
            self.ob_rms.update(np.asarray(ob))
            ob = np.clip(
                (ob - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob, self.clipob)
            new_obs.append(ob)

        return new_obs

    def save_ms(self, path):
        np.save(path + "_mean", self.ob_rms.mean)
        np.save(path + "_var", self.ob_rms.var)
        np.save(path + "_count", self.ob_rms.count)

    def load_ms(self, path):
        ob_mean = np.load(path + "_mean.npy")
        ob_var = np.load(path + "_var.npy")
        ob_count = np.load(path + "_count.npy")
        self.ob_rms.mean = ob_mean
        self.ob_rms.var = ob_var
        self.ob_rms.count = ob_count

    def _build_ph(self):
        self.obs_ph = tf.placeholder(
            tf.float32,
            (None, self.neighbor_num*6+3),
            'obs_ph'
            )

        self.act_ph = tf.placeholder(
            tf.float32, (None, 3), 'act_ph'
            )

        self.ret_ph = tf.placeholder(
            tf.float32, (None, ), 'ret_ph'
            )
        self.weights_ph = tf.placeholder(
            tf.float32, (None, ), 'weights_ph'
            )

        self.q_grads_ph = tf.placeholder(
            tf.float32, (None, 3), 'q_grads_ph'
            )

    def _build_training(self):
        self.mean_q = tf.reduce_mean(self.critic.q_eval)
        self.q_gradients = tf.gradients(self.critic.q_eval, self.act_ph)
        with tf.variable_scope('td_error'):
            self.td_error = tf.squeeze(self.critic.q_eval) - self.ret_ph 

        with tf.variable_scope('huber_loss'):
            errors = U.huber_loss(self.td_error)
        
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(self.weights_ph * errors)

        self.critic_opt = tf.train.AdamOptimizer(1e-3).minimize(self.critic_loss)
        actor_param = []
        actor_param = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="actor_net"
        )
        grads = tf.gradients(self.actor.means, actor_param, -self.q_grads_ph)
        self.actor_opt = tf.train.AdamOptimizer(1e-4).apply_gradients(
            zip(grads, actor_param)
            )

        # update target network
        eval_vars, target_vars = [], []

        eval_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_net"
        )

        eval_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_net"
        )

        target_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_target"
        )

        target_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_target"
        )

        update_target = []
        for var, var_target in zip(eval_vars, target_vars):
            update_target.append(var_target.assign(var))

        self.update_target = tf.group(*update_target)
        
    def _build_tensorboard(self):
        self.reward_tb = tf.placeholder(
            tf.float32,
            name='reward_tb')
        self.loss_tb = tf.placeholder(
            tf.float32,
            name='loss_tb')
        self.buffer_tb = tf.placeholder(
            tf.float32,
            name='buffer_tb')
        self.beta_tb = tf.placeholder(
            tf.float32,
            name='beta_tb')

        tf.summary.scalar('reward', self.reward_tb)
        tf.summary.scalar('loss', self.loss_tb)
        tf.summary.scalar('buffer_length', self.buffer_tb)
        tf.summary.scalar('beta', self.beta_tb)

    def perceive(self, obs, acts, rews, new_obs, dones_id):
        self.counter += 1
        if self.args.train:
            for i in range(self.args.num_agents):
                if dones_id[i] == 0:
                    done = False
                else:
                    done = True

                self.buffer.add(
                        obs[i], 
                        acts[i], rews[i],
                        new_obs[i],
                        done)
                    
            if self.counter % 10 == 0:
                self.train() 

    def train(self):
        self.time_step += 1
        self.beta += self.speed_beta
        experience = self.buffer.sample(self.args.batch_size, self.beta)
        (obs, acts, rews, new_obs, dones, weights, idxes) = experience

        obs = np.array(obs)
        acts = np.array(acts)
        rews = np.array(rews)
        new_obs = np.array(new_obs)

        act_target = self.sess.run(
            self.actor.means_target, 
            feed_dict={
                self.obs_ph: obs
                })

        q_target = self.sess.run(
            self.critic.q_target, 
            feed_dict={
                self.obs_ph: new_obs,
                self.act_ph: act_target
                })

        rets = []
        for i in range(self.args.batch_size):
            if dones[i]:
                rets.append(rews[i])
            else:
                rets.append(rews[i] + self.args.gamma*q_target[i][0])

        rets = np.asarray(rets)

        feed_dict = {
        self.obs_ph: obs,
        self.act_ph: acts,
        self.ret_ph: rets,
        self.weights_ph: weights
        }

        td_error, critic_loss, _ = self.sess.run(
            [self.td_error, self.critic_loss, self.critic_opt],
            feed_dict=feed_dict)

        self.means_loss.append(critic_loss)

        act_eval = self.sess.run(
            self.actor.means, 
            feed_dict={
                self.obs_ph: obs
                })

        grads = self.sess.run(
            self.q_gradients, 
            feed_dict={
                self.obs_ph: obs,
                self.act_ph: act_eval})

        self.sess.run(
            self.actor_opt, 
            feed_dict={
                self.obs_ph: obs,
                self.q_grads_ph: grads[0]})

        # update target network
        if self.time_step % 400 == 0:
            self.sess.run(self.update_target)

        new_priorities = np.abs(td_error) + 1e-6
        self.buffer.update_priorities(idxes, new_priorities)

    def visualize(self, paths, 
                  iter_count, 
                  perfect_distance, trajectory, 
                  arrive_flag, collision_flag):

        epRewards = np.array([path["reward"].sum() for path in paths])
        epPathLengths = np.array([len(path["reward"]) for path in paths])
        stats = OrderedDict()
        stats["EpRewardsMean"] = epRewards.mean()
        stats["EpRewardsMax"] = epRewards.max()
        stats["EpRewardsMin"] = epRewards.min()
        stats["EpPathLengthsMean"] = epPathLengths.mean()

        if self.args.train:
            with U.timed("train"):
                for _ in range(10):
                    self.train()

            self.means_loss = np.array(self.means_loss)
            # stats["EpPathLengthsMax"] = epPathLengths.max()
            # stats["EpPathLengthsMin"] = epPathLengths.min()
            stats["EpPathLengthsSum"] = epPathLengths.sum()
            stats["RewardPerStep"] = epRewards.sum() / epPathLengths.sum()
            stats["CriticLoss"] = np.mean(self.means_loss)
            stats["MemorySize"] = len(self.buffer)

            feed_dict = {
            self.reward_tb: epRewards.mean(),
            self.loss_tb: np.mean(self.means_loss),
            self.buffer_tb: len(self.buffer),
            self.beta_tb: self.beta
            }

            summary = self.sess.run(self.merge_all, feed_dict)
            self.writer.add_summary(summary, self.time_step)

            self.means_loss = []

            if iter_count % 10 == 0:
                self.actor.save_net(
                    './ddpg/models/actor_{}'.format(iter_count)
                    )
                self.critic.save_net(
                    './ddpg/models/critic_{}'.format(iter_count)
                    )
                self.save_ms(
                    "./ddpg/models/{}".format(iter_count)
                )

        else:
            succ_agent = np.zeros(len(arrive_flag))
            stuck_agent = np.zeros(len(arrive_flag))
            for i, (arrive, collision) in enumerate(zip(arrive_flag,
                                                        collision_flag)):
                if arrive:
                    succ_agent[i] = 1
                elif not collision:
                    stuck_agent[i] = 1

            stats["SuccessNum"] = succ_agent.sum()
            stats["SuccessRate"] = succ_agent.sum() / succ_agent.shape[0]
            stats["StuckNum"] = stuck_agent.sum()
            stats["StuckRate"] = stuck_agent.sum() / stuck_agent.shape[0]
            stats["MeanDistance"] = (perfect_distance * succ_agent).sum() / succ_agent.sum()
            stats["MeanTrajectory"] = (trajectory * succ_agent).sum() / succ_agent.sum()
            stats["ExtraDistance"] = stats["MeanTrajectory"] - stats["MeanDistance"]
            stats["MeanVelocity-D"] = stats["MeanDistance"] / (stats["EpPathLengthsMean"] / 10.)
            stats["MeanTime"] = (stats["EpPathLengthsMean"] / 10.)
            stats["ExtraTime"] = stats["MeanTime"] - stats["MeanDistance"] / 1.72

        return stats

class Actor(object):
    def __init__(self, sess, obs_ph, args):
        self.sess = sess

        self.obs_ph = obs_ph

        self.args = args
        self.sigma = args.sigma
        self.theta = args.theta
        self.vel_smoother = VelocitySmoother()

        self._build_ph()
        self.means = self._build_net("actor_net")
        self.means_target = self._build_net("actor_target")

    def _build_ph(self):
        self.q_gradients_ph = tf.placeholder(tf.float32, (None, 3), 'q_gradients_ph')

    def _build_net(self, net_name):
        with tf.variable_scope(net_name):
            x = tf.layers.dense(
                self.obs_ph, 256, activation=tf.nn.tanh
            )
            x = tf.layers.dense(
                x, 128, activation=tf.nn.tanh
            )
            means = tf.layers.dense(
                x, 3, activation=tf.nn.tanh
            )

        return means

    def act(self, obs, terminated):
        actions = self.sess.run(
            self.means, 
            feed_dict={
                self.obs_ph: obs
                })

        if self.args.train:
            actions = actions + self.ou_noise()

        for action in actions:
            action = np.clip(action, -1., 1.)
            # action = self.vel_smoother.step(action[0], action[1], 0.1)

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(3)

        return actions

    def test(self, obs, terminated):
        actions = self.sess.run(
            self.means, 
            feed_dict={
                self.obs_ph: obs
            })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(3)

        return actions

    def reset_noise(self):
        self.noise = np.ones((self.args.num_agents, 3))

    def ou_noise(self):
        x = self.noise
        dx = self.theta * (-x) + self.sigma * np.random.randn(self.args.num_agents, 3)
        self.noise = x + dx
        return self.noise

    def save_net(self, save_path):
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_net"
        )
        
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_net"
        )
        
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

class Critic(object):
    def __init__(self, sess, obs_ph, act_ph):
        self.sess = sess

        self.obs_ph = obs_ph
        self.act_ph = act_ph

        self.q_eval = self._build_net('critic_net')
        self.q_target = self._build_net('critic_target')

    def _build_net(self, net_name):
        with tf.variable_scope(net_name):
            x_act = tf.layers.dense(
                self.act_ph, 256, activation=tf.nn.tanh
            )
            x = tf.layers.dense(
                self.obs_ph, 256, activation=tf.nn.tanh
            )
            x = tf.layers.dense(
                x + x_act, 128, activation=tf.nn.tanh
            )
            value = tf.layers.dense(
                x, 1
            )
        return value

    def save_net(self, save_path):
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_net"
        )
        
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_net"
        )
        
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

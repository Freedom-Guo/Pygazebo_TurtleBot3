import os
import time

import numpy as np
import tensorflow as tf
import ppo.tf_utils as U

from collections import OrderedDict
from sklearn.utils import shuffle
from ppo.utils import discount


class PPO(object):
    def __init__(self, agent):
        self.agent = agent
        self.sess = agent.sess

        self.obs_ph = agent.obs_ph
        self.act_dim = 3

        self.time_step = 0
        self.best_score = 0
        self.total_timesteps = 0

        # rl param
        self.gamma = 0.99
        self.lamb = 0.95

        # actor param
        self.batch_size = 4096
        self.actor_epochs = 20
        self.beta = 1.0
        self.eta = 50.0
        self.kl_targ = 0.0025
        self.actor_lr = 3e-4
        self.lr_multiplier = 1.0

        # critic param
        self.buffer_obs = None
        self.buffer_ret = None

        self.critic_epochs = 10
        self.critic_lr = 1e-3


        self.actor = agent.actor
        self.means = agent.actor.means
        self.log_vars = agent.actor.log_vars
        self.critic = agent.critic

        # save network
        self._build_ph()
        self._build_ppo()
        self._build_critic_opt()
        self._build_tensorboard()

    def _build_ph(self):
        self.act_ph = tf.placeholder(
            tf.float32, [None, 3], 'act_ph'
            )
        self.adv_ph = tf.placeholder(
            tf.float32, [None, ], 'adv_ph'
            )
        self.ret_ph = tf.placeholder(
            tf.float32, [None, ], 'ret_ph'
            )
        self.old_log_vars_ph = tf.placeholder(
            tf.float32, [self.act_dim, ], 'old_log_vars_ph'
            )
        self.old_means_ph = tf.placeholder(
            tf.float32, [None, self.act_dim], 'old_means_ph'
            )

        self.beta_ph = tf.placeholder(tf.float32, name='beta')
        self.eta_ph = tf.placeholder(tf.float32, name='eta')
        self.lr_ph = tf.placeholder(tf.float32, name='lr')

    def _build_ppo(self):
        # compute logprob
        self.logp = -0.5 * tf.reduce_sum(
            self.log_vars) + -0.5 * tf.reduce_sum(
                tf.square(self.act_ph - self.means) / tf.exp(self.log_vars),
                axis=1)
        self.logp_old = -0.5 * tf.reduce_sum(
            self.old_log_vars_ph) + -0.5 * tf.reduce_sum(
                tf.square(self.act_ph - self.old_means_ph) \
                / tf.exp(self.old_log_vars_ph),
                axis=1)
        # compute kl
        with tf.variable_scope('kl'):
            self.kl = 0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    tf.exp(self.old_log_vars_ph - self.log_vars)) +
                tf.reduce_sum(
                    tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars),
                    axis=1) - self.act_dim + tf.reduce_sum(self.log_vars) -
                tf.reduce_sum(self.old_log_vars_ph))
        # compute entropy
        with tf.variable_scope('entropy'):
            self.entropy = 0.5 * (
                self.act_dim *
                (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))
        # compute actor loss
        with tf.variable_scope('actor_loss'):
            loss1 = -tf.reduce_mean(
                self.adv_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(
                tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.actor_loss = loss1 + loss2 + loss3
        # opt actor loss
        self.actor_opt = tf.train.AdamOptimizer(self.lr_ph).minimize(
            self.actor_loss)

    def _build_critic_opt(self):
        with tf.variable_scope("critic_loss"):
            self.critic_loss = tf.reduce_mean(
                tf.square(self.agent.critic.value - self.ret_ph)
                )

        self.critic_opt = tf.train.AdamOptimizer(1e-3).minimize(
            self.critic_loss
            )

    def _build_tensorboard(self):
        self.reward_tb = tf.placeholder(
            tf.float32, name="reward_tb"
        )
        self.kl_tb = tf.placeholder(
            tf.float32, name="kl_tb"
        )
        self.entropy_tb = tf.placeholder(
            tf.float32, name="entropy_tb"
        )
        self.succ_rate_tb = tf.placeholder(
            tf.float32, name="succ_rate_tb"
        )
        self.stuck_rate_tb = tf.placeholder(
            tf.float32, name="stuck_rate_tb"
        )
        self.critic_loss_tb = tf.placeholder(
            tf.float32, name="critic_loss_tb"
        )
        self.actor_loss_tb = tf.placeholder(
            tf.float32, name="actor_loss_tb"
        )

        with tf.name_scope('param'):
            tf.summary.scalar('reward', self.reward_tb)
            tf.summary.scalar('kl', self.kl_tb)
            tf.summary.scalar('entropy', self.entropy_tb)
            tf.summary.scalar('beta', self.beta_ph)
            tf.summary.scalar('actor_lr', self.lr_ph)
            tf.summary.scalar('succ_rate', self.succ_rate_tb)
            tf.summary.scalar('stuck_rate', self.stuck_rate_tb)

        with tf.name_scope('loss'):
            tf.summary.scalar('critic_loss', self.critic_loss_tb)
            tf.summary.scalar('actor_loss', self.actor_loss_tb)

        self.merge_all = tf.summary.merge_all()

        time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.writer = tf.summary.FileWriter("./log/tb_{}".format(time_stamp), 
            self.sess.graph)

    def update(self, paths, arrive_flag, collision_flag):
        self.time_step += 1

        acts = np.concatenate([path["acts"] for path in paths])
        values = np.concatenate([path["values"] for path in paths])
        obses = np.concatenate([path["obses"] for path in paths])

        last_path_size = 0
        for i, path in enumerate(paths):
            np.array(path["reward"])
            path["return"] = discount(path["reward"], self.gamma)
            b = path["baseline"] = values[
                last_path_size:last_path_size + path["done_id"]]
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + self.gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(
                deltas, self.gamma * self.lamb)
            last_path_size = path["done_id"]

        rets = np.concatenate([path["return"] for path in paths])
        advs = np.concatenate([path["advantage"] for path in paths])
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)

        self.total_timesteps += len(rets)

        if self.time_step > 1: # train acotr after trained critic
            lossvals = self.train_actor(
                obses, acts, advs, rets
                )
                 
        self.train_critic(obses, rets)

        stats = OrderedDict()
        epRewards = np.array([path["reward"].sum() for path in paths])
        epPathLengths = np.array([len(path["reward"]) for path in paths])
        succ_agent = np.zeros(len(arrive_flag))
        stuck_agent = np.zeros(len(arrive_flag))
        for i, (arrive, collision) in enumerate(zip(arrive_flag,
                                                    collision_flag)):
            if arrive:
                succ_agent[i] = 1
            elif not collision:
                stuck_agent[i] = 1
        
        stats["SuccessNum"] = succ_agent.sum()
        stats["StuckNum"] = stuck_agent.sum()
        stats["SuccessRate"] = succ_agent.sum() / succ_agent.shape[0]
        stats["StuckRate"] = stuck_agent.sum() / succ_agent.shape[0]
        stats["TotalTimesteps"] = self.total_timesteps
        stats["EpRewardsMean"] = epRewards.mean()
        stats["EpPathLengthsMean"] = epPathLengths.mean()
        # stats["EpPathLengthsMax"] = epPathLengths.max()
        # stats["EpPathLengthsMin"] = epPathLengths.min()
        stats["EpPathLengthsSum"] = epPathLengths.sum()
        stats["RewardPerStep"] = epRewards.sum() / epPathLengths.sum()

        if self.time_step > 1:
            stats["Beta"] = self.beta
            stats["ActorLearningRate"] = self.actor_lr * self.lr_multiplier
            stats["KL-Divergence"] = lossvals[3]
            stats["ActorLoss"] = lossvals[0]
            stats["CriticLoss"] = lossvals[1]
            stats["Entropy"] = lossvals[2]

            feed_dict = {
                self.reward_tb: epRewards.mean(),
                self.kl_tb: lossvals[3],
                self.succ_rate_tb: stats["SuccessRate"],
                self.stuck_rate_tb: stats["StuckRate"],
                self.actor_loss_tb: lossvals[0],
                self.critic_loss_tb: lossvals[1],
                self.entropy_tb: lossvals[2],
                self.beta_ph: self.beta,
                self.eta_ph: self.eta,
                self.lr_ph: self.actor_lr * self.lr_multiplier
            }

            summary = self.sess.run(self.merge_all, feed_dict)
            self.writer.add_summary(summary, self.time_step)

        if epRewards.mean() > self.best_score:
            self.agent.save_net(
                os.path.join("./log/model/", "best"))
            self.best_score = epRewards.mean()

        if self.time_step % 20 == 0:
            self.agent.critic.save_net(
                os.path.join("./log/model",
                "critic_{}".format(self.time_step))
            )
            self.agent.actor.save_net(
                os.path.join("./log/model", 
                "actor_{}".format(self.time_step))
                )
            self.agent.save_ms(
                os.path.join("./log/model",
                "{}".format(self.time_step))
            )

        return stats

    def actor_update(self, old_log_vars, 
                     obs, acts, advs, rets,
                     old_means):

        feed_dict = {
            self.obs_ph: obs,
            self.act_ph: acts,
            self.adv_ph: advs,
            self.ret_ph: rets,
            self.old_log_vars_ph: old_log_vars,
            self.old_means_ph: old_means,
            self.beta_ph: self.beta,
            self.eta_ph: self.eta,
            self.lr_ph: self.actor_lr * self.lr_multiplier
        }

        loss = self.sess.run(
                [self.actor_loss, self.critic_loss, 
                self.entropy, self.kl, self.actor_opt], feed_dict=feed_dict
                )[:-1]
                    
        return loss

    def train_actor(self, obs, acts, advs, rets):
        feed_dict = {
            self.obs_ph: obs,
            self.act_ph: acts,
            self.adv_ph: advs,
            self.beta_ph: self.beta,
            self.eta_ph: self.eta,
            self.lr_ph: self.actor_lr * self.lr_multiplier
        }

        old_means, old_log_vars = self.sess.run(
            [self.means, self.log_vars], feed_dict)

        total_len = len(advs)
        inds = np.arange(total_len)
        for _ in range(self.actor_epochs):
            np.random.shuffle(inds)
            mblossvals = []
            for start in range(0, total_len, self.batch_size):
                end = start + self.batch_size
                if (end + self.batch_size) > total_len:
                    end = total_len
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, acts, advs, rets, old_means))
                mblossvals.append(self.actor_update(old_log_vars, *slices))
            lossvals = np.mean(mblossvals, axis=0)
            kl = lossvals[3]
            if  kl > self.kl_targ * 4:  # early stopping
                break

        if kl > self.kl_targ * 2:
            self.beta = np.minimum(35, 1.5 * self.beta)
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2.0:
            self.beta = np.maximum(1.0 / 35.0, self.beta / 1.5)
            if self.beta < (1.0 / 30.0) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        return lossvals

    def critic_update(self, obs, returns):
        feed_dict = {
            self.obs_ph: obs,
            self.ret_ph: returns}

        self.sess.run(self.critic_opt, feed_dict)

    def train_critic(self, obs, rets):
        num_batches = max(rets.shape[0] // 256, 1)
        batch_size = rets.shape[0] // num_batches
        if self.buffer_obs is None:
            obs_train, ret_train = obs, rets
        else:
            obs_train = np.concatenate(
                [obs, self.buffer_obs])
            ret_train = np.concatenate(
                [rets, self.buffer_ret])

        self.buffer_obs = obs.copy()
        self.buffer_ret = rets.copy()

        for _ in range(self.critic_epochs):
            obs_train, ret_train = shuffle(obs_train, ret_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs_set = obs_train[start:end, :]
                ret_set = ret_train[start:end]
                self.critic_update(obs_set, ret_set)

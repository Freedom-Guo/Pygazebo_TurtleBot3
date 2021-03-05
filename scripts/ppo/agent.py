import os
import joblib

import numpy as np
import tensorflow as tf

from ppo.ac import Actor, Critic
from ppo.utils import RunningMeanStd


class Agent(object):
    def __init__(self, sess, neighbor_num=5):
        self.neighbor_num = neighbor_num
        self.sess = sess
        self._build_ph()

        self.actor = Actor(sess, self.obs_ph)
        self.critic = Critic(sess, self.obs_ph)

        self.ob_rms = RunningMeanStd(shape=(neighbor_num*6+3, ))
        self.clipob = 10.
        self.epsilon = 1e-8

    def _build_ph(self):
        self.obs_ph = tf.placeholder(
            tf.float32, [None, self.neighbor_num*6+3], 'obs_ph'
        )

    def save_net(self, save_path):
        params = tf.trainable_variables()
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = tf.trainable_variables()
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

    def step(self, obs, terminated, is_training):
        if is_training:
            acts, values = self.sess.run(
                [self.actor.act, self.critic.value],
                feed_dict = {
                    self.obs_ph: obs
                }
            )
        else:
            acts, values = self.sess.run(
                [self.actor.means, self.critic.value],
                feed_dict = {
                    self.obs_ph: obs
                }
            )

        for i, t in enumerate(terminated):
            if t:
                acts[i] = np.zeros(3)

        return acts, values

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
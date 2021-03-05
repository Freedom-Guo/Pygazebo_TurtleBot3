import joblib

import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, sess, obs_ph):
        self.sess = sess
        self.obs_ph = obs_ph

        self.means, self.act, self.log_vars = self._build_net()

    def _build_net(self):
        with tf.variable_scope("actor_net"):
            x = tf.layers.dense(
                self.obs_ph, 256, activation=tf.nn.tanh
            )
            x = tf.layers.dense(
                x, 128, activation=tf.nn.tanh
            )
            means = tf.layers.dense(
                x, 3, activation=tf.nn.tanh
            )

        logvar_speed = (10 * 128) // 48
        log_vars = tf.get_variable(
            'logvars', (logvar_speed, 3), tf.float32,
            tf.constant_initializer(0.0))
        log_vars = tf.reduce_sum(log_vars, axis=0) -1.0

        sampled_act = means + \
            tf.exp(log_vars / 2.0) * tf.random_normal(shape=(3,))

        return means, sampled_act, log_vars

    def get_acts(self, obs, terminated, batch=True):
        if not batch:
            obs = [np.expand_dims(x, 0) for x in obs]

        actions = self.sess.run(self.act, feed_dict={
                self.obs_ph: obs
            })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(2)

        return actions

    def get_means(self, obs, terminated):
        actions = self.sess.run(self.means, feed_dict={
                self.obs_ph: obs
            })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(2)
        return actions

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
    def __init__(self, sess, obs_ph):
        self.sess = sess
        self.obs_ph = obs_ph
        
        self.value = self._build_net()
        
    def _build_net(self):
        with tf.variable_scope("critic_net"):
            x = tf.layers.dense(
                self.obs_ph, 256, activation=tf.nn.tanh
            )
            x = tf.layers.dense(
                x, 128, activation=tf.nn.tanh
            )
            value = tf.layers.dense(
                x, 1
            )

        return value

    def predict(self, obs):
        value = self.sess.run(
            self.value,
            feed_dict={
                self.obs_ph: obs
                })
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
import os

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from utils import discount

class DDPG(object):
    def __init__(self, args, agent, session, obs_shape):
        self.sess = session
        self.args = args

        self._build_tensorboard()

    def _build_tensorboard(self):
        self.visual_reward = tf.placeholder(tf.float32, name='visual_reward')
        tf.summary.scalar('reward', self.visual_reward)

    def update()


        

import numpy as np

from gym.utils import seeding
import math


class Scenarios(object):
    def __init__(self, num_agents, agent_size, env_size):
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.env_size = env_size

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_scene(self, 
                     num_agents=None,
                     env_size=None, 
                     center_x=0.0, 
                     center_y=0.0):

        if num_agents is None:
            num_agents = self.num_agents
        if env_size is None:
            env_size = self.env_size

        starts, goals = [], []
        for _ in range(num_agents):
            succ = False
            sx, sy, sz, gx, gy, gz = 0., 0., 0., 0., 0., 0.
            while not succ:
                sx, sy, gx, gy = self.np_random.uniform(-env_size,
                                                        env_size, 4)
                sz, gz = self.np_random.uniform(1.0, env_size, 2)
                s = np.array([sx + center_x, sy + center_y, sz])
                g = np.array([gx + center_x, gy + center_y, gz])

                succ = True

                if np.linalg.norm(s - g) < 5.0:
                    succ = False

                if starts:
                    for other_s in starts:
                        if np.linalg.norm(s - other_s) < self.agent_size * 5.:
                            succ = False

                if goals:
                    for other_g in goals:
                        if np.linalg.norm(g - other_g) < self.agent_size * 5.:
                            succ = False

            starts.append(s)
            goals.append(g)

        return starts, goals

    def uniform_circle_scene(self,
                     num_agents=None,
                     env_size=None,
                     center_x=0.0,
                     center_y=0.0, 
                     altitude=10):
        if num_agents is None:
            num_agents = self.num_agents
        
        if env_size is None:
            env_size = self.env_size

        starts, goals = [], []
        for i in range(num_agents):
            angle = i * 2 * np.pi / num_agents
            sx = env_size * np.cos(angle)
            sy = env_size * np.sin(angle)
            sz = altitude
            starts.append(np.array([sx + center_x, sy + center_y, sz]))
            goals.append(np.array([-sx + center_x, -sy + center_y, sz]))

        return starts, goals

    def random_circle_scene(self,
                     num_agents=None,
                     env_size=None,
                     center_x=0.0,
                     center_y=0.0, 
                     altitude=10):
        if num_agents is None:
            num_agents = self.num_agents
        
        if env_size is None:
            env_size = self.env_size

        starts, goals = [], []
        for _ in range(num_agents):
            succ = False
            while not succ:
                angle = self.np_random.uniform(0., 2*np.pi)
                sx = env_size * np.cos(angle)
                sy = env_size * np.sin(angle)
                sz = altitude

                succ = True

                if starts:
                    for s in starts:
                        if np.hypot(sx + center_x - s[0],
                                    sy + center_y - s[1]) < self.agent_size * 3.:
                            succ = False

            starts.append(np.array([sx + center_x, sy + center_y, sz]))
            goals.append(np.array([-sx + center_x, -sy + center_y, sz]))

        return starts, goals
    
    def random_ball_scene(self,
                   num_agents=None,
                   env_size=None,
                   center_x=0.0,
                   center_y=0.0,
                   altitude=2.0):

        if num_agents is None:
            num_agents = self.num_agents
        
        if env_size is None:
            env_size = self.env_size

        starts, goals = [], []
        
        for _ in range(num_agents):
            succ = False
            while not succ:
                theta = 2 * np.pi * self.np_random.uniform()
                phi = np.arccos(2 * self.np_random.uniform() - 1.0)
                
                sx = env_size * np.sin(theta) * np.cos(phi)
                sy = env_size * np.sin(theta) * np.sin(phi)
                sz = env_size * np.cos(theta)
                s_candidate = np.array([sx + center_x, 
                                        sy + center_y,
                                        sz + altitude + env_size])
                succ = True

                if starts:
                    for s in starts:
                        if np.linalg.norm(s_candidate - s) < self.agent_size * 3.:
                            succ = False

            starts.append(
                np.array([sx + center_x, 
                          sy + center_y,
                          sz + altitude + env_size])
                          )
            goals.append(
                np.array([-sx + center_x, 
                          -sy + center_y, 
                          -sz + altitude + env_size])
                          )

        return starts, goals

    def multi_scenes(self,
                    num_agents=None,
                    env_size=None,
                    center_x=0.0,
                    center_y=0.0):

        if num_agents is None:
            num_agents = self.num_agents
        if env_size is None:
            env_size = self.env_size

        starts, goals = [], []
        random_agents = int(num_agents / 3)
        ball_agents = int(num_agents / 3)
        circle_agents = num_agents - random_agents - ball_agents

        random_s, random_g = self.random_scene(
            num_agents=random_agents,
            env_size=env_size / 1.5,
            center_x=-2 * env_size
            )

        ball_s, ball_g = self.random_ball_scene(
            num_agents=ball_agents,
            env_size=env_size / 1.5
        )

        circle_s, circle_g = self.random_circle_scene(
            num_agents=circle_agents,
            env_size=env_size,
            center_x=2 * env_size
        )

        starts.extend(random_s)
        starts.extend(ball_s)
        starts.extend(circle_s)
        goals.extend(random_g)
        goals.extend(ball_g)
        goals.extend(circle_g)

        return starts, goals
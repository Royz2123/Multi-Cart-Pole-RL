"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

import multi_cart.envs.single_cart as single_cart
import multi_cart.envs.constants as constants


class MultiCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Crate a SingleCart object per CARTPOLE
        self.cartpoles = [
            single_cart.SingleCart(offset=i * constants.CARTDIST)
            for i in range(constants.CARTPOLES)
        ]

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        # define spaces
        self.action_space = spaces.MultiBinary(constants.CARTPOLES)
        self.observation_space = spaces.Tuple(tuple([
            spaces.Box(-high, high, dtype=np.float32)
            for _ in range(constants.CARTPOLES)
        ]))

        # self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        for i in range(constants.CARTPOLES):
            self.cartpoles[i].step(action[i])

        done = any([cartpole.is_done() for cartpole in self.cartpoles])

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # One of the poles has fallen
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        # return results
        return np.array(self.get_state()), reward, done, {}

    def get_state(self):
        return [cartpole.state for cartpole in self.cartpoles]

    def reset(self):
        for cartpole in self.cartpoles:
            cartpole.reset()

        self.steps_beyond_done = None
        return self.get_state()

    def render(self, mode='human'):
        screen_width = 1000
        screen_height = 400

        init = False
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            init = True

        if self.get_state() is None:
            return None

        for cartpole in self.cartpoles:
            cartpole.render(viewer=self.viewer, screen_width=screen_width, init=init)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



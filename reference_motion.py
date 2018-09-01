from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
import gym
import random as r


class StandUprightWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StandUprightWrapper, self).__init__(env)
        self.ES = 0.5
        self.observation_space = gym.spaces.Box(np.zeros(160), np.zeros(160))

    def reset(self, project=True, **kwargs):
        self.env.reset(project, **kwargs)
        self.set_init_state()
        return self.env.get_observation() if project else self.env.get_state_desc()

    def step(self, action, **kwargs):
        observation, _, done, info = self.env.step(action, **kwargs)
        return observation, self.alive_bonus(), done, info

    def alive_bonus(self):
        return 4.0 if self.env.get_state_desc()['body_pos']['pelvis'][1] > 0.6 else -1

    def set_init_state(self):
        state = self.env.osim_model.get_state()

        for joint in self.env.osim_model.model.getJointSet():
            for i in range(min(3, joint.numCoordinates())):
                joint.get_coordinates(i).setValue(state, (r.random() - 0.5) * self.ES)

        self.env.osim_model.set_state(state)
        self.env.osim_model.model.equilibrateMuscles(self.env.osim_model.get_state())
        self.env.osim_model.state_desc_istep = None

if __name__ == '__main__':

    env = ProstheticsEnv(visualize=True)
    wrapped_env = StandUprightWrapper(env)
    done = True

    for _ in range(100):
        frame = score = 0
        obs = wrapped_env.reset(project=False)
        for _ in range(3):
            a = wrapped_env.action_space.sample()
            wrapped_env.step(a)

    wrapped_env.close()
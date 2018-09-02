from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
import gym
import random as r

def matrix_from_euler(phi, theta, psi):
    return np.array([[m.cos(psi)*m.cos(theta), m.cos(psi)*m.sin(theta)*m.sin(phi)-m.cos(phi)*m.sin(psi), m.sin(psi)*m.sin(phi)+m.cos(psi)*m.cos(phi)*m.sin(theta)],
                    [m.cos(theta)*m.sin(psi), m.cos(psi)*m.cos(phi)+m.sin(psi)*m.sin(theta)*m.sin(phi), m.cos(phi)*m.sin(psi)*m.sin(theta)-m.cos(psi)*m.sin(phi)],
                    [-m.sin(theta), m.cos(theta)*m.sin(phi), m.cos(theta)*m.cos(phi)]])


class StandUprightWrapper(gym.Wrapper):
    def __init__(self, env, es=0.1):
        super(StandUprightWrapper, self).__init__(env)
        self.ES = 0.5
        self.observation_space = gym.spaces.Box(np.zeros(160), np.zeros(160))

    def reset(self, project=True, **kwargs):
        self.env.reset(project, **kwargs)
        self.set_init_state()
        return self.env.get_observation() if project else self.env.get_state_desc()

    def step(self, action, **kwargs):
        observation, _, done, info = self.env.step(action, **kwargs)
        state = self.env.get_state_desc()
        reward = self.alive(state) + self.pain(state) + self.crossed_legs(state)
        return observation, reward, done, info

    def pain(self, state):
        # don't hyperextend your knee
        return -1.0 if any([state['joint_pos'][knee][0] > 0.1 for knee in ['knee_l','knee_r']]) else 0.0

    def crossed_legs(self, state):
        z_axis = matrix_from_euler(*np.array(state['joint_pos']['ground_pelvis'])[[1,2,0]])[:,2]
        foot_width = z_axis.dot(state['body_pos']['pros_foot_r']) - z_axis.dot(state['body_pos']['calcn_l'])
        return -1.0 if foot_width < 0.0 else 0.0

    def alive(self, state):
        return 4.0 if state['body_pos']['pelvis'][1] > 0.6 else -1

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
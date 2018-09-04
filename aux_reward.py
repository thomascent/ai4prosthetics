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


def quaternion_from_euler(phi, theta, psi):
    cos_p = m.cos(theta * 0.5)
    sin_p = m.sin(theta * 0.5)
    cos_r = m.cos(phi * 0.5)
    sin_r = m.sin(phi * 0.5)
    cos_y = m.cos(psi * 0.5)
    sin_y = m.sin(psi * 0.5)

    qw = cos_p * cos_r * cos_y + sin_p * sin_r * sin_y
    qx = cos_p * sin_r * cos_y - sin_p * cos_r * sin_y
    qy = sin_p * cos_r * cos_y + cos_p * sin_r * sin_y
    qz = cos_p * cos_r * sin_y - sin_p * sin_r * cos_y

    return np.array([qw, qx, qy, qz])


def quaternion_dist(q1, q2):
    return m.acos(q1.dot(q2.T) ** 2 * 2 - 1)

class AuxRewardWrapper(gym.Wrapper):
    def __init__(self, env, es=0.1):
        super(AuxRewardWrapper, self).__init__(env)
        self.ES = es
        self.observation_space = gym.spaces.Box(np.zeros(160), np.zeros(160))
        self.aux_rewards = [self.alive, self.knee_hyperextension, self.crossed_legs, self.balance]
        self.target_posture = quaternion_from_euler(0., 0. ,0.)

    def reset(self, project=True, **kwargs):
        self.env.reset(project, **kwargs)
        # self.set_init_state()
        return self.env.get_observation() if project else self.env.get_state_desc()

    def step(self, action, **kwargs):
        observation, velocity_reward, done, info = self.env.step(action, **kwargs)
        state = self.env.get_state_desc()
        reward = velocity_reward + sum([aux(state) for aux in self.aux_rewards])
        return observation, reward, done, info

    def knee_hyperextension(self, state):
        return -1.0 if any([state['joint_pos'][knee][0] > 0.1 for knee in ['knee_l','knee_r']]) else 0.0

    def balance(self, state):
        return -quaternion_dist(self.target_posture, quaternion_from_euler(*state['joint_pos']['ground_pelvis'][:3])) * 10.

    def crossed_legs(self, state):
        z_axis = matrix_from_euler(*np.array(state['joint_pos']['ground_pelvis'])[[1,2,0]])[:,2]
        foot_width = z_axis.dot(state['body_pos']['pros_foot_r']) - z_axis.dot(state['body_pos']['calcn_l'])
        return -1.0 if foot_width < 0.0 or foot_width > 0.8 else 0.0

    def alive(self, state):
        return 4.0 if state['body_pos']['pelvis'][1] > 0.8 else -1

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
    wrapped_env = AuxRewardWrapper(env)
    done = True

    for _ in range(100):
        frame = score = 0
        obs = wrapped_env.reset(project=False)
        for _ in range(300):
            a = wrapped_env.action_space.sample()
            wrapped_env.step(a)

    wrapped_env.close()
from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import math as m
from bvh import Bvh
import gym
import random as r
import pickle as pkl

def flatten(lol):
    return [i for l in lol for i in l]

class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file, omega=1.0, rsi=False):
        super(ReferenceMotionWrapper, self).__init__(env)

        with open(motion_file, 'rb') as f:
            self.motion = pkl.load(f)

        self.target_frame = 0

         # If on average the bodies are within 10cm of their target positions then that's fine
        self.CLOSE_ENOUGH = 0.1 * len(self.motion[0]['body_pos'].keys())

        env_obs_shape = env.observation_space.shape[0] + len(flatten(list(self.motion[0]['body_pos'].values()))) + 2
        self.observation_space = gym.spaces.Box(low=np.zeros([env_obs_shape,]), high=np.zeros([env_obs_shape,]), dtype=np.float32)

    def reset(self, project=True, frame=None, **kwargs):
        observation = self.env.reset(project, **kwargs)

        self.target_frame = 0

        if frame is not None:
            self.set_state_desc(self.motion[frame]['joint_pos'])

        return self.observation(observation)

    def step(self, action, **kwargs):
        observation, task_reward, done, info = self.env.step(action, **kwargs)
        imitation_reward = self.imitation_reward()
        observation = self.observation(observation)

        # while the guy is close enough to the next frame, move to the next frame
        while self.dist_to_target() < self.CLOSE_ENOUGH:
            self.target_frame += 1

        info['task_reward'] = task_reward
        info['imitation_reward'] = imitation_reward
        info['target_frame'] = self.target_frame

        print(imitation_reward, self.target_frame)

        return observation, imitation_reward, done, info

    def observation(self, observation):
        if isinstance(observation, dict):
            for k,v in self.motion[self.target_frame]['body_pos'].items():
                observation['target_' + k] = v
        elif isinstance(observation, list):
            observation += flatten(list(self.motion[self.target_frame]['body_pos'].values()))
        else:
            raise ValueError

        return observation

    def imitation_reward(self):
        return np.exp(-self.dist_to_target())

    def dist_to_target(self):
        ref_desc, curr_desc = self.motion[self.target_frame]['body_pos'], self.env.get_state_desc()['body_pos']
        return np.sum([norm(np.array(ref_desc[name]) - curr_desc[name]) for name in set(ref_desc).intersection(set(curr_desc))])

    def set_state_desc(self, state_desc):
        state = self.env.osim_model.get_state()

        for joint in self.env.osim_model.model.getJointSet():
            name = joint.getName()
            if name in state_desc.keys():
                [joint.get_coordinates(i).setValue(state, state_desc[name][i]) for i in range(len(state_desc[name]))]

        self.env.osim_model.set_state(state)
        self.env.osim_model.model.equilibrateMuscles(self.env.osim_model.get_state())
        self.env.osim_model.state_desc_istep = None

if __name__ == '__main__':
    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy.bvh.pkl')
    done = True

    for i in range(200):
        # print('setting frame: ' + str(i))
        obs = wrapped_env.reset(project=True, frame=0)
        for j in range(50):
            obs, rew, done, info = wrapped_env.step(env.action_space.sample(), project=True)
            print(rew, wrapped_env.dist_to_target())
            if done: env.reset()

    env.close()

from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
from bvh import Bvh
import gym
import random as r


class ReferenceMotion:
    def __init__(self, filename):
        with open(filename) as f:
            mocap = Bvh(f.read())
        
        self.len = mocap.nframes

        joints = {}
        joints['knee_l'] = mocap.frames_joint_channels('LeftLeg',['Xrotation'])
        joints['knee_r'] = mocap.frames_joint_channels('RightLeg',['Xrotation'])
        joints['ankle_l'] = mocap.frames_joint_channels('LeftFoot',['Xrotation'])
        joints['hip_l'] = mocap.frames_joint_channels('LeftUpLeg',['Xrotation'])
        joints['hip_r'] = mocap.frames_joint_channels('RightUpLeg',['Xrotation'])

        self.joints = {k: [[m.radians(-e) for e in l] for l in v] for (k, v) in joints.items() }
        self.joints['phase'] = [(i) / self.len for i in range(self.len)]

    def __getitem__(self, frame):
        frame = frame % self.len
        return {k: v[frame] for (k, v) in self.joints.items()}

    def __iter__(self):
        for frame in range(self.len):
            yield {k: v[frame] for (k, v) in self.joints.items()}
    
    def __len__(self):
        return self.len


class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file, omega=1., rsi=False):
        super(ReferenceMotionWrapper, self).__init__(env)

        env_obs_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=np.zeros([env_obs_shape+1,]), high=np.zeros([env_obs_shape+1,]), dtype=np.float32)       
        
        self.motion = ReferenceMotion(motion_file)
        self.frame = 0
        self.omega = omega
        self.rsi = rsi
        self.near_enough = 0.95

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation, reward = self.observation(observation), self.reward(reward)

        if self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos']) > self.near_enough:
            self.frame += 1
            print('next!')

        return observation, reward, done, info

    def reset(self, project=True, frame=None, **kwargs):
        observation = self.env.reset(project, **kwargs)
        
        if self.rsi or frame:
            self.frame = r.randint(0,len(self.motion)-1) if frame is None else frame
            self.set_state_desc(self.motion[self.frame])
            self.env.osim_model.model.equilibrateMuscles(self.env.osim_model.get_state())
            self.env.osim_model.state_desc_istep = None

            if project:
                observation = self.env.get_observation()
            else:
                observation = self.env.get_state_desc()
    
        return self.observation(observation)

    def observation(self, observation):
        if isinstance(observation, dict):
            observation['phase'] = self.frame / len(self.motion)
        elif isinstance(observation, list):
            observation += [self.frame / len(self.motion)]

        return observation

    def reward(self, reward):
        reward += self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos']) * self.omega
        return reward

    def calculate_similarity(self, ref_state_desc, curr_state_desc):
        ref, curr = set(ref_state_desc), set(curr_state_desc)
        cos_sim = [np.cos(np.subtract(ref_state_desc[name], curr_state_desc[name][0:len(ref_state_desc[name])])) for name in ref.intersection(curr)]
        return np.mean(cos_sim)

    def set_state_desc(self, state_desc):
        state = self.env.osim_model.get_state()

        for joint in self.env.osim_model.model.getJointSet():
            name = joint.getName()
            if name in state_desc.keys():
                [joint.get_coordinates(i).setValue(state, state_desc[name][i]) for i in range(len(state_desc[name]))]

        self.env.osim_model.set_state(state)


if __name__ == '__main__':

    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_loop.bvh')
    done = True

    while True:
        if done:
            obs = wrapped_env.reset(project=False, frame=i)

    for i in range(200):
        print('setting frame: ' + str(i))
        obs = wrapped_env.reset(project=False, frame=i)
        obs, rew, done, _ = wrapped_env.step(env.action_space.sample(), project=False)


    env.close()

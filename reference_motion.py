from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
from bvh import Bvh
import gym
import random as r


class ReferenceMotion:
    def __init__(self, filename, period=-1, offset=0):
        with open(filename) as f:
            mocap = Bvh(f.read())
        
        self.mocap = mocap
        self.period = period
        self.offset = offset

        if self.period == -1:
            self.period = len(mocap.frames)
    
        joints = {}
        joints['knee_l'] = mocap.frames_joint_channels('LeftLeg',['Xrotation'])
        joints['knee_r'] = mocap.frames_joint_channels('RightLeg',['Xrotation'])
        joints['ankle_l'] = mocap.frames_joint_channels('LeftFoot',['Xrotation'])
        joints['hip_l'] = mocap.frames_joint_channels('LeftUpLeg',['Xrotation', 'Yrotation', 'Zrotation'])
        joints['hip_r'] = mocap.frames_joint_channels('RightUpLeg',['Xrotation', 'Yrotation', 'Zrotation'])

        self.joints = {k: [[m.radians(-e) for e in l] for l in v][self.offset:self.offset+self.period] for (k, v) in joints.items() }
        
        self.joints['phase'] = [(i) / self.period for i in range(self.period)]
        
    def __getitem__(self, frame):
        return {k: v[frame] for (k, v) in self.joints.items()}

    def __iter__(self):
        for frame in range(self.period):
            yield {k: v[frame] for (k, v) in self.joints.items()}
    
    def __len__(self):
        return self.period


class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file, period=-1, offset=0, omega=1., rsi=False):
        super(ReferenceMotionWrapper, self).__init__(env)

        env_obs_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=np.zeros([env_obs_shape+1,]), high=np.zeros([env_obs_shape+1,]), dtype=np.float32)       
        
        self.motion = ReferenceMotion(motion_file, period=period, offset=offset)
        self.frame = 0
        self.omega = omega
        self.rsi = rsi

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation, reward = self.observation(observation), self.reward(reward)
        
        idx = np.argmax([self.calculate_similarity(frame, self.env.get_state_desc()['joint_pos']) for frame in self.motion])
        self.frame = (idx + 1) % len(self.motion) # set the target frame to the nearest state + 1, might want to rate limit this at some point
        
        print(self.frame)

        return observation, reward, done, info

    def reset(self, project=True, **kwargs):
        observation = self.env.reset(project, **kwargs)
        
        if self.rsi:
            self.frame = r.randint(0,len(self.motion)-1)
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
        reward = self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos']) * self.omega ## !! replace the += here
        return reward

    def calculate_similarity(self, ref_state_desc, curr_state_desc):
        ref = set(ref_state_desc)
        curr = set(curr_state_desc)
        return np.mean([np.mean(np.cos(np.subtract(ref_state_desc[name], curr_state_desc[name]))) for name in ref.intersection(curr)])

    def set_state_desc(self, state_desc):
        state = self.env.osim_model.get_state()

        for joint in self.env.osim_model.model.getJointSet():
            name = joint.getName()
            if name in state_desc.keys():
                [joint.get_coordinates(i).setValue(state, state_desc[name][i]) for i in range(joint.numCoordinates())]

        self.env.osim_model.set_state(state)

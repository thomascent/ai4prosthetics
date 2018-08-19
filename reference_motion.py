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
        joints['hip_l'] = mocap.frames_joint_channels('LeftUpLeg',['Xrotation','Yrotation','Zrotation'])
        joints['hip_r'] = mocap.frames_joint_channels('RightUpLeg',['Xrotation','Yrotation','Zrotation'])
        joints['ground_pelvis'] = mocap.frames_joint_channels('RightUpLeg',['Xrotation','Yrotation','Zrotation'])

        self.joints = {k: [[m.radians(-e) for e in l] for l in v] for (k, v) in joints.items() }

        # @todo: make this less of a filthy hack
        for i in range(self.len):
            self.joints['hip_l'][i][1] = 0.
            self.joints['hip_l'][i][2] = 0.
            self.joints['hip_r'][i][1] = 0.
            self.joints['hip_r'][i][2] = 0.
            self.joints['ground_pelvis'][i][0] = m.radians(-10.) # note this won' work for non-zero yaw angles... ugh
            self.joints['ground_pelvis'][i][1] = 0.
            self.joints['ground_pelvis'][i][2] = 0.
            self.joints['hip_l'][i][0] += m.radians(10.)
            self.joints['hip_r'][i][0] += m.radians(10.)

        self.joints['phase'] = [(i) / self.len for i in range(self.len)]

    def __getitem__(self, frame):
        frame = frame % self.len
        return {k: v[frame] for (k, v) in self.joints.items()}

    def __iter__(self):
        for frame in range(self.len):
            yield {k: v[frame] for (k, v) in self.joints.items()}
    
    def frame_to_phase(self, frame):
        return (frame % self.len) / self.len

    def __len__(self):
        return self.len


class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file, omega=1.0, rsi=False):
        super(ReferenceMotionWrapper, self).__init__(env)

        env_obs_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=np.zeros([env_obs_shape+1,]), high=np.zeros([env_obs_shape+1,]), dtype=np.float32)       
        
        self.motion = ReferenceMotion(motion_file)
        self.frame = 0
        self.prev_similarity = 0.0
        self.omega = omega
        self.rsi = rsi
        self.CLOSE_ENOUGH = 0.95

    def reset(self, project=True, frame=None, **kwargs):
        observation = self.env.reset(project, **kwargs)
        
        if self.rsi or frame:
            self.frame = r.randint(0,len(self.motion)-1) if frame is None else frame
            self.set_state_desc(self.motion[self.frame-1])
            self.env.osim_model.model.equilibrateMuscles(self.env.osim_model.get_state())
            self.env.osim_model.state_desc_istep = None

            if project:
                observation = self.env.get_observation()
            else:
                observation = self.env.get_state_desc()

        self.prev_similarity = self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos'])
    
        return self.observation(observation)

    def step(self, action, **kwargs):
        observation, task_reward, done, info = self.env.step(action, **kwargs)
        imitation_reward = self.imitation_reward()
        observation = self.observation(observation)

        info['task_reward'] = task_reward
        info['imitation_reward'] = imitation_reward
        info['frame'] = self.frame

        return observation, task_reward + imitation_reward, done, info

    def observation(self, observation):
        if isinstance(observation, dict):
            observation['phase'] = self.motion.frame_to_phase(self.frame)
        elif isinstance(observation, list):
            observation += [self.motion.frame_to_phase(self.frame)]

        return observation

    def imitation_reward(self):
        curr_similarity = self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos'])
        imitation_reward = (curr_similarity - self.prev_similarity) * self.omega

        if curr_similarity > self.CLOSE_ENOUGH:
            self.frame += 1 # note that we're incrementing the frame here so we need to recalculate similarity
            self.prev_similarity = self.calculate_similarity(self.motion[self.frame], self.env.get_state_desc()['joint_pos'])
        else:
            self.prev_similarity = curr_similarity

        return imitation_reward

    def calculate_similarity(self, ref_state_desc, curr_state_desc):
        ref, curr = set(ref_state_desc), set(curr_state_desc)

        cos_sim = []

        for name in ref.intersection(curr):
            for idx in range(len(ref_state_desc[name])):
                cos_sim += [ np.cos(np.subtract(ref_state_desc[name][idx], curr_state_desc[name][idx]))]

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

    for i in range(200):
        print('setting frame: ' + str(i))
        obs = wrapped_env.reset(project=False, frame=i)
        obs, rew, done, info = wrapped_env.step(env.action_space.sample(), project=False)

    env.close()

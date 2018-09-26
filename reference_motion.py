from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import math as m
from bvh import Bvh
import gym
import random as r
from retarget import MocapDataLoop, set_osim_joint_pos
from transforms import flatten, pad_zeros, quaternion_from_euler as q_from_e, quaternion_distance as q_dist


class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file):
        super(ReferenceMotionWrapper, self).__init__(env)

        self.tracked_positions = ['head','pelvis','toes_l','pros_foot_r']
        self.tracked_joints = ['ground_pelvis', 'hip_l', 'hip_r', 'knee_l', 'knee_r']

        self.CLOSE_ENOUGH = 0.1 * len(self.tracked_positions)

        self.ref_motion = MocapDataLoop(motion_file)
        self.ref_motion_it = iter(self.ref_motion)
        self.target = next(self.ref_motion_it)
        self.accumulated_reward = 0.

        env_obs = np.zeros([len(self.observation(list(env.observation_space.sample()))) + 2])
        self.observation_space = gym.spaces.Box(low=env_obs, high=env_obs, dtype=np.float32)

    def reset(self, project=True, **kwargs):
        observation = self.env.reset(project, **kwargs)

        self.target = self.ref_motion.reset()
        self.accumulated_reward = 0.

        return self.observation(observation)

    def reward(self):
        return np.exp(-self.end_effector_dist()) + 0.25 * np.exp(-self.joint_dist())

    def step(self, action, **kwargs):
        obs, task_reward, done, info = self.env.step(action, **kwargs)
        obs = self.observation(obs)

        frames_completed = 0

        # while the guy is close enough to the next frame, move to the next frame
        while self.end_effector_dist() < self.CLOSE_ENOUGH:
            self.accumulated_reward += self.reward()
            self.target = next(self.ref_motion_it)
            frames_completed += 1

        imitation_reward = self.reward()

        info['task_reward'] = task_reward
        info['imitation_reward'] = imitation_reward
        info['frames_completed'] = frames_completed

        return obs, imitation_reward + self.accumulated_reward, done, info

    def observation(self, obs):
        if isinstance(obs, dict):
            for k in filter(lambda k: k in self.tracked_positions, self.target['body_pos'].keys()): obs['target_' + k] = self.target['body_pos'][k]
            for k in filter(lambda k: k in self.tracked_joints, self.target['joint_pos'].keys()): obs['target_' + k] = self.target['joint_pos'][k]
        else: # it's a list
            obs += flatten(list([v for k, v in self.target['body_pos'].items() if k in self.tracked_positions]))
            obs += flatten(list([v for k, v in self.target['joint_pos'].items() if k in self.tracked_joints]))

        return obs

    def end_effector_dist(self):
        ref_pos = {k: v for k, v in self.target['body_pos'].items() if k in self.tracked_positions}
        curr_pos = self.env.get_state_desc()['body_pos']

        return np.sum([norm(np.array(ref_pos[name]) - curr_pos[name]) for name in set(ref_pos).intersection(set(curr_pos))])

    def joint_dist(self):
        ref_pos = {k: v for k, v in self.target['joint_pos'].items() if k in self.tracked_joints}
        curr_pos = self.env.get_state_desc()['joint_pos']

        return np.sum([q_dist(q_from_e(*pad_zeros(ref_pos[name])), q_from_e(*pad_zeros(curr_pos[name]))) for name in set(ref_pos).intersection(set(curr_pos))])


if __name__ == '__main__':
    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_keyframes.pkl')
    done = True

    for i in range(200):
        obs = wrapped_env.reset()
        set_osim_joint_pos(env, wrapped_env.target['joint_pos'])
        for j in range(50):
            obs, rew, done, info = wrapped_env.step(env.action_space.sample(), project=False)
            if done: continue

    env.close()

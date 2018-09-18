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
from transforms import flatten


class ReferenceMotionWrapper(gym.Wrapper):
    def __init__(self, env, motion_file, RSI=True):
        super(ReferenceMotionWrapper, self).__init__(env)

        self.CLOSE_ENOUGH = 1.0

        self.RSI = False
        self.ref_motion = MocapDataLoop(motion_file)
        self.ref_motion_it = iter(self.ref_motion)
        self.target = next(self.ref_motion_it)

        env_obs = np.zeros([env.observation_space.shape[0] + len(self.target['body_pos'].keys()) * 3 + 2,])
        self.observation_space = gym.spaces.Box(low=env_obs, high=env_obs, dtype=np.float32)

    def reset(self, project=True, **kwargs):
        observation = self.env.reset(project, **kwargs)

        self.target = self.ref_motion.reset(r.randint(0, len(self.ref_motion)) if self.RSI else 0)

        if self.RSI:
            set_osim_joint_pos(self.env, self.target['joint_pos'])
            self.target = next(self.ref_motion_it)

        return self.observation(observation)

    def step(self, action, **kwargs):
        obs, task_reward, done, info = self.env.step(action, **kwargs)
        obs = self.observation(obs)

        imitation_reward = np.exp(-self.dist_to_target())

        # while the guy is close enough to the next frame, move to the next frame
        while self.dist_to_target() < self.CLOSE_ENOUGH:
            self.target = next(self.ref_motion_it)

        info['task_reward'] = task_reward
        info['imitation_reward'] = imitation_reward
        info['target_frame'] = self.ref_motion.curr_frame

        return obs, imitation_reward, done, info

    def observation(self, obs):
        if isinstance(obs, dict):
            for k,v in self.target['body_pos'].items(): obs['target_' + k] = v
        else: # it's a list
            obs += flatten(list(self.target['body_pos'].values()))

        return obs

    def dist_to_target(self):
        ref_pos = {k: v for k, v in self.target['body_pos'].items() if not k in ['calcn_l','talus_l']}
        curr_pos = self.env.get_state_desc()['body_pos']

        return np.mean([norm(np.array(ref_pos[name]) - curr_pos[name]) for name in set(ref_pos).intersection(set(curr_pos))])


if __name__ == '__main__':
    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_keyframes.pkl')
    done = True

    for i in range(200):
        obs = wrapped_env.reset()
        for j in range(50):
            obs, rew, done, info = wrapped_env.step(env.action_space.sample(), project=False)
            if done: continue

    env.close()

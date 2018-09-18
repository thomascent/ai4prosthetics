import tensorflow as tf
import gym
import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import argparse
import numpy as np
from mpi4py import MPI
from distutils.util import strtobool
from baselines import logger
from YAPPO.ppo import PPO
from YAPPO.network import MlpPolicy, MlpCritic
from YAPPO.util import throttle, Saver
from datetime import datetime
import os
from reference_motion import ReferenceMotionWrapper


def visualise(pi, env):
    """Renders a pretrained policy behaving in an environment.

    Args:
        pi: A policy which implements an act method
        env: An environment which implements an openai gym interface

    """
    while 1:
        frame = score = 0
        obs = env.reset()

        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1

            if not done: continue

            print("score=%0.2f in %i frames" % (score, frame))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit running guy to the grader')
    parser.add_argument('--model', default='l2r_ref_motion_v4_0', type=str, help='the name under which the checkpoint file will be saved') 
    parser.add_argument('--rsi', type=lambda x:bool(strtobool(x)), default=False, help='use reference state initialisation')
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model)
    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_keyframes.pkl', RSI=args.rsi)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=wrapped_env.action_space.shape, observation_shape=wrapped_env.observation_space.shape, hid_size=64, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=wrapped_env.observation_space.shape, hid_size=64, num_hid_layers=3)

        saver = Saver(model_dir, sess)
        saver.try_restore()

        visualise(pi, wrapped_env)
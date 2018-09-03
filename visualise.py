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
from YAPPO.util import throttle
from datetime import datetime
import os
from aux_reward import AuxRewardWrapper


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

            # if not env.render("human"): return
            if not done: continue

            print("score=%0.2f in %i frames" % (score, frame))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit running guy to the grader')
    parser.add_argument('--model', default='l2r_standup', type=str, help='the name under which the checkpoint file will be saved')
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model)
    env = ProstheticsEnv(visualize=True)
    wrapped_env = AuxRewardWrapper(env, es=0.0)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=wrapped_env.action_space.shape, observation_shape=wrapped_env.observation_space.shape, hid_size=400, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=wrapped_env.observation_space.shape, hid_size=400, num_hid_layers=3)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Model not found!')
            raise NotImplementedError

        visualise(pi, wrapped_env)
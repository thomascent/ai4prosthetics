import tensorflow as tf
import gym
import opensim as osim
import argparse
import numpy as np
import os
from osim.env import ProstheticsEnv
from mpi4py import MPI
from distutils.util import strtobool
from baselines import logger
from YAPPO.ppo import PPO
from YAPPO.network import MlpPolicy, MlpCritic
from YAPPO.util import throttle, Saver
from datetime import datetime
from reference_motion import ReferenceMotionWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train our running guy!')
    parser.add_argument('--ntimesteps', default=10000000, type=int, help='number of timesteps to run on the environment')
    parser.add_argument('--model', default='l2r_ref_motion_v1', type=str, help='the name under which the checkpoint file will be saved') 
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model)
    log_dir = 'runs/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout'])

    env = ProstheticsEnv(visualize=False)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_loop.bvh', rsi=False)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=wrapped_env.action_space.shape, observation_shape=wrapped_env.observation_space.shape, hid_size=64, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=wrapped_env.observation_space.shape, hid_size=64, num_hid_layers=3)
        ppo = PPO(pi=pi, critic=critic, env=wrapped_env, timesteps_per_actorbatch=512)
        
        saver = Saver(model_dir, sess)
        saver.try_restore()
        callback = saver.save if MPI.COMM_WORLD.Get_rank() == 0 else None

        ppo.train(max_timesteps=args.ntimesteps, optimizer_stepsize=1e-4, user_callback=callback)

        env.close()

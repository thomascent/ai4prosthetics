import tensorflow as tf
import gym
import opensim as osim
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
from reference_motion import ReferenceMotionWrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train our running guy!')
    parser.add_argument('--ntimesteps', default=10000000, type=int, help='number of timesteps to run on the environment')
    parser.add_argument('--model', default='l2r_ref_motion_v0', type=str, help='the name under which the checkpoint file will be saved') 
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model)
    log_dir = 'runs/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = ProstheticsEnv(visualize=False)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_keyframes.bvh', rsi=False)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=(19,), observation_shape=(159,), hid_size=64, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=(159,), hid_size=64, num_hid_layers=3)
        ppo = PPO(pi=pi, critic=critic, env=wrapped_env, timesteps_per_actorbatch=512)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)

        if MPI.COMM_WORLD.Get_rank() == 0:
            @throttle(minutes=5)
            def save_model(*args):
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
            callback = save_model
        else:
            callback = None

        ppo.train(max_timesteps=args.ntimesteps, optimizer_stepsize=5e-5, user_callback=callback)

        env.close()

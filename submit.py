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


def desc_to_list(state_desc):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        if body_part in ["toes_r","talus_r"]:
            res += [0] * 9
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res


def submit(pi):
    remote_base = "http://grader.crowdai.org:1729"
    crowdai_token = "0dd7c22f5eb61cb4453b5a5b8e510656"

    client = Client(remote_base)
    observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")
    
    frame = score = 0

    while True:

        a = pi.act(desc_to_list(observation))

        [observation, reward, done, _] = client.env_step(a.tolist(), True)
        score += reward
        frame += 1

        if done:
            print("score=%0.2f in %i frames" % (score, frame))
            frame = score = 0

            observation = client.env_reset()
            if not observation:
                break

    client.submit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit running guy to the grader')
    parser.add_argument('--model', default='l2r_v0', type=str, help='the name under which the checkpoint file will be saved') 
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model)

    with tf.Session() as sess:
        pi = MlpPolicy(name='pi', action_shape=(19,), observation_shape=(158,), hid_size=64, num_hid_layers=3)
        critic = MlpCritic(name='critic', observation_shape=(158,), hid_size=64, num_hid_layers=3)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Model not found!')
            raise NotImplementedError

        submit(pi)
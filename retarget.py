from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
from bvh import Bvh
import gym
import random as r
import pickle as pkl
from transforms import matrix_from_euler, euler_from_matrix
import copy


class MocapData:
    def __init__(self, bvh_filename):
        with open(bvh_filename) as f:
            mocap = Bvh(f.read())

        MOCAP_TO_OSIM_HEIGHT = 0.94 / 17.6356
        MOCAP_TO_OSIM_TRANSLATION = 1.27441 # the relative position of the left toes in the first frame of mocap data

        self.nframes = mocap.nframes
        self.frame_time = mocap.frame_time

        self.joints = {}
        self.joints['hip_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['hip_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['knee_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['knee_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ankle_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftFoot',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ground_pelvis'] = np.array([mocap.frames_joint_channels('Pelvis',['Xrotation','Yrotation','Zrotation','Xposition','Yposition','Zposition'])]).squeeze()

        # now make a few tweaks to the reference motion to zero out some joints
        self.joints['ground_pelvis'][:,3:] = self.joints['ground_pelvis'][:,3:].dot(matrix_from_euler(0,-90, 0)) * MOCAP_TO_OSIM_HEIGHT
        self.joints['ground_pelvis'][:,3] += MOCAP_TO_OSIM_TRANSLATION
        self.joints['ground_pelvis'][:,:3] = self.joints['hip_l'][:,1:] = self.joints['hip_r'][:,1:] = 0

    def align_mocap_to_osim(self, eul_mocap, eul_align):
        # convert the mocap euler angle to a rotation matrix then apply the alignment
        R_mocap = matrix_from_euler(*eul_align).dot(matrix_from_euler(*eul_mocap))
        # apply the rotation defined about the mocap frame to the osim frame
        R_osim = matrix_from_euler(0, 90, 0).dot(R_mocap.dot(matrix_from_euler(0,-90, 0)))
        # convert back to euler angles
        eul_osim = euler_from_matrix(R_osim, units='rad')
        # and we're done (I still don't know why the order is reversed here though)
        return np.flip(eul_osim)

    def __iter__(self):
        for i in range(self.nframes):
            yield {k: v[i] for (k, v) in self.joints.items()}


class MocapDataLoop:
    def __init__(self, pkl_filename):

        with open(pkl_filename, 'rb') as f:
            self.states = pkl.load(f)

        self.curr_state = copy.deepcopy(self.states[0])
        self.curr_frame = 0

    def reset(self, frame_id):
        self.curr_state = copy.deepcopy(self.states[0])
        self.curr_frame = 0

        for i, _ in enumerate(self):
            if i == frame_id:
                return self.curr_state

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        while True:
            yield self.curr_state

            for pos_key, pos_val in self.curr_state.items():
                for joint_key in pos_val.keys():
                    self.curr_state[pos_key][joint_key] += np.subtract(self.states[self.curr_frame+1][pos_key][joint_key], self.states[self.curr_frame][pos_key][joint_key])

            self.curr_frame = (self.curr_frame + 1) % (len(self.states) - 1)


def set_osim_joint_pos(env, joint_pos):
    state = env.osim_model.get_state()

    for joint in env.osim_model.model.getJointSet():
        name = joint.getName()
        if name in joint_pos.keys():
            [joint.get_coordinates(i).setValue(state, joint_pos[name][i]) for i in range(len(joint_pos[name]))]

    env.osim_model.set_state(state)
    env.osim_model.model.equilibrateMuscles(env.osim_model.get_state())
    env.osim_model.state_desc_istep = None

    return env.get_state_desc()


if __name__ == '__main__':

    env = ProstheticsEnv(visualize=True)
    mocap_data = MocapData(bvh_filename='mocap_data/running_guy_keyframes.bvh')

    states = []

    for i, frame in enumerate(mocap_data):
        print('setting frame: ' + str(i))
        env.reset(project=False)
        state = set_osim_joint_pos(env, frame)
        states.append({k: state[k] for k in ['body_pos', 'joint_pos']})
        env.step(env.action_space.sample())

    with open('mocap_data/running_guy_keyframes.pkl', 'wb') as f:
        pkl.dump(states, f)

    loop = MocapDataLoop('mocap_data/running_guy_keyframes.pkl')

    for i, frame in enumerate(loop):
        if i == 20: break

        env.reset(project=False)
        state = set_osim_joint_pos(env, frame['joint_pos'])
        env.step(env.action_space.sample())

    env.close()

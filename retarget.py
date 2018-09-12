from osim.env import ProstheticsEnv
import opensim
import matplotlib.pyplot as plt
import numpy as np
import math as m
from bvh import Bvh
import gym
import random as r
import pickle as pkl


class ReferenceMotion:
    def __init__(self, filename):
        with open(filename) as f:
            mocap = Bvh(f.read())

        MOCAP_TO_OSIM_HEIGHT = 0.94 / 17.6356
        MOCAP_TO_OSIM_TRANSLATION = 1.27441 # the relative position of the left toes in the first frame of mocap data

        self.len = mocap.nframes
        self.frame_time = mocap.frame_time

        self.joints = {}
        self.joints['hip_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['hip_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['knee_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['knee_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ankle_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftFoot',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ground_pelvis'] = np.array([mocap.frames_joint_channels('Pelvis',['Xrotation','Yrotation','Zrotation','Xposition','Yposition','Zposition'])]).squeeze()

        # now make a few tweaks to the reference motion to zero out some joints
        self.joints['ground_pelvis'][:,3:] = self.joints['ground_pelvis'][:,3:].dot(self.matrix_from_euler(0,-90, 0)) * MOCAP_TO_OSIM_HEIGHT
        self.joints['ground_pelvis'][:,3] += MOCAP_TO_OSIM_TRANSLATION
        self.joints['ground_pelvis'][:,:3] = 0

        self.joints['hip_l'][:,1:] = 0
        self.joints['hip_r'][:,1:] = 0

        self.joints['delta_ground_pelvis'] = np.diff(np.vstack([self.joints['ground_pelvis'],self.joints['ground_pelvis'][0]]),axis=0)
        self.joints['delta_ground_pelvis'][-1][3] = self.joints['delta_ground_pelvis'][-2][3]

    def __iter__(self):
        frame = 0
        ground_pelvis = self.joints['ground_pelvis'][frame]

        while True:
            ret = {k: v[frame] for (k, v) in self.joints.items()}
            ret['ground_pelvis'] = ground_pelvis
            yield ret

            frame = (frame + 1) % self.len
            ground_pelvis += self.joints['delta_ground_pelvis'][frame]


    def align_mocap_to_osim(self, eul_mocap, eul_align):
        # convert the mocap euler angle to a rotation matrix then apply the alignment
        R_mocap = self.matrix_from_euler(*eul_align).dot(self.matrix_from_euler(*eul_mocap))
        # apply the rotation defined about the mocap frame to the osim frame
        R_osim = self.matrix_from_euler(0, 90, 0).dot(R_mocap.dot(self.matrix_from_euler(0,-90, 0)))
        # convert back to euler angles
        eul_osim = self.euler_from_matrix(R_osim, units='rad')
        # and we're done (I still don't know why the order is reversed here though)
        return [eul_osim[2], eul_osim[1], eul_osim[0]]

    def matrix_from_euler(self, phi, theta=0, psi=0, units='deg'):
        assert(units == 'deg' or units == 'rad')

        if units == 'deg':
            [phi, theta, psi] = [m.radians(i) for i in [phi, theta, psi]]

        return np.array([[m.cos(psi)*m.cos(theta), m.cos(psi)*m.sin(theta)*m.sin(phi)-m.cos(phi)*m.sin(psi), m.sin(psi)*m.sin(phi)+m.cos(psi)*m.cos(phi)*m.sin(theta)],
                        [m.cos(theta)*m.sin(psi), m.cos(psi)*m.cos(phi)+m.sin(psi)*m.sin(theta)*m.sin(phi), m.cos(phi)*m.sin(psi)*m.sin(theta)-m.cos(psi)*m.sin(phi)],
                        [-m.sin(theta), m.cos(theta)*m.sin(phi), m.cos(theta)*m.cos(phi)]])

    def euler_from_matrix(self, R, units='deg'):
        assert(units == 'deg' or units == 'rad')

        sy = m.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular:
            phi = m.atan2(R[2,1], R[2,2])
            theta = m.atan2(-R[2,0], sy)
            psi = m.atan2(R[1,0], R[0,0])
        else:
            phi = m.atan2(-R[1,2], R[1,1])
            theta = m.atan2(-R[2,0], sy)
            psi = 0

        if units == 'deg':
            [phi, theta, psi] = [m.degrees(i) for i in [phi, theta, psi]]

        return np.array([phi, theta, psi])


def set_state_desc(env, state_desc):
    state = env.osim_model.get_state()

    for joint in env.osim_model.model.getJointSet():
        name = joint.getName()
        if name in state_desc.keys():
            [joint.get_coordinates(i).setValue(state, state_desc[name][i]) for i in range(len(state_desc[name]))]

    env.osim_model.set_state(state)
    env.osim_model.model.equilibrateMuscles(env.osim_model.get_state())
    env.osim_model.state_desc_istep = None


if __name__ == '__main__':

    filename = 'mocap_data/running_guy.bvh'
    env = ProstheticsEnv(visualize=True)
    ref_motion = ReferenceMotion(filename=filename)

    states = []

    env.reset(project=False)
    env.step(env.action_space.sample())

    for i, frame in enumerate(ref_motion):
        if i > 1000:
            break

        print('setting frame: ' + str(i))
        env.reset(project=False)
        set_state_desc(env, frame)
        states.append(env.get_state_desc())
        env.step(env.action_space.sample())

    env.close()

    with open(filename + '.pkl', 'wb') as f:
        pkl.dump(states, f)

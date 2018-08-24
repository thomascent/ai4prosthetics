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

        self.LEAN = -20.
        self.len = mocap.nframes

        self.joints = {}
        self.joints['hip_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['hip_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightUpLeg',['Xrotation','Yrotation','Zrotation'])])
        self.joints['knee_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['knee_r'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,-21]) for eul_mocap in mocap.frames_joint_channels('RightLeg',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ankle_l'] = np.array([self.align_mocap_to_osim(eul_mocap,[0,0,21]) for eul_mocap in mocap.frames_joint_channels('LeftFoot',['Xrotation','Yrotation','Zrotation'])])[:,:1]
        self.joints['ground_pelvis'] = np.zeros([self.len,3])

        # now make a few tweaks to the reference motion to zero out some joints and add a lean so the guy can accelerate
        self.joints['hip_l'][:,1:] = self.joints['hip_r'][:,1:] = 0
        self.joints['ground_pelvis'][:,0] = m.radians(self.LEAN)
        self.joints['hip_l'][:,0] += m.radians(-self.LEAN)
        self.joints['hip_r'][:,0] += m.radians(-self.LEAN)

        # @todo: scale imitation reward of each joint to represent their relative importance (ie pelvis is greater than ankle)... I guess the most reasonable way to do this is to take the jacobian of the end site wrt the joint angle
        # so now I need the jacobian of the joint. I guess I can do that analytically, it probably needs to be static anyway so that the reward doesn't change too much mid-episode

        self.joints['phase'] = [(i) / self.len for i in range(self.len)] # I'm pretty sure I don't need this

    def make_frame(self):
        frame = {}
        frame['hip_l'] = np.zeros([3])
        frame['hip_r'] = np.zeros([3])
        frame['knee_l'] = np.zeros([1])
        frame['knee_r'] = np.zeros([1])
        frame['ankle_l'] = np.zeros([1])
        frame['ground_pelvis'] = np.zeros([3])
        frame['phase'] = 0
        return frame

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

        self.start_state = self.motion.make_frame()
        self.start_state['ankle_l'][0] = -m.radians(self.motion.LEAN)
        self.start_state['ground_pelvis'][0] = m.radians(self.motion.LEAN)


    def reset(self, project=True, frame=None, **kwargs):
        observation = self.env.reset(project, **kwargs)
        
        if self.rsi or frame:
            self.frame = r.randint(0,len(self.motion)-1) if frame is None else frame
            self.set_state_desc(self.motion[self.frame])

            if project:
                observation = self.env.get_observation()
            else:
                observation = self.env.get_state_desc()
        else:
            self.set_state_desc(self.start_state)

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
            cos_sim += [*np.cos(ref_state_desc[name] - curr_state_desc[name][:len(ref_state_desc[name])])]

        return np.mean(cos_sim)

    def set_state_desc(self, state_desc):
        state = self.env.osim_model.get_state()

        for joint in self.env.osim_model.model.getJointSet():
            name = joint.getName()
            if name in state_desc.keys():
                [joint.get_coordinates(i).setValue(state, state_desc[name][i]) for i in range(len(state_desc[name]))]

        self.env.osim_model.set_state(state)
        self.env.osim_model.model.equilibrateMuscles(self.env.osim_model.get_state())
        self.env.osim_model.state_desc_istep = None

if __name__ == '__main__':

    env = ProstheticsEnv(visualize=True)
    wrapped_env = ReferenceMotionWrapper(env, motion_file='mocap_data/running_guy_keyframes.bvh')
    done = True

    for i in range(200):
        print('setting frame: ' + str(i))
        obs = wrapped_env.reset(project=False, frame=i)
        obs, rew, done, info = wrapped_env.step(env.action_space.sample(), project=False)

    env.close()

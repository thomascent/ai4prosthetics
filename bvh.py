import re
import math as m
import numpy as np

class BvhNode:
    def __init__(self, hierarchy, channels_it, parent=None):

        class Zero:
            def __getitem__(self,t):
                return 0.

        self.parent = parent
        self.children = []

        self.channels = {x: Zero() for x in ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']}
        self.offset = np.zeros([3])

        self.name = hierarchy['name']
        self.type = hierarchy['type']

        if 'OFFSET' in hierarchy.keys():
            self.offset = np.array(hierarchy['OFFSET']).astype(np.float)

        if 'CHANNELS' in hierarchy.keys():
            for channel in hierarchy['CHANNELS'][1:]:
                self.channels[channel] = next(channels_it)

        for joint in hierarchy['joints']:
            self.children.append(BvhNode(joint, channels_it, self))

    def get_relative_pose(self, t):
        ret = np.eye(4)
        ret[:3,:3] = self.rot(*[m.radians(self.channels[ch][t]) for ch in ['Xrotation', 'Yrotation', 'Zrotation']])
        ret[:3,3] = self.offset + np.array([self.channels[ch][t] for ch in ['Xposition', 'Yposition', 'Zposition']])

        return ret

    def lookup_transform(self, t, dest, H=np.eye(4)):

        H = H.dot(self.get_relative_pose(t))

        if self.name == dest:
            return H
        else:
            for child in self.children:
                H_child = child.lookup_transform(t, dest, H)
                if H_child is not None:
                    return H_child

        return None

    @staticmethod
    def rot(x, y, z):
        return np.array([[m.cos(y)*m.cos(z)+m.sin(y)*m.sin(x)*m.sin(z), m.cos(z)*m.sin(y)*m.sin(x)-m.cos(y)*m.sin(z), m.cos(x)*m.sin(y)],
                        [m.cos(x)*m.sin(z), m.cos(x)*m.cos(z), -m.sin(x)],
                        [m.cos(y)*m.sin(x)*m.sin(z)-m.cos(z)*m.sin(y), m.cos(y)*m.cos(z)*m.sin(x)+m.sin(y)*m.sin(z), m.cos(y)*m.cos(x)]])

class Bvh:
    def __init__(self, data):
        hierarchy = {'joints': []}
        stack = [hierarchy]
        frames = []
        it = iter(self.tokenise(data))

        for line in it:
            if line[0] == 'HIERARCHY':
                for line in it:
                    if line[0] in ['ROOT', 'JOINT', 'End']:
                        sub = {'joints': [], 'name': line[1], 'type': line[0] }
                        stack[-1]['joints'].append(sub)
                        stack.append(sub)
                    elif line[0] == '}':
                        stack.pop()
                        if len(stack) == 1: break
                    elif line[0] in ['OFFSET', 'CHANNELS']:
                        stack[-1][line[0]] = line[1:]

            if line[0] == 'MOTION':
                self.no_of_frames = next(it)[-1]
                self.frame_time = next(it)[-1]

                for line in it:
                    frames.append(line)

        self.root = BvhNode(hierarchy['joints'][0], iter(np.array(frames).astype(np.float).T))

    def tokenise(self, data):
        accumulator = ''
        for char in data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                yield re.split('\\s+', re.sub(':','',accumulator.strip()))
                accumulator = ''

        raise StopIteration

    def get_relative_pose(self, src, dest, time):
        pass

if __name__ == '__main__':
    with open('running_guy.bvh','r') as f:
        b = Bvh(f.read())

    print(b.root.lookup_transform(0, 'Site'))
import numpy as np
import math as m


def matrix_from_euler(phi, theta=0, psi=0, units='deg'):
    assert(units == 'deg' or units == 'rad')

    if units == 'deg':
        [phi, theta, psi] = [m.radians(i) for i in [phi, theta, psi]]

    return np.array([[m.cos(psi)*m.cos(theta), m.cos(psi)*m.sin(theta)*m.sin(phi)-m.cos(phi)*m.sin(psi), m.sin(psi)*m.sin(phi)+m.cos(psi)*m.cos(phi)*m.sin(theta)],
                    [m.cos(theta)*m.sin(psi), m.cos(psi)*m.cos(phi)+m.sin(psi)*m.sin(theta)*m.sin(phi), m.cos(phi)*m.sin(psi)*m.sin(theta)-m.cos(psi)*m.sin(phi)],
                    [-m.sin(theta), m.cos(theta)*m.sin(phi), m.cos(theta)*m.cos(phi)]])


def euler_from_matrix(R, units='deg'):
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


def quaternion_from_euler(phi, theta, psi, units='deg'):
    assert(units == 'deg' or units == 'rad')

    if units == 'deg':
        phi, theta, psi = (m.radians(i) for i in [phi, theta, psi])

    i = 0
    j = 1
    k = 2

    phi /= 2.0
    theta /= 2.0
    psi /= 2.0
    ci = m.cos(phi)
    si = m.sin(phi)
    cj = m.cos(theta)
    sj = m.sin(theta)
    ck = m.cos(psi)
    sk = m.sin(psi)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)

    quaternion[i] = cj*sc - sj*cs
    quaternion[j] = cj*ss + sj*cc
    quaternion[k] = cj*cs - sj*sc
    quaternion[3] = cj*cc + sj*ss

    return quaternion


def quaternion_from_matrix(matrix):
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M) + 1.0
    if t > 1.0:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + 1.0
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / m.sqrt(t)
    return q


def quaternion_distance(q1, q2):
    return m.acos(2 * q1.dot(q2)**2 - 1)

def flatten(lol):
    return [i for l in lol for i in l]


def pad_zeros(pad, size=3):
    ret = np.zeros(shape=[size])
    ret[:min(size, len(pad))] = pad[:min(size, len(pad))]
    return ret


if __name__ == '__main__':

    R = matrix_from_euler(15,20,3)
    q = quaternion_from_euler(15,20,3)
    q_R = quaternion_from_matrix(R)

    print(q, q_R)

    q1 = quaternion_from_euler(0,0,90)
    q2 = quaternion_from_euler(0,0,0)

    print(m.degrees(quaternion_distance(q1,q2)))

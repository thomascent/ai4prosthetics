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

# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: State.py
@Description: Records current A/C state.
@Package dependency:
"""
import numpy as np
import CrossPlatformDev


class DragModel(object):
    """Defines the drag model used for quadcopter kinematics simulation.
    Effective C_d values estimated from https://journals.sagepub.com/doi/figure/10.1177/1756829320923565?,
    2kg drone achieved cruise speed of ~20m/s at 40-45 deg pitch down attitude. Assume slip angle does not
    significantly contribute to C_d. Refer to get_Cd mtd for more details on C_d estimation.
    """
    def __init__(self, AircraftType, prop_diameter=9, disabled=False):
        self.disabled = disabled
        self.AircraftType = AircraftType
        if self.AircraftType.prop_diameter:
            self.prop_diameter = self.AircraftType.prop_diameter
        elif prop_diameter:
            self.prop_diameter = prop_diameter
            AircraftType.prop_diameter = prop_diameter
            print('Changing A/C prop diameter to ', string(prop_diameter), '. Please check if base A/C type class has '
                                                                           'correct parameters initialized.')

        self.scaling = (self.prop_diameter / 9) ** 2

    def get_Cd(self, alpha, beta):
        """Assume Cd is linear in size (cross-sectional area), and a sinusoidal function in alpha (AoA)
        due to high symmetry of multirotor config. For now, assume sideslip angle beta to not factor into C_d
        determination. Note: alpha, beta are in radians. Assume C_d is positive definite."""
        if self.disabled:
            return 0
        else:
            #return 19.62 * self.scaling*(0.035+0.017*(np.sin(alpha))**2) #<--- this works but drag is wayyyy too high
            return self.scaling * (0.035 + 0.017 * (np.sin(alpha)) ** 2)

    def get_angles(self, air_vel, rpy):
        """Gets air_vel using ground frame orientation, and aircraft RPY relative to ground. """
        airspd = np.linalg.norm(air_vel)
        if airspd == 0:
            return np.zeros(3)
        else:
            x, y, z = rpy_trans(air_vel, rpy)
            if z == 0:
                alpha = np.pi / 2
            else:
                alpha = np.pi / 2 - np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
            if x == 0:
                beta = np.pi / 2
            else:
                beta = np.arctan(y / x)
        if alpha > np.pi/2:
            alpha = np.pi - alpha
        return np.array([alpha, beta])

    def get_drag(self, air_vel, rpy):
        """Estimates drag from effective C_d and air velocity of aircraft in the ground frame."""
        airspd = np.linalg.norm(air_vel)
        if airspd == 0:
            return np.zeros(3)
        else:
            x, y, z = rpy_trans(air_vel, rpy)
            if z == 0:
                alpha = np.pi/2
            else:
                alpha = np.pi / 2 - np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
            if x == 0:
                beta = np.pi/2
            else:
                beta = np.arctan(y / x)
            Cd = self.get_Cd(alpha, beta)
            return -airspd*air_vel*Cd


def r_x(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    rx = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
    return rx


def r_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    ry = np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])
    return ry


def r_z(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    rz = np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])
    return rz


def rpy(phi, theta, psi):
    ryrx = np.matmul(r_y(theta), r_x(psi))
    rzryrx = np.matmul(r_z(phi), ryrx)
    return rzryrx


def rpy_inv(phi, theta, psi):
    return np.linalg.inv(rpy(phi, theta, psi))


def rpy_trans(air_vel, rpy):
    r, p, y = rpy
    return np.matmul(rpy_inv(r, p, y), air_vel)
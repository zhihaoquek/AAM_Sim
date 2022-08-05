# -*- coding: utf-8 -*-
"""
@Time    : 09/15/2021 9:41 AM
@Author  : Wei Dai
@FileName: Aircraft.py
@Description:
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print


class AircraftType(object):
    def __init__(self,
                 model_file=None, max_cruise_spd=5, prop_diameter=9, mass=1.2):

        if not model_file:
            # generate basic multirotor model
            self.mass = mass

            # self.max_thrust = 200  # not in use at the moment... (deprecated)
            self.max_thrust = 3 * self.mass * 9.81
            # self.max_roll_pitch = np.pi / 4 # Original value (deprecated)
            # self.max_roll_pitch = np.pi / 2.2  # Previous Best
            self.max_roll_pitch = np.pi / 2.05

            self.max_cruise_speed = max_cruise_spd  # default was 20, now changed to 5
            self.max_ascend_speed = 2  # default was 5
            self.max_descend_speed = 3  # default was 3
            self.max_xy_torque = 5
            self.max_z_torque = 5

            self.prop_diameter = prop_diameter
            self.type = 'Multirotor'
        else:  # TODO: import drone type information from model_file, a .xml file
            pass

# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: Sensors.py
@Description: Package for modelling various types of sensor errors. Used for modelling NSE, airspeed sensors, etc.
@Package dependency:
"""

import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent

# NOTE: No intention to model air velocity error for now due to highly integrated
# nature of controller/sensor architecture.


class NavUpdate(Agent):
    def __init__(self, update_rate, start_time, err=np.zeros(3), phase_delay=0):
        super().__init__(update_rate, start_time, phase_delay)
        self.err = err

    def update_error(self, time):
        if super().check_time(time):
            self.err = np.zeros(3)
        return self.err


class GPSPosNavUpdate(NavUpdate):
    def __init__(self, update_rate, start_time,
                 x_auto, y_auto, z_auto,
                 x_sigma, y_sigma, z_sigma,
                 x_mean=0, y_mean=0, z_mean=0,
                 phase_delay=None):
        self.auto = np.array([x_auto, y_auto, z_auto])
        self.auto_sq = self.auto * self.auto
        self.x_sigma, self.y_sigma, self.z_sigma = x_sigma, y_sigma, z_sigma
        self.x_mean, self.y_mean, self.z_mean = x_mean, y_mean, z_mean
        err = np.array([np.random.normal(x_mean, x_sigma),
                        np.random.normal(y_mean, y_sigma),
                        np.random.normal(z_mean, z_sigma)])
        if isinstance(phase_delay, type(None)):
            super().__init__(update_rate, start_time, err, phase_delay=np.random.uniform(0, 1/update_rate))
        else:
            super().__init__(update_rate, start_time, err, phase_delay=phase_delay)

    def update_error(self, time):
        if super().check_time(time):
            self.err = self.auto * self.err + np.array([np.random.normal(self.x_mean,
                                                                         self.x_sigma * np.sqrt(1-self.auto_sq[0])),
                                                        np.random.normal(self.y_mean,
                                                                         self.y_sigma * np.sqrt(1-self.auto_sq[1])),
                                                        np.random.normal(self.z_mean,
                                                                         self.z_sigma * np.sqrt(1-self.auto_sq[2]))
                                                        ])
        return self.err

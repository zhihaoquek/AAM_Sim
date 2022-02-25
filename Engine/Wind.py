# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: Wind.py
@Description: Records current A/C state.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent


class WindField(Agent):
    def __init__(self, update_rate, start_time,
                 auto_x=0, auto_y=0, auto_z=0,
                 std_x=0, std_y=0, std_z=0,
                 mean_x=0, mean_y=0, mean_z=0
                 ):
        super().__init__(update_rate, start_time)
        self.auto = np.array([auto_x, auto_y, auto_z])
        self.std_x, self.std_y, self.std_z = std_x, std_y, std_z
        self.mean_x, self.mean_y, self.mean_z = mean_x, mean_y, mean_z
        self.auto_sq = self.auto * self.auto
        self.wind_spd = np.zeros(3)
        self.last_wind_spd = np.array([np.random.normal(mean_x, std_x),
                                       np.random.normal(mean_y, std_y),
                                       np.random.normal(mean_z, std_z)
                                       ])

    def get_windspd(self, time, pos):
        if self.check_time(time):
            # self.wind_spd = np.zeros(3)
            self.last_wind_spd = self.wind_spd
            self.wind_spd = self.auto * self.last_wind_spd + np.array([np.random.normal(self.mean_x,
                                                                                        self.std_x
                                                                                        * np.sqrt(1 - self.auto_sq[0])),
                                                                       np.random.normal(self.mean_y,
                                                                                        self.std_x
                                                                                        * np.sqrt(1 - self.auto_sq[1])),
                                                                       np.random.normal(self.mean_z,
                                                                                        self.std_z
                                                                                        * np.sqrt(1 - self.auto_sq[2]))
                                                                       ])
        return self.wind_spd
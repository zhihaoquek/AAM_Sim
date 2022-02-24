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
                 std_x=0, std_y=0, std_z=0,
                 mean_x=0, mean_y=0, mean_z=0,
                 auto_x=0, auto_y=0, auto_z=0):
        super().__init__(update_rate, start_time)
        self.wind_spd = np.zeros(3)

    def get_windspd(self, time, pos):
        if self.check_time(time):
            self.wind_spd = np.zeros(3)
        return self.wind_spd
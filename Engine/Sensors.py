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
    def __init__(self, update_rate, start_time, err=np.zeros(3)):
        super().__init__(update_rate, start_time)
        self.err = err

    def update_error(self, time):
        if super().check_time(time):
            self.err = np.zeros(3)
        return self.err